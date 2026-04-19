"""
build_az_graph.py
=================
Arizona-representative graph builder for IEEE 123-bus feeder.
Multi-year time series (2019–2023) with all features integrated.

WORKFLOW:
  Step 0  Verify DSS files
  Step 1  Parse DSS files → nodes_df, edges_df
  Step 2  Classify buses (substation / solar / EV / storage / regulator)
  Step 3  For each year: generate synthetic AZ time series (8760 h × 128 buses)
  Step 4  For each year: fetch real NOAA Phoenix weather → merge into temp_f
  Step 5  For each year: fetch EIA-930 AZPS bulk demand (real API or synthetic)
  Step 6  For each year: add EV scenarios, battery storage, heat dome flags
  Step 7  Concatenate all years → single timeseries_df
  Step 8  Build PyTorch Geometric Data object
  Step 9  Save outputs

RUN:
  pip install pandas numpy torch torch-geometric requests pyarrow
  python build_az_graph.py

OUTPUT FILES:
  az_feeder_nodes.csv              -- one row per bus, static features
  az_feeder_edges.csv              -- directed edges with electrical properties
  az_feeder_timeseries.parquet     -- all years combined, ~4.5M rows
  az_feeder_graph.pt               -- PyTorch Geometric Data object

TRAIN / VAL / TEST SPLIT (by year — no leakage):
  Train : 2019, 2021, 2022
  Val   : 2023 Jan–Jun
  Test  : 2023 Jul–Dec  (includes heat dome as held-out stress period)
  (2020 included but flagged with is_covid_year=1 for the model to handle)
"""

import re
import math
import warnings
import requests
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────

DSS_DIR     = Path(r".\OpenDSS\Distrib\IEEETestCases\123Bus")
MASTER_FILE = DSS_DIR / "IEEE123Master.dss"
OUT_DIR     = Path(r".\data")

# API keys — both optional, scripts fall back to calibrated synthetic data
NOAA_TOKEN  = "AmaawOYAtZtlcmzfMzfOeHkwqnUlDexY"
EIA_KEY     = ""          # get free at eia.gov/opendata

# Years to build — NSRDB only goes to 2023, so cap there
SIM_YEARS   = [2019, 2020, 2021, 2022, 2023]

# Arizona load scaling (IEEE feeder ~3.5 MVA → APS feeder ~10 MVA)
LOAD_SCALE  = 3.0

# EV adoption assumptions (NREL EVI-DiST Maricopa County)
EV_ADOPTION = {
    "2024_baseline":   0.08,
    "2030_projection": 0.28,
    "2035_aggressive": 0.45,
}
EV_MULT_2030 = EV_ADOPTION["2030_projection"] / EV_ADOPTION["2024_baseline"]  # 3.5×
EV_MULT_2035 = EV_ADOPTION["2035_aggressive"] / EV_ADOPTION["2024_baseline"]  # 5.6×

# Battery storage specs (Tesla Powerwall 2 representative)
STORAGE_KW_RATE    = 5.0
STORAGE_EFFICIENCY = 0.90

# Heat dome windows per year (derived from NOAA Phoenix heat alert records)
# Flag is_heatdome=1 during these windows; heat multiplier is still
# driven by the actual temperature value, not just the flag.
HEATDOME_WINDOWS = {
    2019: ("2019-06-10", "2019-06-14"),
    2020: ("2020-08-09", "2020-08-19"),
    2021: ("2021-06-12", "2021-06-20"),
    2022: ("2022-07-11", "2022-07-22"),
    2023: ("2023-06-30", "2023-08-04"),   # 31 consecutive days ≥110°F
}

# ── STEP 0: VERIFY FILES ───────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 0 — Verifying DSS files")
print("="*60)

for f in ["IEEE123Master.dss", "IEEELineCodes.DSS", "IEEE123Loads.DSS",
          "BusCoords.dat", "IEEE123Regulators.DSS"]:
    path = DSS_DIR / f
    exists = path.exists() or any(True for _ in DSS_DIR.glob(f"*{f.split('.')[0]}*"))
    print(f"  {'✓' if exists else '✗'} {f}")


# ── STEP 1: PARSE DSS FILES ───────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 1 — Parsing DSS files")
print("="*60)

def parse_buscoords(filepath):
    rows = []
    coord_path = next((f for f in filepath.parent.iterdir()
                       if f.name.lower() == "buscoords.dat"), None)
    if coord_path is None:
        print("  ! BusCoords.dat not found — using zero coordinates")
        return pd.DataFrame(columns=["bus_name", "x", "y"])
    with open(coord_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                rows.append({"bus_name": parts[0].lower(),
                             "x": float(parts[1]), "y": float(parts[2])})
    df = pd.DataFrame(rows)
    print(f"  Parsed {len(df)} bus coordinates")
    return df


def parse_loads(filepath):
    rows = []
    loads_path = next((f for f in filepath.parent.iterdir()
                       if f.name.lower() == "ieee123loads.dss"), None)
    if loads_path is None:
        print("  ! IEEE123Loads.DSS not found")
        return pd.DataFrame()
    with open(loads_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("!") or not line.lower().startswith("new load"):
                continue
            nm  = re.search(r"new\s+load\.(\S+)", line, re.I)
            bus = re.search(r"bus1=(\S+)",         line, re.I)
            kw  = re.search(r"kw=([\d.]+)",        line, re.I)
            kvar= re.search(r"kvar=([\d.]+)",      line, re.I)
            ph  = re.search(r"phases=(\d+)",       line, re.I)
            mod = re.search(r"model=(\d+)",        line, re.I)
            if nm and bus:
                rows.append({
                    "load_name": nm.group(1),
                    "bus_name":  bus.group(1).split(".")[0].lower(),
                    "kw":   float(kw.group(1))   if kw   else 0.0,
                    "kvar": float(kvar.group(1)) if kvar else 0.0,
                    "phases": int(ph.group(1))   if ph   else 1,
                    "model":  int(mod.group(1))  if mod  else 1,
                })
    df = pd.DataFrame(rows)
    print(f"  Parsed {len(df)} loads")
    return df


def parse_linecodes(filepath):
    codes = {}
    lc_path = next((f for f in filepath.parent.iterdir()
                    if f.name.lower() == "ieeelinecodes.dss"), None)
    if lc_path is None:
        return codes
    with open(lc_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("!") or not line.lower().startswith("new linecode"):
                continue
            nm = re.search(r"new\s+linecode\.(\S+)", line, re.I)
            r1 = re.search(r"\bR1=([\d.]+)", line, re.I)
            x1 = re.search(r"\bX1=([\d.]+)", line, re.I)
            r0 = re.search(r"\bR0=([\d.]+)", line, re.I)
            x0 = re.search(r"\bX0=([\d.]+)", line, re.I)
            if nm:
                codes[nm.group(1).lower()] = {
                    "r1": float(r1.group(1)) if r1 else 0.1,
                    "x1": float(x1.group(1)) if x1 else 0.1,
                    "r0": float(r0.group(1)) if r0 else 0.2,
                    "x0": float(x0.group(1)) if x0 else 0.2,
                }
    print(f"  Parsed {len(codes)} line codes")
    return codes


def parse_lines(master_path, linecodes):
    rows = []
    with open(master_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            if not re.match(r"new\s+line\.", line, re.I):
                continue
            nm   = re.search(r"new\s+line\.(\S+)",    line, re.I)
            bus1 = re.search(r"bus1=(\S+)",            line, re.I)
            bus2 = re.search(r"bus2=(\S+)",            line, re.I)
            ln   = re.search(r"\blength=([\d.e+-]+)",  line, re.I)
            ph   = re.search(r"phases=(\d+)",          line, re.I)
            lc   = re.search(r"linecode=(\S+)",        line, re.I)
            r1m  = re.search(r"\br1=([\d.e+-]+)",      line, re.I)
            x1m  = re.search(r"\bx1=([\d.e+-]+)",      line, re.I)
            na   = re.search(r"normamps=([\d.]+)",      line, re.I)
            if not (nm and bus1 and bus2):
                continue
            from_bus = bus1.group(1).split(".")[0].lower()
            to_bus   = bus2.group(1).split(".")[0].lower()
            length   = float(ln.group(1)) if ln else 0.001
            phases   = int(ph.group(1))   if ph else 3
            if lc:
                code = lc.group(1).lower()
                lc_data = linecodes.get(code, {"r1": 0.1, "x1": 0.1})
                r1, x1 = lc_data["r1"], lc_data["x1"]
            else:
                r1 = float(r1m.group(1)) if r1m else 0.001
                x1 = float(x1m.group(1)) if x1m else 0.001
            is_switch = (r1 <= 0.01 and length <= 0.01)
            norm_amps = float(na.group(1)) if na else 400.0
            rows.append({
                "line_name": nm.group(1), "from_bus": from_bus, "to_bus": to_bus,
                "length_mi": length, "phases": phases, "r1": r1, "x1": x1,
                "norm_amps": norm_amps, "is_switch": is_switch,
            })
    df = pd.DataFrame(rows)
    print(f"  Parsed {len(df)} lines ({df['is_switch'].sum()} switches)")
    return df


coords_df = parse_buscoords(MASTER_FILE)
loads_df  = parse_loads(MASTER_FILE)
linecodes = parse_linecodes(MASTER_FILE)
lines_df  = parse_lines(MASTER_FILE, linecodes)


# ── STEP 2: BUILD NODES AND EDGES ─────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 2 — Building nodes_df and edges_df")
print("="*60)

all_buses = sorted(set(lines_df["from_bus"]) | set(lines_df["to_bus"]) |
                   set(loads_df["bus_name"] if len(loads_df) else []) |
                   set(coords_df["bus_name"] if len(coords_df) else []))
bus_index = {name: i for i, name in enumerate(all_buses)}
n_buses   = len(all_buses)
print(f"  Total buses: {n_buses}")

coord_lookup = dict(zip(coords_df["bus_name"],
                        zip(coords_df["x"], coords_df["y"]))) if len(coords_df) else {}

if len(loads_df):
    load_agg = loads_df.groupby("bus_name")[["kw", "kvar"]].sum().reset_index()
    load_agg[["kw", "kvar"]] *= LOAD_SCALE
    load_lookup_kw   = dict(zip(load_agg["bus_name"], load_agg["kw"]))
    load_lookup_kvar = dict(zip(load_agg["bus_name"], load_agg["kvar"]))
else:
    load_lookup_kw = load_lookup_kvar = {}

load_bus_names = set(loads_df["bus_name"]) if len(loads_df) else set()

if load_lookup_kw:
    sorted_load_buses = sorted(load_lookup_kw.items(), key=lambda x: x[1], reverse=True)
    heavy_buses  = [b for b, _ in sorted_load_buses[:25]]
    medium_buses = [b for b, _ in sorted_load_buses[25:60]]
    light_buses  = [b for b, _ in sorted_load_buses[60:]]
else:
    heavy_buses = medium_buses = light_buses = []

solar_candidates = medium_buses + light_buses
solar_buses      = set(solar_candidates[:17]) if len(solar_candidates) >= 17 else set(solar_candidates)

ev_candidates = [b for b in medium_buses if b not in solar_buses]
ev_buses      = set(ev_candidates[:8]) if len(ev_candidates) >= 8 else set(ev_candidates)

# Storage buses: 5 of the 17 solar buses (higher-income, paired Powerwall homes)
# Pick the 5 solar buses with the highest base load — wealthier = more storage
solar_sorted_by_load = sorted(solar_buses, key=lambda b: load_lookup_kw.get(b, 0), reverse=True)
storage_buses = set(solar_sorted_by_load[:5])

print(f"  Solar buses:   {len(solar_buses)}")
print(f"  EV buses:      {len(ev_buses)}")
print(f"  Storage buses: {len(storage_buses)} (subset of solar)")

nodes_rows = []
for bus in all_buses:
    x, y   = coord_lookup.get(bus, (0.0, 0.0))
    kw     = load_lookup_kw.get(bus, 0.0)
    kvar   = load_lookup_kvar.get(bus, 0.0)
    kv_base = 0.48 if bus == "610" else 4.16
    nodes_rows.append({
        "bus_id":           bus_index[bus],
        "bus_name":         bus,
        "x":                x,
        "y":                y,
        "kv_base":          kv_base,
        "load_kw":          kw,
        "load_kvar":        kvar,
        "is_substation":    int(bus == "150"),
        "is_reg_secondary": int(bus.endswith("r")),
        "is_solar_bus":     int(bus in solar_buses),
        "is_ev_bus":        int(bus in ev_buses),
        "is_storage_bus":   int(bus in storage_buses),
        "has_load":         int(bus in load_bus_names),
    })
nodes_df = pd.DataFrame(nodes_rows)
print(f"  nodes_df: {nodes_df.shape}")

edge_rows = []
for _, row in lines_df.iterrows():
    fb, tb = row["from_bus"], row["to_bus"]
    if fb not in bus_index or tb not in bus_index:
        continue
    z_mag   = math.sqrt(row["r1"]**2 + row["x1"]**2)
    z_total = z_mag * max(row["length_mi"], 1e-6)
    weight  = 1.0 / (z_total + 1e-6)
    base = {"from_bus": fb, "to_bus": tb,
            "from_idx": bus_index[fb], "to_idx": bus_index[tb],
            "line_name": row["line_name"], "length_mi": row["length_mi"],
            "phases": row["phases"], "r1": row["r1"], "x1": row["x1"],
            "z_mag": z_mag, "z_total": z_total, "weight": weight,
            "norm_amps": row["norm_amps"], "is_switch": int(row["is_switch"])}
    edge_rows.append({**base, "direction": "forward"})
    rev = {**base, "from_bus": tb, "to_bus": fb,
           "from_idx": bus_index[tb], "to_idx": bus_index[fb],
           "direction": "reverse"}
    edge_rows.append(rev)
edges_df = pd.DataFrame(edge_rows)
print(f"  edges_df: {edges_df.shape}  ({len(edges_df)//2} physical lines × 2 directions)")


# ── HELPER FUNCTIONS — WEATHER, SOLAR, LOAD ───────────────────────────────────

def synthetic_phoenix_temperature(timestamps):
    """Hourly Phoenix dry-bulb temperature (°F). Calibrated to Sky Harbor normals."""
    temps = np.zeros(len(timestamps))
    for i, ts in enumerate(timestamps):
        doy, hour = ts.day_of_year, ts.hour
        annual_mean = 75.0
        annual_amp  = 27.0
        daily_mean  = annual_mean - annual_amp * math.cos(2 * math.pi * (doy - 15) / 365)
        daily_amp   = 18.0 if daily_mean > 90 else 14.0
        temp        = daily_mean - daily_amp * math.cos(2 * math.pi * (hour - 5) / 24)
        np.random.seed(i % 1000)
        temp += np.random.normal(0, 1.5)
        temps[i] = temp
    return temps


def synthetic_ghi(timestamps):
    """Hourly GHI (W/m²) for Phoenix lat=33.45°N with monsoon cloud cover."""
    lat_rad = math.radians(33.45)
    ghi = np.zeros(len(timestamps))
    for i, ts in enumerate(timestamps):
        doy  = ts.day_of_year
        hour = ts.hour + 0.5
        decl = math.radians(23.45 * math.sin(math.radians(360 / 365 * (doy - 81))))
        ha   = math.radians((hour - 12) * 15)
        sin_elev = (math.sin(lat_rad) * math.sin(decl) +
                    math.cos(lat_rad) * math.cos(decl) * math.cos(ha))
        elev = math.degrees(math.asin(max(sin_elev, 0)))
        if elev <= 0:
            continue
        airmass  = min(1 / (math.sin(math.radians(elev)) +
                            0.50572 * (elev + 6.07995)**-1.6364), 38)
        tau_beam = 0.56 * (math.exp(-0.065 * airmass) + math.exp(-0.095 * airmass))
        ghi_cs   = 1353 * tau_beam * math.sin(math.radians(elev))
        cloud_factor = 1.0
        if ts.month in [7, 8, 9]:
            np.random.seed(i % 500 + 5000)
            if ts.hour >= 13 and np.random.random() < 0.30:
                cloud_factor = np.random.uniform(0.40, 0.75)
        ghi[i] = max(ghi_cs * cloud_factor, 0)
    return ghi


def az_load_multiplier(hour, temp_f_val, is_weekend):
    """Residential load shape × temperature sensitivity."""
    base_shape = [
        0.41, 0.38, 0.36, 0.35, 0.36, 0.40,
        0.48, 0.55, 0.62, 0.67, 0.71, 0.74,
        0.76, 0.75, 0.77, 0.81, 0.87, 0.94,
        1.00, 0.97, 0.91, 0.82, 0.70, 0.55,
    ]
    base = base_shape[hour]
    if is_weekend:
        base = base * 0.90 + 0.10
    if temp_f_val > 72:
        temp_boost = 1.0 + (temp_f_val - 72) * 0.008
        if temp_f_val > 105:
            temp_boost *= 1.15
    else:
        temp_boost = max(1.0 - (72 - temp_f_val) * 0.003, 0.85)
    return base * temp_boost


def storage_profile_for_hour(hour, ghi_normalized, is_storage_bus):
    """
    Net storage power in kW for storage-equipped buses.
    Negative = charging (absorbing), Positive = discharging (supplying).
    Non-storage buses always return 0.
    """
    if not is_storage_bus:
        return 0.0
    if 10 <= hour <= 14 and ghi_normalized > 0.3:
        return -STORAGE_KW_RATE * min(ghi_normalized, 1.0)   # charging
    elif 17 <= hour <= 21:
        return STORAGE_KW_RATE * STORAGE_EFFICIENCY            # discharging
    elif hour >= 22 or hour <= 5:
        return -0.05                                           # vampire drain
    return 0.0


def heat_scenario_multiplier(ts_val, temp_f_val, heatdome_start, heatdome_end):
    """
    Nonlinear AC load multiplier driven by temperature.
    Heat enters as a continuous driver, not just a binary flag.
    """
    is_heatdome = heatdome_start <= ts_val <= heatdome_end
    if not is_heatdome:
        return 1.0, 0
    if temp_f_val >= 115:
        return 1.22, 1
    elif temp_f_val >= 110:
        return 1.15, 1
    elif temp_f_val >= 105:
        return 1.08, 1
    else:
        return 1.04, 1


# ── STEP 3+4+5+6: PER-YEAR TIME SERIES CONSTRUCTION ──────────────────────────

print("\n" + "="*60)
print("STEPS 3–6 — Building time series for all years")
print("="*60)

# EV shape (NREL EVI-Pro Maricopa County L2 residential)
EV_SHAPE = np.array([
    0.02, 0.01, 0.01, 0.01, 0.01, 0.02,
    0.03, 0.04, 0.04, 0.05, 0.05, 0.04,
    0.05, 0.05, 0.06, 0.10, 0.35, 0.75,
    1.00, 0.95, 0.80, 0.55, 0.30, 0.10,
])
EV_CHARGER_KW    = 7.2
EV_ADOPTION_BASE = 0.08
PANEL_AREA_M2    = 44.4
PANEL_EFF        = 0.18
PANEL_DERATING   = 0.85


def fetch_noaa_weather(year, token):
    """
    Fetch hourly Phoenix Sky Harbor temperature for a given year.
    Returns a Series indexed by timestamp (hourly), values in °F.
    Returns None on failure.
    """
    if not token:
        return None
    url = "https://www.ncei.noaa.gov/access/services/data/v1"
    params = {
        "dataset":   "global-hourly",
        "stations":  "72278023183",
        "startDate": f"{year}-01-01T00:00:00",
        "endDate":   f"{year}-12-31T23:59:59",
        "dataTypes": "TMP",
        "units":     "metric",
        "format":    "json",
        "includeAttributes": "false",
    }
    try:
        resp = requests.get(url, params=params,
                            headers={"token": token}, timeout=60)
        if resp.status_code != 200:
            print(f"    NOAA {year}: HTTP {resp.status_code} — using synthetic")
            return None
        records = resp.json()
        if not records:
            return None
        df = pd.DataFrame(records)
        # TMP field format: "0256,1" → 25.6 °C
        df["temp_c"] = df["TMP"].str.split(",").str[0].astype(float) / 10
        df["temp_f"] = df["temp_c"] * 9 / 5 + 32
        df["timestamp"] = pd.to_datetime(df["DATE"]).dt.floor("h")
        series = df.groupby("timestamp")["temp_f"].mean()
        print(f"    NOAA {year}: fetched {len(series)} hourly records")
        return series
    except Exception as e:
        print(f"    NOAA {year}: failed ({e}) — using synthetic")
        return None


def fetch_eia930_azps(year, api_key):
    """
    Fetch AZPS balancing-authority hourly demand from EIA-930.
    Returns a Series indexed by timestamp, values in MW.
    Returns None on failure or missing key.
    """
    if not api_key:
        return None
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key":              api_key,
        "frequency":            "hourly",
        "data[0]":              "value",
        "facets[respondent][]": "AZPS",
        "facets[type][]":       "D",
        "start":                f"{year}-01-01T00",
        "end":                  f"{year}-12-31T23",
        "length":               5000,
        "offset":               0,
    }
    all_records = []
    try:
        while True:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"    EIA {year}: HTTP {resp.status_code} — using synthetic")
                return None
            data = resp.json().get("response", {}).get("data", [])
            if not data:
                break
            all_records.extend(data)
            if len(data) < 5000:
                break
            params["offset"] += 5000
    except Exception as e:
        print(f"    EIA {year}: failed ({e}) — using synthetic")
        return None
    if not all_records:
        return None
    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["period"], utc=True).dt.tz_localize(None)
    df["aps_demand_mw"] = pd.to_numeric(df["value"], errors="coerce")
    series = df.set_index("timestamp")["aps_demand_mw"].sort_index()
    print(f"    EIA {year}: fetched {len(series)} records  "
          f"range {series.min():.0f}–{series.max():.0f} MW")
    return series


def synthetic_eia930(timestamps, year):
    """
    Synthetic AZPS demand calibrated to APS annual pattern.
    APS: ~1.3M customers, summer peak ~8,000 MW, winter ~4,500 MW.
    """
    aps_demand = np.zeros(len(timestamps))
    for i, ts_val in enumerate(timestamps):
        doy  = ts_val.day_of_year
        hour = ts_val.hour
        annual     = 6000 + 2000 * np.sin(2 * np.pi * (doy - 80) / 365)
        daily_mult = 0.70 + 0.30 * np.sin(2 * np.pi * (hour - 4) / 24 - np.pi / 2)
        # Year-specific heat adjustments based on actual Phoenix summers
        if year == 2023 and ts_val.month == 7:
            heat_boost = 1.12
        elif year == 2022 and ts_val.month in [7, 8]:
            heat_boost = 1.06
        elif year == 2021 and ts_val.month == 6:
            heat_boost = 1.08
        elif year == 2020 and ts_val.month == 8:
            heat_boost = 1.07
        else:
            heat_boost = 1.0
        aps_demand[i] = annual * daily_mult * heat_boost
    return pd.Series(aps_demand, index=timestamps)


def build_year_timeseries(year):
    """
    Build the complete (bus × hour) feature matrix for one calendar year.
    Returns a DataFrame with all features including EIA-930, EV scenarios,
    battery storage, and heat dome flags.
    """
    print(f"\n  ── Year {year} ──")

    timestamps = pd.date_range(
        start=f"{year}-01-01 00:00", periods=8760, freq="h"
    )
    n_hours = len(timestamps)

    # ── STEP 3: Synthetic weather + solar ─────────────────────────────────────
    print(f"    Generating synthetic temperature and GHI...")
    temp_f_synth = synthetic_phoenix_temperature(timestamps)
    temp_f       = temp_f_synth.copy()   # will be overwritten by NOAA if available
    ghi_wm2      = synthetic_ghi(timestamps)

    # ── STEP 4: Fetch NOAA real temperature and merge ─────────────────────────
    print(f"    Fetching NOAA weather for {year}...")
    noaa_series = fetch_noaa_weather(year, NOAA_TOKEN)
    if noaa_series is not None:
        noaa_lookup = noaa_series.to_dict()
        replaced = 0
        for i, ts_val in enumerate(timestamps):
            if ts_val in noaa_lookup and not np.isnan(noaa_lookup[ts_val]):
                temp_f[i] = noaa_lookup[ts_val]
                replaced += 1
        # Fill any remaining gaps with synthetic
        for i in range(len(temp_f)):
            if np.isnan(temp_f[i]):
                temp_f[i] = temp_f_synth[i]
        print(f"    Merged {replaced}/{n_hours} real NOAA hours "
              f"({replaced/n_hours*100:.0f}% real data)")
    else:
        print(f"    Using fully synthetic temperature for {year}")

    temp_c = (temp_f - 32) * 5 / 9

    # PV output with temperature derating
    cell_temp_c = temp_c + (ghi_wm2 / 1000) * 25
    temp_derating = 1.0 - np.maximum(cell_temp_c - 25, 0) * 0.004
    pv_output_kw  = (ghi_wm2 / 1000) * PANEL_AREA_M2 * PANEL_EFF * PANEL_DERATING * temp_derating

    # Load multipliers
    load_multipliers = np.array([
        az_load_multiplier(ts.hour, temp_f[i], ts.dayofweek >= 5)
        for i, ts in enumerate(timestamps)
    ])

    # EV load shape
    ev_load_kw = EV_SHAPE[timestamps.hour] * EV_CHARGER_KW * EV_ADOPTION_BASE
    weekend_mask = timestamps.dayofweek >= 5
    ev_load_kw = ev_load_kw.copy()
    ev_load_kw[weekend_mask] = (
        np.roll(EV_SHAPE, -2)[timestamps[weekend_mask].hour]
        * EV_CHARGER_KW * EV_ADOPTION_BASE
    )

    # ── STEP 5: EIA-930 AZPS bulk demand ─────────────────────────────────────
    print(f"    Fetching EIA-930 AZPS demand for {year}...")
    eia_series = fetch_eia930_azps(year, EIA_KEY)
    if eia_series is None:
        print(f"    Using synthetic AZPS demand for {year}")
        eia_series = synthetic_eia930(timestamps, year)

    # Reindex to match our exact timestamps (fill gaps with interpolation)
    eia_series = eia_series.reindex(timestamps).interpolate("time").bfill().ffill()
    aps_max = eia_series.max()
    aps_lookup      = eia_series.to_dict()
    aps_norm_lookup = (eia_series / aps_max).to_dict()
    aps_lag24       = eia_series.shift(24).bfill()
    aps_lag24_lookup = aps_lag24.to_dict()

    # ── STEP 6a: Heat dome flags ──────────────────────────────────────────────
    hd_start_str, hd_end_str = HEATDOME_WINDOWS.get(
        year, (f"{year}-07-01", f"{year}-07-14")   # default: 2 weeks
    )
    hd_start = pd.Timestamp(hd_start_str)
    hd_end   = pd.Timestamp(hd_end_str)

    heatdome_flag_arr = np.zeros(n_hours, dtype=np.int8)
    heatdome_mult_arr = np.ones(n_hours)
    duck_flag_arr     = np.zeros(n_hours, dtype=np.int8)
    storage_by_ts_arr = np.zeros(n_hours)   # system-level (per solar bus), applied below

    ghi_max  = ghi_wm2.max()

    for i, ts_val in enumerate(timestamps):
        hour     = ts_val.hour
        ghi_norm = ghi_wm2[i] / max(ghi_max, 1)
        mult, flag = heat_scenario_multiplier(ts_val, temp_f[i], hd_start, hd_end)
        heatdome_flag_arr[i] = flag
        heatdome_mult_arr[i] = mult
        duck_flag_arr[i]     = int(17 <= hour <= 21)
        storage_by_ts_arr[i] = storage_profile_for_hour(hour, ghi_norm, True)
        # Note: is_storage_bus=True here; the per-bus flag is applied during assembly

    # ── STEP 6b: Assemble per-bus rows ───────────────────────────────────────
    print(f"    Assembling per-bus features ({n_buses} buses × {n_hours} hours)...")
    all_records = []

    is_covid_year = int(year == 2020)

    for bus in all_buses:
        bus_id   = bus_index[bus]
        base_kw  = nodes_df.loc[nodes_df["bus_name"] == bus, "load_kw"].values[0]
        is_solar = int(bus in solar_buses)
        is_ev    = int(bus in ev_buses)
        is_stor  = int(bus in storage_buses)
        has_load = int(bus in load_bus_names)
        kv_base  = nodes_df.loc[nodes_df["bus_name"] == bus, "kv_base"].values[0]

        load_kw_series = base_kw * load_multipliers if (has_load and base_kw > 0) \
                         else np.zeros(n_hours)
        solar_kw_series = pv_output_kw * is_solar
        ev_kw_series    = ev_load_kw * is_ev

        # Storage: only active on is_storage_bus buses
        storage_kw_series = storage_by_ts_arr * is_stor

        net_load_kw = load_kw_series + ev_kw_series - solar_kw_series

        # Thermal utilisation estimate
        connected_lines = edges_df[edges_df["from_bus"] == bus]
        max_rating = connected_lines["norm_amps"].max() if len(connected_lines) else 400.0
        rated_kw   = max_rating * kv_base * math.sqrt(3) * 0.9
        thermal_pct = np.clip(np.abs(net_load_kw) / max(rated_kw, 1) * 100, 0, 200)

        # Lag features
        lag1_kw   = np.roll(load_kw_series, 1);  lag1_kw[0]    = load_kw_series[0]
        lag24_kw  = np.roll(load_kw_series, 24); lag24_kw[:24] = load_kw_series[:24]
        lag168_kw = np.roll(load_kw_series,168); lag168_kw[:168] = load_kw_series[:168]

        load_pd      = pd.Series(load_kw_series)
        roll7d_mean  = load_pd.rolling(168, min_periods=1).mean().values
        roll7d_std   = load_pd.rolling(168, min_periods=1).std().fillna(0).values

        temp_load_interaction = temp_f * load_multipliers

        # EV scenario columns
        ev_2030_kw = ev_kw_series * EV_MULT_2030
        ev_2035_kw = ev_kw_series * EV_MULT_2035

        # Scenario target columns
        net_load_heatdome_kw = (
            net_load_kw * heatdome_mult_arr + storage_kw_series
        )
        net_load_2030_kw = net_load_kw + (ev_2030_kw - ev_kw_series)

        for i, ts_val in enumerate(timestamps):
            all_records.append({
                # Identifiers
                "bus_name":          bus,
                "bus_id":            bus_id,
                "timestamp":         ts_val,
                "year":              year,
                "is_covid_year":     is_covid_year,
                # Time encoding
                "hour":              ts_val.hour,
                "month":             ts_val.month,
                "day_of_week":       ts_val.dayofweek,
                "is_weekend":        int(ts_val.dayofweek >= 5),
                "hour_sin":          math.sin(2 * math.pi * ts_val.hour / 24),
                "hour_cos":          math.cos(2 * math.pi * ts_val.hour / 24),
                "month_sin":         math.sin(2 * math.pi * ts_val.month / 12),
                "month_cos":         math.cos(2 * math.pi * ts_val.month / 12),
                # Weather (NOAA real where available, synthetic fallback)
                "temp_f":            temp_f[i],
                "temp_c":            temp_c[i],
                "temp_lag1":         temp_f[i-1] if i > 0    else temp_f[0],
                "temp_lag24":        temp_f[i-24] if i >= 24  else temp_f[0],
                # Solar (synthetic GHI calibrated to Phoenix clear-sky model)
                "ghi_wm2":           ghi_wm2[i],
                "clearsky_ratio":    min(ghi_wm2[i] / max(ghi_wm2.max(), 1), 1.0),
                "pv_output_kw":      solar_kw_series[i],
                # Load
                "load_kw":           load_kw_series[i],
                "load_lag1":         lag1_kw[i],
                "load_lag24":        lag24_kw[i],
                "load_lag168":       lag168_kw[i],
                "load_roll7d_mean":  roll7d_mean[i],
                "load_roll7d_std":   roll7d_std[i],
                # EV (current + future scenarios)
                "ev_charging_kw":    ev_kw_series[i],
                "ev_load_2030_kw":   ev_2030_kw[i],
                "ev_load_2035_kw":   ev_2035_kw[i],
                # Battery storage
                "storage_net_kw":    storage_kw_series[i],
                # EIA-930 bulk system demand
                "aps_demand_mw":     aps_lookup.get(ts_val, np.nan),
                "aps_demand_mw_norm":aps_norm_lookup.get(ts_val, np.nan),
                "aps_demand_lag24_mw":aps_lag24_lookup.get(ts_val, np.nan),
                # Heat dome scenario
                "is_heatdome":       int(heatdome_flag_arr[i]),
                "heat_scenario_mult":heatdome_mult_arr[i],
                "is_duck_curve_window": int(duck_flag_arr[i]),
                # Interaction term
                "temp_x_load_mult":  temp_load_interaction[i],
                # Target columns
                "net_load_kw":           net_load_kw[i],
                "net_load_heatdome_kw":  net_load_heatdome_kw[i],
                "net_load_2030_kw":      net_load_2030_kw[i],
                "thermal_pct":           thermal_pct[i],
                # Bus type flags (static, repeated per timestep for tensor assembly)
                "is_solar_bus":      is_solar,
                "is_ev_bus":         is_ev,
                "is_storage_bus":    is_stor,
                "is_substation":     int(bus == "150"),
                "is_reg_secondary":  int(bus.endswith("r")),
            })

    year_df = pd.DataFrame(all_records)

    # ── Validation spot checks ────────────────────────────────────────────────
    hd_hours = year_df[year_df["is_heatdome"] == 1]["timestamp"].nunique()
    peak_ts  = year_df.groupby("timestamp")["net_load_kw"].sum().idxmax()
    peak_kw  = year_df.groupby("timestamp")["net_load_kw"].sum().max()
    ev_peak  = year_df["ev_charging_kw"].max()
    ev_2030p = year_df["ev_load_2030_kw"].max()
    stor_noon = year_df[(year_df["timestamp"].dt.hour == 12) &
                        (year_df["is_storage_bus"] == 1)]["storage_net_kw"].mean()
    stor_6pm  = year_df[(year_df["timestamp"].dt.hour == 18) &
                        (year_df["is_storage_bus"] == 1)]["storage_net_kw"].mean()
    print(f"    Heat dome hours: {hd_hours} ({hd_hours/24:.0f} days)")
    print(f"    Peak feeder load: {peak_kw/1000:.2f} MVA at {peak_ts}")
    print(f"    EV peak: {ev_peak:.2f} kW → 2030: {ev_2030p:.2f} kW (×{EV_MULT_2030:.1f})")
    print(f"    Storage noon (expect <0 charging): {stor_noon:.2f} kW")
    print(f"    Storage 6pm  (expect >0 discharging): {stor_6pm:.2f} kW")
    print(f"    APS demand: {year_df['aps_demand_mw'].min():.0f}–"
          f"{year_df['aps_demand_mw'].max():.0f} MW")

    return year_df


# Build all years
all_year_dfs = []
for year in SIM_YEARS:
    df = build_year_timeseries(year)
    all_year_dfs.append(df)

print("\n" + "="*60)
print("STEP 7 — Concatenating all years")
print("="*60)

timeseries_df = pd.concat(all_year_dfs, ignore_index=True)
print(f"  Combined shape: {timeseries_df.shape}")
print(f"  Years: {sorted(timeseries_df['year'].unique().tolist())}")
print(f"  Total rows: {len(timeseries_df):,}")

# ── Train / Val / Test split labels ──────────────────────────────────────────
def assign_split(row):
    y = row["year"]
    if y in [2019, 2020, 2021, 2022]:
        return "train"
    if y == 2023 and row["month"] <= 6:
        return "val"
    return "test"   # 2023 Jul–Dec — unseen heat dome period

timeseries_df["split"] = timeseries_df.apply(assign_split, axis=1)
split_counts = timeseries_df.groupby("split")["timestamp"].nunique()
print(f"  Split hours — train: {split_counts.get('train',0):,}  "
      f"val: {split_counts.get('val',0):,}  "
      f"test: {split_counts.get('test',0):,}")


# ── STEP 8: BUILD PYTORCH GEOMETRIC GRAPH ─────────────────────────────────────

print("\n" + "="*60)
print("STEP 8 — Building PyTorch Geometric graph")
print("="*60)

try:
    import torch
    from torch_geometric.data import Data

    for col in ["x", "y", "load_kw", "load_kvar"]:
        col_max = nodes_df[col].abs().max()
        nodes_df[f"{col}_norm"] = nodes_df[col] / col_max if col_max > 0 else 0.0

    x_node = torch.tensor(
        nodes_df[["x_norm", "y_norm", "kv_base", "load_kw_norm", "load_kvar_norm",
                  "is_substation", "is_reg_secondary", "is_solar_bus",
                  "is_ev_bus", "is_storage_bus", "has_load"]].values.astype(np.float32),
        dtype=torch.float
    )

    for col in ["r1", "x1", "z_total", "weight", "norm_amps", "length_mi"]:
        col_max = edges_df[col].abs().max()
        edges_df[f"{col}_norm"] = edges_df[col] / col_max if col_max > 0 else 0.0

    src = torch.tensor(edges_df["from_idx"].values, dtype=torch.long)
    dst = torch.tensor(edges_df["to_idx"].values,   dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    edge_attr = torch.tensor(
        edges_df[["r1_norm", "x1_norm", "z_total_norm",
                  "weight_norm", "norm_amps_norm", "length_mi_norm",
                  "phases", "is_switch"]].values.astype(np.float32),
        dtype=torch.float
    )

    graph = Data(x=x_node, edge_index=edge_index, edge_attr=edge_attr,
                 num_nodes=n_buses)
    graph.bus_names    = all_buses
    graph.bus_index    = bus_index
    graph.solar_buses  = list(solar_buses)
    graph.ev_buses     = list(ev_buses)
    graph.storage_buses = list(storage_buses)
    graph.sim_years    = SIM_YEARS

    self_loops = (graph.edge_index[0] == graph.edge_index[1]).sum().item()
    print(f"  num_nodes:      {graph.num_nodes}")
    print(f"  num_edges:      {graph.num_edges}  ({graph.num_edges//2} physical lines)")
    print(f"  node features:  {graph.x.shape[1]}")
    print(f"  edge features:  {graph.edge_attr.shape[1]}")
    print(f"  self-loops:     {self_loops}  (expect 0)")
    pyg_available = True

except ImportError:
    print("  PyTorch / PyG not installed — skipping graph object")
    print("  pip install torch torch-geometric")
    pyg_available = False
    graph = None


# ── STEP 9: SAVE OUTPUTS ──────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 9 — Saving outputs")
print("="*60)

OUT_DIR.mkdir(parents=True, exist_ok=True)

nodes_df.to_csv(OUT_DIR / "az_feeder_nodes.csv", index=False)
print(f"  Saved: az_feeder_nodes.csv  ({len(nodes_df)} rows)")

edges_df.to_csv(OUT_DIR / "az_feeder_edges.csv", index=False)
print(f"  Saved: az_feeder_edges.csv  ({len(edges_df)} rows)")

try:
    ts_path = OUT_DIR / "az_feeder_timeseries.parquet"
    timeseries_df.to_parquet(ts_path, index=False)
    print(f"  Saved: az_feeder_timeseries.parquet  ({len(timeseries_df):,} rows)")
except Exception:
    csv_path = OUT_DIR / "az_feeder_timeseries.csv"
    timeseries_df.to_csv(csv_path, index=False)
    print(f"  Saved: az_feeder_timeseries.csv (parquet not available)")

if pyg_available and graph is not None:
    import torch
    torch.save(graph, OUT_DIR / "az_feeder_graph.pt")
    print(f"  Saved: az_feeder_graph.pt")

print("\n" + "="*60)
print("DONE")
print("="*60)
print(f"  Buses:            {n_buses}")
print(f"  Physical lines:   {len(lines_df)}")
print(f"  Solar buses:      {len(solar_buses)}")
print(f"  EV buses:         {len(ev_buses)}")
print(f"  Storage buses:    {len(storage_buses)}")
print(f"  Years built:      {SIM_YEARS}")
print(f"  Total rows:       {len(timeseries_df):,}")
print(f"  Features/row:     {len([c for c in timeseries_df.columns if c not in ['bus_name','bus_id','timestamp']])}")
print()
print("  Load in your GNN training script:")
print("    graph = torch.load('az_feeder_graph.pt')")
print("    ts    = pd.read_parquet('az_feeder_timeseries.parquet')")
print("    train = ts[ts['split'] == 'train']")
print("    val   = ts[ts['split'] == 'val']")
print("    test  = ts[ts['split'] == 'test']")
print()

# ── Feature inventory ─────────────────────────────────────────────────────────
print("--- FEATURE INVENTORY ---")
feature_groups = {
    "Identifiers":    ["bus_name", "bus_id", "timestamp", "year", "is_covid_year", "split"],
    "Time encoding":  ["hour", "month", "day_of_week", "is_weekend",
                       "hour_sin", "hour_cos", "month_sin", "month_cos"],
    "Weather (NOAA)": ["temp_f", "temp_c", "temp_lag1", "temp_lag24"],
    "Solar":          ["ghi_wm2", "clearsky_ratio", "pv_output_kw"],
    "Load":           ["load_kw", "load_lag1", "load_lag24", "load_lag168",
                       "load_roll7d_mean", "load_roll7d_std"],
    "EV":             ["ev_charging_kw", "ev_load_2030_kw", "ev_load_2035_kw"],
    "Storage":        ["storage_net_kw"],
    "EIA-930":        ["aps_demand_mw", "aps_demand_mw_norm", "aps_demand_lag24_mw"],
    "Heat scenario":  ["is_heatdome", "heat_scenario_mult", "temp_x_load_mult"],
    "Duck curve":     ["is_duck_curve_window"],
    "Targets":        ["net_load_kw", "net_load_heatdome_kw", "net_load_2030_kw", "thermal_pct"],
    "Bus flags":      ["is_solar_bus", "is_ev_bus", "is_storage_bus",
                       "is_substation", "is_reg_secondary"],
}
for group, cols in feature_groups.items():
    present = [c for c in cols if c in timeseries_df.columns]
    print(f"  {group:<18} {len(present):>2} features")