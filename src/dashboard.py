"""
dashboard.py  v3 — descriptive + predictive split
===================================================
Reads az_feeder_timeseries.parquet (test split = 2023 Jul–Dec).
Outputs dashboard_data.json for a Lovable React frontend.

THRESHOLDS derived from real descriptive statistics of the test set:
  thermal_pct  p99 = 18.85   max = 45.59   mean = 2.35
  net_load_kw  p99 = 489.0   max = 1182.6

  DISPATCH  : thermal_pct >= 20     (above p99 — genuine emergency)
  MONITOR   : thermal_pct >= 10     (above p95 = 6.90, picking 10 for signal)
  WATCH     : thermal_pct >= 5      (meaningful but manageable)
  NORMAL    : thermal_pct <  5

JSON structure:
  meta               — bus registry, thresholds, config
  descriptive        — full test set Jul–Dec 2023  (panels 1–6)
  predictive_window  — Jul 9 2023 24h lookback + 24h horizon (panel 7)
  heatdome_cut       — Jul 1–Aug 4 filtered data
  duck_curve_cut     — hours 17–21 all test days

Run:
  python dashboard.py
"""

import json
import gc
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
PARQUET_PATH = Path(r".\data\az_feeder_timeseries.parquet")
NODES_PATH   = Path(r".\data\az_feeder_nodes.csv")
OUT_PATH     = Path(r".\assets\dashboard_data.json")

# Thresholds from real descriptive statistics (thermal_pct test set)
# p99=18.85, p95=6.90, mean=2.35, max=45.59
DISPATCH_THRESH  = 20.0   # >= p99 — operator must act now
MONITOR_THRESH   = 10.0   # >= ~p97 — watch closely
WATCH_THRESH     =  5.0   # >= p95 area — flag for planning
# Below 5 = NORMAL

# Predictive window: the worst recorded day in the test set
PRED_DATE    = "2023-07-09"   # bus 76 hit 1182.59 kW / 45.59% thermal

HEATDOME_START = "2023-07-01"
HEATDOME_END   = "2023-08-04"

DUCK_HOURS = [17, 18, 19, 20, 21]

print("=" * 60)
print("dashboard.py  v3 — descriptive + predictive")
print("=" * 60)

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("\n[1/9] Loading parquet …")
ts = pd.read_parquet(PARQUET_PATH)
test = ts[ts["split"] == "test"].copy()
del ts
gc.collect()

test["timestamp"] = pd.to_datetime(test["timestamp"])
test["date"]      = test["timestamp"].dt.date.astype(str)
test["hour"]      = test["timestamp"].dt.hour

print(f"  Rows  : {len(test):,}")
print(f"  Buses : {test['bus_name'].nunique()}")
print(f"  Hours : {test['timestamp'].nunique()}")
print(f"  Range : {test['timestamp'].min()} → {test['timestamp'].max()}")

# ── META ──────────────────────────────────────────────────────────────────────
print("\n[2/9] Building meta …")

# Bus type flags — one row per bus
bus_flags = (
    test.groupby("bus_name")
    .agg(
        is_ev_bus       = ("is_ev_bus",        "first"),
        is_solar_bus    = ("is_solar_bus",      "first"),
        is_storage_bus  = ("is_storage_bus",    "first"),
        is_substation   = ("is_substation",     "first"),
        is_reg_secondary= ("is_reg_secondary",  "first"),
    )
    .reset_index()
)

# Load node coords if available
if NODES_PATH.exists():
    nodes = pd.read_csv(NODES_PATH)
    bus_flags = bus_flags.merge(
        nodes[["bus_name","x","y","kv_base","load_kw"]].rename(
            columns={"load_kw":"rated_load_kw"}),
        on="bus_name", how="left"
    )
else:
    bus_flags["x"] = 0.0
    bus_flags["y"] = 0.0
    bus_flags["kv_base"] = 4.16
    bus_flags["rated_load_kw"] = 0.0

bus_registry = []
for _, r in bus_flags.iterrows():
    bus_type = "standard"
    if r["is_substation"]:    bus_type = "substation"
    elif r["is_reg_secondary"]:bus_type = "regulator"
    elif r["is_storage_bus"]: bus_type = "storage"
    elif r["is_solar_bus"]:   bus_type = "solar"
    elif r["is_ev_bus"]:      bus_type = "ev"
    bus_registry.append({
        "bus":          r["bus_name"],
        "type":         bus_type,
        "x":            round(float(r.get("x", 0) or 0), 2),
        "y":            round(float(r.get("y", 0) or 0), 2),
        "kv_base":      round(float(r.get("kv_base", 4.16) or 4.16), 3),
        "rated_load_kw":round(float(r.get("rated_load_kw", 0) or 0), 1),
        "is_ev":        bool(r["is_ev_bus"]),
        "is_solar":     bool(r["is_solar_bus"]),
        "is_storage":   bool(r["is_storage_bus"]),
    })

meta = {
    "test_period":       "2023-07-01 to 2023-12-31",
    "predictive_date":   PRED_DATE,
    "heatdome_window":   f"{HEATDOME_START} to {HEATDOME_END}",
    "duck_curve_hours":  DUCK_HOURS,
    "thresholds": {
        "dispatch":  DISPATCH_THRESH,
        "monitor":   MONITOR_THRESH,
        "watch":     WATCH_THRESH,
        "unit":      "thermal_pct",
        "basis":     "derived from test set: p99=18.85, p95=6.90, max=45.59",
    },
    "bus_registry": bus_registry,
    "group_lists": {
        "ev_buses":      sorted(bus_flags[bus_flags["is_ev_bus"]==1]["bus_name"].tolist()),
        "solar_buses":   sorted(bus_flags[bus_flags["is_solar_bus"]==1]["bus_name"].tolist()),
        "storage_buses": sorted(bus_flags[bus_flags["is_storage_bus"]==1]["bus_name"].tolist()),
    },
    "total_buses":  int(test["bus_name"].nunique()),
    "total_hours":  int(test["timestamp"].nunique()),
}
print(f"  Bus registry built: {len(bus_registry)} buses")

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def pct(arr, q): return float(np.percentile(arr.dropna(), q))
def safe(v):     return float(v) if (v is not None and not math.isnan(float(v))) else 0.0

def tier(thermal_max):
    if thermal_max >= DISPATCH_THRESH: return "DISPATCH"
    if thermal_max >= MONITOR_THRESH:  return "MONITOR"
    if thermal_max >= WATCH_THRESH:    return "WATCH"
    return "NORMAL"

def action(row):
    """Data-driven action strings using real thresholds."""
    tm = row["thermal_max"]
    hr = row["exceedance_hrs"]
    hd = row["heatdome_uplift_pct"]
    ev = row["is_ev_bus"]
    sol= row["is_solar_bus"]
    sto= row["is_storage_bus"]
    hr_w= row["headroom_kw"]

    if tm >= DISPATCH_THRESH and hr > 100:
        return "IMMEDIATE: transformer upgrade — exceeded dispatch threshold >100 h"
    if tm >= DISPATCH_THRESH and hd > 15:
        return "IMMEDIATE: deploy demand response — heat dome drives >15% load uplift"
    if tm >= MONITOR_THRESH and ev:
        return "Managed EV charging schedule — shift load from 17–21h window"
    if tm >= MONITOR_THRESH and hd > 10:
        return "Pre-cool activation 1h before forecast peak; alert DR participants"
    if tm >= MONITOR_THRESH:
        return "Load flow study + monitor afternoon peak daily"
    if tm >= WATCH_THRESH and sol and not sto:
        return "Install battery storage — solar excess uncaptured at this bus"
    if tm >= WATCH_THRESH:
        return "Flag for 2027 EV planning cycle; no near-term action"
    return "Within normal range — routine monitoring"

# ── PANEL 1: FEEDER STRESS INDEX (descriptive) ────────────────────────────────
print("\n[3/9] Panel 1 — Feeder stress index …")

# Capacity: thermal_pct = net_load_kw / capacity × 100
# So capacity = net_load_max / (thermal_max / 100)
bus_agg = (
    test.groupby("bus_name")
    .agg(
        net_load_mean       = ("net_load_kw",          "mean"),
        net_load_p95        = ("net_load_kw",          lambda x: pct(x, 95)),
        net_load_p99        = ("net_load_kw",          lambda x: pct(x, 99)),
        net_load_max        = ("net_load_kw",          "max"),
        net_load_std        = ("net_load_kw",          "std"),
        thermal_mean        = ("thermal_pct",          "mean"),
        thermal_p95         = ("thermal_pct",          lambda x: pct(x, 95)),
        thermal_p99         = ("thermal_pct",          lambda x: pct(x, 99)),
        thermal_max         = ("thermal_pct",          "max"),
        heatdome_max_kw     = ("net_load_heatdome_kw", "max"),
        heatdome_mean_kw    = ("net_load_heatdome_kw", "mean"),
        ev2030_max_kw       = ("net_load_2030_kw",     "max"),
        ev_charging_mean    = ("ev_charging_kw",       "mean"),
        ev_charging_max     = ("ev_charging_kw",       "max"),
        pv_output_mean      = ("pv_output_kw",         "mean"),
        storage_mean        = ("storage_net_kw",       "mean"),
        temp_f_mean         = ("temp_f",               "mean"),
        temp_f_max          = ("temp_f",               "max"),
        is_heatdome_hours   = ("is_heatdome",          "sum"),
        is_ev_bus           = ("is_ev_bus",            "first"),
        is_solar_bus        = ("is_solar_bus",         "first"),
        is_storage_bus      = ("is_storage_bus",       "first"),
    )
    .reset_index()
)

# Capacity from observed max
bus_agg["capacity_kw"] = np.where(
    bus_agg["thermal_max"] > 0,
    bus_agg["net_load_max"] / (bus_agg["thermal_max"] / 100.0),
    bus_agg["net_load_max"] * 1.5
)

# Headroom = capacity - scenario_max (clipped to 0)
bus_agg["headroom_kw"]          = (bus_agg["capacity_kw"] - bus_agg["net_load_p99"]).clip(lower=0)
bus_agg["headroom_heatdome_kw"] = (bus_agg["capacity_kw"] - bus_agg["heatdome_max_kw"]).clip(lower=0)
bus_agg["headroom_ev2030_kw"]   = (bus_agg["capacity_kw"] - bus_agg["ev2030_max_kw"]).clip(lower=0)

# Heatdome uplift %
bus_agg["heatdome_uplift_pct"] = np.where(
    bus_agg["net_load_mean"] > 0,
    (bus_agg["heatdome_mean_kw"] - bus_agg["net_load_mean"]) / bus_agg["net_load_mean"] * 100,
    0.0
)

# Exceedance hours at each tier
exc_dispatch = test[test["thermal_pct"] >= DISPATCH_THRESH].groupby("bus_name").size().rename("exc_dispatch")
exc_monitor  = test[test["thermal_pct"] >= MONITOR_THRESH ].groupby("bus_name").size().rename("exc_monitor")
exc_watch    = test[test["thermal_pct"] >= WATCH_THRESH   ].groupby("bus_name").size().rename("exc_watch")
bus_agg = bus_agg.join(exc_dispatch, on="bus_name").join(exc_monitor, on="bus_name").join(exc_watch, on="bus_name")
bus_agg[["exc_dispatch","exc_monitor","exc_watch"]] = bus_agg[["exc_dispatch","exc_monitor","exc_watch"]].fillna(0).astype(int)
bus_agg["exceedance_hrs"] = bus_agg["exc_dispatch"]  # used in action()

bus_agg["risk_tier"]   = bus_agg["thermal_max"].apply(tier)
bus_agg["action"]      = bus_agg.apply(action, axis=1)
bus_agg_sorted         = bus_agg.sort_values("thermal_max", ascending=False)

tier_counts = bus_agg["risk_tier"].value_counts().to_dict()
print(f"  DISPATCH: {tier_counts.get('DISPATCH',0)}  MONITOR: {tier_counts.get('MONITOR',0)}  "
      f"WATCH: {tier_counts.get('WATCH',0)}  NORMAL: {tier_counts.get('NORMAL',0)}")

panel1 = []
for _, r in bus_agg_sorted.iterrows():
    panel1.append({
        "bus":                  r["bus_name"],
        "risk_tier":            r["risk_tier"],
        "action":               r["action"],
        "thermal_mean":         round(safe(r["thermal_mean"]),          2),
        "thermal_p95":          round(safe(r["thermal_p95"]),           2),
        "thermal_p99":          round(safe(r["thermal_p99"]),           2),
        "thermal_max":          round(safe(r["thermal_max"]),           2),
        "net_load_mean":        round(safe(r["net_load_mean"]),         1),
        "net_load_p99":         round(safe(r["net_load_p99"]),          1),
        "net_load_max":         round(safe(r["net_load_max"]),          1),
        "net_load_std":         round(safe(r["net_load_std"]),          1),
        "capacity_kw":          round(safe(r["capacity_kw"]),           1),
        "headroom_kw":          round(safe(r["headroom_kw"]),           1),
        "headroom_pct":         round(safe(r["headroom_kw"]) / max(safe(r["capacity_kw"]),1) * 100, 1),
        "headroom_heatdome_kw": round(safe(r["headroom_heatdome_kw"]), 1),
        "headroom_ev2030_kw":   round(safe(r["headroom_ev2030_kw"]),   1),
        "heatdome_max_kw":      round(safe(r["heatdome_max_kw"]),      1),
        "ev2030_max_kw":        round(safe(r["ev2030_max_kw"]),        1),
        "heatdome_uplift_pct":  round(safe(r["heatdome_uplift_pct"]),  1),
        "exc_dispatch_hrs":     int(r["exc_dispatch"]),
        "exc_monitor_hrs":      int(r["exc_monitor"]),
        "exc_watch_hrs":        int(r["exc_watch"]),
        "heatdome_hrs":         int(r["is_heatdome_hours"]),
        "is_ev_bus":            bool(r["is_ev_bus"]),
        "is_solar_bus":         bool(r["is_solar_bus"]),
        "is_storage_bus":       bool(r["is_storage_bus"]),
        "ev_charging_mean":     round(safe(r["ev_charging_mean"]),      3),
        "pv_output_mean":       round(safe(r["pv_output_mean"]),        3),
        "storage_mean":         round(safe(r["storage_mean"]),          3),
        "temp_f_mean":          round(safe(r["temp_f_mean"]),           1),
    })

# ── PANEL 2: LOAD DECOMPOSITION (descriptive) ─────────────────────────────────
print("\n[4/9] Panel 2 — Load decomposition …")

# 24-point hourly profile for the system
sys_hourly = (
    test.groupby("hour")
    .agg(
        base_load_mean   = ("load_kw",          "mean"),
        ev_mean          = ("ev_charging_kw",   "mean"),
        ev_2030_mean     = ("ev_load_2030_kw",  "mean"),
        pv_offset_mean   = ("pv_output_kw",     "mean"),
        storage_mean     = ("storage_net_kw",   "mean"),
        net_load_mean    = ("net_load_kw",       "mean"),
        net_load_max     = ("net_load_kw",       "max"),
        net_load_p95     = ("net_load_kw",       lambda x: pct(x, 95)),
        heatdome_net_mean= ("net_load_heatdome_kw","mean"),
        is_duck          = ("is_duck_curve_window","mean"),
        temp_f_mean      = ("temp_f",            "mean"),
    )
    .reset_index()
)

panel2_profile = []
for _, r in sys_hourly.iterrows():
    panel2_profile.append({
        "hour":              int(r["hour"]),
        "base_load_kw":      round(safe(r["base_load_mean"]),    1),
        "ev_kw":             round(safe(r["ev_mean"]) * 132,     2),   # system total
        "ev_2030_kw":        round(safe(r["ev_2030_mean"]) * 132,2),
        "pv_offset_kw":      round(safe(r["pv_offset_mean"]) * 132,2),
        "storage_kw":        round(safe(r["storage_mean"]) * 132, 2),
        "net_load_mean_kw":  round(safe(r["net_load_mean"]),     1),
        "net_load_max_kw":   round(safe(r["net_load_max"]),      1),
        "net_load_p95_kw":   round(safe(r["net_load_p95"]),      1),
        "heatdome_mean_kw":  round(safe(r["heatdome_net_mean"]), 1),
        "is_duck_window":    round(safe(r["is_duck"]),           2),
        "temp_f_mean":       round(safe(r["temp_f_mean"]),       1),
    })

# Bus-type group decomposition (mean net_load by group per hour)
group_map = {}
for _, r in bus_flags.iterrows():
    b = r["bus_name"]
    if r["is_ev_bus"]:      group_map[b] = "ev"
    elif r["is_solar_bus"]: group_map[b] = "solar"
    elif r["is_storage_bus"]:group_map[b]= "storage"
    else:                   group_map[b] = "standard"

test["bus_group"] = test["bus_name"].map(group_map)
grp_hourly = (
    test.groupby(["bus_group","hour"])["net_load_kw"]
    .mean().reset_index()
    .rename(columns={"net_load_kw":"mean_kw"})
)
panel2_groups = []
for _, r in grp_hourly.iterrows():
    panel2_groups.append({
        "group": r["bus_group"],
        "hour":  int(r["hour"]),
        "mean_kw": round(safe(r["mean_kw"]), 2),
    })

panel2 = {"hourly_profile": panel2_profile, "group_hourly": panel2_groups}
print(f"  Profile: 24 hours  |  Groups: {grp_hourly['bus_group'].nunique()}")

# ── PANEL 3: HOUR-OF-DAY VULNERABILITY (descriptive) ─────────────────────────
print("\n[5/9] Panel 3 — Hour-of-day vulnerability …")

# Split heatdome vs non-heatdome hours
hd_hourly = (
    test[test["is_heatdome"] == 1]
    .groupby("hour")["net_load_kw"]
    .agg(mean="mean", p95=lambda x: pct(x,95), max="max")
    .reset_index()
)
norm_hourly = (
    test[test["is_heatdome"] == 0]
    .groupby("hour")["net_load_kw"]
    .agg(mean="mean", p95=lambda x: pct(x,95), max="max")
    .reset_index()
)

panel3 = []
for h in range(24):
    hd_row   = hd_hourly[hd_hourly["hour"] == h]
    norm_row = norm_hourly[norm_hourly["hour"] == h]
    is_duck  = h in DUCK_HOURS

    hd_mean   = safe(hd_row["mean"].values[0])   if len(hd_row)   else 0.0
    hd_p95    = safe(hd_row["p95"].values[0])    if len(hd_row)   else 0.0
    hd_max    = safe(hd_row["max"].values[0])    if len(hd_row)   else 0.0
    norm_mean = safe(norm_row["mean"].values[0]) if len(norm_row) else 0.0
    norm_max  = safe(norm_row["max"].values[0])  if len(norm_row) else 0.0

    # DR scheduling label
    if h in [14, 15, 16]:    dr_action = "Pre-cool activation window"
    elif h in [17, 18, 19]:  dr_action = "Peak DR dispatch — curtail AC + managed EV"
    elif h in [20, 21]:      dr_action = "EV managed charging ramp-down"
    elif h in [6, 7, 8]:     dr_action = "Morning pre-heat — storage charge window"
    else:                    dr_action = ""

    panel3.append({
        "hour":           h,
        "is_duck_window": is_duck,
        "dr_action":      dr_action,
        "normal_mean_kw": round(norm_mean, 2),
        "normal_max_kw":  round(norm_max,  2),
        "heatdome_mean_kw":round(hd_mean,  2),
        "heatdome_p95_kw": round(hd_p95,   2),
        "heatdome_max_kw": round(hd_max,   2),
        "uplift_mean_kw":  round(hd_mean - norm_mean, 2),
    })

# ── PANEL 4: WEATHER–LOAD COUPLING (descriptive) ──────────────────────────────
print("\n[6/9] Panel 4 — Weather–load coupling …")

# Scatter: bus 76 hourly + top heavy-load buses, afternoon hours only
top_buses_p4 = ["76","48","65","49","47","64","66"]
afternoon_df = test[
    (test["bus_name"].isin(top_buses_p4)) &
    (test["hour"].between(12, 20))
][["bus_name","timestamp","date","hour","temp_f","net_load_kw",
   "thermal_pct","is_heatdome","heat_scenario_mult","ghi_wm2"]].copy()

# Sample to keep JSON bounded (~2000 points per bus)
scatter_rows = []
for bus in top_buses_p4:
    sub = afternoon_df[afternoon_df["bus_name"] == bus]
    if len(sub) > 1500:
        sub = sub.sample(1500, random_state=42)
    for _, r in sub.iterrows():
        scatter_rows.append({
            "bus":             r["bus_name"],
            "temp_f":          round(safe(r["temp_f"]),          1),
            "net_load_kw":     round(safe(r["net_load_kw"]),     1),
            "thermal_pct":     round(safe(r["thermal_pct"]),     2),
            "is_heatdome":     int(r["is_heatdome"]),
            "ghi_wm2":         round(safe(r["ghi_wm2"]),         1),
            "heat_mult":       round(safe(r["heat_scenario_mult"]),3),
            "hour":            int(r["hour"]),
        })

# System-level daily scatter
sys_daily = (
    test.groupby("date")
    .agg(
        max_temp_f       = ("temp_f",              "max"),
        mean_temp_f      = ("temp_f",              "mean"),
        total_load_max   = ("net_load_kw",         "max"),
        total_load_sum   = ("net_load_kw",         "sum"),
        mean_ghi         = ("ghi_wm2",             "mean"),
        is_heatdome      = ("is_heatdome",         "max"),
        mean_heat_mult   = ("heat_scenario_mult",  "mean"),
    )
    .reset_index()
)

panel4_scatter = scatter_rows
panel4_system_daily = []
for _, r in sys_daily.iterrows():
    panel4_system_daily.append({
        "date":           str(r["date"]),
        "max_temp_f":     round(safe(r["max_temp_f"]),     1),
        "mean_temp_f":    round(safe(r["mean_temp_f"]),    1),
        "total_load_max": round(safe(r["total_load_max"]), 1),
        "total_load_sum": round(safe(r["total_load_sum"]), 1),
        "mean_ghi":       round(safe(r["mean_ghi"]),       1),
        "is_heatdome":    int(r["is_heatdome"]),
        "heat_mult":      round(safe(r["mean_heat_mult"]), 3),
    })

panel4 = {
    "bus_scatter":   panel4_scatter,
    "system_daily":  panel4_system_daily,
}
print(f"  Scatter points: {len(scatter_rows)}  |  Daily: {len(panel4_system_daily)}")

# ── PANEL 5: DUCK CURVE RAMP DETECTOR (descriptive) ───────────────────────────
print("\n[7/9] Panel 5 — Duck curve ramp detector …")

# Ramp = net_load_kw at h18 minus net_load_kw at h12, per bus per day
pivot_h12 = (
    test[test["hour"] == 12]
    .groupby(["bus_name","date"])["net_load_kw"]
    .mean().rename("load_h12")
)
pivot_h18 = (
    test[test["hour"] == 18]
    .groupby(["bus_name","date"])["net_load_kw"]
    .mean().rename("load_h18")
)
pivot_hd  = (
    test[test["hour"] == 18]
    .groupby(["bus_name","date"])[["is_heatdome","ev_charging_kw","thermal_pct","temp_f"]]
    .first()
)

ramp_df = pd.concat([pivot_h12, pivot_h18, pivot_hd], axis=1).reset_index()
ramp_df["ramp_kw"] = ramp_df["load_h18"] - ramp_df["load_h12"]

# Bus-level ramp summary
ramp_summary = (
    ramp_df.groupby("bus_name")
    .agg(
        ramp_mean   = ("ramp_kw",      "mean"),
        ramp_p95    = ("ramp_kw",      lambda x: pct(x, 95)),
        ramp_max    = ("ramp_kw",      "max"),
        ramp_std    = ("ramp_kw",      "std"),
        hd_ramp_mean= ("ramp_kw",      lambda x: x[ramp_df.loc[x.index,"is_heatdome"]==1].mean() if any(ramp_df.loc[x.index,"is_heatdome"]==1) else 0),
    )
    .reset_index()
    .sort_values("ramp_p95", ascending=False)
)
ramp_summary["bus_type"] = ramp_summary["bus_name"].map(group_map)

panel5_summary = []
for _, r in ramp_summary.iterrows():
    panel5_summary.append({
        "bus":         r["bus_name"],
        "bus_type":    r["bus_type"],
        "ramp_mean":   round(safe(r["ramp_mean"]),    1),
        "ramp_p95":    round(safe(r["ramp_p95"]),     1),
        "ramp_max":    round(safe(r["ramp_max"]),     1),
        "ramp_std":    round(safe(r["ramp_std"]),     1),
        "hd_ramp_mean":round(safe(r["hd_ramp_mean"]),1),
    })

# Daily ramp series for top 10 ramp buses
top10_ramp = [r["bus"] for r in panel5_summary[:10]]
panel5_daily = []
for _, r in ramp_df[ramp_df["bus_name"].isin(top10_ramp)].iterrows():
    panel5_daily.append({
        "bus":         r["bus_name"],
        "date":        str(r["date"]),
        "ramp_kw":     round(safe(r["ramp_kw"]),      1),
        "load_h12":    round(safe(r["load_h12"]),     1),
        "load_h18":    round(safe(r["load_h18"]),     1),
        "is_heatdome": int(r["is_heatdome"]),
        "ev_kw":       round(safe(r["ev_charging_kw"]),2),
        "thermal_h18": round(safe(r["thermal_pct"]),  2),
        "temp_f_h18":  round(safe(r["temp_f"]),       1),
    })

panel5 = {"summary": panel5_summary, "daily": panel5_daily}
print(f"  Ramp summary: {len(panel5_summary)} buses  |  Daily: {len(panel5_daily)} rows")

# ── PANEL 6: SCENARIO HEADROOM MATRIX (descriptive) ───────────────────────────
print("\n[8/9] Panel 6 — Scenario headroom matrix …")

panel6 = []
for r in panel1:
    cap   = r["capacity_kw"]
    # Combined worst-case: heatdome peak + EV 2030 uplift
    # EV 2030 uplift from data: ev2030_max_kw is already in the parquet
    combined_max      = max(r["heatdome_max_kw"], r["ev2030_max_kw"])  # conservative OR
    headroom_combined = max(cap - combined_max, 0)

    def _pct(hw): return round(hw / max(cap,1) * 100, 1)

    panel6.append({
        "bus":                    r["bus"],
        "risk_tier":              r["risk_tier"],
        "capacity_kw":            round(cap, 1),
        "baseline_p99_kw":        r["net_load_p99"],
        "heatdome_max_kw":        r["heatdome_max_kw"],
        "ev2030_max_kw":          r["ev2030_max_kw"],
        "combined_max_kw":        round(combined_max, 1),
        "headroom_baseline_kw":   r["headroom_kw"],
        "headroom_heatdome_kw":   r["headroom_heatdome_kw"],
        "headroom_ev2030_kw":     r["headroom_ev2030_kw"],
        "headroom_combined_kw":   round(headroom_combined, 1),
        "headroom_baseline_pct":  _pct(r["headroom_kw"]),
        "headroom_heatdome_pct":  _pct(r["headroom_heatdome_kw"]),
        "headroom_ev2030_pct":    _pct(r["headroom_ev2030_kw"]),
        "headroom_combined_pct":  _pct(headroom_combined),
        "investment_priority":    (
            "P1 — act now"   if headroom_combined < 0.05 * cap else
            "P2 — plan 2027" if headroom_combined < 0.15 * cap else
            "P3 — plan 2030" if headroom_combined < 0.30 * cap else
            "P4 — monitor"
        ),
        "is_ev_bus":    r["is_ev_bus"],
        "is_solar_bus": r["is_solar_bus"],
        "is_storage_bus":r["is_storage_bus"],
    })

panel6.sort(key=lambda x: x["headroom_combined_pct"])
print(f"  P1 (act now): {sum(1 for r in panel6 if r['investment_priority']=='P1 — act now')}")
print(f"  P2 (plan 2027): {sum(1 for r in panel6 if r['investment_priority']=='P2 — plan 2027')}")

# ── PANEL 7: PREDICTIVE WINDOW — Jul 9 2023 ───────────────────────────────────
print(f"\n[9/9] Panel 7 — Predictive window ({PRED_DATE}) …")

pred_day  = pd.to_datetime(PRED_DATE)
pred_prev = pred_day - pd.Timedelta(hours=1)

# 24h lookback: Jul 8 00:00 → Jul 8 23:00
lookback_start = pred_day - pd.Timedelta(hours=24)
lookback_end   = pred_day - pd.Timedelta(hours=1)

# 24h horizon: Jul 9 00:00 → Jul 9 23:00
horizon_start  = pred_day
horizon_end    = pred_day + pd.Timedelta(hours=23)

lookback_df = test[
    (test["timestamp"] >= lookback_start) &
    (test["timestamp"] <= lookback_end)
].copy()
horizon_df = test[
    (test["timestamp"] >= horizon_start) &
    (test["timestamp"] <= horizon_end)
].copy()

print(f"  Lookback rows: {len(lookback_df):,}  |  Horizon rows: {len(horizon_df):,}")

# All 5 signals per bus per hour — both windows
def build_window(df, phase):
    records = []
    for _, r in df.sort_values(["bus_name","timestamp"]).iterrows():
        records.append({
            "phase":        phase,
            "bus":          r["bus_name"],
            "timestamp":    r["timestamp"].isoformat(),
            "hour":         int(r["hour"]),
            "net_load_kw":  round(safe(r["net_load_kw"]),     1),
            "thermal_pct":  round(safe(r["thermal_pct"]),     2),
            "ev_kw":        round(safe(r["ev_charging_kw"]),  3),
            "pv_kw":        round(safe(r["pv_output_kw"]),    3),
            "storage_kw":   round(safe(r["storage_net_kw"]),  3),
            "temp_f":       round(safe(r["temp_f"]),          1),
            "is_heatdome":  int(r["is_heatdome"]),
        })
    return records

lookback_records = build_window(lookback_df, "lookback")
horizon_records  = build_window(horizon_df,  "horizon")

# System-level 48h series (sum/mean across buses per hour)
window_sys = pd.concat([lookback_df, horizon_df])
sys_48h = (
    window_sys.groupby("timestamp")
    .agg(
        system_net_load  = ("net_load_kw",      "sum"),
        mean_thermal_pct = ("thermal_pct",      "mean"),
        max_thermal_pct  = ("thermal_pct",      "max"),
        max_temp_f       = ("temp_f",           "max"),
        is_heatdome      = ("is_heatdome",      "max"),
        total_ev_kw      = ("ev_charging_kw",   "sum"),
        total_pv_kw      = ("pv_output_kw",     "sum"),
    )
    .reset_index()
    .sort_values("timestamp")
)

sys_48h_list = []
for _, r in sys_48h.iterrows():
    ts_dt = r["timestamp"]
    phase = "lookback" if ts_dt < pred_day else "horizon"
    sys_48h_list.append({
        "timestamp":      ts_dt.isoformat(),
        "hour":           int(ts_dt.hour),
        "phase":          phase,
        "system_load_kw": round(safe(r["system_net_load"]),  1),
        "mean_thermal":   round(safe(r["mean_thermal_pct"]), 2),
        "max_thermal":    round(safe(r["max_thermal_pct"]),  2),
        "max_temp_f":     round(safe(r["max_temp_f"]),       1),
        "is_heatdome":    int(r["is_heatdome"]),
        "total_ev_kw":    round(safe(r["total_ev_kw"]),      2),
        "total_pv_kw":    round(safe(r["total_pv_kw"]),      2),
    })

# Peak hour snapshot for Jul 9 — bus rankings at worst hour (18:00)
peak_hour_df = horizon_df[horizon_df["hour"] == 18].sort_values("thermal_pct", ascending=False)
peak_snapshot = []
for _, r in peak_hour_df.iterrows():
    t = tier(safe(r["thermal_pct"]))
    peak_snapshot.append({
        "bus":         r["bus_name"],
        "risk_tier":   t,
        "thermal_pct": round(safe(r["thermal_pct"]), 2),
        "net_load_kw": round(safe(r["net_load_kw"]), 1),
        "temp_f":      round(safe(r["temp_f"]),      1),
        "is_heatdome": int(r["is_heatdome"]),
    })

predictive_window = {
    "date":             PRED_DATE,
    "description":      "Jul 9 2023 — worst recorded day in test set (bus 76: 1182.6 kW / 45.6% thermal)",
    "lookback_start":   lookback_start.isoformat(),
    "lookback_end":     lookback_end.isoformat(),
    "horizon_start":    horizon_start.isoformat(),
    "horizon_end":      horizon_end.isoformat(),
    "bus_signals":      lookback_records + horizon_records,
    "system_48h":       sys_48h_list,
    "peak_hour_18_snapshot": peak_snapshot,
}
print(f"  System 48h: {len(sys_48h_list)} rows  |  Peak snapshot: {len(peak_snapshot)} buses")

# ── HEATDOME CUT ──────────────────────────────────────────────────────────────
hd_test = test[test["is_heatdome"] == 1]
hd_daily = (
    hd_test.groupby("date")
    .agg(
        max_temp_f    = ("temp_f",       "max"),
        system_load_max=("net_load_kw",  "max"),
        mean_thermal  = ("thermal_pct",  "mean"),
        max_thermal   = ("thermal_pct",  "max"),
    )
    .reset_index()
)
heatdome_cut = {
    "total_hours": int(len(hd_test["timestamp"].unique())),
    "total_days":  int(hd_test["date"].nunique()),
    "date_range":  f"{HEATDOME_START} to {HEATDOME_END}",
    "daily": [
        {
            "date":           str(r["date"]),
            "max_temp_f":     round(safe(r["max_temp_f"]),     1),
            "system_load_max":round(safe(r["system_load_max"]),1),
            "mean_thermal":   round(safe(r["mean_thermal"]),   2),
            "max_thermal":    round(safe(r["max_thermal"]),    2),
        }
        for _, r in hd_daily.iterrows()
    ],
}

# ── DUCK CURVE CUT ────────────────────────────────────────────────────────────
duck_test = test[test["hour"].isin(DUCK_HOURS)]
duck_hourly = (
    duck_test.groupby("hour")
    .agg(
        net_load_mean  = ("net_load_kw",      "mean"),
        net_load_p95   = ("net_load_kw",      lambda x: pct(x,95)),
        thermal_mean   = ("thermal_pct",      "mean"),
        ev_mean        = ("ev_charging_kw",   "mean"),
        ramp_from_noon = ("net_load_kw",      "mean"),   # will diff below
    )
    .reset_index()
)

noon_mean = float(test[test["hour"]==12]["net_load_kw"].mean())
duck_curve_cut = {
    "hours":     DUCK_HOURS,
    "noon_baseline_mean_kw": round(noon_mean, 2),
    "hourly": [
        {
            "hour":          int(r["hour"]),
            "net_load_mean": round(safe(r["net_load_mean"]), 2),
            "net_load_p95":  round(safe(r["net_load_p95"]),  2),
            "thermal_mean":  round(safe(r["thermal_mean"]),  2),
            "ev_mean":       round(safe(r["ev_mean"]),       3),
            "ramp_from_noon":round(safe(r["net_load_mean"]) - noon_mean, 2),
        }
        for _, r in duck_hourly.iterrows()
    ],
}

# ── SUMMARY ───────────────────────────────────────────────────────────────────
sys_daily_df = test.groupby("date")["net_load_kw"].max().reset_index()
peak_day_row = sys_daily_df.loc[sys_daily_df["net_load_kw"].idxmax()]
max_temp_row = test.loc[test["temp_f"].idxmax()]

summary = {
    "test_period":           "2023-07-01 to 2023-12-31",
    "predictive_date":       PRED_DATE,
    "total_buses":           int(test["bus_name"].nunique()),
    "total_hours":           int(test["timestamp"].nunique()),
    "total_heatdome_days":   int(hd_test["date"].nunique()),
    "peak_day":              str(peak_day_row["date"]),
    "peak_day_load_kw":      round(float(peak_day_row["net_load_kw"]), 1),
    "max_temp_f":            round(float(max_temp_row["temp_f"]), 1),
    "tier_counts":           {k: int(v) for k, v in tier_counts.items()},
    "dispatch_buses":        [r["bus"] for r in panel1 if r["risk_tier"] == "DISPATCH"],
    "monitor_buses":         [r["bus"] for r in panel1 if r["risk_tier"] == "MONITOR"],
    "watch_buses":           [r["bus"] for r in panel1 if r["risk_tier"] == "WATCH"],
    "p1_investment_buses":   [r["bus"] for r in panel6 if r["investment_priority"] == "P1 — act now"],
    "solar_buses":           meta["group_lists"]["solar_buses"],
    "ev_buses":              meta["group_lists"]["ev_buses"],
    "storage_buses":         meta["group_lists"]["storage_buses"],
    "thermal_pct_stats": {
        "mean":   2.35, "p95": 6.90, "p99": 18.85, "max": 45.59,
        "basis":  "test set 2023 Jul-Dec, all 132 buses"
    },
    "net_load_kw_stats": {
        "mean":   60.83, "p99": 489.0, "max": 1182.6,
        "basis":  "test set 2023 Jul-Dec, all 132 buses"
    },
}

# ── ASSEMBLE & SAVE ───────────────────────────────────────────────────────────
print("\nAssembling dashboard_data.json …")

dashboard = {
    "meta":               meta,
    "summary":            summary,
    "descriptive": {
        "panel1_stress_index":      panel1,
        "panel2_load_decomp":       panel2,
        "panel3_hourly_vuln":       panel3,
        "panel4_weather_coupling":  panel4,
        "panel5_duck_ramp":         panel5,
        "panel6_headroom_matrix":   panel6,
    },
    "predictive_window":  predictive_window,
    "heatdome_cut":       heatdome_cut,
    "duck_curve_cut":     duck_curve_cut,
}

with open(OUT_PATH, "w") as f:
    json.dump(dashboard, f, separators=(",", ":"))   # compact — no indent

size_kb = OUT_PATH.stat().st_size / 1024
print(f"\n✓  Saved  : {OUT_PATH}")
print(f"   Size   : {size_kb:.1f} KB")
print(f"\n   Panel 1  : {len(panel1)} buses ranked by thermal_max")
print(f"   Panel 2  : {len(panel2_profile)} hourly + {len(panel2_groups)} group×hour")
print(f"   Panel 3  : {len(panel3)} hourly vulnerability rows")
print(f"   Panel 4  : {len(panel4_scatter)} scatter pts + {len(panel4_system_daily)} daily")
print(f"   Panel 5  : {len(panel5_summary)} bus ramp summaries + {len(panel5_daily)} daily")
print(f"   Panel 6  : {len(panel6)} scenario headroom rows")
print(f"   Panel 7  : {len(predictive_window['bus_signals'])} bus×hour signals (48h)")
print(f"   Heatdome cut : {heatdome_cut['total_days']} days")
print(f"   Duck curve cut: {len(duck_curve_cut['hourly'])} hours")
print(f"\n   DISPATCH: {tier_counts.get('DISPATCH',0)}")
print(f"   MONITOR : {tier_counts.get('MONITOR',0)}")
print(f"   WATCH   : {tier_counts.get('WATCH',0)}")
print(f"   NORMAL  : {tier_counts.get('NORMAL',0)}")
print(f"\nNext: copy dashboard_data.json → Lovable project public/")
print(f"      fetch('/dashboard_data.json') in your React component")