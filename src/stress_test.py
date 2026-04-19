"""
stress_test.py  (memory-safe edition)
======================================
Runs baseline → delete → heatdome → delete → ev2027
Never holds more than ONE scenario array in RAM at a time.

Scenario A — Arizona Mega Heat Dome (July 8–21 2023 window)
  +8 F on temp features, force is_heatdome=1, boost heat_scenario_mult

Scenario B — EV Surge 2027 (full test period, duck-curve 17-21h)
  ev_charging_kw x2.75, is_duck_curve_window forced 17-21h

OUTPUT: stress_test_outputs/
  stress_summary.csv
  overload_alerts.csv
  scenario_A_heatdome_results.csv
  scenario_B_ev2027_results.csv
  stress_test_plots.png
"""

import gc
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# ── PATHS ──────────────────────────────────────────────────────────────────────
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent  # project root

DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "model1_outputs"
OUT_DIR   = BASE_DIR / "results" / "stress_test_outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── CONFIG (must match training exactly) ──────────────────────────────────────
LOOKBACK   = 24
HORIZON    = 24
BATCH_SIZE = 16          # halved from training to save GPU memory
HIDDEN_GNN = 64
HIDDEN_RNN = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "temp_f", "temp_lag1", "temp_lag24",
    "ghi_wm2", "clearsky_ratio",
    "load_lag1", "load_lag24", "load_lag168",
    "load_roll7d_mean", "load_roll7d_std",
    "ev_charging_kw", "storage_net_kw",
    "aps_demand_mw_norm", "aps_demand_lag24_mw",
    "is_heatdome", "heat_scenario_mult", "is_duck_curve_window",
    "temp_x_load_mult",
    "is_solar_bus", "is_ev_bus", "is_storage_bus",
    "is_substation", "is_reg_secondary",
    "is_weekend",
]
TARGET_COLS = ["net_load_kw", "thermal_pct"]
N_FEATURES  = len(INPUT_FEATURES)
N_TARGETS   = len(TARGET_COLS)
IDX = {f: i for i, f in enumerate(INPUT_FEATURES)}

# ── SCENARIO PARAMS ────────────────────────────────────────────────────────────
HEATDOME_START   = "2023-07-08"
HEATDOME_END     = "2023-07-21"
TEMP_AMPLIFY_F   = 8.0
HEAT_MULT_BOOST  = 0.35
EV_SCALE_FACTOR  = 2.75
DUCK_START_H     = 17
DUCK_END_H       = 21
THERMAL_ALERT    = 0.90

print("=" * 58)
print("STRESS TEST — memory-safe single-pass mode")
print(f"Device : {DEVICE}")
print(f"Scenario A : Heat Dome +{TEMP_AMPLIFY_F}F  ({HEATDOME_START} -> {HEATDOME_END})")
print(f"Scenario B : EV 2027  x{EV_SCALE_FACTOR}  duck-curve {DUCK_START_H}-{DUCK_END_H}h")
print("=" * 58)

# ── LOAD DATA (once, shared) ───────────────────────────────────────────────────
print("\n[1/7] Loading parquet + graph...")
ts  = pd.read_parquet(DATA_DIR / "az_feeder_timeseries.parquet")
gph = torch.load(DATA_DIR / "az_feeder_graph.pt", weights_only=False)

bus_names = sorted(ts["bus_name"].unique())
n_buses   = len(bus_names)
ts = ts.sort_values(["timestamp", "bus_name"]).reset_index(drop=True)
timestamps_all = sorted(ts["timestamp"].unique())
T_all = len(timestamps_all)
ts_pd = pd.DatetimeIndex(timestamps_all)

print(f"  Buses: {n_buses}  |  Timestamps: {T_all:,}")

# Split indices
split_ser  = ts.drop_duplicates("timestamp").set_index("timestamp")["split"].reindex(timestamps_all)
train_idx  = np.where([split_ser.iloc[i] == "train" for i in range(T_all)])[0]
test_idx   = np.where([split_ser.iloc[i] == "test"  for i in range(T_all)])[0]
print(f"  Train: {len(train_idx):,}  |  Test: {len(test_idx):,}")

# Graph tensors (stay on device the whole time)
edge_index = gph.edge_index.to(DEVICE)
edge_attr  = gph.edge_attr.to(DEVICE)
node_feat  = gph.x.to(DEVICE)

# ── BUILD & FIT SCALERS FROM TRAIN (pivot once, fit, delete) ──────────────────
print("\n[2/7] Fitting scalers on train split...")

def pivot_feature(col):
    p = ts.pivot(index="timestamp", columns="bus_name", values=col)
    return p.reindex(index=timestamps_all, columns=bus_names).ffill().bfill().values.astype(np.float32)

X_train_rows = []
y_train_rows = []
for ci, col in enumerate(INPUT_FEATURES):
    vals = pivot_feature(col)[train_idx]
    X_train_rows.append(vals.reshape(-1))
X_train_2d = np.stack(X_train_rows, axis=1)

for ci, col in enumerate(TARGET_COLS):
    vals = pivot_feature(col)[train_idx]
    y_train_rows.append(vals.reshape(-1))
y_train_2d = np.stack(y_train_rows, axis=1)

scaler_X = StandardScaler().fit(X_train_2d)
scaler_y = StandardScaler().fit(y_train_2d)

del X_train_rows, X_train_2d, y_train_rows, y_train_2d
gc.collect()
print("  Scalers fitted and train arrays freed.")

# ── BUILD FULL RAW ARRAYS ─────────────────────────────────────────────────────
print("\n[3/7] Building raw feature + target arrays...")

feat_raw   = np.zeros((T_all, n_buses, N_FEATURES), dtype=np.float32)
target_raw = np.zeros((T_all, n_buses, N_TARGETS),  dtype=np.float32)

for ci, col in enumerate(INPUT_FEATURES):
    feat_raw[:, :, ci] = pivot_feature(col)
    if (ci + 1) % 7 == 0 or ci == N_FEATURES - 1:
        print(f"  Features pivoted: {ci+1}/{N_FEATURES}")

for ci, col in enumerate(TARGET_COLS):
    target_raw[:, :, ci] = pivot_feature(col)

target_norm = scaler_y.transform(
    target_raw.reshape(-1, N_TARGETS)).reshape(T_all, n_buses, N_TARGETS)
target_tensor = torch.tensor(target_norm, dtype=torch.float32)

del target_raw, target_norm
gc.collect()
print(f"  feat_raw shape : {feat_raw.shape}  |  target tensor ready.")

# ── MODEL ─────────────────────────────────────────────────────────────────────
print("\n[4/7] Loading model weights...")

try:
    from torch_geometric.nn import GCNConv as _GCNConv
except ImportError:
    raise ImportError("pip install torch-geometric")

class GNNLSTM(nn.Module):
    def __init__(self, n_feat, n_static, n_targets, hidden_gnn, hidden_rnn, horizon):
        super().__init__()
        self.horizon = horizon
        self.n_buses = n_buses
        self.node_embed = nn.Linear(n_static, hidden_gnn)
        self.gcn1 = _GCNConv(n_feat + hidden_gnn, hidden_gnn)
        self.gcn2 = _GCNConv(hidden_gnn, hidden_gnn)
        self.lstm = nn.LSTM(
            input_size=hidden_gnn * n_buses,
            hidden_size=hidden_rnn,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_rnn, hidden_rnn),
            nn.ReLU(),
            nn.Linear(hidden_rnn, horizon * n_buses * n_targets),
        )

    def forward(self, x_seq, edge_index, edge_attr, node_feat):
        B, L, N, F_ = x_seq.shape
        static_emb = F.relu(self.node_embed(node_feat))
        gcn_out = []
        for t in range(L):
            xt = x_seq[:, t].reshape(B * N, F_)
            st = static_emb.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
            node_in = torch.cat([xt, st], dim=-1)
            batch_ei = torch.cat([edge_index + b * N for b in range(B)], dim=1)
            h = F.relu(self.gcn1(node_in, batch_ei))
            h = F.relu(self.gcn2(h, batch_ei))
            gcn_out.append(h.reshape(B, N * self.gcn2.out_channels))
        seq = torch.stack(gcn_out, dim=1)
        lstm_out, _ = self.lstm(seq)
        out = self.head(lstm_out[:, -1])
        return out.reshape(B, self.horizon, N, -1)

model = GNNLSTM(
    n_feat=N_FEATURES,
    n_static=node_feat.shape[1],
    n_targets=N_TARGETS,
    hidden_gnn=HIDDEN_GNN,
    hidden_rnn=HIDDEN_RNN,
    horizon=HORIZON,
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_DIR / "best_model1.pt",
                                 map_location=DEVICE, weights_only=True))
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model loaded  ({n_params:,} params)")

# ── DATASET + INFERENCE ───────────────────────────────────────────────────────
class ScenarioDataset(Dataset):
    def __init__(self, feat_t):
        self.windows = [i for i in test_idx if i >= LOOKBACK and i + HORIZON <= T_all]
        self.feat_t  = feat_t
    def __len__(self):  return len(self.windows)
    def __getitem__(self, idx):
        t = self.windows[idx]
        return self.feat_t[t - LOOKBACK:t], target_tensor[t:t + HORIZON]

def run_inference(feat_norm_arr, label):
    print(f"\n  [{label}] Normalising array...")
    feat_norm = scaler_X.transform(
        feat_norm_arr.reshape(-1, N_FEATURES)).reshape(T_all, n_buses, N_FEATURES)
    feat_t = torch.tensor(feat_norm, dtype=torch.float32)
    del feat_norm
    gc.collect()

    ds     = ScenarioDataset(feat_t)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  [{label}] Inference: {len(ds)} windows, {len(loader)} batches...")

    preds_list, truth_list = [], []
    with torch.no_grad():
        for bi, (X, y) in enumerate(loader):
            pred = model(X.to(DEVICE), edge_index, edge_attr, node_feat)
            preds_list.append(pred.cpu().numpy())
            truth_list.append(y.numpy())
            if (bi + 1) % 50 == 0:
                print(f"    batch {bi+1}/{len(loader)}")
            if (bi + 1) % 50 == 0 and DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    del feat_t
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    preds_norm = np.concatenate(preds_list, axis=0)
    truth_norm = np.concatenate(truth_list, axis=0)
    shape = preds_norm.shape

    preds_kw = scaler_y.inverse_transform(preds_norm.reshape(-1, N_TARGETS)).reshape(shape)
    truth_kw = scaler_y.inverse_transform(truth_norm.reshape(-1, N_TARGETS)).reshape(shape)

    mae_load  = mean_absolute_error(truth_kw[..., 0].ravel(), preds_kw[..., 0].ravel())
    mae_therm = mean_absolute_error(truth_kw[..., 1].ravel(), preds_kw[..., 1].ravel())
    print(f"  [{label}] DONE  MAE net_load={mae_load:.2f} kW  MAE thermal={mae_therm:.4f}")
    return preds_kw

# ── PERTURBATION FUNCTIONS ────────────────────────────────────────────────────
def perturb_heatdome(arr):
    out = arr.copy()
    hd_mask = (ts_pd >= HEATDOME_START) & (ts_pd <= HEATDOME_END)
    idx = np.where(hd_mask)[0]
    print(f"  Heat dome mask: {len(idx)} timestamps  ({len(idx)//24} days)")
    for f in ["temp_f", "temp_lag1", "temp_lag24"]:
        out[idx, :, IDX[f]] += TEMP_AMPLIFY_F
    out[idx, :, IDX["is_heatdome"]]        = 1.0
    out[idx, :, IDX["heat_scenario_mult"]] = np.clip(
        out[idx, :, IDX["heat_scenario_mult"]] + HEAT_MULT_BOOST, 0, 1.5)
    T_old = arr[idx, :, IDX["temp_f"]]
    T_new = out[idx, :, IDX["temp_f"]]
    ratio = np.where(np.abs(T_old) > 1e-3, T_new / T_old, 1.0)
    out[idx, :, IDX["temp_x_load_mult"]] *= ratio
    return out

def perturb_ev2027(arr):
    out = arr.copy()
    out[:, :, IDX["ev_charging_kw"]] *= EV_SCALE_FACTOR
    duck_idx = np.where((ts_pd.hour >= DUCK_START_H) & (ts_pd.hour < DUCK_END_H))[0]
    out[duck_idx, :, IDX["is_duck_curve_window"]] = 1.0
    print(f"  EV scale x{EV_SCALE_FACTOR}  |  duck-curve timestamps: {len(duck_idx):,}")
    return out

# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — BASELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*58)
print("[5/7] PASS 1 — BASELINE")
print("="*58)
preds_base = run_inference(feat_raw, "Baseline")

N_PROF = min(14 * 24, preds_base.shape[0])
base_load_profile   = preds_base[:N_PROF, 0, :, 0].sum(axis=1)
base_therm_profile  = preds_base[:N_PROF, 0, :, 1].mean(axis=1)
base_bus_avg_load   = preds_base[..., 0].mean(axis=(0, 1))
base_bus_peak_load  = preds_base[..., 0].max(axis=(0, 1))
base_bus_avg_therm  = preds_base[..., 1].mean(axis=(0, 1))
base_bus_peak_therm = preds_base[..., 1].max(axis=(0, 1))

del preds_base
gc.collect()
print("  Baseline summarised and freed.")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — HEAT DOME
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*58)
print("[6/7] PASS 2 — SCENARIO A: MEGA HEAT DOME")
print("="*58)
feat_heat  = perturb_heatdome(feat_raw)
preds_heat = run_inference(feat_heat, "HeatDome")
del feat_heat
gc.collect()

heat_load_profile   = preds_heat[:N_PROF, 0, :, 0].sum(axis=1)
heat_therm_profile  = preds_heat[:N_PROF, 0, :, 1].mean(axis=1)
heat_bus_avg_load   = preds_heat[..., 0].mean(axis=(0, 1))
heat_bus_peak_load  = preds_heat[..., 0].max(axis=(0, 1))
heat_bus_peak_therm = preds_heat[..., 1].max(axis=(0, 1))

pd.DataFrame({
    "hour_offset":          np.arange(N_PROF),
    "baseline_load_kw":     base_load_profile,
    "heatdome_load_kw":     heat_load_profile,
    "delta_load_kw":        heat_load_profile - base_load_profile,
    "baseline_thermal_pct": base_therm_profile,
    "heatdome_thermal_pct": heat_therm_profile,
}).to_csv(OUT_DIR / "scenario_A_heatdome_results.csv", index=False)
print(f"  Saved: scenario_A_heatdome_results.csv")

del preds_heat
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# PASS 3 — EV 2027
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*58)
print("[7/7] PASS 3 — SCENARIO B: EV SURGE 2027")
print("="*58)
feat_ev   = perturb_ev2027(feat_raw)
preds_ev  = run_inference(feat_ev, "EV-2027")
del feat_ev, feat_raw
gc.collect()

ev_load_profile   = preds_ev[:N_PROF, 0, :, 0].sum(axis=1)
ev_therm_profile  = preds_ev[:N_PROF, 0, :, 1].mean(axis=1)
ev_bus_avg_load   = preds_ev[..., 0].mean(axis=(0, 1))
ev_bus_peak_load  = preds_ev[..., 0].max(axis=(0, 1))
ev_bus_peak_therm = preds_ev[..., 1].max(axis=(0, 1))

pd.DataFrame({
    "hour_offset":          np.arange(N_PROF),
    "baseline_load_kw":     base_load_profile,
    "ev2027_load_kw":       ev_load_profile,
    "delta_load_kw":        ev_load_profile - base_load_profile,
    "baseline_thermal_pct": base_therm_profile,
    "ev2027_thermal_pct":   ev_therm_profile,
}).to_csv(OUT_DIR / "scenario_B_ev2027_results.csv", index=False)
print(f"  Saved: scenario_B_ev2027_results.csv")

del preds_ev
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY + ALERTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*58)
print("BUILDING SUMMARY + ALERT TABLES")
print("="*58)

summary_df = pd.DataFrame({
    "bus_name":              bus_names,
    "base_avg_load_kw":      base_bus_avg_load,
    "base_peak_load_kw":     base_bus_peak_load,
    "base_peak_thermal_pct": base_bus_peak_therm,
    "heat_avg_load_kw":      heat_bus_avg_load,
    "heat_peak_load_kw":     heat_bus_peak_load,
    "heat_peak_thermal_pct": heat_bus_peak_therm,
    "heat_delta_avg_kw":     heat_bus_avg_load  - base_bus_avg_load,
    "heat_delta_peak_kw":    heat_bus_peak_load - base_bus_peak_load,
    "ev_avg_load_kw":        ev_bus_avg_load,
    "ev_peak_load_kw":       ev_bus_peak_load,
    "ev_peak_thermal_pct":   ev_bus_peak_therm,
    "ev_delta_avg_kw":       ev_bus_avg_load  - base_bus_avg_load,
    "ev_delta_peak_kw":      ev_bus_peak_load - base_bus_peak_load,
})
summary_df = summary_df.sort_values("heat_peak_thermal_pct", ascending=False)
summary_df.to_csv(OUT_DIR / "stress_summary.csv", index=False)
print(f"  Saved: stress_summary.csv  ({len(summary_df)} buses)")

alerts_heat = summary_df[summary_df["heat_peak_thermal_pct"] >= THERMAL_ALERT][
    ["bus_name","base_peak_thermal_pct","heat_peak_thermal_pct","heat_delta_peak_kw"]
].copy()
alerts_heat["scenario"] = "Mega_HeatDome"

alerts_ev = summary_df[summary_df["ev_peak_thermal_pct"] >= THERMAL_ALERT][
    ["bus_name","base_peak_thermal_pct","ev_peak_thermal_pct","ev_delta_peak_kw"]
].copy()
alerts_ev["scenario"] = "EV_2027"

overload_df = pd.concat([
    alerts_heat.rename(columns={"heat_peak_thermal_pct": "scenario_thermal",
                                "heat_delta_peak_kw":    "delta_peak_kw"}),
    alerts_ev.rename(  columns={"ev_peak_thermal_pct":   "scenario_thermal",
                                "ev_delta_peak_kw":      "delta_peak_kw"}),
], ignore_index=True).sort_values("scenario_thermal", ascending=False)

overload_df.to_csv(OUT_DIR / "overload_alerts.csv", index=False)
print(f"  Saved: overload_alerts.csv")
print(f"  Heat dome buses >= 90% thermal : {len(alerts_heat)}")
print(f"  EV 2027 buses   >= 90% thermal : {len(alerts_ev)}")

# ══════════════════════════════════════════════════════════════════════════════
# 6-PANEL FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating 6-panel figure...")

hours = np.arange(N_PROF)
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("APS-like Feeder — Stress Test Results", fontsize=14, fontweight="bold")

ax = axes[0, 0]
ax.plot(hours, base_load_profile / 1000, label="Baseline",              color="steelblue",  lw=1.8)
ax.plot(hours, heat_load_profile / 1000, label=f"Heat Dome +{TEMP_AMPLIFY_F}F",
        color="tomato", lw=1.8, linestyle="--")
ax.fill_between(hours, base_load_profile/1000, heat_load_profile/1000, alpha=0.15, color="tomato")
ax.set_title("Scenario A — Feeder Total Load (MW)")
ax.set_xlabel("Hour offset"); ax.set_ylabel("MW")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(hours, base_load_profile / 1000, label="Baseline",               color="steelblue",  lw=1.8)
ax.plot(hours, ev_load_profile   / 1000, label=f"EV 2027 x{EV_SCALE_FACTOR}",
        color="darkorange", lw=1.8, linestyle="--")
ax.fill_between(hours, base_load_profile/1000, ev_load_profile/1000, alpha=0.15, color="darkorange")
ax.set_title("Scenario B — Feeder Total Load (MW)")
ax.set_xlabel("Hour offset"); ax.set_ylabel("MW")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(hours, base_therm_profile, label="Baseline",              color="steelblue", lw=1.8)
ax.plot(hours, heat_therm_profile, label=f"Heat Dome +{TEMP_AMPLIFY_F}F",
        color="tomato", lw=1.8, linestyle="--")
ax.axhline(THERMAL_ALERT, color="red", linestyle=":", lw=1.4, label="90% alert")
ax.set_title("Scenario A — Mean Thermal Loading")
ax.set_xlabel("Hour offset"); ax.set_ylabel("Thermal fraction")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(hours, base_therm_profile, label="Baseline",             color="steelblue",  lw=1.8)
ax.plot(hours, ev_therm_profile,   label=f"EV 2027 x{EV_SCALE_FACTOR}",
        color="darkorange", lw=1.8, linestyle="--")
ax.axhline(THERMAL_ALERT, color="red", linestyle=":", lw=1.4, label="90% alert")
ax.set_title("Scenario B — Mean Thermal Loading")
ax.set_xlabel("Hour offset"); ax.set_ylabel("Thermal fraction")
ax.legend(); ax.grid(True, alpha=0.3)

top20_h = summary_df.head(20)
top20_e = summary_df.sort_values("ev_delta_peak_kw", ascending=False).head(20)

ax = axes[2, 0]
colors_h = ["tomato" if v >= THERMAL_ALERT else "salmon"
            for v in top20_h["heat_peak_thermal_pct"]]
ax.barh(top20_h["bus_name"], top20_h["heat_delta_peak_kw"],
        color=colors_h, edgecolor="white", height=0.7)
ax.axvline(0, color="gray", lw=0.8)
ax.set_title("Scenario A — Top 20 Buses: Peak Load Increase")
ax.set_xlabel("Delta Peak kW vs Baseline")
ax.grid(True, alpha=0.3, axis="x")
for i, row in enumerate(top20_h.itertuples()):
    if row.heat_peak_thermal_pct >= THERMAL_ALERT:
        ax.text(row.heat_delta_peak_kw + 0.3, i, "!", va="center", fontsize=9, color="red")

ax = axes[2, 1]
colors_e = ["darkorange" if v >= THERMAL_ALERT else "moccasin"
            for v in top20_e["ev_peak_thermal_pct"]]
ax.barh(top20_e["bus_name"], top20_e["ev_delta_peak_kw"],
        color=colors_e, edgecolor="white", height=0.7)
ax.axvline(0, color="gray", lw=0.8)
ax.set_title("Scenario B — Top 20 Buses: Peak Load Increase")
ax.set_xlabel("Delta Peak kW vs Baseline")
ax.grid(True, alpha=0.3, axis="x")
for i, row in enumerate(top20_e.itertuples()):
    if row.ev_peak_thermal_pct >= THERMAL_ALERT:
        ax.text(row.ev_delta_peak_kw + 0.3, i, "!", va="center", fontsize=9, color="darkorange")

plt.tight_layout()
fig.savefig(OUT_DIR / "stress_test_plots.png", dpi=130, bbox_inches="tight")
print(f"  Saved: stress_test_plots.png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# FINAL PRINT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*58)
print("STRESS TEST COMPLETE")
print("="*58)

print(f"\n  Scenario A — Mega Heat Dome (+{TEMP_AMPLIFY_F}F)")
print(f"    Avg feeder load delta : {(heat_bus_avg_load - base_bus_avg_load).mean():.1f} kW/bus")
print(f"    Peak load delta       : {(heat_bus_peak_load - base_bus_peak_load).mean():.1f} kW/bus")
print(f"    Buses >= 90% thermal  : {len(alerts_heat)}")
if len(alerts_heat):
    print(f"    Top 3 at-risk buses   : {', '.join(alerts_heat['bus_name'].head(3).tolist())}")

print(f"\n  Scenario B — EV Surge 2027 (x{EV_SCALE_FACTOR})")
print(f"    Avg feeder load delta : {(ev_bus_avg_load - base_bus_avg_load).mean():.1f} kW/bus")
print(f"    Peak load delta       : {(ev_bus_peak_load - base_bus_peak_load).mean():.1f} kW/bus")
print(f"    Buses >= 90% thermal  : {len(alerts_ev)}")
if len(alerts_ev):
    print(f"    Top 3 at-risk buses   : {', '.join(alerts_ev['bus_name'].head(3).tolist())}")

print(f"\n  Output files in: {OUT_DIR}/")
for f in sorted(OUT_DIR.iterdir()):
    size_kb = f.stat().st_size // 1024
    print(f"    {f.name:<45} {size_kb:>5} KB")

print("\nDone.")