import time
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
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent  # project root

DATA_DIR  = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "models" / "model1_outputs"
OUT_DIR.mkdir(exist_ok=True)

LOOKBACK   = 24    # hours of history fed to LSTM
HORIZON    = 24    # hours to forecast
BATCH_SIZE = 32    # number of windows per batch
EPOCHS     = 20
LR         = 1e-3
HIDDEN_GNN = 64    # GCN hidden dim
HIDDEN_RNN = 128   # LSTM hidden dim
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input features (time-varying, per bus, per hour)
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

print(f"Device: {DEVICE}")
print(f"Input features: {N_FEATURES}  Targets: {N_TARGETS}")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("\nLoading data...")
ts  = pd.read_parquet(DATA_DIR / "az_feeder_timeseries.parquet")
gph = torch.load(DATA_DIR / "az_feeder_graph.pt", weights_only=False)

bus_names  = sorted(ts["bus_name"].unique())
n_buses    = len(bus_names)
bus2idx    = {b: i for i, b in enumerate(bus_names)}
print(f"  Buses: {n_buses}  |  Years: {sorted(ts['year'].unique())}")
print(f"  Total rows: {len(ts):,}")

# Sort so pivot is deterministic
ts = ts.sort_values(["timestamp", "bus_name"]).reset_index(drop=True)

# Pivot to (T × N_buses × features)  — much faster than iterating
print("  Pivoting to 3-D array (timestamps × buses × features)...")
timestamps_all = sorted(ts["timestamp"].unique())
T_all = len(timestamps_all)
ts2idx = {t: i for i, t in enumerate(timestamps_all)}

# Build feature array: shape (T, N_buses, N_features)
feat_arr   = np.zeros((T_all, n_buses, N_FEATURES), dtype=np.float32)
target_arr = np.zeros((T_all, n_buses, N_TARGETS),  dtype=np.float32)

for col_i, col in enumerate(INPUT_FEATURES):
    pivot = ts.pivot(index="timestamp", columns="bus_name", values=col)
    pivot = pivot.reindex(index=timestamps_all, columns=bus_names).ffill().bfill()
    feat_arr[:, :, col_i] = pivot.values.astype(np.float32)

for col_i, col in enumerate(TARGET_COLS):
    pivot = ts.pivot(index="timestamp", columns="bus_name", values=col)
    pivot = pivot.reindex(index=timestamps_all, columns=bus_names).ffill().bfill()
    target_arr[:, :, col_i] = pivot.values.astype(np.float32)

# Split mask by split column
split_ser = ts.drop_duplicates("timestamp").set_index("timestamp")["split"].reindex(timestamps_all)
train_mask = np.array([split_ser.iloc[i] == "train" for i in range(T_all)])
val_mask   = np.array([split_ser.iloc[i] == "val"   for i in range(T_all)])
test_mask  = np.array([split_ser.iloc[i] == "test"  for i in range(T_all)])

train_idx = np.where(train_mask)[0]
val_idx   = np.where(val_mask)[0]
test_idx  = np.where(test_mask)[0]
print(f"  Train steps: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

# Normalise features using train statistics only
print("  Normalising features...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_2d = feat_arr[train_idx].reshape(-1, N_FEATURES)
y_train_2d = target_arr[train_idx].reshape(-1, N_TARGETS)

scaler_X.fit(X_train_2d)
scaler_y.fit(y_train_2d)

feat_arr_norm   = scaler_X.transform(feat_arr.reshape(-1, N_FEATURES)).reshape(T_all, n_buses, N_FEATURES)
target_arr_norm = scaler_y.transform(target_arr.reshape(-1, N_TARGETS)).reshape(T_all, n_buses, N_TARGETS)

feat_tensor   = torch.tensor(feat_arr_norm,   dtype=torch.float32)
target_tensor = torch.tensor(target_arr_norm, dtype=torch.float32)

# Graph topology — move to device
edge_index = gph.edge_index.to(DEVICE)
edge_attr  = gph.edge_attr.to(DEVICE)
node_feat  = gph.x.to(DEVICE)      # static node features, shape (N, 11)

# ── DATASET ───────────────────────────────────────────────────────────────────
class SlidingWindowDataset(Dataset):
    """
    Each sample: (X_seq, y_seq)
      X_seq : (LOOKBACK, N_buses, N_features)
      y_seq : (HORIZON,  N_buses, N_targets)
    """
    def __init__(self, indices):
        # Only keep windows that are fully contained (no cross-year bleed)
        valid = [i for i in indices
                 if i >= LOOKBACK and i + HORIZON <= T_all]
        self.windows = valid

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        t = self.windows[idx]
        X = feat_tensor[t - LOOKBACK : t]           # (L, N, F)
        y = target_tensor[t : t + HORIZON]          # (H, N, T)
        return X, y

train_ds = SlidingWindowDataset(train_idx)
val_ds   = SlidingWindowDataset(val_idx)
test_ds  = SlidingWindowDataset(test_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"  Train batches: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}")

# ── MODEL ─────────────────────────────────────────────────────────────────────
class GNNLSTM(nn.Module):
    """
    Step 1 — GCN: at each timestep, mix node features across the graph.
    Step 2 — LSTM: encode the resulting sequence of spatial embeddings.
    Step 3 — Linear head: decode to (HORIZON, N_buses, N_targets).
    """
    def __init__(self, n_feat, n_static, n_targets, hidden_gnn, hidden_rnn, horizon):
        super().__init__()
        self.horizon = horizon
        self.n_buses = n_buses

        # GCN layers — combine dynamic features with static node embedding
        self.node_embed = nn.Linear(n_static, hidden_gnn)
        self.gcn1 = GCNConv(n_feat + hidden_gnn, hidden_gnn)
        self.gcn2 = GCNConv(hidden_gnn, hidden_gnn)

        # LSTM over time
        self.lstm = nn.LSTM(
            input_size=hidden_gnn * n_buses,
            hidden_size=hidden_rnn,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Output head: hidden_rnn → horizon × n_buses × n_targets
        self.head = nn.Sequential(
            nn.Linear(hidden_rnn, hidden_rnn),
            nn.ReLU(),
            nn.Linear(hidden_rnn, horizon * n_buses * n_targets),
        )

    def forward(self, x_seq, edge_index, edge_attr, node_feat):
        """
        x_seq    : (B, L, N, F)   dynamic features
        node_feat: (N, S)          static node features
        Returns  : (B, H, N, T)
        """
        B, L, N, num_dynamic_features = x_seq.shape

        # Embed static node features once
        static_emb = F.relu(self.node_embed(node_feat))  # (N, hidden_gnn)

        # Apply GCN at every timestep
        gcn_out = []
        for t in range(L):
            xt = x_seq[:, t, :, :]          # (B, N, F)
            xt_flat = xt.reshape(B * N, num_dynamic_features)  # flatten batch for GCN

            # Tile static embedding across batch
            static_t = static_emb.unsqueeze(0).expand(B, -1, -1)
            static_t = static_t.reshape(B * N, -1)

            # Concatenate dynamic + static
            node_in = torch.cat([xt_flat, static_t], dim=-1)  # (B*N, F+S)

            # Tile edge_index across batch (offset node ids per sample)
            batch_edge_index = edge_index.clone()
            all_edges = []
            for b in range(B):
                offset = b * N
                all_edges.append(edge_index + offset)
            batch_ei = torch.cat(all_edges, dim=1)

            h = F.relu(self.gcn1(node_in, batch_ei))
            h = F.relu(self.gcn2(h, batch_ei))     # (B*N, hidden_gnn)
            h = h.reshape(B, N * self.gcn2.out_channels)  # (B, N*hidden)
            gcn_out.append(h)

        # Stack along time → (B, L, N*hidden)
        seq = torch.stack(gcn_out, dim=1)

        # LSTM over time
        lstm_out, _ = self.lstm(seq)        # (B, L, hidden_rnn)
        last = lstm_out[:, -1, :]           # (B, hidden_rnn)

        # Decode
        out = self.head(last)               # (B, H*N*T)
        out = out.reshape(B, self.horizon, N, -1)  # (B, H, N, T)
        return out


model = GNNLSTM(
    n_feat=N_FEATURES,
    n_static=node_feat.shape[1],
    n_targets=N_TARGETS,
    hidden_gnn=HIDDEN_GNN,
    hidden_rnn=HIDDEN_RNN,
    horizon=HORIZON,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5
)

# ── TRAINING ──────────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches  = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X = X.to(DEVICE)   # (B, L, N, F)
            y = y.to(DEVICE)   # (B, H, N, T)

            pred = model(X, edge_index, edge_attr, node_feat)
            loss = F.mse_loss(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
    return total_loss / max(n_batches, 1)


print("\n" + "="*55)
print("TRAINING — GNN + LSTM")
print("="*55)

history = {"train_loss": [], "val_loss": [], "epoch_time": []}
best_val  = float("inf")
best_path = OUT_DIR / "best_model1.pt"

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss  = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader,   train=False)
    elapsed  = time.time() - t0

    scheduler.step(val_loss)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(val_loss)
    history["epoch_time"].append(elapsed)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), best_path)
        marker = " ◀ best"
    else:
        marker = ""

    print(f"  Epoch {epoch:>2}/{EPOCHS}  "
          f"train={tr_loss:.4f}  val={val_loss:.4f}  "
          f"time={elapsed:.1f}s{marker}")

total_train_time = sum(history["epoch_time"])
print(f"\nTotal training time: {total_train_time/60:.1f} min")

# ── EVALUATION ────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("EVALUATION — Test set (2023 Jul–Dec)")
print("="*55)

model.load_state_dict(torch.load(best_path, weights_only=True))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(DEVICE)
        pred = model(X, edge_index, edge_attr, node_feat)
        all_preds.append(pred.cpu().numpy())
        all_true.append(y.numpy())

preds_norm = np.concatenate(all_preds, axis=0)   # (n_samples, H, N, T)
truth_norm = np.concatenate(all_true,  axis=0)

# Inverse-transform
shape = preds_norm.shape
preds_kw = scaler_y.inverse_transform(
    preds_norm.reshape(-1, N_TARGETS)
).reshape(shape)
truth_kw = scaler_y.inverse_transform(
    truth_norm.reshape(-1, N_TARGETS)
).reshape(shape)

# Per-target metrics (averaged over buses and horizon)
print(f"\n{'Metric':<20} {'net_load_kw':>14} {'thermal_pct':>14}")
print("-" * 50)
for ti, tname in enumerate(TARGET_COLS):
    p = preds_kw[..., ti].ravel()
    t = truth_kw[..., ti].ravel()
    mae  = mean_absolute_error(t, p)
    rmse = math.sqrt(mean_squared_error(t, p))
    r2   = r2_score(t, p)
    mape = np.mean(np.abs((t - p) / (np.abs(t) + 1e-6))) * 100
    if ti == 0:
        row = f"  {'MAE (kW)':18} {mae:>14.2f}"
        row2 = f"  {'RMSE (kW)':18} {rmse:>14.2f}"
        row3 = f"  {'R²':18} {r2:>14.4f}"
        row4 = f"  {'MAPE (%)':18} {mape:>14.2f}"
        print(row);print(row2);print(row3);print(row4)
    else:
        pass   # printed below

print()
for ti, tname in enumerate(TARGET_COLS):
    p = preds_kw[..., ti].ravel()
    t = truth_kw[..., ti].ravel()
    mae  = mean_absolute_error(t, p)
    rmse = math.sqrt(mean_squared_error(t, p))
    r2   = r2_score(t, p)
    mape = np.mean(np.abs((t - p) / (np.abs(t) + 1e-6))) * 100
    print(f"  [{tname}]  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  MAPE={mape:.2f}%")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
axes[0].plot(history["train_loss"], label="Train MSE", marker="o", markersize=3)
axes[0].plot(history["val_loss"],   label="Val MSE",   marker="s", markersize=3)
axes[0].set_title("GNN+LSTM — Training Loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE (normalised)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Forecast vs actual for bus 0, first test sample, net_load_kw
sample_pred = preds_kw[0, :, 0, 0]   # (H,)
sample_true = truth_kw[0, :, 0, 0]
axes[1].plot(sample_true, label="Actual",    color="steelblue")
axes[1].plot(sample_pred, label="Forecast",  color="tomato", linestyle="--")
axes[1].set_title(f"24-h Forecast — Bus 0 (net_load_kw)")
axes[1].set_xlabel("Hour ahead"); axes[1].set_ylabel("kW")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "model1_results.png", dpi=120)
print(f"\nPlot saved → {OUT_DIR / 'model1_results.png'}")

# Per-bus summary
bus_maes = []
for b in range(n_buses):
    p = preds_kw[:, :, b, 0].ravel()
    t = truth_kw[:, :, b, 0].ravel()
    bus_maes.append(mean_absolute_error(t, p))

bus_mae_df = pd.DataFrame({
    "bus_name": bus_names,
    "mae_net_load_kw": bus_maes
}).sort_values("mae_net_load_kw", ascending=False)
bus_mae_df.to_csv(OUT_DIR / "model1_per_bus_mae.csv", index=False)
print(f"Per-bus MAE saved → {OUT_DIR / 'model1_per_bus_mae.csv'}")
print(f"\nWorst 5 buses:\n{bus_mae_df.head()}")

print("\nDone — Model 1 (GNN + LSTM)")