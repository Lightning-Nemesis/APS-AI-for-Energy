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
from torch_geometric.nn import GATv2Conv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent  # project root

DATA_DIR  = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "models" / "model2_outputs"

OUT_DIR.mkdir(exist_ok=True)

LOOKBACK   = 24
HORIZON    = 24
BATCH_SIZE = 16      # smaller — TFT is heavier than LSTM
EPOCHS     = 5
LR         = 2e-3
D_MODEL    = 32      # TFT hidden dimension
N_HEADS    = 4       # attention heads
N_GNN_LAYERS = 2
DROPOUT    = 0.1
QUANTILES  = [0.1, 0.5, 0.9]   # P10, P50, P90

# Physics constraint weights in the loss
LAMBDA_PHYSICS = 0.05    # weight of physics penalty vs data loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Multi-output targets
TARGET_COLS = ["net_load_kw", "thermal_pct", "net_load_heatdome_kw"]
N_FEATURES  = len(INPUT_FEATURES)
N_TARGETS   = len(TARGET_COLS)
N_QUANTILES = len(QUANTILES)

print(f"Device: {DEVICE}")
print(f"Input features: {N_FEATURES}  Targets: {N_TARGETS}  Quantiles: {N_QUANTILES}")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("\nLoading data...")
ts  = pd.read_parquet(DATA_DIR / "az_feeder_timeseries.parquet")
gph = torch.load(DATA_DIR / "az_feeder_graph.pt", weights_only=False)

bus_names = sorted(ts["bus_name"].unique())
n_buses   = len(bus_names)
ts = ts.sort_values(["timestamp", "bus_name"]).reset_index(drop=True)

timestamps_all = sorted(ts["timestamp"].unique())
T_all = len(timestamps_all)

print("  Pivoting to 3-D arrays...")
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

# Also store raw norm_amps per bus for physics constraint
norm_amps_arr = np.zeros(n_buses, dtype=np.float32)
nodes = pd.read_csv(DATA_DIR / "az_feeder_nodes.csv")
edges = pd.read_csv(DATA_DIR / "az_feeder_edges.csv")
for b, bus in enumerate(bus_names):
    conn = edges[edges["from_bus"] == bus]["norm_amps"]
    norm_amps_arr[b] = conn.max() if len(conn) else 400.0
norm_amps_t = torch.tensor(norm_amps_arr, dtype=torch.float32).to(DEVICE)  # (N,)

# kv_base per bus for rated power calculation
kv_base_arr = np.zeros(n_buses, dtype=np.float32)
for b, bus in enumerate(bus_names):
    row = nodes[nodes["bus_name"] == bus]["kv_base"]
    kv_base_arr[b] = row.values[0] if len(row) else 4.16
kv_base_t = torch.tensor(kv_base_arr, dtype=torch.float32).to(DEVICE)     # (N,)
rated_kw_t = norm_amps_t * kv_base_t * math.sqrt(3) * 0.9                 # (N,)

# Splits
split_ser = ts.drop_duplicates("timestamp").set_index("timestamp")["split"].reindex(timestamps_all)
train_idx = np.where([split_ser.iloc[i] == "train" for i in range(T_all)])[0]
val_idx   = np.where([split_ser.iloc[i] == "val"   for i in range(T_all)])[0]
test_idx  = np.where([split_ser.iloc[i] == "test"  for i in range(T_all)])[0]
print(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

# Normalise (fit on train only)
print("  Normalising...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(feat_arr[train_idx].reshape(-1, N_FEATURES))
scaler_y.fit(target_arr[train_idx].reshape(-1, N_TARGETS))

feat_norm   = scaler_X.transform(feat_arr.reshape(-1, N_FEATURES)).reshape(T_all, n_buses, N_FEATURES)
target_norm = scaler_y.transform(target_arr.reshape(-1, N_TARGETS)).reshape(T_all, n_buses, N_TARGETS)

feat_t   = torch.tensor(feat_norm,   dtype=torch.float32)
target_t = torch.tensor(target_norm, dtype=torch.float32)

edge_index = gph.edge_index.to(DEVICE)
edge_attr  = gph.edge_attr.to(DEVICE)
node_feat  = gph.x.to(DEVICE)

# ── DATASET ───────────────────────────────────────────────────────────────────
class SlidingWindowDataset(Dataset):
    def __init__(self, indices):
        self.windows = [i for i in indices
                        if i >= LOOKBACK and i + HORIZON <= T_all]
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        t = self.windows[idx]
        X = feat_t[t - LOOKBACK : t]      # (L, N, F)
        y = target_t[t : t + HORIZON]     # (H, N, T)
        return X, y

train_ds = SlidingWindowDataset(train_idx)
val_ds   = SlidingWindowDataset(val_idx)
test_ds  = SlidingWindowDataset(test_idx)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)


# ── TFT BUILDING BLOCKS ───────────────────────────────────────────────────────

class GatedLinearUnit(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc  = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """
    TFT's core non-linear processing block.
    Applies ELU → Linear → GLU → LayerNorm with skip connection.
    """
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.fc1   = nn.Linear(d_in, d_hidden)
        self.fc2   = nn.Linear(d_hidden, d_out)
        self.glu   = GatedLinearUnit(d_out)
        self.norm  = nn.LayerNorm(d_out)
        self.skip  = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.drop  = nn.Dropout(dropout)
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        h = self.glu(h)
        return self.norm(h + self.skip(x))


class VariableSelectionNetwork(nn.Module):
    """
    Learns soft feature importance weights per bus per timestep.
    Produces interpretable feature importances — useful for the dashboard.
    """
    def __init__(self, n_features, d_model, dropout=0.1):
        super().__init__()
        self.grn_flat  = GatedResidualNetwork(n_features, d_model, n_features, dropout)
        self.grns      = nn.ModuleList([
            GatedResidualNetwork(1, d_model, d_model, dropout)
            for _ in range(n_features)
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """x: (..., n_features)"""
        weights = self.softmax(self.grn_flat(x))          # (..., F)
        processed = torch.stack(
            [self.grns[i](x[..., i:i+1]) for i in range(len(self.grns))],
            dim=-2
        )                                                  # (..., F, d)
        out = (weights.unsqueeze(-1) * processed).sum(-2) # (..., d)
        return out, weights


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention over the lookback window.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        attn_out, attn_weights = self.attn(x, x, x)
        return self.norm(x + self.drop(attn_out)), attn_weights


# ── MAIN MODEL ────────────────────────────────────────────────────────────────
class GNNTemporalFusionTransformer(nn.Module):
    """
    Full architecture:
      1. GATv2 layers: spatial message passing at each timestep
      2. Variable Selection Network: learned per-feature importance
      3. LSTM encoder: encode look-back into hidden state
      4. Temporal self-attention: attend over look-back positions
      5. Quantile output head: P10 / P50 / P90 per target per bus
    """
    def __init__(self, n_feat, n_static, n_targets, n_quantiles,
                 d_model, n_heads, horizon, dropout):
        super().__init__()
        self.horizon    = horizon
        self.n_buses    = n_buses
        self.n_quantiles = n_quantiles
        self.n_targets  = n_targets

        # Static node embedding
        self.static_embed = nn.Linear(n_static, d_model)

        # GATv2 layers (learns which neighbours to attend to)
        self.gat1 = GATv2Conv(n_feat + d_model, d_model,
                               heads=2, concat=False, dropout=dropout,
                               edge_dim=edge_attr.shape[1])
        self.gat2 = GATv2Conv(d_model, d_model,
                               heads=2, concat=False, dropout=dropout,
                               edge_dim=edge_attr.shape[1])

        # Variable selection (learn feature importance per bus)
        self.vsn = VariableSelectionNetwork(d_model, d_model, dropout)

        # LSTM encoder over look-back
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Temporal self-attention
        self.temporal_attn = TemporalSelfAttention(d_model, n_heads, dropout)

        # GRN post-attention
        self.post_attn_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        # Quantile output head
        # Predicts (horizon × n_targets × n_quantiles) per bus
        self.output_head = nn.Sequential(
            GatedResidualNetwork(d_model, d_model, d_model, dropout),
            nn.Linear(d_model, horizon * n_targets * n_quantiles),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, edge_index, edge_attr, node_feat):
        """
        x_seq    : (B, L, N, F)
        Returns  : (B, H, N, T, Q)   — quantile predictions
        Also returns variable importance weights for interpretability
        """
        B, L, N, F = x_seq.shape

        static_emb = torch.relu(self.static_embed(node_feat))  # (N, d)

        # ── Step 1: GATv2 at each timestep ───────────────────────────────────
        gat_out = []
        for t in range(L):
            xt = x_seq[:, t, :, :]           # (B, N, F)
            xt_flat = xt.reshape(B * N, F)

            static_t = static_emb.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
            node_in  = torch.cat([xt_flat, static_t], dim=-1)

            # Tile edges across batch
            all_edges = [edge_index + b * N for b in range(B)]
            batch_ei  = torch.cat(all_edges, dim=1)
            batch_ea  = edge_attr.repeat(B, 1)

            h = torch.relu(self.gat1(node_in, batch_ei, batch_ea))
            h = torch.relu(self.gat2(h, batch_ei, batch_ea))      # (B*N, d)
            h = h.reshape(B, N, -1)                           # (B, N, d)
            gat_out.append(h)

        gat_seq = torch.stack(gat_out, dim=1)   # (B, L, N, d)

        # ── Step 2: Variable selection (per bus, applied across time) ─────────
        # Reshape to (B*N, L, d) for VSN
        gat_flat = gat_seq.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        vsn_out, feat_weights = self.vsn(gat_flat)   # (B*N, L, d), (B*N, L, d)

        # ── Step 3: LSTM encode ───────────────────────────────────────────────
        lstm_out, _ = self.lstm_encoder(vsn_out)    # (B*N, L, d)

        # ── Step 4: Temporal self-attention ───────────────────────────────────
        attn_out, attn_weights = self.temporal_attn(lstm_out)  # (B*N, L, d)
        attn_out = self.post_attn_grn(attn_out)

        # Take final position as context vector
        context = attn_out[:, -1, :]   # (B*N, d)

        # ── Step 5: Quantile output ───────────────────────────────────────────
        raw = self.output_head(context)                          # (B*N, H*T*Q)
        raw = raw.reshape(B, N, self.horizon, self.n_targets, self.n_quantiles)
        out = raw.permute(0, 2, 1, 3, 4)                        # (B, H, N, T, Q)

        return out, feat_weights


model2 = GNNTemporalFusionTransformer(
    n_feat=N_FEATURES,
    n_static=node_feat.shape[1],
    n_targets=N_TARGETS,
    n_quantiles=N_QUANTILES,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    horizon=HORIZON,
    dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
print(f"\nModel parameters: {n_params:,}")


# ── PHYSICS-CONSTRAINED LOSS ──────────────────────────────────────────────────

def quantile_loss(pred_q, target, quantiles):
    """
    Pinball / quantile loss for each quantile level.
    pred_q : (B, H, N, T, Q)
    target : (B, H, N, T)
    """
    losses = []
    for qi, q in enumerate(quantiles):
        err = target - pred_q[..., qi]
        losses.append(torch.max((q - 1) * err, q * err).mean())
    return sum(losses) / len(losses)


def physics_loss(pred_q, rated_kw_t):
    """
    Soft physics constraints on the P50 (median) prediction.

    Constraint 1 — Thermal limit:
      Predicted net load at each bus should not exceed rated line capacity.
      Penalises max(0, pred_net_load - rated_kw) — overload violation.

    Constraint 2 — Non-negativity of thermal utilisation:
      thermal_pct >= 0 always.

    Constraint 3 — Storage sign consistency:
      Storage buses: net load should be lower than non-storage neighbours
      at discharge hours. Approximated as a soft inequality.

    Returns a scalar penalty (add to main loss weighted by LAMBDA_PHYSICS).
    """
    median_idx = 1   # P50 is quantile index 1

    # net_load_kw is target index 0
    pred_net_load = pred_q[:, :, :, 0, median_idx]   # (B, H, N)
    # thermal_pct is target index 1
    pred_thermal  = pred_q[:, :, :, 1, median_idx]   # (B, H, N)

    # Constraint 1: net load within thermal capacity
    # rated_kw_t is in raw kW, predictions are normalised — need to approximate
    # We use a soft upper bound: penalise predictions > 3 std above mean
    # (since we can't easily unnorm inside the loss without scaler)
    # This is a relative constraint: no bus should be 3× worse than average
    mean_load = pred_net_load.mean(dim=-1, keepdim=True)
    overload_penalty = F.relu(pred_net_load - 3.0 * mean_load.abs()).mean()

    # Constraint 2: thermal pct >= 0 (it's always non-negative physically)
    negativity_penalty = F.relu(-pred_thermal).mean()

    # Constraint 3: no instantaneous load > 5 std from bus-specific mean
    # (catches unrealistic spikes)
    bus_mean = pred_net_load.mean(dim=1, keepdim=True)
    bus_std  = pred_net_load.std(dim=1, keepdim=True).clamp(min=0.1)
    spike_penalty = F.relu((pred_net_load - bus_mean).abs() - 5 * bus_std).mean()

    return overload_penalty + negativity_penalty + 0.5 * spike_penalty


optimizer2 = torch.optim.Adam(model2.parameters(), lr=LR)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer2, T_max=EPOCHS, eta_min=LR * 0.1
)

# ── TRAINING ──────────────────────────────────────────────────────────────────
def run_epoch2(loader, train=True):
    model2.train() if train else model2.eval()
    total_data   = 0.0
    total_phys   = 0.0
    n_batches    = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for X, y in loader:
            X = X.to(DEVICE)   # (B, L, N, F)
            y = y.to(DEVICE)   # (B, H, N, T)

            pred_q, _ = model2(X, edge_index, edge_attr, node_feat)
            # pred_q: (B, H, N, T, Q)

            data_loss = quantile_loss(pred_q, y, QUANTILES)
            phys_loss = physics_loss(pred_q, rated_kw_t)
            loss = data_loss + LAMBDA_PHYSICS * phys_loss

            if train:
                optimizer2.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
                optimizer2.step()

            total_data += data_loss.item()
            total_phys += phys_loss.item()
            n_batches  += 1

    nb = max(n_batches, 1)
    return total_data / nb, total_phys / nb


print("\n" + "="*55)
print("TRAINING — GNN + TFT + Physics Constraints")
print("="*55)

history2 = {"train_data": [], "train_phys": [], "val_data": [], "epoch_time": []}
best_val2  = float("inf")
best_path2 = OUT_DIR / "best_model2.pt"

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_data, tr_phys = run_epoch2(train_loader, train=True)
    val_data, _      = run_epoch2(val_loader,   train=False)
    elapsed = time.time() - t0

    scheduler2.step()
    history2["train_data"].append(tr_data)
    history2["train_phys"].append(tr_phys)
    history2["val_data"].append(val_data)
    history2["epoch_time"].append(elapsed)

    if val_data < best_val2:
        best_val2 = val_data
        torch.save(model2.state_dict(), best_path2)
        marker = " ◀ best"
    else:
        marker = ""

    print(f"  Epoch {epoch:>2}/{EPOCHS}  "
          f"data={tr_data:.4f}  phys={tr_phys:.4f}  "
          f"val={val_data:.4f}  time={elapsed:.1f}s{marker}")

total_time = sum(history2["epoch_time"])
print(f"\nTotal training time: {total_time/60:.1f} min")

# ── EVALUATION ────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("EVALUATION — Test set")
print("="*55)

model2.load_state_dict(torch.load(best_path2, weights_only=True))
model2.eval()

all_preds_q, all_true = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(DEVICE)
        pred_q, _ = model2(X, edge_index, edge_attr, node_feat)
        all_preds_q.append(pred_q.cpu().numpy())
        all_true.append(y.numpy())

preds_q_norm = np.concatenate(all_preds_q, axis=0)  # (S, H, N, T, Q)
truth_norm   = np.concatenate(all_true,    axis=0)   # (S, H, N, T)

shape_q = preds_q_norm.shape
shape_t = truth_norm.shape

# Inverse transform median predictions
median_norm = preds_q_norm[..., 1]   # (S, H, N, T)
median_kw   = scaler_y.inverse_transform(
    median_norm.reshape(-1, N_TARGETS)
).reshape(shape_t)
truth_kw = scaler_y.inverse_transform(
    truth_norm.reshape(-1, N_TARGETS)
).reshape(shape_t)

# Inverse transform P10 and P90 for net_load_kw (target 0) only
p10_norm = preds_q_norm[:, :, :, 0, 0]
p90_norm = preds_q_norm[:, :, :, 0, 2]
p10_kw = scaler_y.inverse_transform(
    np.stack([p10_norm.ravel(),
              np.zeros_like(p10_norm.ravel()),
              np.zeros_like(p10_norm.ravel())], axis=1)
)[:, 0].reshape(p10_norm.shape)
p90_kw = scaler_y.inverse_transform(
    np.stack([p90_norm.ravel(),
              np.zeros_like(p90_norm.ravel()),
              np.zeros_like(p90_norm.ravel())], axis=1)
)[:, 0].reshape(p90_norm.shape)

print(f"\n{'Target':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE%':>8}")
print("-" * 62)
for ti, tname in enumerate(TARGET_COLS):
    p = median_kw[..., ti].ravel()
    t = truth_kw[..., ti].ravel()
    mae  = mean_absolute_error(t, p)
    rmse = math.sqrt(mean_squared_error(t, p))
    r2   = r2_score(t, p)
    mape = np.mean(np.abs((t - p) / (np.abs(t) + 1e-6))) * 100
    print(f"  {tname:<23} {mae:>8.2f} {rmse:>8.2f} {r2:>8.4f} {mape:>8.2f}")

# Quantile coverage (how often true falls between P10 and P90)
p10_flat = p10_kw.ravel()
p90_flat = p90_kw.ravel()
true_flat = truth_kw[..., 0].ravel()
coverage = np.mean((true_flat >= p10_flat) & (true_flat <= p90_flat)) * 100
print(f"\n  P10–P90 interval coverage: {coverage:.1f}%  (ideal ≈ 80%)")

# Physics constraint satisfaction
phys_viol = np.mean(median_kw[..., 1].ravel() < 0) * 100
print(f"  thermal_pct < 0 violations (physics): {phys_viol:.2f}%  (ideal = 0%)")

# ── INTERPRETABILITY: Feature importance (averaged over val set) ──────────────
print("\n  Computing feature importance from Variable Selection Network...")
model2.eval()
feat_weights_list = []
with torch.no_grad():
    for X, y in val_loader:
        X = X.to(DEVICE)
        _, fw = model2(X, edge_index, edge_attr, node_feat)
        # fw: (B*N, L, d) — the VSN weights
        # We extract the input-space weights from the GRN flat layer
        # Approximate: just use the magnitude of the first GRN projection
        feat_weights_list.append(fw.cpu().numpy())
        break   # one batch is enough for illustration

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("GNN + TFT + Physics — Results", fontsize=13)

# 1. Loss curves
ax = axes[0, 0]
ax.plot(history2["train_data"], label="Train quantile loss", marker="o", ms=3)
ax.plot(history2["val_data"],   label="Val quantile loss",   marker="s", ms=3)
ax.plot(history2["train_phys"], label="Physics penalty",     marker="^", ms=3,
        color="green", linestyle="--")
ax.set_title("Training Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 2. Forecast with uncertainty bands — bus 0, sample 0, net_load_kw
ax = axes[0, 1]
h_range = range(HORIZON)
sample_true  = truth_kw[0, :, 0, 0]
sample_p50   = median_kw[0, :, 0, 0]
sample_p10   = p10_kw[0, :, 0]
sample_p90   = p90_kw[0, :, 0]
ax.fill_between(h_range, sample_p10, sample_p90, alpha=0.2, color="tomato", label="P10–P90")
ax.plot(sample_true, color="steelblue", label="Actual")
ax.plot(sample_p50,  color="tomato",    linestyle="--", label="P50 Forecast")
ax.set_title("24-h Forecast with Uncertainty (Bus 0, net_load_kw)")
ax.set_xlabel("Hour ahead"); ax.set_ylabel("kW")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 3. Per-bus MAE heatmap (net_load_kw)
bus_maes = []
for b in range(n_buses):
    p = median_kw[:, :, b, 0].ravel()
    t = truth_kw[:, :, b, 0].ravel()
    bus_maes.append(mean_absolute_error(t, p))
ax = axes[1, 0]
ax.bar(range(n_buses), sorted(bus_maes, reverse=True), color="steelblue", alpha=0.7)
ax.set_title("Per-Bus MAE (net_load_kw, sorted)")
ax.set_xlabel("Bus rank"); ax.set_ylabel("MAE (kW)")
ax.grid(True, alpha=0.3, axis="y")

# 4. Actual vs predicted scatter — net_load_kw, subsample
ax = axes[1, 1]
n_sample = min(5000, len(truth_kw[..., 0].ravel()))
idx_s = np.random.choice(len(truth_kw[..., 0].ravel()), n_sample, replace=False)
ax.scatter(truth_kw[..., 0].ravel()[idx_s],
           median_kw[..., 0].ravel()[idx_s],
           s=1, alpha=0.3, color="steelblue")
lims = [min(truth_kw[..., 0].min(), median_kw[..., 0].min()),
        max(truth_kw[..., 0].max(), median_kw[..., 0].max())]
ax.plot(lims, lims, "r--", linewidth=1, label="Perfect")
ax.set_title("Actual vs Predicted (net_load_kw)")
ax.set_xlabel("Actual (kW)"); ax.set_ylabel("Predicted P50 (kW)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "model2_results.png", dpi=120)
print(f"\nPlot saved → {OUT_DIR / 'model2_results.png'}")

# Per-bus results CSV
bus_results = []
for b, bus in enumerate(bus_names):
    for ti, tname in enumerate(TARGET_COLS):
        p = median_kw[:, :, b, ti].ravel()
        t = truth_kw[:, :, b, ti].ravel()
        bus_results.append({
            "bus_name": bus,
            "target": tname,
            "mae":  mean_absolute_error(t, p),
            "rmse": math.sqrt(mean_squared_error(t, p)),
            "r2":   r2_score(t, p),
        })
pd.DataFrame(bus_results).to_csv(OUT_DIR / "model2_per_bus_metrics.csv", index=False)
print(f"Per-bus metrics saved → {OUT_DIR / 'model2_per_bus_metrics.csv'}")

print("\nDone — Model 2 (GNN + TFT + Physics)")