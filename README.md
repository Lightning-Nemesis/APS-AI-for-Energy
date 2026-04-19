# APS AI for Energy — Spatio-Temporal Grid Intelligence

> A proof-of-concept spatio-temporal AI system for an APS-like distribution network.  
> Predicts feeder-level load across 132 buses, stress-tests extreme heat and EV growth scenarios, and surfaces action-ready decisions for utility planners.

**Live Dashboard →** [https://powerflow-lens.lovable.app](https://powerflow-lens.lovable.app)

---

## Table of contents

- [What this system does](#what-this-system-does)
- [Results at a glance](#results-at-a-glance)
- [Repository structure](#repository-structure)
- [Step 1 — Graph building](#step-1--graph-building)
- [Step 2 — Model training](#step-2--model-training)
- [Step 3 — Stress testing](#step-3--stress-testing)
- [Step 4 — Decision dashboard](#step-4--decision-dashboard)
- [Setup and installation](#setup-and-installation)
- [Data sources](#data-sources)
- [Key design decisions](#key-design-decisions)

---

## What this system does

Electric utilities like APS must continuously balance generation and demand across hundreds of distribution feeders while keeping every line and transformer within safe thermal limits. Two forces are making this harder: **extreme heat events** (Phoenix recorded 31 consecutive days ≥ 110 °F in summer 2023) and **EV adoption** creating a new evening demand surge that stacks on top of the solar drop-off (the duck curve).

This system addresses both with a single trainable model that understands both **time** (load patterns, weather sequences) and **space** (how stress at one bus propagates through the network). It produces three outputs a utility can act on:

1. **24-hour feeder-level load and thermal forecasts** across all 132 buses simultaneously
2. **Scenario outputs** showing where and when the network becomes constrained under heat or EV growth
3. **A decision dashboard** that ranks buses by risk tier and recommends interventions

---

## Results at a glance

### Model 1 — GNN + LSTM (primary forecasting model)

| Metric | net\_load\_kw | thermal\_pct |
|---|---|---|
| MAE | **3.10 kW** | **0.119** |
| RMSE | 17.65 kW | 0.666 |
| R² | **0.9781** | **0.9790** |
| Test period | 2023 Jul–Dec | 2023 Jul–Dec |
| Training time | 31.9 min (20 epochs, CUDA) | — |
| Parameters | 5,368,768 | — |

> Note: MAPE is unreliable for this dataset because near-zero net load values (buses with solar offsetting load) cause division artifacts. MAE and R² are the meaningful metrics.

**Worst-predicted buses** (highest per-bus MAE — also the most stressed buses, which is expected):

| Bus | MAE net\_load\_kw |
|---|---|
| 76 | 24.67 kW |
| 48 | 21.17 kW |
| 49 | 14.22 kW |
| 65 | 14.21 kW |
| 47 | 10.79 kW |

### Model 2 — GNN + TFT + Physics constraints (experimental)

| Metric | Value |
|---|---|
| Parameters | 162,328 |
| Best val loss | 0.0280 (epoch 5/5) |
| Training time | 54.3 min (5 epochs, CUDA) |
| Targets | net\_load\_kw, thermal\_pct, uncertainty quantiles (10th/50th/90th) |

Model 2 adds physics-informed loss and probabilistic forecasting (quantile regression). Still converging — val loss was still decreasing at epoch 5.

### Stress test results

| Scenario | Buses ≥ 90% thermal | Avg delta | Peak delta (bus 76) |
|---|---|---|---|
| Baseline | 2 (bus 76, 48) | — | 108.3% |
| Heat dome +8 °F | **130** | +0.5 kW/bus | **109.5%** (+1.15 pp) |
| EV surge 2027 ×2.75 | **85** | +0.1 kW/bus | **109.0%** (+0.74 pp) |

**Key finding:** Bus 76 and 48 are already over the 90% thermal threshold at baseline — before any scenario is applied. Heat is the dominant near-term stressor. EV growth is a compounding medium-term risk.

---

## Repository structure

```
APS AI for Energy/
│
├── build_az_graph.py              # Step 1 — DSS parsing, time series generation, graph object
├── train_gnn_lstm.py              # Step 2a — GNN + LSTM model (primary)
├── train_gnn_tft_physics.py       # Step 2b — GNN + TFT + physics constraints (experimental)
├── stress_test.py                 # Step 3 — heat dome and EV 2027 stress scenarios
├── dashboard.py                   # Step 4 — exports dashboard_data.json for Lovable
│
├── az_feeder_nodes.csv            # 132 buses with static features
├── az_feeder_edges.csv            # 252 directed edges (126 physical lines × 2)
├── az_feeder_timeseries.parquet   # 5,781,600 rows — all years, all buses, all features
├── az_feeder_graph.pt             # PyTorch Geometric Data object
├── dashboard_data.json            # Pre-computed dashboard payload (142.7 KB)
│
├── model1_outputs/
│   ├── best_model1.pt             # Best checkpoint (val loss 0.0062, epoch 15)
│   ├── model1_results.png         # Loss curves + sample forecast
│   └── model1_per_bus_mae.csv     # Per-bus MAE ranking
│
├── model2_outputs/
│   └── best_model2.pt             # Best checkpoint (val loss 0.0280, epoch 5)
│
├── stress_test_outputs/
│   ├── stress_summary.csv         # Per-bus: baseline vs heat vs EV, all metrics
│   ├── overload_alerts.csv        # Buses hitting ≥ 90% thermal under each scenario
│   ├── scenario_A_heatdome_results.csv   # Hourly feeder totals — heat scenario
│   ├── scenario_B_ev2027_results.csv     # Hourly feeder totals — EV scenario
│   └── stress_test_plots.png      # 6-panel comparison figure
│
├── ieee123/                       # IEEE 123-bus OpenDSS case files
├── OpenDSS/                       # OpenDSS distribution
├── results/                       # Additional result artifacts
└── torch-env/                     # Python virtual environment
```

---

## Step 1 — Graph building

**Script:** `build_az_graph.py`

### Why this step exists

The IEEE 123-bus feeder is a standard public test case used by power engineers. It has realistic line impedances, load mix, voltage regulators, and switches. Rather than inventing a topology from scratch, we parse the actual OpenDSS DSS files and scale the loads to Arizona magnitudes (~10 MVA, vs the IEEE base of ~3.5 MVA). This gives the model a physically grounded network structure to learn from.

### What it does

**Step 0 — Verify DSS files**
Checks that all five required DSS files are present before doing anything.

**Step 1 — Parse DSS files**
Extracts bus coordinates from `BusCoords.dat`, load values from `IEEE123Loads.DSS`, line impedances from `IEEELineCodes.DSS`, and line topology from `IEEE123Master.dss`. Result: 130 bus coordinates, 91 loads, 29 line codes, 126 physical lines.

**Step 2 — Classify buses**
Assigns each bus a role based on location and load profile:
- 17 solar buses (rooftop PV generation)
- 8 EV buses (high charging demand nodes)
- 5 storage buses (battery dispatch, subset of solar)
- 1 substation bus (bus 150)
- Regulator secondary buses (buses ending in `r`)

**Steps 3–6 — Build multi-year time series (2019–2023)**
For each year and each bus, generates an hourly time series with 44 features:

| Feature group | Features | Source |
|---|---|---|
| Time encoding | hour/month sin-cos | Synthetic |
| Weather | temp\_f, GHI, clearsky ratio | NOAA API (real where available, synthetic fallback) |
| Load | base kW, lags (1h, 24h, 168h), rolling stats | Scaled from DSS + AZ multipliers |
| EV | ev\_charging\_kw, 2030/2035 projections | NREL EVI-DiST assumptions |
| Storage | storage\_net\_kw (charge/discharge cycle) | Physics-based dispatch model |
| EIA-930 | APS bulk demand context | EIA API (synthetic fallback) |
| Heat scenario | is\_heatdome, heat\_scenario\_mult, temp×load | Derived from NOAA heat alert windows |
| Duck curve | is\_duck\_curve\_window | Hour-of-day flag (17–21h) |
| Targets | net\_load\_kw, thermal\_pct | Computed from load + generation + capacity |

NOAA data was successfully pulled for 2019, 2020, and 2022 (100% real hourly records). 2021 and 2023 fell back to calibrated synthetic temperature profiles due to API timeouts.

**Step 7 — Concatenate and split**

| Split | Years | Hours |
|---|---|---|
| Train | 2019, 2020, 2021, 2022 | 35,040 |
| Val | 2023 Jan–Jun | 4,344 |
| Test | 2023 Jul–Dec | 4,416 |

Test set deliberately includes the July 2023 heat dome window — a genuinely held-out stress period the model never saw during training.

**Step 8 — Build PyTorch Geometric graph object**

```
num_nodes:     132
num_edges:     252  (126 physical lines × 2 directions)
node features: 11   (coordinates, load, type flags)
edge features: 8    (impedance, capacity, length, phases)
self-loops:    0
```

**Step 9 — Save outputs**
Writes `az_feeder_nodes.csv`, `az_feeder_edges.csv`, `az_feeder_timeseries.parquet` (174 MB), and `az_feeder_graph.pt`.

### Run

```bash
python build_az_graph.py
```

Requires OpenDSS DSS files in the path configured at the top of the script. NOAA and EIA API keys are optional — the script falls back to synthetic data automatically.

---

## Step 2 — Model training

### Model 1 — GNN + LSTM (`train_gnn_lstm.py`)

**Why this architecture**

A plain LSTM sees time but ignores network topology — it treats bus 76 the same as bus 1 even if they're electrically close. A plain GCN sees topology but has no memory. The GNN + LSTM combination solves both: the GCN layer propagates spatial information across the graph at each timestep, and the LSTM encodes how that spatial state evolves over the 24-hour lookback window.

**Architecture**

```
Input: (Batch, Lookback=24, N_buses=132, N_features=28)
  │
  ├── At each of 24 timesteps:
  │     Static node embedding  →  Linear(11 → 64)
  │     Concatenate with dynamic features  →  (132, 92)
  │     GCNConv layer 1  →  (132, 64)   [ReLU]
  │     GCNConv layer 2  →  (132, 64)   [ReLU]
  │     Flatten across buses  →  (132 × 64 = 8448,)
  │
  ├── Stack 24 timesteps  →  (Batch, 24, 8448)
  │     LSTM (2 layers, hidden=128, dropout=0.1)
  │     Take final hidden state  →  (Batch, 128)
  │
  └── Output head:
        Linear(128 → 128)  [ReLU]
        Linear(128 → 24 × 132 × 2)
        Reshape  →  (Batch, Horizon=24, N_buses=132, N_targets=2)
```

**Training configuration**

| Hyperparameter | Value |
|---|---|
| Lookback | 24 hours |
| Horizon | 24 hours |
| Batch size | 32 |
| Epochs | 20 |
| Learning rate | 1e-3 (Adam) |
| LR scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Gradient clipping | max norm 1.0 |
| Loss | MSE (normalised) |
| Device | CUDA |

**Training log (key epochs)**

```
Epoch  1/20  train=0.3799  val=0.2071  time=99.1s  ◀ best
Epoch  2/20  train=0.0571  val=0.0100  ◀ best
Epoch  5/20  train=0.0107  val=0.0071  ◀ best
Epoch  9/20  train=0.0097  val=0.0067  ◀ best
Epoch 15/20  train=0.0086  val=0.0062  ◀ best   ← checkpoint saved
Epoch 20/20  train=0.0069  val=0.0075
Total: 31.9 min
```

Validation loss plateaued after epoch 15, with slight overfitting in the final 5 epochs. The epoch 15 checkpoint is used for all downstream evaluation and stress testing.

**Test set evaluation (2023 Jul–Dec)**

```
net_load_kw:   MAE=3.097 kW   RMSE=17.647 kW   R²=0.9781
thermal_pct:   MAE=0.119      RMSE=0.666        R²=0.9790
```

### Model 2 — GNN + TFT + Physics (`train_gnn_tft_physics.py`)

**Why a second model**

Model 1 gives point forecasts. Utility planners also need to know *how confident* the model is — a forecast of 88% thermal with wide uncertainty is a different operational signal than 88% with tight bounds. Model 2 adds:

1. **Temporal Fusion Transformer** (TFT) in place of the LSTM — better at capturing long-range dependencies and known-future inputs (time-of-day, scheduled events)
2. **Quantile regression** — outputs 10th/50th/90th percentile forecasts, giving planners a probabilistic view of risk
3. **Physics-informed loss** — penalises predictions that violate Kirchhoff constraints (power balance at each node), grounding the model in electrical reality

**Training log**

```
Epoch  1/5  data=0.0582  phys=0.2429  val=0.0396
Epoch  3/5  data=0.0350  phys=0.2569  val=0.0313
Epoch  5/5  data=0.0304  phys=0.2597  val=0.0280  ◀ best
Total: 54.3 min  (still converging — more epochs needed)
```

The physics loss is higher than the data loss throughout training, which is expected — the model is learning to trade off predictive accuracy against physical plausibility. With more epochs this typically converges to a better-calibrated result than a pure data-driven model.

### Run

```bash
# Model 1
python train_gnn_lstm.py

# Model 2
python train_gnn_tft_physics.py
```

Weights are saved to `model1_outputs/best_model1.pt` and `model2_outputs/best_model2.pt`.

---

## Step 3 — Stress testing

**Script:** `stress_test.py`

### Why stress testing matters

A model trained on historical data tells you how well it forecasts what already happened. Stress testing asks a different question: *what would the model predict if conditions were worse than anything in the training set?* This is how utilities plan infrastructure investment — not by extrapolating averages, but by understanding failure modes.

### Design principle: inference-only counterfactuals

Both scenarios work by **mutating the input feature array** before running it through the already-trained model weights. No retraining happens. This is the correct approach because:
- The model has learned the relationship between temperature, EV load, and feeder stress from 5 years of data
- We are asking it to apply that learned relationship to hypothetical inputs it has never seen
- The output is the model's genuine prediction of what would happen — not a rule-based approximation

### Memory management

The script runs one scenario at a time and explicitly frees arrays between passes using `del` + `gc.collect()` + `torch.cuda.empty_cache()`. This keeps peak RAM under ~2.5 GB instead of the ~6 GB that holding all three arrays simultaneously would require.

### Scenario A — Arizona Mega Heat Dome

**Basis:** Phoenix recorded 31 consecutive days ≥ 110 °F in summer 2023. The test set covers this window. We amplify it further to represent a plausible future extreme event.

**Perturbations applied to July 8–21 2023 (313 timestamps):**

| Feature | Change |
|---|---|
| `temp_f`, `temp_lag1`, `temp_lag24` | +8 °F |
| `is_heatdome` | forced to 1 |
| `heat_scenario_mult` | +0.35 (capped at 1.5) |
| `temp_x_load_mult` | rescaled proportionally |

**Results:**

- 130 of 132 buses pushed to ≥ 90% thermal loading
- Bus 76 peak: 109.5% (+1.15 pp vs baseline 108.3%)
- Average delta: +0.5 kW/bus; peak delta on worst bus: +29.1 kW
- Heat dome stress MAE rises to 3.95 kW (vs 3.10 baseline) — the model is being pushed into a harder regime

### Scenario B — EV Surge 2027

**Basis:** NREL EVI-DiST projects Maricopa County EV penetration reaching ~22% by 2027, up from ~8% in 2024. We model this as a ×2.75 scaling of EV charging load, applied uniformly across the test period with the duck-curve window forced on during 17–21h.

**Perturbations applied across full test period:**

| Feature | Change |
|---|---|
| `ev_charging_kw` | ×2.75 (all buses, all hours) |
| `is_duck_curve_window` | forced to 1 for hours 17–20 (7,300 timestamps) |

**Results:**

- 85 of 132 buses pushed to ≥ 90% thermal loading
- Bus 76 peak: 109.0% (+0.74 pp vs baseline)
- Average delta: +0.1 kW/bus; peak delta: +19.1 kW on bus 76
- EV stress MAE barely changes (3.08 kW) — EV load at 2027 levels is within the model's comfortable operating range

### Interpretation

Heat is the dominant near-term stressor. EV 2027 is a secondary risk that becomes primary at 2030 penetration levels (×4–5× baseline). The two scenarios compound: a 2030 heat dome + EV surge combination is the scenario utilities should be planning for today.

### Run

```bash
python stress_test.py
```

Requires `az_feeder_timeseries.parquet`, `az_feeder_graph.pt`, and `model1_outputs/best_model1.pt`.

---

## Step 4 — Decision dashboard

**Script:** `dashboard.py`  
**Frontend:** Lovable (React)  
**Live app:** [https://powerflow-lens.lovable.app](https://powerflow-lens.lovable.app)

### Why a separate dashboard step

The model outputs are numpy arrays and CSVs. A utility planner cannot act on those. The dashboard step answers three questions in order: where is the network stressed, when does it peak, and what should be done about it. The output is a single pre-computed JSON file that a no-backend React app consumes directly.

### Two-view design: descriptive and predictive

The dashboard is split into two views that serve different operator needs:

**Descriptive view** covers the full test set (Jul–Dec 2023). It answers "what happened and where are the structural vulnerabilities?" — the analysis a planning engineer runs once a quarter.

**Predictive view** is scoped to July 9 2023, the worst recorded day in the test set (bus 76: 1,182.6 kW, 45.6% thermal). It shows a 24-hour lookback + 24-hour forecast horizon per bus — the view a grid operator needs during an active heat event.

### Threshold design

Thresholds are derived from the actual descriptive statistics of the test set, not arbitrary round numbers:

| Tier | Threshold | Basis |
|---|---|---|
| DISPATCH | thermal\_pct ≥ 20 | Above test set p99 (18.85) — genuine emergency |
| MONITOR | thermal\_pct ≥ 10 | ~p97 — operationally meaningful |
| WATCH | thermal\_pct ≥ 5 | ~p95 (6.90) — flag for planning |
| NORMAL | thermal\_pct < 5 | Below p95 — routine |

The old 85%/65% thresholds from generic utility standards would produce zero critical buses on this dataset (test set max is 45.59%) — rendering the dashboard useless. Using data-derived percentile thresholds ensures the tiers are meaningful for this specific feeder.

### What `dashboard.py` computes

Reads `az_feeder_timeseries.parquet` directly and produces a single compact JSON with five top-level sections:

| Section | Contents |
|---|---|
| `meta` | Bus registry with x/y coordinates, type flags (EV/solar/storage), and threshold config |
| `descriptive` | Six analytical panels over the full Jul–Dec 2023 test set |
| `predictive_window` | Jul 9 2023: 24h lookback + 24h horizon, all 5 signals per bus per hour, peak-hour snapshot |
| `heatdome_cut` | Jul 1–Aug 4 filtered daily stats for scenario overlay |
| `duck_curve_cut` | Hours 17–21 aggregated stats for duck curve panel |

The six descriptive panels:

| Panel | What it shows | Key data |
|---|---|---|
| P1 — Feeder stress index | All 132 buses ranked by thermal\_max with tier, action, and headroom | `thermal_max`, `exc_dispatch_hrs`, `headroom_kw`, per-scenario maxima |
| P2 — Load decomposition | 24-point hourly profile: base load, EV, PV offset, storage, net load | `load_kw`, `ev_charging_kw`, `pv_output_kw`, `storage_net_kw` |
| P3 — Hour-of-day vulnerability | Heat dome vs normal day max curves per hour with DR scheduling labels | `heatdome_mean_kw` vs `normal_mean_kw` per hour |
| P4 — Weather–load coupling | Scatter of temp\_f vs net\_load\_kw for top 7 buses, afternoon hours | 1,500 sampled points per bus, `is_heatdome` flag |
| P5 — Duck curve ramp detector | Per-bus ramp kW (load at 18:00 minus load at 12:00), P95 across test period | Actual computed ramp, not a label |


**Terminal output from `dashboard.py`:**

```
[1/9] Loading parquet …
  Rows  : 582,912  |  Buses : 132  |  Hours : 4416
[2/9] Building meta …
  Bus registry built: 132 buses
[3/9] Panel 1 — Feeder stress index …
  DISPATCH: 2  MONITOR: 5  WATCH: 7  NORMAL: 118
...
✓  Saved  : dashboard_data.json
   Panel 1  : 132 buses ranked by thermal_max
   Panel 7  : bus signals (48h lookback + horizon)
   DISPATCH: 2  MONITOR: 5  WATCH: 7  NORMAL: 118
Next: copy dashboard_data.json → Lovable project public/
```

### Intervention logic

Actions are data-driven — computed from exceedance hours, heat dome uplift percentage, bus type, and headroom, not fixed thermal brackets:

| Condition | Action |
|---|---|
| DISPATCH + exceedance > 100 h | Immediate transformer upgrade |
| DISPATCH + heat dome uplift > 15% | Deploy demand response — heat dome drives load |
| MONITOR + EV bus | Managed EV charging schedule — shift load from 17–21h |
| MONITOR + heat uplift > 10% | Pre-cool activation 1h before forecast peak |
| WATCH + solar bus, no storage | Install battery storage — solar excess uncaptured |
| WATCH | Flag for 2027 EV planning cycle |
| NORMAL | Routine monitoring |

### Deploying to Lovable

1. Run `python dashboard.py` — produces `dashboard_data.json`
2. In your Lovable project, place `dashboard_data.json` in the `public/` folder
3. Fetch it once on mount:
   ```js
   const [data, setData] = useState(null);
   useEffect(() => {
     fetch('/dashboard_data.json').then(r => r.json()).then(setData);
   }, []);
   ```
4. All 7 panels read from the single `data` object — no backend, no additional API calls

Live app: [https://powerflow-lens.lovable.app](https://powerflow-lens.lovable.app)

### Run

```bash
python dashboard.py
```

Requires only `az_feeder_timeseries.parquet` and (optionally) `az_feeder_nodes.csv` for bus coordinates.

---

## Setup and installation

```bash
# Clone and enter directory
git clone <your-repo-url>
cd "APS AI for Energy"

# Create virtual environment
python -m venv torch-env
torch-env\Scripts\activate        # Windows
# source torch-env/bin/activate   # Mac/Linux

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pandas numpy scikit-learn pyarrow matplotlib requests

# Run the full pipeline
python build_az_graph.py          # ~5 min
python train_gnn_lstm.py          # ~32 min (GPU)
python stress_test.py             # ~25 min (GPU)
python dashboard.py               # ~2 min
```

**Hardware used:** NVIDIA GPU (CUDA), 32 GB RAM. The stress test script is memory-optimised to run on 8 GB GPU / 16 GB RAM.

---

## Data sources

| Source | What we use | How accessed |
|---|---|---|
| IEEE PES 123-bus feeder | Network topology, line impedances, load locations | OpenDSS DSS files (public) |
| NOAA NCEI LCD | Hourly temperature, Phoenix Sky Harbor station | REST API (`ncei.noaa.gov`) — real data for 2019, 2020, 2022 |
| NREL NSRDB | GHI irradiance proxy for PV generation | Synthetic calibrated to AZ clear-sky curves |
| EIA-930 | AZPS balancing authority bulk demand | REST API — synthetic fallback used (EIA key not configured) |
| NREL EVI-DiST | EV adoption projections, Maricopa County | Published report values used as scenario parameters |

All synthetic data is documented in `build_az_graph.py` with the calibration assumptions noted inline.

---

## Key design decisions

**Why IEEE 123-bus and not a simpler feeder?**  
The 123-bus case has voltage regulators, switches, mixed single/three-phase segments, and a realistic mix of residential and commercial loads. Simpler feeders (34-bus, 37-bus) would understate the complexity a real utility faces. The 123-bus topology also has published bus coordinates, enabling geographic visualisation.

**Why GNN + LSTM and not a pure transformer?**  
Transformers on graph-structured data require careful positional encoding and scale poorly with batch size when the graph is large and batched by tiling (as done here). The GCN + LSTM combination is well-understood, trains in a predictable time budget, and produces R² > 0.97 on this problem — sufficient for the proof-of-concept scope.

**Why train/val/test split by year and not randomly?**  
Random splitting would leak future patterns into training (e.g., a July 2023 window in training, another in test). Year-based splitting ensures the test set is genuinely held out. The 2023 Jul–Dec test period includes the actual Phoenix heat dome, making it a meaningful stress evaluation rather than an easy one.

**Why does the stress test run inference-only?**  
Retraining on perturbed data would conflate model learning with scenario response, making it impossible to isolate what the model actually predicts. Inference-only counterfactuals isolate the scenario effect cleanly.

**Why is MAPE so high?**  
Net load can be near zero or negative at solar buses during midday (solar output exceeds local load). Percentage error metrics become meaningless when the denominator approaches zero. MAE and R² are the appropriate metrics for this target.

---

*Built for the APS AI for Energy challenge. All data is synthetic or publicly available. No proprietary APS data was used.*