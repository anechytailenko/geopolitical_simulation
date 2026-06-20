# services/ml — geopolitic spatio-temporal GNN

Predicts the 5-class country↔country relationship one month ahead, from
`geopolitic_aggregated`. Design: [`../../plans/03-ml-workflow.md`](../../plans/03-ml-workflow.md).
Commands & expected outputs: [`../../COMMANDS.md`](../../COMMANDS.md) §10.

## Layout

```
ml/                     canonical package (used by tests AND the FastAPI server)
  config.py timestep.py   dependency-free (stdlib) — importable/testable without torch
  features.py             Preprocess: scalers (train-only), region one-hot, edge features
  dataset.py              Parquet -> per-month HeteroData windows + neg-sampled labeled pairs
  model.py                HeteroConv(TransformerConv/GATv2) + GRU + attn + concat-MLP decoder
  losses.py metrics.py    weighted-CE / focal ; macro-F1, per-class F1, confusion, ECE
  calibrate.py            temperature scaling
  explain.py              Integrated Gradients + GNNExplainer-style edge mask
  train.py infer.py       training loop (checkpoints + W&B) ; serving
export/neo4j_to_parquet.py   read-only Neo4j(7688) -> Parquet
notebooks/build_notebook.py  -> train_geopolitic_gnn.ipynb (self-contained, for Kaggle)
app.py                   FastAPI inference server
tests/                   pure-stdlib tier + torch-gated tier (synthetic fixtures, no DB)
```

## Quick start

```bash
cd services/ml
python3 -m pip install -e .
python3 -m export.neo4j_to_parquet --out dataset_parquet     # read-only export
python3 -m pytest -q                                         # safe: synthetic fixtures only
python3 -m ml.train --data-dir dataset_parquet --epochs 30   # or run the notebook on Kaggle
```

W&B: project `geosimulation` under entity `anna-nechytailenko-kyiv-school-of-economics`
(`WANDB_API_KEY` from `.env` locally, or a Kaggle Secret on Kaggle). Tests never touch Neo4j
or delete data; the export is read-only.

## Train on Kaggle (free GPU)

### 1. Build the dataset + the notebook (locally)

```bash
cd services/ml
python3 -m export.neo4j_to_parquet --out dataset_parquet   # read-only; needs the agg DB up (§COMMANDS 1)
python3 notebooks/build_notebook.py                        # -> notebooks/train_geopolitic_gnn.ipynb
```

`build_notebook.py` regenerates the **self-contained** notebook from the `ml/` sources
(base64-embedded, so there is a single source of truth — the same code the tests and the
FastAPI server use). **Re-run it after any change under `ml/`.** You upload exactly two
things to Kaggle: the `dataset_parquet/` folder and `train_geopolitic_gnn.ipynb`.

### 2. Upload the dataset

kaggle.com → **Datasets → New Dataset** → drag in the whole `services/ml/dataset_parquet/`
folder (the three `*.parquet` files) → **Create**. On a later data refresh: open the dataset
→ **New Version** → re-upload.

### 3. Upload the notebook

kaggle.com → **Code → New Notebook → File → Import Notebook** → pick
`services/ml/notebooks/train_geopolitic_gnn.ipynb`.

### 4. Configure the session (right-hand panel)

- **Add Input** → attach the Dataset from step 2.
- **Accelerator** → **GPU** (T4 ×2 or P100).
- **Internet** → **On** (needed for `pip install` + W&B).
- **Add-ons → Secrets** → add a secret named exactly **`WANDB_API_KEY`** with your key.

### 5. Run

**Run All.** The five cells: (1) `pip install torch-geometric torchmetrics captum wandb`
(torch/pandas/sklearn/pyarrow are preinstalled), (2) write the `ml/` package to disk, (3)
load `WANDB_API_KEY` from the Secret, (4) auto-find the attached `dataset_parquet`, (5) train.

Outputs land in **`/kaggle/working/artifacts`** (`best.pt`, `last.pt`, `preprocess.pkl`,
`calibrator.pkl`, `metrics.json`, `node_index.json`) and stream to W&B
(`anna-nechytailenko-kyiv-school-of-economics/geosimulation`). Download them from the
notebook's **Output** tab, or restore the best checkpoint anywhere:

```python
import wandb
wandb.Api().artifact(
    "anna-nechytailenko-kyiv-school-of-economics/geosimulation/geosimulation-model:best"
).download("artifacts")
```
