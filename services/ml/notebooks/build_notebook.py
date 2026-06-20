"""Generate the self-contained Kaggle notebook from the canonical ml/ sources.

The notebook embeds the ml package (base64-encoded, so quoting/unicode in the sources can
never corrupt the JSON), writes it to disk on the Kaggle VM, then runs training. This keeps
ONE source of truth (the ml/ modules used by tests + the FastAPI server) while still letting
the user upload exactly two things to Kaggle: the dataset_parquet folder + this .ipynb.

Run (stdlib only):  python build_notebook.py
"""

from __future__ import annotations

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(HERE, "..", "ml")
OUT = os.path.join(HERE, "train_geopolitic_gnn.ipynb")

# dependency order (so a plain exec would also work); __init__ written last
MODULES = ["config", "timestep", "features", "losses", "metrics",
           "dataset", "model", "calibrate", "explain", "train", "infer"]


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def build() -> dict:
    files = {}
    for m in MODULES:
        with open(os.path.join(ML_DIR, f"{m}.py")) as f:
            files[f"ml/{m}.py"] = base64.b64encode(f.read().encode()).decode()
    files["ml/__init__.py"] = base64.b64encode(b"")  # empty package marker
    files["ml/__init__.py"] = files["ml/__init__.py"].decode()

    materialize = (
        "import base64, pathlib\n"
        "pathlib.Path('ml').mkdir(exist_ok=True)\n"
        "FILES = " + json.dumps(files, indent=0) + "\n"
        "for _p, _b in FILES.items():\n"
        "    open(_p, 'w').write(base64.b64decode(_b).decode())\n"
        "print('wrote', len(FILES), 'ml package files')\n"
    )

    pip = ("# Kaggle GPU images ship torch, numpy, pandas, scikit-learn, pyarrow, tqdm.\n"
           "!pip -q install torch-geometric torchmetrics captum wandb tqdm\n")

    wandb_cell = (
        "import os\n"
        "try:\n"
        "    from kaggle_secrets import UserSecretsClient\n"
        "    os.environ['WANDB_API_KEY'] = UserSecretsClient().get_secret('WANDB_API_KEY')\n"
        "    print('W&B key loaded from Kaggle Secret')\n"
        "except Exception as e:\n"
        "    print('Add a Kaggle Secret named WANDB_API_KEY to enable logging:', e)\n"
        "os.environ.setdefault('WANDB_ENTITY', 'anna-nechytailenko-kyiv-school-of-economics')\n"
        "os.environ.setdefault('WANDB_PROJECT', 'geosimulation')\n"
    )

    run_cell = (
        "import os\n"
        "def find_data():\n"
        "    for root in ['/kaggle/input', '.']:\n"
        "        if not os.path.isdir(root):\n"
        "            continue\n"
        "        for dp, _dn, fn in os.walk(root):\n"
        "            if 'node_snapshots.parquet' in fn:\n"
        "                return dp\n"
        "    return 'dataset_parquet'\n"
        "os.environ['GEO_DATA_DIR'] = find_data()\n"
        "os.environ['GEO_ARTIFACTS_DIR'] = '/kaggle/working/artifacts' if os.path.isdir('/kaggle/working') else 'artifacts'\n"
        "from ml.config import Config\n"
        "from ml.train import train\n"
        "cfg = Config.from_env()\n"
        "print('data_dir =', cfg.data_dir, '| artifacts =', cfg.artifacts_dir)\n"
        "metrics = train(cfg)\n"
        "metrics\n"
    )

    intro = (
        "# Geopolitic GNN — training (Kaggle)\n\n"
        "Self-contained: upload **two** things to Kaggle and *Run All* —\n"
        "1. the `dataset_parquet/` folder as a **Dataset** (Add Input), and\n"
        "2. this notebook.\n\n"
        "Set **Accelerator = GPU**, **Internet = On**, and add a **Secret** `WANDB_API_KEY`.\n"
        "Outputs (best.pt, preprocess.pkl, calibrator.pkl, metrics.json) land in "
        "`/kaggle/working/artifacts` and stream to W&B "
        "(`anna-nechytailenko-kyiv-school-of-economics/geosimulation`)."
    )

    nb = {
        "cells": [
            md_cell(intro),
            code_cell(pip),
            code_cell(materialize),
            code_cell(wandb_cell),
            code_cell(run_cell),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    return nb


def main() -> None:
    nb = build()
    with open(OUT, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
