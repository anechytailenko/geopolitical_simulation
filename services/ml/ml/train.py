"""Training entrypoint (plans/03 §2.4, §4). Chronological splits, per-epoch negative
resampling (train) vs frozen negatives (val/test), class-weighted/focal loss, early stopping
on val macro-F1, temperature calibration on val, test report, and W&B tracking + artifacts.

Checkpoint policy (plans/03 §4.4):
- best.pt  — written ONLY when val macro-F1 strictly improves (> best + 1e-4).
- last.pt  — overwritten every epoch (resume after a cut-off Kaggle session).
Each checkpoint carries {model_state, optimizer_state, epoch, best_macro_f1, preprocess,
calibrator, config, git_sha} so it both resumes training and serves consistently.

Run:  python -m ml.train --data-dir dataset_parquet --artifacts-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # tqdm is optional — fall back to a no-op so training still runs
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kwargs):
            self.iterable = [] if iterable is None else iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *a, **k):
            pass

        def close(self):
            pass

from .calibrate import TemperatureScaler
from .config import Config
from .dataset import GeopoliticDataset, C
from .losses import inverse_frequency_weights, make_loss
from .metrics import Evaluator
from .model import SpatioTemporalEdgeClassifier


# --------------------------------------------------------------------------- utilities
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(cfg: Config) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def move_window(window: list, device: torch.device) -> list:
    return [d.to(device) for d in window]


@torch.no_grad()
def collect_logits(model, ds, samples, device, desc: str | None = None):
    model.eval()
    logits_all, labels_all = [], []
    iterator = tqdm(samples, desc=desc, leave=False, unit="month") if desc else samples
    for s in iterator:
        window = move_window(ds.build_window(s.target_ts), device)
        logits = model(window, s.pair_index.to(device), s.pair_attr.to(device))
        logits_all.append(logits.cpu()); labels_all.append(s.labels.cpu())
    if not logits_all:
        return torch.zeros((0, len(ds.cfg.class_names))), torch.zeros((0,), dtype=torch.long)
    return torch.cat(logits_all), torch.cat(labels_all)


def evaluate(model, ds, samples, device, calibrator: TemperatureScaler | None = None,
             desc: str | None = None):
    logits, labels = collect_logits(model, ds, samples, device, desc=desc)
    probs = calibrator.probs(logits) if calibrator is not None else torch.softmax(logits, dim=-1)
    ev = Evaluator("cpu");
    if labels.numel():
        ev.update(probs, labels)
    return ev.compute(), logits, labels


def save_checkpoint(path, model, optimizer, epoch, best, cfg, pp, calibrator_temp=None):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_macro_f1": best,
        "preprocess": pp,
        "calibrator_temperature": calibrator_temp,
        "config": cfg.to_dict(),
        "git_sha": git_sha(),
        "class_names": list(cfg.class_names),
    }, path)


# ------------------------------------------------------------------------- W&B wrapper
class WandbRun:
    """Thin wrapper: no-ops cleanly if wandb is missing or mode='disabled'."""

    def __init__(self, cfg: Config):
        self.run = None
        if cfg.wandb_mode == "disabled":
            return
        try:
            import wandb
            self._wandb = wandb
            key = os.environ.get("WANDB_API_KEY")
            if not key:
                try:
                    from kaggle_secrets import UserSecretsClient
                    key = UserSecretsClient().get_secret("WANDB_API_KEY")
                except Exception:
                    key = None
            if key:
                wandb.login(key=key)
            self.run = wandb.init(
                project=cfg.wandb_project, entity=cfg.wandb_entity, mode=cfg.wandb_mode,
                name=(cfg.run_name or None), config=cfg.to_dict(), job_type="train",
            )
        except Exception as e:  # never let tracking break training
            print(f"[wandb] disabled ({e})")
            self.run = None

    def log(self, data: dict, step: int | None = None) -> None:
        if self.run is not None:
            self._wandb.log(data, step=step)

    def confusion(self, y_true, preds, class_names):
        if self.run is not None:
            try:
                self._wandb.log({"test/confusion": self._wandb.plot.confusion_matrix(
                    y_true=y_true, preds=preds, class_names=class_names)})
            except Exception:
                pass

    def log_artifact(self, files: list[str], name: str, metadata: dict, aliases: list[str]) -> None:
        if self.run is None:
            return
        try:
            art = self._wandb.Artifact(name, type="model", metadata=metadata)
            for f in files:
                if os.path.exists(f):
                    art.add_file(f)
            self.run.log_artifact(art, aliases=aliases)
        except Exception as e:
            print(f"[wandb] artifact log failed ({e})")

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


# ------------------------------------------------------------------------------- train
def train(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = pick_device(cfg)
    print(f"[train] device={device} data_dir={cfg.data_dir}")

    ds = GeopoliticDataset.from_parquet(cfg)
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    pp_path = os.path.join(cfg.artifacts_dir, "preprocess.pkl")
    ds.pp.save(pp_path)
    with open(os.path.join(cfg.artifacts_dir, "node_index.json"), "w") as f:
        json.dump({"country_ids": ds.pp.country_ids, "actor_ids": ds.pp.actor_ids,
                   "class_names": cfg.class_names}, f)

    model = SpatioTemporalEdgeClassifier.from_dataset(ds, cfg).to(device)
    counts = ds.class_counts("train", seed=cfg.seed)
    weights = inverse_frequency_weights(counts).to(device)
    print(f"[train] class counts={counts.tolist()} weights={[round(w,2) for w in weights.tolist()]}")
    criterion = make_loss(cfg.loss, weights, cfg.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    val_samples = ds.make_samples("val", np.random.default_rng(cfg.seed + 1))  # frozen
    best_path = os.path.join(cfg.artifacts_dir, "best.pt")
    last_path = os.path.join(cfg.artifacts_dir, "last.pt")
    wb = WandbRun(cfg)

    best, best_epoch, patience = -1.0, -1, 0
    for epoch in range(cfg.epochs):
        model.train()
        rng = np.random.default_rng(cfg.seed * 1000 + epoch)   # fresh train negatives each epoch
        train_samples = ds.make_samples("train", rng)
        order = list(range(len(train_samples))); rng.shuffle(order)
        t0 = time.time(); total, nb = 0.0, 0
        pbar = tqdm(order, desc=f"epoch {epoch:02d}/{cfg.epochs - 1}", leave=False, unit="batch")
        for i in pbar:
            s = train_samples[i]
            window = move_window(ds.build_window(s.target_ts), device)
            logits = model(window, s.pair_index.to(device), s.pair_attr.to(device))
            loss = criterion(logits, s.labels.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.item()); nb += 1
            pbar.set_postfix(loss=f"{total / nb:.4f}", best_f1=f"{max(best, 0.0):.3f}")
        pbar.close()
        train_loss = total / max(1, nb)

        val_res, _, _ = evaluate(model, ds, val_samples, device)
        dt = time.time() - t0
        print(f"[epoch {epoch:02d}] loss={train_loss:.4f} val_macroF1={val_res.macro_f1:.4f} "
              f"val_ece={val_res.ece:.4f} ({dt:.1f}s)")
        log = {"epoch": epoch, "train/loss": train_loss, "val/macro_f1": val_res.macro_f1,
               "val/ece": val_res.ece}
        for c, f1 in zip(cfg.class_names, val_res.per_class_f1):
            log[f"val/f1_{c}"] = f1
        wb.log(log, step=epoch)

        save_checkpoint(last_path, model, optimizer, epoch, best, cfg, ds.pp)  # every epoch
        if val_res.macro_f1 > best + 1e-4:                                     # strict improvement only
            best, best_epoch, patience = val_res.macro_f1, epoch, 0
            save_checkpoint(best_path, model, optimizer, epoch, best, cfg, ds.pp)
        else:
            patience += 1
            if patience >= cfg.patience:
                print(f"[train] early stop at epoch {epoch} (best={best:.4f} @ {best_epoch})")
                break

    # ---- load best, calibrate on val, evaluate test --------------------------------
    if os.path.exists(best_path):
        # weights_only=False: our checkpoints embed the Preprocess bundle (trusted, self-written).
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state"])

    _, val_logits, val_labels = evaluate(model, ds, val_samples, device, desc="calibrate (val)")
    calibrator = TemperatureScaler()
    if val_labels.numel():
        calibrator.fit(val_logits, val_labels)
    cal_path = os.path.join(cfg.artifacts_dir, "calibrator.pkl")
    calibrator.save(cal_path)
    save_checkpoint(best_path, model, optimizer, best_epoch, best, cfg, ds.pp, calibrator.temperature)

    test_samples = ds.make_samples("test", np.random.default_rng(cfg.seed + 2))
    test_res_uncal, test_logits, test_labels = evaluate(model, ds, test_samples, device, desc="test")
    test_res_cal, _, _ = evaluate(model, ds, test_samples, device, calibrator)

    metrics = {
        "best_val_macro_f1": best, "best_epoch": best_epoch,
        "temperature": calibrator.temperature,
        "test_macro_f1": test_res_cal.macro_f1,
        "test_per_class_f1": dict(zip(cfg.class_names, test_res_cal.per_class_f1)),
        "test_ece_uncalibrated": test_res_uncal.ece,
        "test_ece_calibrated": test_res_cal.ece,
        "test_confusion": test_res_cal.confusion,
        "test_n": test_res_cal.n,
        "git_sha": git_sha(),
    }
    metrics_path = os.path.join(cfg.artifacts_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] TEST macro-F1={test_res_cal.macro_f1:.4f} "
          f"ECE {test_res_uncal.ece:.4f}->{test_res_cal.ece:.4f} T={calibrator.temperature:.3f}")

    if test_labels.numel():
        probs = calibrator.probs(test_logits)
        wb.confusion(test_labels.tolist(), probs.argmax(-1).tolist(), cfg.class_names)
    wb.log_artifact(
        [best_path, pp_path, cal_path, metrics_path],
        name=f"{cfg.wandb_project}-model",
        metadata={"test_macro_f1": test_res_cal.macro_f1, "best_val_macro_f1": best, "git_sha": git_sha()},
        aliases=["best", "latest", f"f1-{test_res_cal.macro_f1:.3f}"],
    )
    wb.finish()
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the geopolitic spatio-temporal GNN.")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--loss", default=None, choices=["weighted_ce", "focal"])
    p.add_argument("--device", default=None)
    p.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config.from_env()
    if args.data_dir: cfg.data_dir = args.data_dir
    if args.artifacts_dir: cfg.artifacts_dir = args.artifacts_dir
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.loss: cfg.loss = args.loss
    if args.device: cfg.device = args.device
    if args.wandb_mode: cfg.wandb_mode = args.wandb_mode
    train(cfg)


if __name__ == "__main__":
    main()
