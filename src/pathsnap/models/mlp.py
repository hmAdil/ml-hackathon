"""Model: PyTorch MLP on GPU (median-impute + standardise inside, BCE, AdamW, early stopping).

Early stopping monitors a held-out contiguous tail of the training rows (whole trips, since
rows are ordered by trip). Epoch-wise train/val loss is kept in history_ and written to
results/models/mlp_history.json (+ mlp_loss_curve.png) at the end of every fit; the last fit
in pathsnap.train is the final full-train fit.
"""
import json

import numpy as np
import torch
from torch import nn

from ..config import RESULTS

NAME = "mlp"
FAMILY = "neural"
DEFAULT_PARAMS = {"hidden": (128, 64), "dropout": 0.1, "lr": 2e-3, "weight_decay": 1e-4,
                  "batch_size": 8192, "max_epochs": 60, "patience": 6, "val_frac": 0.1,
                  "amp": True, "seed": 42}
SEARCH_SPACE = {"lr": ("log", 1e-4, 1e-2), "weight_decay": ("log", 1e-6, 1e-2),
                "dropout": ("float", 0.0, 0.4), "batch_size": ("int", 4096, 16384)}


def _net(d_in, hidden, dropout):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, 1))


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fwd(net, Z, bs=65536):
    return torch.cat([net(Z[i:i + bs]).squeeze(1).float() for i in range(0, len(Z), bs)])


class MLPScorer:
    def __init__(self, **params):
        self.params = {**DEFAULT_PARAMS, **params}
        self.history_ = []
        self._net = None

    def _prep(self, X):
        X = np.asarray(X, np.float32)
        X = np.where(np.isnan(X), self.med_, X)
        return ((X - self.mu_) / self.sd_).astype(np.float32)

    def fit(self, X, y):
        p = self.params
        torch.manual_seed(p["seed"])
        np.random.seed(p["seed"])
        X, y = np.asarray(X, np.float32), np.asarray(y, np.float32)
        cut = int(len(X) * (1 - p["val_frac"]))
        # preprocessing statistics from the training part only
        self.med_ = np.nan_to_num(np.nanmedian(X[:cut], axis=0)).astype(np.float32)
        Xi = np.where(np.isnan(X[:cut]), self.med_, X[:cut])
        self.mu_ = Xi.mean(0)
        self.sd_ = Xi.std(0) + 1e-6
        Z = self._prep(X)
        dev = _dev()
        amp = bool(p["amp"]) and dev.type == "cuda"
        Zt, yt = torch.tensor(Z, device=dev), torch.tensor(y, device=dev)
        self.d_in_ = Z.shape[1]
        net = _net(self.d_in_, p["hidden"], p["dropout"]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        lossf = nn.BCEWithLogitsLoss()
        g = torch.Generator(device=dev).manual_seed(p["seed"])
        best, best_state, bad, self.history_ = np.inf, None, 0, []
        for ep in range(p["max_epochs"]):
            net.train()
            perm = torch.randperm(cut, device=dev, generator=g)
            tot = 0.0
            for i in range(0, cut, p["batch_size"]):
                idx = perm[i:i + p["batch_size"]]
                with torch.autocast(dev.type, enabled=amp):
                    out = net(Zt[idx]).squeeze(1)
                loss = lossf(out.float(), yt[idx])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(idx)
            net.eval()
            with torch.no_grad():
                tr_l = float(lossf(_fwd(net, Zt[:cut]), yt[:cut]))
                va_l = float(lossf(_fwd(net, Zt[cut:]), yt[cut:])) if cut < len(X) else tr_l
            self.history_.append({"epoch": ep + 1, "train_loss": tr_l, "val_loss": va_l,
                                  "train_loss_running": tot / cut})
            if va_l < best - 1e-5:
                best, bad = va_l, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        self.best_epoch_ = int(np.argmin([h["val_loss"] for h in self.history_]) + 1)
        self.state_ = best_state
        self._net = None
        self._write_history()
        return self

    def _write_history(self):
        out = RESULTS / "models"
        out.mkdir(parents=True, exist_ok=True)
        (out / "mlp_history.json").write_text(json.dumps(
            {"params": {**self.params, "hidden": list(self.params["hidden"])},
             "best_epoch": self.best_epoch_, "history": self.history_}, indent=2))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            e = [h["epoch"] for h in self.history_]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(e, [h["train_loss"] for h in self.history_], label="train")
            ax.plot(e, [h["val_loss"] for h in self.history_], label="validation")
            ax.axvline(self.best_epoch_, ls="--", color="gray", label="best epoch")
            ax.set_xlabel("epoch")
            ax.set_ylabel("BCE loss")
            ax.set_title("MLP loss curve")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "mlp_loss_curve.png", dpi=150)
            plt.close(fig)
        except Exception as exc:  # plotting must never break training
            print("loss curve not drawn:", exc)

    def _get_net(self):
        if self._net is None:
            net = _net(self.d_in_, self.params["hidden"], self.params["dropout"])
            net.load_state_dict(self.state_)
            self._net = net.to(_dev()).eval()
        return self._net

    def predict_proba(self, X):
        net = self._get_net()
        Z = torch.tensor(self._prep(X), device=_dev())
        with torch.no_grad():
            p1 = torch.sigmoid(_fwd(net, Z)).cpu().numpy().astype(np.float64)
        return np.column_stack([1 - p1, p1])

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_net"] = None
        return d


def build(**params):
    return MLPScorer(**params)
