"""A5 - Ridge regression from scratch: predict target_bps (final auction price vs indicative).

Loss:      L(w) = mean((y - Xw)^2) + lam * sum(w^2)
Closed:    w = (X'X + n*lam*I)^-1 X'y
Gradient:  dL/dw = -(2/n) X'(y - Xw) + 2*lam*w      ->  w <- w - lr * dL/dw

Features are standardized with train-set stats; the intercept is the train mean of y (not penalized).
Testing is walk-forward: train on earlier days, score on later days the model never saw.

    uv run python -m noi_tool.model
    uv run python -m noi_tool.model --auction C --test-frac 0.3
"""

import argparse
import json

import numpy as np
import pandas as pd

from noi_tool.dataset import FEATURES, OUT_PATH as TRAINING_PATH
from shared.config import DATA_DIR, REPORTS_DIR

LAMBDAS = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
MODEL_PATH = DATA_DIR / "noi_ridge_model.json"


class Ridge:
    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def _prep(self, X):
        return (X - self.mu) / self.sd

    def fit(self, X, y, method: str = "closed", lr: float | None = None, epochs: int = 100_000, tol: float = 1e-12):
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        Z, self.b = self._prep(X), y.mean()
        yc = y - self.b
        n, p = Z.shape

        if method == "closed":
            self.w = np.linalg.solve(Z.T @ Z + n * self.lam * np.eye(p), Z.T @ yc)
        else:
            # Largest safe step = 1 / L, where L = max curvature of the loss = 2*eig_max(Z'Z/n) + 2*lam
            lr = lr or 1 / (2 * np.linalg.eigvalsh(Z.T @ Z / n)[-1] + 2 * self.lam)
            self.w, self.loss_history = np.zeros(p), []
            for _ in range(epochs):
                resid = yc - Z @ self.w
                self.loss_history.append(np.mean(resid**2) + self.lam * np.sum(self.w**2))
                self.w -= lr * (-(2 / n) * Z.T @ resid + 2 * self.lam * self.w)
                if len(self.loss_history) > 1 and abs(self.loss_history[-2] - self.loss_history[-1]) < tol:
                    break
        return self

    def predict(self, X):
        return self._prep(X) @ self.w + self.b

    def save(self, path):
        path.write_text(json.dumps({
            "lam": self.lam, "features": FEATURES, "b": float(self.b),
            "mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist(),
        }, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(path.read_text())
        m = cls(d["lam"])
        m.b, m.mu, m.sd, m.w = d["b"], np.array(d["mu"]), np.array(d["sd"]), np.array(d["w"])
        return m


def metrics(y, pred) -> dict:
    nonzero = (y != 0) & (pred != 0)  # the baseline predicts 0 = no direction call
    return {
        "mae_bps": float(np.mean(np.abs(y - pred))),
        "rmse_bps": float(np.sqrt(np.mean((y - pred) ** 2))),
        "direction_hit": float(np.mean(np.sign(pred[nonzero]) == np.sign(y[nonzero]))) if nonzero.any() else float("nan"),
        "r2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def split_by_date(df: pd.DataFrame, test_frac: float):
    dates = sorted(df["date"].unique())
    if len(dates) < 3:
        raise SystemExit(f"Need at least 3 days of data to split train/test (have {len(dates)}).")
    cut = dates[max(1, int(len(dates) * (1 - test_frac)))]
    return df[df["date"] < cut], df[df["date"] >= cut]


def tune_lambda(train: pd.DataFrame) -> float:
    """Pick lambda on the last 20% of TRAIN days, so the test days stay untouched."""
    fit_part, val_part = split_by_date(train, 0.2)
    Xf, yf = fit_part[FEATURES].to_numpy(float), fit_part["target_bps"].to_numpy(float)
    Xv, yv = val_part[FEATURES].to_numpy(float), val_part["target_bps"].to_numpy(float)
    scores = {lam: metrics(yv, Ridge(lam).fit(Xf, yf).predict(Xv))["mae_bps"] for lam in LAMBDAS}
    return min(scores, key=scores.get)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auction", choices=["C", "M"], help="Only close (C) or open (M) auctions")
    parser.add_argument("--test-frac", type=float, default=0.3)
    args = parser.parse_args()

    df = pd.read_csv(TRAINING_PATH)
    if args.auction:
        df = df[df["auction"] == args.auction]
    train, test = split_by_date(df, args.test_frac)
    Xtr, ytr = train[FEATURES].to_numpy(float), train["target_bps"].to_numpy(float)
    Xte, yte = test[FEATURES].to_numpy(float), test["target_bps"].to_numpy(float)

    lam = tune_lambda(train) if train["date"].nunique() >= 3 else 1.0
    closed = Ridge(lam).fit(Xtr, ytr, method="closed")
    gd = Ridge(lam).fit(Xtr, ytr, method="gd")

    results = {
        "train_days": int(train["date"].nunique()), "test_days": int(test["date"].nunique()),
        "train_rows": len(train), "test_rows": len(test), "lambda": lam,
        "baseline_trust_nyse": metrics(yte, np.zeros_like(yte)),
        "ridge_closed_form": metrics(yte, closed.predict(Xte)),
        "ridge_gradient_descent": metrics(yte, gd.predict(Xte)),
        "gd_vs_closed_max_weight_diff": float(np.max(np.abs(closed.w - gd.w))),
        "gd_epochs": len(gd.loss_history),
        "coefficients_per_1sd": dict(zip(FEATURES, map(float, closed.w))),
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "noi_ridge_results.json").write_text(json.dumps(results, indent=2))
    # Live tools use a model refit on ALL days with the tuned lambda
    Ridge(lam).fit(df[FEATURES].to_numpy(float), df["target_bps"].to_numpy(float)).save(MODEL_PATH)

    print(f"Train {results['train_days']} days / {len(train):,} rows | Test {results['test_days']} days / {len(test):,} rows | lambda={lam}")
    print(f"\n{'model':24} {'MAE bps':>8} {'RMSE':>8} {'dir hit':>8} {'R2':>7}")
    for name in ["baseline_trust_nyse", "ridge_closed_form", "ridge_gradient_descent"]:
        m = results[name]
        hit = "n/a" if np.isnan(m["direction_hit"]) else f"{m['direction_hit']:.1%}"
        print(f"{name:24} {m['mae_bps']:8.2f} {m['rmse_bps']:8.2f} {hit:>8} {m['r2']:7.3f}")
    print(f"\nGD matched closed form within {results['gd_vs_closed_max_weight_diff']:.2e} ({results['gd_epochs']} epochs)")
    print("\nCoefficients (bps per 1 std dev of feature):")
    for f, w in sorted(results["coefficients_per_1sd"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {f:16} {w:+.3f}")


if __name__ == "__main__":
    main()
