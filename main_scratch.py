# main_scratch.py
# Regresión lineal SIN FRAMEWORK (solo numpy/pandas/matplotlib).
# - Train/Validation/Test split
# - Normal Equation con L2 opcional (ridge desde cero)
# - Métricas: MSE, RMSE, MAE, R2 en train/val/test
# - Reporte Markdown + JSON + gráficas (pred vs real, residuales)

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Utilidades
# ---------------------------

def ensure_outdir(path="out"):
    os.makedirs(path, exist_ok=True)
    return path

def set_seed(seed: int = 42):
    np.random.seed(seed)

def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split manual sin sklearn."""
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    X, y = X[idx], y[idx]
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_val - n_test

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test

def add_bias(X: np.ndarray) -> np.ndarray:
    """Agrega columna de 1s para el sesgo b."""
    return np.c_[np.ones((X.shape[0], 1)), X]

def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z-score: (x - mu)/sigma. Regresa X_std, mu, sigma."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma, mu, sigma

def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma

# ---------------------------
# Modelo: Normal Equation con L2
# ---------------------------

@dataclass
class LinRegScratch:
    l2: float = 0.0               # lambda de regularización L2 (ridge)
    theta: np.ndarray = None      # parámetros [b, w1, w2, ...]
    mu_: np.ndarray = None        # para estandarización
    sigma_: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray, standardize: bool = True):
        """
        X: (n, d) sin bias, y: (n,) o (n,1)
        Si standardize=True, ajusta z-score a X (no al bias).
        """
        y = y.reshape(-1, 1)

        if standardize:
            Xs, mu, sigma = standardize_fit(X)
            self.mu_, self.sigma_ = mu, sigma
        else:
            Xs = X
            self.mu_ = None
            self.sigma_ = None

        Xb = add_bias(Xs)  # (n, d+1)

        # Normal Equation con L2 (no regularizamos el bias):
        # theta = (X^T X + λ * I')^{-1} X^T y
        n_params = Xb.shape[1]
        I = np.eye(n_params)
        I[0, 0] = 0.0  # no regularizar el término de sesgo
        A = Xb.T @ Xb + self.l2 * I
        b = Xb.T @ y
        self.theta = np.linalg.pinv(A) @ b  # pinv por estabilidad

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.theta is not None, "Primero llama fit()"
        if self.mu_ is not None and self.sigma_ is not None:
            X = standardize_apply(X, self.mu_, self.sigma_)
        Xb = add_bias(X)
        return (Xb @ self.theta).ravel()


# ---------------------------
# Métricas
# ---------------------------

def mse(y_true, y_pred): return float(np.mean((y_true - y_pred) ** 2))
def rmse(y_true, y_pred): return float(np.sqrt(mse(y_true, y_pred)))
def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

def compute_all_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }

# ---------------------------
# Gráficas
# ---------------------------

def plot_pred_vs_true(y_true, y_pred, out_png):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_v = float(min(np.min(y_true), np.min(y_pred)))
    max_v = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v])
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.title("Predicción vs. Valor real (Test)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

def plot_residuals(y_true, y_pred, out_png):
    resid = y_true - y_pred
    plt.figure()
    plt.hist(resid, bins=30)
    plt.xlabel("Residuales")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de residuales (Test)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

# ---------------------------
# Reporte
# ---------------------------

def write_report_markdown(
    path_md: str,
    args: argparse.Namespace,
    metrics_train: Dict[str, float],
    metrics_val: Dict[str, float],
    metrics_test: Dict[str, float],
    l2_used: float,
    pred_png: str,
    resid_png: str,
):
    bias_text = diagnose_bias(metrics_train, metrics_val)
    var_text = diagnose_variance(metrics_train, metrics_val)
    fit_text = diagnose_fit(metrics_train, metrics_val)

    md = f"""# Reporte: Regresión lineal desde cero (sin framework)

**Dataset:** `{os.path.basename(args.data)}`  
**Target:** `{args.target}`  
**Features:** {("todas menos target" if args.features is None else ", ".join(args.features.split(",")))}
**Split:** train={int((1-args.val_size-args.test_size)*100)}%  val={int(args.val_size*100)}%  test={int(args.test_size*100)}%  
**Estandarización:** {args.standardize}  
**Regularización L2 (lambda):** {l2_used}

## Métricas

| Conjunto | MSE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| Train | {metrics_train["MSE"]:.6f} | {metrics_train["RMSE"]:.6f} | {metrics_train["MAE"]:.6f} | {metrics_train["R2"]:.6f} |
| Val   | {metrics_val["MSE"]:.6f}   | {metrics_val["RMSE"]:.6f}   | {metrics_val["MAE"]:.6f}   | {metrics_val["R2"]:.6f}   |
| Test  | {metrics_test["MSE"]:.6f}  | {metrics_test["RMSE"]:.6f}  | {metrics_test["MAE"]:.6f}  | {metrics_test["R2"]:.6f}  |

## Diagnóstico

- **Sesgo (Bias):** {bias_text}  
- **Varianza:** {var_text}  
- **Nivel de ajuste:** {fit_text}

## Gráficas
- Predicción vs Valor real (Test): `{os.path.basename(pred_png)}`
- Distribución de residuales (Test): `{os.path.basename(resid_png)}`

> Nota: Este experimento implementa regresión lineal mediante ecuación normal
> con regularización L2 opcional, sin uso de frameworks de ML (no sklearn).
"""
    with open(path_md, "w", encoding="utf-8") as f:
        f.write(md)

def diagnose_bias(m_train: Dict[str, float], m_val: Dict[str, float]) -> str:
    # Heurística simple: sesgo alto si ambos errores son altos y R2 bajos
    hi_err = (m_train["RMSE"] > 0.75 * (m_val["RMSE"] + m_train["RMSE"]) / 2) and (m_val["R2"] < 0.5 and m_train["R2"] < 0.6)
    if hi_err:
        return "alto (errores elevados en train y val; el modelo sub-ajusta patrones)."
    med = abs(m_train["RMSE"] - m_val["RMSE"]) < 0.2 * m_val["RMSE"]
    if med:
        return "medio (errores similares entre train y val; puede haber submodelado leve)."
    return "bajo (errores controlados y desempeño estable)."

def diagnose_variance(m_train: Dict[str, float], m_val: Dict[str, float]) -> str:
    gap = m_val["RMSE"] - m_train["RMSE"]
    if gap > 0.25 * m_val["RMSE"]:
        return "alta (gran diferencia entre train y val; posible sobreajuste)."
    if gap > 0.1 * m_val["RMSE"]:
        return "media (cierta diferencia entre train y val)."
    return "baja (desempeño similar entre train y val)."

def diagnose_fit(m_train: Dict[str, float], m_val: Dict[str, float]) -> str:
    # Underfit: R2 bajo en ambos; Overfit: train muy bueno y val/test mucho peor
    if m_train["R2"] < 0.4 and m_val["R2"] < 0.4:
        return "underfit (modelo poco expresivo; altos errores en train y val)."
    if (m_train["R2"] - m_val["R2"]) > 0.25:
        return "overfit (train notablemente mejor que validación)."
    return "fit adecuado (balance entre error de entrenamiento y validación)."

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Linear Regression from scratch (no ML frameworks).")
    parser.add_argument("--data", type=str, required=True, help="Ruta al CSV.")
    parser.add_argument("--target", type=str, required=True, help="Nombre de la columna objetivo.")
    parser.add_argument("--features", type=str, default=None,
                        help="Lista de columnas feature separadas por coma. Si se omite, usa todas menos target.")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l2", type=float, default=0.0, help="Lambda de regularización L2 (0.0 = sin regularizar).")
    parser.add_argument("--standardize", action="store_true", help="Estandariza features (recomendado).")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir("out")

    # Carga de datos
    df = pd.read_csv(args.data)
    if args.features is None:
        feature_cols = [c for c in df.columns if c != args.target]
    else:
        feature_cols = [c.strip() for c in args.features.split(",")]

    X = df[feature_cols].values.astype(float)
    y = df[args.target].values.astype(float)

    # Split
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, val_size=args.val_size, test_size=args.test_size, shuffle=True, seed=args.seed
    )

    # Fit
    model = LinRegScratch(l2=args.l2)
    model.fit(X_train, y_train, standardize=args.standardize)

    # Predicciones
    yhat_train = model.predict(X_train)
    yhat_val = model.predict(X_val)
    yhat_test = model.predict(X_test)

    # Métricas
    metrics_train = compute_all_metrics(y_train, yhat_train)
    metrics_val = compute_all_metrics(y_val, yhat_val)
    metrics_test = compute_all_metrics(y_test, yhat_test)

    # Gráficas
    pred_png = os.path.join(outdir, "pred_vs_true_scratch.png")
    resid_png = os.path.join(outdir, "residuals_scratch.png")
    plot_pred_vs_true(y_test, yhat_test, pred_png)
    plot_residuals(y_test, yhat_test, resid_png)

    # Reportes
    report_md = os.path.join(outdir, "report_scratch.md")
    write_report_markdown(
        report_md, args, metrics_train, metrics_val, metrics_test, args.l2, pred_png, resid_png
    )

    results = {
        "config": {
            "data": args.data,
            "target": args.target,
            "features": feature_cols,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "seed": args.seed,
            "l2": args.l2,
            "standardize": bool(args.standardize),
        },
        "metrics": {
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test,
        },
        "theta_shape": None if model.theta is None else list(model.theta.shape),
    }
    with open(os.path.join(outdir, "results_scratch.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("OK - entrenamiento SIN framework completado.")
    print("Reporte:", report_md)
    print("Resultados JSON:", os.path.join(outdir, "results_scratch.json"))
    print("Gráficas:", pred_png, "|", resid_png)


if __name__ == "__main__":
    main()
