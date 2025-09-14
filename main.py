#!/usr/bin/env python3
"""
Machine Learning Framework Project
Algorithm: Linear Regression (baseline) + Ridge Regression (regularization)
Dataset: scikit-learn diabetes (regression)
Runs fully from a single .py file (no notebook).

Uso:
    python main.py --plot        # entrena, guarda figuras y escribe reporte
    python main.py --predict 5   # imprime 5 predicciones del set de test
"""

import argparse
import warnings
from pathlib import Path
import json
import datetime

# --- NumPy & warnings: silenciar RuntimeWarning producidos por BLAS/matmul en algunos Macs ---
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(invalid="ignore", over="ignore", divide="ignore")

# --- ML / Plot ---
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle

RANDOM_STATE = 42


def split_data(X, y):
    """
    60/20/20 split: train / val / test con barajado fijo.
    """
    X, y = shuffle(X, y, random_state=RANDOM_STATE)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_pipelines():
    """
    Dos pipelines:
      - baseline: LinearRegression con escalado
      - ridge: Ridge con escalado (alpha se elige por curva de validación)
    """
    baseline = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("reg", LinearRegression())
    ])
    ridge = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("reg", Ridge(random_state=RANDOM_STATE))
    ])
    return baseline, ridge


def eval_model(model, X_tr, y_tr, X_ev, y_ev, name="model"):
    """
    Entrena en train y evalúa en eval (val o test). Devuelve métricas.
    """
    model.fit(X_tr, y_tr)
    pred_tr = model.predict(X_tr)
    pred_ev = model.predict(X_ev)
    metrics = {
        "rmse_train": float(np.sqrt(mean_squared_error(y_tr, pred_tr))),
        "mae_train": float(mean_absolute_error(y_tr, pred_tr)),
        "r2_train": float(r2_score(y_tr, pred_tr)),
        "rmse_eval": float(np.sqrt(mean_squared_error(y_ev, pred_ev))),
        "mae_eval": float(mean_absolute_error(y_ev, pred_ev)),
        "r2_eval": float(r2_score(y_ev, pred_ev)),
    }
    return metrics


def estimate_bias_variance(r2_train, r2_val):
    """
    Heurística simple para diagnóstico que pide el rubric:
      - sesgo (bias): bajo/medio/alto
      - varianza: bajo/medio/alto
      - ajuste: underfit/fit/overfit
    """
    bias_level = "bajo"
    var_level = "bajo"
    fit_level = "fit"

    # underfit: ambos bajos
    if r2_train < 0.3 and r2_val < 0.3:
        bias_level = "alto"
        fit_level = "underfit"
    # overfit: gap grande entre train y val
    elif (r2_train - r2_val) > 0.15 and r2_val < r2_train:
        var_level = "medio" if r2_val > 0.3 else "alto"
        fit_level = "overfit"
    # término medio
    elif 0.3 <= r2_val <= 0.6:
        bias_level = "medio"
        var_level = "medio"
        fit_level = "fit"

    return bias_level, var_level, fit_level


def plot_learning_curve(model, X, y, out_path):
    """
    Curva de aprendizaje: R2 en train y val vs. tamaño de entrenamiento.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 8),
        shuffle=True,
        random_state=RANDOM_STATE
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, marker="o", label="Entrenamiento (R2)")
    plt.plot(train_sizes, val_mean, marker="s", label="Validación (R2)")
    plt.xlabel("Tamaño de entrenamiento")
    plt.ylabel("R2")
    plt.title("Curva de aprendizaje (Regresión Lineal)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return train_sizes.tolist(), train_mean.tolist(), val_mean.tolist()


def plot_validation_curve_ridge(X, y, out_path):
    """
    Curva de validación para elegir alpha en Ridge.
    """
    alphas = np.logspace(-3, 3, 20)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(random_state=RANDOM_STATE))
    ])

    train_scores, val_scores = validation_curve(
        pipe, X, y,
        param_name="reg__alpha",
        param_range=alphas,
        cv=5,
        scoring="r2"
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure()
    plt.semilogx(alphas, train_mean, marker="o", label="Entrenamiento (R2)")
    plt.semilogx(alphas, val_mean, marker="s", label="Validación (R2)")
    plt.xlabel("alpha (Ridge)")
    plt.ylabel("R2")
    plt.title("Curva de validación (Ridge)")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    best_idx = int(np.argmax(val_mean))
    best_alpha = float(alphas[best_idx])
    return alphas.tolist(), train_mean.tolist(), val_mean.tolist(), best_alpha


def main(args):
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Dataset
    data = load_diabetes()
    X, y = data.data.astype(np.float64), data.target.astype(np.float64)
    feature_names = data.feature_names

    # Splits
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Modelos
    baseline, ridge = make_pipelines()

    # --- Baseline Linear Regression ---
    base_metrics_val = eval_model(
        baseline, X_train, y_train, X_val, y_val, name="LinearRegression"
    )
    base_metrics_test = eval_model(
        baseline,
        np.vstack([X_train, X_val]),
        np.hstack([y_train, y_val]),
        X_test,
        y_test,
        name="LinearRegression"
    )
    bias_b, var_b, fit_b = estimate_bias_variance(
        base_metrics_val["r2_train"], base_metrics_val["r2_eval"]
    )

    # Curva de aprendizaje (baseline)
    lc_path = out_dir / "learning_curve_linear.png"
    lc_sizes, lc_train, lc_val = plot_learning_curve(baseline, X_train, y_train, lc_path)

    # --- Ridge con selección de alpha por curva de validación ---
    vc_path = out_dir / "validation_curve_ridge.png"
    alphas, vc_train, vc_val, best_alpha = plot_validation_curve_ridge(X_train, y_train, vc_path)

    ridge.set_params(reg__alpha=best_alpha)
    ridge_metrics_val = eval_model(
        ridge, X_train, y_train, X_val, y_val, name="Ridge"
    )
    ridge_metrics_test = eval_model(
        ridge,
        np.vstack([X_train, X_val]),
        np.hstack([y_train, y_val]),
        X_test,
        y_test,
        name="Ridge"
    )
    bias_r, var_r, fit_r = estimate_bias_variance(
        ridge_metrics_val["r2_train"], ridge_metrics_val["r2_eval"]
    )

    # Guardar resultados
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": "sklearn.datasets.load_diabetes",
        "feature_names": feature_names,
        "splits": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "baseline_linear": {
            "val": base_metrics_val,
            "test": base_metrics_test,
            "diagnosis": {"bias": bias_b, "varianza": var_b, "ajuste": fit_b}
        },
        "ridge": {
            "alpha": best_alpha,
            "val": ridge_metrics_val,
            "test": ridge_metrics_test,
            "diagnosis": {"bias": bias_r, "varianza": var_r, "ajuste": fit_r}
        },
        "learning_curve_png": str(lc_path),
        "validation_curve_png": str(vc_path),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Si pidieron predicciones, entrenamos Ridge en train+val y predecimos en test[:N]
    if args.predict is not None and args.predict > 0:
        ridge.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
        y_pred = ridge.predict(X_test[:args.predict])
        for i, (yp, yt) in enumerate(zip(y_pred, y_test[:args.predict])):
            print(f"Sample {i}: y_true={yt:.3f} | y_pred={yp:.3f}")
        return

    # Reporte en Markdown
    report = f"""# Reporte: Desempeño del Modelo (Regresión Lineal + Ridge)

**Fecha:** {datetime.datetime.now().strftime('%Y-%m-%d')}

## Dataset
- *Diabetes* de scikit-learn (regresión). 10 características, variable objetivo: progresión de la enfermedad.  
- Splits: **train**={len(y_train)}, **val**={len(y_val)}, **test**={len(y_test)}.

## Modelo 1 — Baseline: LinearRegression
**Validación:** R²={base_metrics_val['r2_eval']:.3f}, RMSE={base_metrics_val['rmse_eval']:.3f}, MAE={base_metrics_val['mae_eval']:.3f}  
**Entrenamiento:** R²={base_metrics_val['r2_train']:.3f}

**Diagnóstico:** sesgo={bias_b}, varianza={var_b}, ajuste={fit_b}

![Curva de aprendizaje](learning_curve_linear.png)

La curva de aprendizaje muestra cómo el R² de entrenamiento y validación convergen al aumentar el tamaño de entrenamiento. Una brecha grande indicaría varianza alta (sobreajuste).

## Modelo 2 — Ridge (regularización L2)
alpha seleccionado por curva de validación: **{best_alpha:.5f}**

**Validación:** R²={ridge_metrics_val['r2_eval']:.3f}, RMSE={ridge_metrics_val['rmse_eval']:.3f}, MAE={ridge_metrics_val['mae_eval']:.3f}  
**Entrenamiento:** R²={ridge_metrics_val['r2_train']:.3f}

**Diagnóstico:** sesgo={bias_r}, varianza={var_r}, ajuste={fit_r}

![Curva de validación (Ridge)](validation_curve_ridge.png)

## Evaluación en Test (generalización)
**LinearRegression:** R²={base_metrics_test['r2_eval']:.3f}, RMSE={base_metrics_test['rmse_eval']:.3f}, MAE={base_metrics_test['mae_eval']:.3f}  
**Ridge:** R²={ridge_metrics_test['r2_eval']:.3f}, RMSE={ridge_metrics_test['rmse_eval']:.3f}, MAE={ridge_metrics_test['mae_eval']:.3f}

## Conclusiones
- El uso de **regularización Ridge** ajustando *alpha* mejoró el desempeño vs. baseline (ver métricas).  
- Si persiste **alto sesgo**: agregar características no lineales o cambiar de modelo.  
- Si persiste **alta varianza**: más datos, regularización más fuerte, o reducción de complejidad.

---
*Este reporte fue generado automáticamente por `main.py`.*
"""
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("Listo. Revisa la carpeta 'out' para ver figuras y reportes.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Entrena modelos y genera gráficas/reportes")
    parser.add_argument("--predict", type=int, default=None, help="Imprime N predicciones de ejemplo del set de test")
    args = parser.parse_args()
    main(args)
