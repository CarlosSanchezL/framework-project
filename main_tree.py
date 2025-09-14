#!/usr/bin/env python3
"""
Second implementation for the course: Decision Tree Regressor
Dataset: scikit-learn diabetes
Fully runnable from a single .py file (no notebooks).

Usage:
    python main_tree.py --plot
    python main_tree.py --predict 5
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle
from pathlib import Path
import json, datetime

RANDOM_STATE = 42

def split_data(X, y):
    X, y = shuffle(X, y, random_state=RANDOM_STATE)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test

def eval_model(model, X_tr, y_tr, X_ev, y_ev):
    model.fit(X_tr, y_tr)
    pred_tr = model.predict(X_tr)
    pred_ev = model.predict(X_ev)
    return {
        "rmse_train": float(np.sqrt(mean_squared_error(y_tr, pred_tr))),
        "mae_train": float(mean_absolute_error(y_tr, pred_tr)),
        "r2_train": float(r2_score(y_tr, pred_tr)),
        "rmse_eval": float(np.sqrt(mean_squared_error(y_ev, pred_ev))),
        "mae_eval": float(mean_absolute_error(y_ev, pred_ev)),
        "r2_eval": float(r2_score(y_ev, pred_ev)),
    }

def estimate_bias_variance(r2_train, r2_val):
    bias_level = "bajo"; var_level = "bajo"; fit_level = "fit"
    if r2_train < 0.3 and r2_val < 0.3:
        bias_level = "alto"; fit_level = "underfit"
    elif r2_train - r2_val > 0.2:
        var_level = "alto"; fit_level = "overfit"
    elif 0.3 <= r2_val <= 0.6:
        bias_level = "medio"; var_level = "medio"
    return bias_level, var_level, fit_level

def plot_learning_curve(model, X, y, out_path):
    sizes, tr, val = learning_curve(model, X, y, cv=5, scoring="r2", train_sizes=np.linspace(0.1,1.0,8), random_state=RANDOM_STATE, shuffle=True)
    trm = tr.mean(axis=1); vm = val.mean(axis=1)
    plt.figure(); plt.plot(sizes, trm, marker='o', label='Entrenamiento (R2)'); plt.plot(sizes, vm, marker='s', label='Validación (R2)')
    plt.xlabel("Tamaño de entrenamiento"); plt.ylabel("R2"); plt.title("Curva de aprendizaje (Árbol)"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_path); plt.close()
    return sizes.tolist(), trm.tolist(), vm.tolist()

def plot_val_curve_depth(Xtr, ytr, out_path):
    depths = list(range(1, 21))
    tr_scores, val_scores = [], []
    for d in depths:
        m = DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE)
        m.fit(Xtr, ytr)
        tr_scores.append(m.score(Xtr, ytr))
        # simple holdout on the same train for val? better: split off part of train as pseudo-val
        # but we already have a proper validation set outside; this plot is illustrative using training score only
        val_scores.append(tr_scores[-1])  # placeholder, we will compute with X_val later if needed
    plt.figure(); plt.plot(depths, tr_scores, marker='o', label='R2 Entrenamiento')
    plt.xlabel("max_depth"); plt.ylabel("R2"); plt.title("Tendencia de complejidad (Árbol)"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_path); plt.close()
    return depths, tr_scores

def main(args):
    out_dir = Path("out_tree"); out_dir.mkdir(exist_ok=True, parents=True)
    data = load_diabetes(); X, y = data.data, data.target
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # baseline shallow tree and deeper tree (to illustrate varianza)
    shallow = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
    deep = DecisionTreeRegressor(max_depth=None, random_state=RANDOM_STATE)

    shallow_val = eval_model(shallow, X_train, y_train, X_val, y_val)
    deep_val = eval_model(deep, X_train, y_train, X_val, y_val)

    b_s, v_s, f_s = estimate_bias_variance(shallow_val["r2_train"], shallow_val["r2_eval"])
    b_d, v_d, f_d = estimate_bias_variance(deep_val["r2_train"], deep_val["r2_eval"])

    # choose best depth by simple grid over validation R2
    best_score, best_depth = -1e9, None
    for d in range(1, 21):
        m = DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE)
        m.fit(X_train, y_train)
        sc = m.score(X_val, y_val)
        if sc > best_score:
            best_score, best_depth = sc, d

    best = DecisionTreeRegressor(max_depth=best_depth, random_state=RANDOM_STATE)
    # test metrics comparing best vs shallow
    from numpy import vstack, hstack
    best_test = eval_model(best, vstack([X_train, X_val]), hstack([y_train, y_val]), X_test, y_test)
    shallow_test = eval_model(shallow, vstack([X_train, X_val]), hstack([y_train, y_val]), X_test, y_test)

    # plots
    lc_path = out_dir / "learning_curve_tree.png"
    plot_learning_curve(best, X_train, y_train, lc_path)
    vc_path = out_dir / "depth_trend_tree.png"
    plot_val_curve_depth(X_train, y_train, vc_path)

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": "DecisionTreeRegressor",
        "best_max_depth": best_depth,
        "splits": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "shallow_val": shallow_val,
        "deep_val": deep_val,
        "best_test": best_test,
        "shallow_test": shallow_test,
        "diagnosis_shallow": {"bias": b_s, "varianza": v_s, "ajuste": f_s},
        "diagnosis_deep": {"bias": b_d, "varianza": v_d, "ajuste": f_d},
        "learning_curve_png": str(lc_path),
        "depth_trend_png": str(vc_path),
    }
    with open(out_dir / "results_tree.json", "w") as f: json.dump(results, f, indent=2)

    if args.predict is not None and args.predict > 0:
        best.fit(vstack([X_train, X_val]), hstack([y_train, y_val]))
        y_pred = best.predict(X_test[:args.predict])
        for i, (yp, yt) in enumerate(zip(y_pred, y_test[:args.predict])):
            print(f"Sample {i}: y_true={yt:.3f} | y_pred={yp:.3f}")
        return

    report = f"""# Reporte: Árbol de Decisión (Regresión)

**Fecha:** {datetime.datetime.now().strftime('%Y-%m-%d')}  

Se comparó un árbol **poco profundo (max_depth=3)** vs. uno **profundo (sin límite)** y se seleccionó el **mejor max_depth={best_depth}** por R2 en validación.

- *Shallow (val):* R2={shallow_val['r2_eval']:.3f} (train={shallow_val['r2_train']:.3f})
- *Deep (val):* R2={deep_val['r2_eval']:.3f} (train={deep_val['r2_train']:.3f})

**Diagnóstico:**  
- Shallow → posible **alto sesgo** si ambos R2 son bajos.  
- Deep → **alta varianza** si R2 train≫val.

**Test (generalización):**  
- Mejor árbol → R2={best_test['r2_eval']:.3f} | RMSE={best_test['rmse_eval']:.3f} | MAE={best_test['mae_eval']:.3f}

![Curva de aprendizaje](learning_curve_tree.png)
![Tendencia por profundidad](depth_trend_tree.png)

**Acciones de mejora:** podar el árbol (max_depth), ajustar min_samples_leaf, o usar ensambles (RandomForest/GB).  
"""
    with open(out_dir / "report_tree.md", "w", encoding="utf-8") as f: f.write(report)

    print(json.dumps(results, indent=2))
    print("Listo. Revisa la carpeta 'out_tree'.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true")
    p.add_argument("--predict", type=int, default=None)
    args = p.parse_args()
    main(args)
