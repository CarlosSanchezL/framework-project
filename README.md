📊 Machine Learning Framework Project
Este proyecto implementa un flujo completo de aprendizaje automático usando scikit-learn. Se centra en resolver un problema de regresión con el dataset de diabetes incluido en la librería, aplicando:

    -Regresión Lineal como baseline
    -Regresión Ridge con regularización para mejorar el desempeño
    -Árboles de decisión como modelo adicional de comparación

Además, el proyecto genera curvas de aprendizaje, curvas de validación, reportes automáticos y predicciones de ejemplo, todo ejecutado desde scripts en Python (sin notebooks).

🎯 Objetivos del proyecto

-Aplicar un framework de machine learning en un caso real
-Implementar algoritmos de regresión lineal y regularizada (Ridge)
-Explorar un modelo basado en árboles de decisión
-Dividir los datos en train / validation / test para evaluar de manera objetiva
-Generar métricas, gráficas y un reporte automático con diagnóstico de:

    -Sesgo (bias)
    -Varianza
    -Nivel de ajuste (underfit / overfit / fit)

📂 Estructura del proyecto
ml-framework-project/
│
├── main.py                      # Script principal: LinearRegression + Ridge
├── main_tree.py                 # Script adicional: Árbol de Decisión
├── requirements.txt             # Dependencias del proyecto
├── README.md                    # Este archivo
│
├── out/                         # Resultados Lineal + Ridge
│   ├── learning_curve_linear.png
│   ├── validation_curve_ridge.png
│   ├── report.md
│   └── results.json
│
└── out_tree/                    # Resultados Árbol de Decisión
    ├── learning_curve_tree.png
    ├── depth_trend_tree.png
    ├── report_tree.md
    └── results_tree.json

⚙️ Instalación y configuración
1. Crear entorno virtual
En macOS / Linux:
bashpython3 -m venv .venv
source .venv/bin/activate

En Windows (PowerShell):
powershellpython -m venv .venv
.venv\Scripts\activate

2. Instalar dependencias
bashpip install --upgrade pip setuptools wheel
pip install -r requirements.txt

▶️ Ejecución
Generar modelos, gráficas y reportes
bashpython main.py --plot
Esto entrenará los modelos Lineal y Ridge, y guardará las figuras y reportes en la carpeta out/.

Hacer predicciones con el conjunto de prueba
bashpython main.py --predict 5
Esto imprimirá 5 predicciones de ejemplo comparando y_true (valor real) y y_pred (predicho).

Árbol de Decisión (opcional)
bashpython main_tree.py --plot
Esto entrenará y evaluará un árbol de decisión, generando resultados en out_tree/.

📈 Resultados esperados

-Curvas de aprendizaje → muestran el desempeño del modelo en función del tamaño de entrenamiento
-Curvas de validación → exploran el impacto de la regularización (Ridge)
-Reportes automáticos (.md y .json) → resumen métricas clave:

    -RMSE (Root Mean Squared Error)
    -MAE (Mean Absolute Error)
    -R² (Coeficiente de determinación)
    -Diagnóstico de sesgo, varianza y ajuste



Ejemplo de salida al predecir:
Sample 0: y_true=97.000  | y_pred=151.386
Sample 1: y_true=96.000  | y_pred=59.120
Sample 2: y_true=273.000 | y_pred=247.484

📌 Notas

El proyecto está diseñado para ser reproducible y correr únicamente con Python estándar (sin notebooks)
Los resultados pueden variar levemente debido a la aleatoriedad del train_test_split
Compatible con macOS (M1/M2/M3), Linux y Windows


👨‍💻 Autor
Carlos Sánchez Llanes
Proyecto académico de aprendizaje automático – 2025    