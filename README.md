# 🍷 Wine Quality ML Pipeline

Pipeline reproducible de machine learning para predecir la calidad del vino tinto, automatizado con GitHub Actions y MLflow.

## 📁 Estructura del proyecto

```
wine-quality-ml/
├── src/
│   ├── train.py          ← Pipeline completo: carga, preprocesamiento, entrenamiento, MLflow
│   └── test_model.py     ← Pruebas básicas de validación
├── data/
│   └── winequality-red.csv  ← Dataset Wine Quality (UCI)
├── mlruns/               ← Almacenamiento local de MLflow (se genera automáticamente)
├── .github/
│   └── workflows/
│       └── ml.yml        ← Workflow de GitHub Actions
├── config.yaml           ← Hiperparámetros y rutas
├── Makefile              ← Comandos automatizados
├── requirements.txt      ← Dependencias
└── README.md
```

## 🚀 Cómo ejecutar localmente

### 1. Instalar dependencias
```bash
make install
```

### 2. Ejecutar pruebas
```bash
make test
```

### 3. Entrenar el modelo
```bash
make train
```

## 📊 Dataset

- **Fuente:** [UCI Machine Learning Repository - Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Archivo:** `winequality-red.csv`
- **Target:** `quality` (valor numérico del 0 al 10)
- **Features:** 11 características químicas del vino

## 🤖 Modelo

- **Algoritmo:** RandomForestRegressor (scikit-learn)
- **Métricas evaluadas:** MSE, RMSE, MAE, R²
- **Tracking:** MLflow (local en `mlruns/`)

## ⚙️ CI/CD

El pipeline se ejecuta automáticamente con cada push a `main` mediante GitHub Actions:

1. Instala dependencias (`make install`)
2. Ejecuta pruebas (`make test`)
3. Entrena el modelo (`make train`)
4. Sube el modelo como artefacto del workflow

## 📦 Entregables

- URL del repositorio público en GitHub
- Evidencia del modelo registrado en MLflow (captura de pantalla)
- Archivo `.zip` del proyecto
