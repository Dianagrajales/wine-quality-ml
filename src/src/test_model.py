"""
Pruebas básicas de validación del pipeline.
Se ejecutan con: make test
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "winequality-red.csv")

RMSE_THRESHOLD = 1.0   # El RMSE debe ser menor a 1.0
R2_THRESHOLD   = 0.3   # El R2 debe ser mayor a 0.3

errors = []

print("=== Ejecutando pruebas de validación ===\n")

# ── Test 1: El archivo de datos existe ────────────────────────────────────────
print("Test 1: Verificando existencia del dataset...")
if not os.path.exists(DATA_PATH):
    errors.append(f"FAIL: No se encontró el dataset en {DATA_PATH}")
    print(f"  ❌ FAIL")
else:
    print(f"  ✅ PASS")

# ── Test 2: El dataset se puede cargar y tiene las columnas esperadas ─────────
print("Test 2: Verificando estructura del dataset...")
try:
    df = pd.read_csv(DATA_PATH, sep=";")
    expected_cols = ["fixed acidity", "volatile acidity", "citric acid",
                     "residual sugar", "chlorides", "free sulfur dioxide",
                     "total sulfur dioxide", "density", "pH", "sulphates",
                     "alcohol", "quality"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        errors.append(f"FAIL: Columnas faltantes: {missing}")
        print(f"  ❌ FAIL - columnas faltantes: {missing}")
    else:
        print(f"  ✅ PASS - {df.shape[0]} filas, {df.shape[1]} columnas")
except Exception as e:
    errors.append(f"FAIL: Error cargando dataset: {e}")
    print(f"  ❌ FAIL")

# ── Test 3: El modelo entrena sin errores ─────────────────────────────────────
print("Test 3: Verificando entrenamiento del modelo...")
try:
    df = pd.read_csv(DATA_PATH, sep=";").dropna()
    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train_s, y_train)
    print(f"  ✅ PASS")
except Exception as e:
    errors.append(f"FAIL: Error en entrenamiento: {e}")
    print(f"  ❌ FAIL: {e}")

# ── Test 4: Las métricas cumplen umbrales mínimos ────────────────────────────
print("Test 4: Verificando métricas del modelo...")
try:
    preds = model.predict(X_test_s)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    print(f"         RMSE={rmse:.4f} (umbral < {RMSE_THRESHOLD}) | R2={r2:.4f} (umbral > {R2_THRESHOLD})")
    if rmse >= RMSE_THRESHOLD:
        errors.append(f"FAIL: RMSE={rmse:.4f} supera el umbral de {RMSE_THRESHOLD}")
        print(f"  ❌ FAIL - RMSE fuera de umbral")
    elif r2 <= R2_THRESHOLD:
        errors.append(f"FAIL: R2={r2:.4f} por debajo del umbral de {R2_THRESHOLD}")
        print(f"  ❌ FAIL - R2 por debajo del umbral")
    else:
        print(f"  ✅ PASS")
except Exception as e:
    errors.append(f"FAIL: Error evaluando métricas: {e}")
    print(f"  ❌ FAIL: {e}")

# ── Resultado final ───────────────────────────────────────────────────────────
print("\n=== Resultado ===")
if errors:
    print(f"❌ {len(errors)} prueba(s) fallaron:")
    for err in errors:
        print(f"   - {err}")
    sys.exit(1)
else:
    print("✅ Todas las pruebas pasaron correctamente.")
    sys.exit(0)
