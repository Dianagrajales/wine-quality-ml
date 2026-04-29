import os
import sys
import traceback
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature

print("=== Iniciando pipeline de entrenamiento ===")

# ─── Rutas ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data", "winequality-red.csv")
MLRUNS_DIR   = os.path.join(BASE_DIR, "mlruns")
TRACKING_URI = "file://" + os.path.abspath(MLRUNS_DIR)

os.makedirs(MLRUNS_DIR, exist_ok=True)

# ─── Hiperparámetros (config) ─────────────────────────────────────────────────
N_ESTIMATORS  = int(os.environ.get("N_ESTIMATORS", 100))
MAX_DEPTH     = int(os.environ.get("MAX_DEPTH", 10))
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
EXPERIMENT    = "wine-quality-regression"

# ─── 1. Carga de datos ────────────────────────────────────────────────────────
print(f"[1/5] Cargando dataset desde: {DATA_PATH}")
try:
    df = pd.read_csv(DATA_PATH, sep=";")
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo {DATA_PATH}")
    sys.exit(1)

print(f"      Shape: {df.shape}")

# ─── 2. Preprocesamiento ──────────────────────────────────────────────────────
print("[2/5] Preprocesando datos...")

# Manejo de nulos
nulos = df.isnull().sum().sum()
print(f"      Valores nulos encontrados: {nulos}")
df = df.dropna()

# Separar features y target
X = df.drop("quality", axis=1)
y = df["quality"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Escalamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"      Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}")

# ─── 3. Entrenamiento ─────────────────────────────────────────────────────────
print("[3/5] Entrenando modelo RandomForestRegressor...")
model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    random_state=RANDOM_STATE
)
model.fit(X_train_scaled, y_train)

# ─── 4. Evaluación ────────────────────────────────────────────────────────────
print("[4/5] Evaluando modelo...")
preds = model.predict(X_test_scaled)
mse   = mean_squared_error(y_test, preds)
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)
rmse  = np.sqrt(mse)

print(f"      MSE:  {mse:.4f}")
print(f"      RMSE: {rmse:.4f}")
print(f"      MAE:  {mae:.4f}")
print(f"      R2:   {r2:.4f}")

# ─── 5. MLflow tracking ───────────────────────────────────────────────────────
print("[5/5] Registrando en MLflow...")
mlflow.set_tracking_uri(TRACKING_URI)

# Crear o recuperar experimento
try:
    experiment_id = mlflow.create_experiment(
        name=EXPERIMENT,
        artifact_location=TRACKING_URI
    )
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        experiment_id = mlflow.get_experiment_by_name(EXPERIMENT).experiment_id
    else:
        raise e

try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Parámetros
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("dataset", "winequality-red")

        # Métricas
        mlflow.log_metric("mse",  mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("r2",   r2)

        # Firma y ejemplo de entrada
        signature    = infer_signature(X_train_scaled, preds)
        input_example = X_test_scaled[:3]

        # Modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        print(f"\n✅ Modelo registrado exitosamente.")
        print(f"   Run ID:      {run.info.run_id}")
        print(f"   Experimento: {EXPERIMENT}")
        print(f"   R2:  {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

except Exception as e:
    print("\nERROR durante MLflow:")
    traceback.print_exc()
    sys.exit(1)
