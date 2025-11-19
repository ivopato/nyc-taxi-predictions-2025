# =========================
# TF6-F: Challenger Flow (Prefect)
# =========================

# --- Imports pedidos ---
import os, mlflow
from dotenv import load_dotenv
import pathlib, time, requests
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from mlflow.models.signature import infer_signature
import pandas as pd, pickle
from mlflow import MlflowClient
from datetime import datetime
import mlflow.pyfunc

# Prefect
from typing import Tuple, Dict, Any
from prefect import flow, task

# =========================
# Constantes TF6-F
# =========================
HW_PREFIX = "TF6-F-"
HW_TAG_PROJECT = "Tarea_TF6"
HW_TAG_PURPOSE = "challenger_selection"

EXPERIMENT_NAME = "/Users/marianasgg19@gmail.com/nyc-taxi-experiment-prefect"
MODEL_NAME = "workspace.default.nyc-taxi-model-prefect"

DATA_DIR = pathlib.Path("../data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Utils simples
# =========================
def _as_dense(x, rows: int = None):
    """Convierte sparse->ndarray; permite cortar filas si rows se especifica."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    else:
        x = np.asarray(x)
    return x if rows is None else x[:rows]

# =========================
# Tareas
# =========================
@task(name="P6F-Setup-MLflow")
def setup_mlflow():
    load_dotenv(override=True)
    mlflow.set_tracking_uri("databricks")
    exp = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Experiment ID:", exp.experiment_id)

@task(name="P6F-Download-March-2025")
def download_march_2025() -> pathlib.Path:
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2025-03.parquet"
    fname = DATA_DIR / "green_tripdata_2025-03.parquet"
    if not fname.exists():
        for attempt in range(3):
            try:
                print(f"Descargando {url} (intento {attempt+1})")
                r = requests.get(url, timeout=60); r.raise_for_status()
                fname.write_bytes(r.content)
                print("Descarga completa.")
                break
            except Exception as e:
                print("Fallo:", e); time.sleep(2)
        if not fname.exists():
            raise RuntimeError("No se pudo descargar green_tripdata_2025-03.parquet")
    else:
        print("Archivo ya existe:", fname)
    return fname

@task(name="P6F-Read-Data")
def read_dataframe(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    df['duration'] = pd.to_datetime(df.lpep_dropoff_datetime) - pd.to_datetime(df.lpep_pickup_datetime)
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    for c in ['PULocationID', 'DOLocationID']:
        df[c] = df[c].astype(str)
    return df

@task(name="P6F-Vectorize")
def vectorize_train_val(df_train: pd.DataFrame, df_val: pd.DataFrame):
    dv = DictVectorizer()

    df_train = df_train.copy()
    df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
    X_train = dv.fit_transform(df_train[['PU_DO','trip_distance']].to_dict(orient='records'))
    y_train = df_train['duration'].values

    df_val = df_val.copy()
    df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']
    X_val = dv.transform(df_val[['PU_DO','trip_distance']].to_dict(orient='records'))
    y_val = df_val['duration'].values

    # Densificar una sola vez para simplificar (ambos modelos trabajan bien con denso)
    X_train = _as_dense(X_train)
    X_val   = _as_dense(X_val)

    # Log opcional de datasets
    try:
        mlflow.data.from_numpy(X_train, targets=y_train, name="green_tripdata_2025-01")
        mlflow.data.from_numpy(X_val,   targets=y_val,   name="green_tripdata_2025-02")
    except Exception:
        pass

    return X_train, X_val, y_train, y_val, dv

# -------------------------
# Entrenamiento GBR
# -------------------------
@task(name="P6F-Train-GBR")
def train_gbr(X_train, X_val, y_train, y_val):
    np.random.seed(42)

    N = X_train.shape[0]
    subset_size = min(10000, N)
    subset_idx = np.random.choice(N, size=subset_size, replace=False)
    X_train_sub = X_train[subset_idx]
    y_train_sub = y_train[subset_idx]

    def sample_gbr_params():
        return {
            "n_estimators": int(np.random.randint(60, 151)),
            "learning_rate": float(np.random.uniform(0.05, 0.30)),
            "max_depth": int(np.random.randint(2, 7)),
            "subsample": float(np.random.uniform(0.7, 1.0)),
            "min_samples_leaf": int(np.random.randint(10, 41)),
            "n_iter_no_change": 10,
            "validation_fraction": 0.1,
            "tol": 1e-3,
            "random_state": 42
        }

    best_model = None
    best_rmse = float("inf")
    best_params = None

    with mlflow.start_run(run_name=f"{HW_PREFIX}_GBR_PARENT"):
        mlflow.set_tags({
            "model_family": "GradientBoostingRegressor",
            "assignment": HW_TAG_PROJECT,
            "purpose": HW_TAG_PURPOSE,
            "parent": "true",
            "acronym": f"{HW_PREFIX}_GBR"
        })
        mlflow.log_dict({"search": "random_small", "trials": 5, "subset_size": int(subset_size)}, "search_meta.json")

        for t in range(3):
            params = sample_gbr_params()
            with mlflow.start_run(run_name=f"{HW_PREFIX}_GBR_TRIAL_{t+1}", nested=True):
                mlflow.log_params(params)
                model = GradientBoostingRegressor(**params)
                model.fit(X_train_sub, y_train_sub)
                y_pred = model.predict(X_val)
                rmse = root_mean_squared_error(y_val, y_pred)
                mlflow.log_metric("validation_rmse", rmse)

                input_example = X_val[:5]
                signature = infer_signature(input_example, y_pred[:5])
                mlflow.sklearn.log_model(sk_model=model, name="model", input_example=input_example, signature=signature)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_params = params

        mlflow.log_metric("best_validation_rmse", best_rmse)
        mlflow.log_dict({"best_params": best_params}, "best_params.json")

    return best_model, float(best_rmse), best_params

# -------------------------
# Entrenamiento RF
# -------------------------
@task(name="P6F-Train-RF")
def train_rf(X_train, X_val, y_train, y_val):
    np.random.seed(42)

    N = X_train.shape[0]
    subset_size = min(20000, N)
    subset_idx = np.random.choice(N, size=subset_size, replace=False)
    X_train_sub = X_train[subset_idx]
    y_train_sub = y_train[subset_idx]

    def sample_rf_params():
        return {
            "n_estimators": int(np.random.randint(50, 121)),
            "max_depth": int(np.random.randint(8, 17)),
            "min_samples_split": int(np.random.randint(2, 6)),
            "min_samples_leaf": int(np.random.randint(5, 21)),
            "max_features": "sqrt",
            "bootstrap": False,
            "random_state": 42,
            "n_jobs": -1
        }

    best_model = None
    best_rmse = float("inf")
    best_params = None

    with mlflow.start_run(run_name=f"{HW_PREFIX}_RF_PARENT"):
        mlflow.set_tags({
            "model_family": "RandomForestRegressor",
            "assignment": HW_TAG_PROJECT,
            "purpose": HW_TAG_PURPOSE,
            "parent": "true",
            "acronym": f"{HW_PREFIX}_RF"
        })
        mlflow.log_dict({"search": "random_small", "trials": 3, "subset_size": int(subset_size)}, "search_meta.json")

        for t in range(2):
            params = sample_rf_params()
            with mlflow.start_run(run_name=f"{HW_PREFIX}_RF_TRIAL_{t+1}", nested=True):
                mlflow.log_params(params)
                model = RandomForestRegressor(**params)
                model.fit(X_train_sub, y_train_sub)
                y_pred = model.predict(X_val)
                rmse = root_mean_squared_error(y_val, y_pred)
                mlflow.log_metric("validation_rmse", rmse)

                input_example = X_val[:5]
                signature = infer_signature(input_example, y_pred[:5])
                mlflow.sklearn.log_model(sk_model=model, name="model", input_example=input_example, signature=signature)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_params = params

        mlflow.log_metric("best_validation_rmse", best_rmse)
        mlflow.log_dict({"best_params": best_params}, "best_params.json")

    return best_model, float(best_rmse), best_params

# =========================
# Registro del CHALLENGER
# =========================
@task(name="P6F-Register-Challenger")
def register_challenger(best_family: str, best_rmse: float, best_model, X_sample, dv: DictVectorizer):
    pathlib.Path("preprocessor").mkdir(exist_ok=True)
    with open("preprocessor/dv.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)

    input_example = _as_dense(X_sample, rows=5)
    signature = infer_signature(input_example, best_model.predict(input_example))

    run_name = f"{HW_PREFIX}_CHALLENGER_REG_{best_family}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "assignment": HW_TAG_PROJECT,
            "purpose": HW_TAG_PURPOSE,
            "role": "challenger_candidate",
            "model_family": best_family,
            "acronym": f"{HW_PREFIX}{'GBR' if best_family=='GradientBoostingRegressor' else 'RF'}_CHAL",
        })
        mlflow.log_metric("validation_rmse", best_rmse)
        mlflow.log_artifact("preprocessor/dv.pkl", artifact_path="preprocessor")
        mlflow.sklearn.log_model(best_model, name="model", input_example=input_example, signature=signature)
        run_id = mlflow.active_run().info.run_id

    result = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(name=MODEL_NAME, alias="challenger", version=result.version)
    client.update_model_version(
        name=MODEL_NAME,
        version=result.version,
        description=f"[{HW_PREFIX}CHAL] {best_family} {datetime.today().isoformat(timespec='seconds')} | RMSE(val)={best_rmse:.4f}"
    )
    print(f"Challenger registrado como versión {result.version}")
    return int(result.version)

# =========================
# Evaluación Champion vs Challenger (marzo 2025)
# =========================
@task(name="P6F-Prepare-March")
def prepare_march_features(march_path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    df_mar = read_dataframe.fn(str(march_path))
    with open("preprocessor/dv.pkl", "rb") as f_in:
        dv_loaded: DictVectorizer = pickle.load(f_in)

    dfc = df_mar.copy()
    dfc['PU_DO'] = dfc['PULocationID'] + '_' + dfc['DOLocationID']
    X_mar = dv_loaded.transform(dfc[['PU_DO','trip_distance']].to_dict(orient='records'))
    y_mar = df_mar['duration'].values
    X_mar = _as_dense(X_mar)
    return X_mar, y_mar

def _load_by_alias_both(alias: str):
    for cand in [alias, alias.capitalize()]:
        try:
            return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{cand}")
        except Exception:
            continue
    raise RuntimeError(f"No se pudo cargar alias {alias}/{alias.capitalize()} en {MODEL_NAME}")

@task(name="P6F-Compare-Champ-Chal")
def compare_champion_challenger(X_mar: np.ndarray, y_mar: np.ndarray) -> Dict[str, float]:
    champion = _load_by_alias_both("champion")
    challenger = _load_by_alias_both("challenger")
    rmse_champ = root_mean_squared_error(y_mar, champion.predict(X_mar))
    rmse_chal  = root_mean_squared_error(y_mar, challenger.predict(X_mar))
    print(f"RMSE marzo 2025 -> champion: {rmse_champ:.4f} | challenger: {rmse_chal:.4f}")
    return {"rmse_champion": float(rmse_champ), "rmse_challenger": float(rmse_chal)}

# =========================
# Decisión y Promoción
# =========================
@task(name="P6F-Promotion-Decision")
def promotion_decision(metrics: Dict[str, float], improvement_threshold: float = 0.005) -> Dict[str, Any]:
    rmse_champion   = metrics["rmse_champion"]
    rmse_challenger = metrics["rmse_challenger"]
    improvement = (rmse_champion - rmse_challenger) / rmse_champion
    decision = "PROMOVER" if improvement >= improvement_threshold else "NO PROMOVER"
    print(f"Mejora relativa: {improvement*100:.2f}% | Decisión: {decision}")
    return {"decision": decision, "improvement": float(improvement)}

@task(name="P6F-Promote-If-Approved")
def promote_if_approved(decision_payload: Dict[str, Any]) -> Dict[str, Any]:
    client = MlflowClient()
    if decision_payload["decision"] == "PROMOVER":
        mv = client.get_model_version_by_alias(name=MODEL_NAME, alias="challenger")
        client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=mv.version)
        client.update_model_version(
            name=MODEL_NAME,
            version=mv.version,
            description=f"[{HW_PREFIX}PROMOTED]->champion | Mejora {decision_payload['improvement']*100:.2f}% en RMSE marzo"
        )
        print(f"Promovido a champion: versión {mv.version}")
        return {"promoted": True, "new_champion_version": int(mv.version)}
    print("Se mantiene el champion actual.")
    return {"promoted": False}

# =========================
# FLOW PRINCIPAL 
# =========================
@flow(name="TF6F-Challenger-Flow-Simple")
def TF6f_challenger_flow():
    # 1) Setup
    setup_mlflow()

    # 2) Train/val (enero/febrero)
    df_train = read_dataframe(str(DATA_DIR / "green_tripdata_2025-01.parquet"))
    df_val   = read_dataframe(str(DATA_DIR / "green_tripdata_2025-02.parquet"))
    X_train, X_val, y_train, y_val, dv = vectorize_train_val(df_train, df_val)

    # 3) Entrenar dos familias
    gbr_model, gbr_rmse, _ = train_gbr(X_train, X_val, y_train, y_val)
    rf_model,  rf_rmse,  _ = train_rf (X_train, X_val, y_train, y_val)

    # 4) Elegir CHALLENGER y registrar
    if gbr_rmse <= rf_rmse:
        best_family, best_rmse, best_model = "GradientBoostingRegressor", gbr_rmse, gbr_model
    else:
        best_family, best_rmse, best_model = "RandomForestRegressor", rf_rmse, rf_model

    print(f"Mejor familia: {best_family} | RMSE(val)={best_rmse:.4f}")
    challenger_version = register_challenger(best_family, best_rmse, best_model, X_val, dv)

    # 5) Marzo-2025 para comparación
    march_path = download_march_2025()
    X_mar, y_mar = prepare_march_features(march_path)

    # 6) Comparar, decidir y (si aplica) promover
    comp = compare_champion_challenger(X_mar, y_mar)
    decision = promotion_decision(comp, improvement_threshold=0.005)
    result = promote_if_approved(decision)

    return {
        "best_family": best_family,
        "best_rmse_val": best_rmse,
        "challenger_version": challenger_version,
        "rmse_march_champion": comp["rmse_champion"],
        "rmse_march_challenger": comp["rmse_challenger"],
        "improvement": decision["improvement"],
        "promoted": result.get("promoted", False),
    }

# Ejecución local
if __name__ == "__main__":
    TF6f_challenger_flow()