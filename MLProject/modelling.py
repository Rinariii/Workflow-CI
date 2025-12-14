import os
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Asumsi tuning.py ada di root atau subfolder

OUTPUT_DIR = "tuning_artifacts"
MLRUNS_DIR = "mlruns"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

DATA_PATH = os.path.join("preprocessing", "loan_clean.csv")

def load_data():
    if not os.path.exists(DATA_PATH):
        if os.path.exists("loan_clean.csv"):
            return pd.read_csv("loan_clean.csv")
        
        print(f"CWD: {os.getcwd()}")
        raise FileNotFoundError(f"File data tidak ditemukan di: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    print(f"Data Loaded: {df.shape}")
    return df

def main():
    mlflow.set_tracking_uri(f"file:///{os.path.abspath(MLRUNS_DIR)}")
    mlflow.sklearn.autolog(disable=True) 

    experiment_name = "Clustering_Tuning_Automated"
    mlflow.set_experiment(experiment_name)

    # Load Data
    df = load_data()

    # Config Tuning
    candidate_k = [2, 3, 4, 5, 6, 7, 8]
    best_score = -1
    best_model = None
    best_k = None

    print("Mulai Tuning...")

    for k in candidate_k:
        with mlflow.start_run(run_name=f"k={k}"):
            
            # Train Model
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(df)

            # Hitung Score
            sil = silhouette_score(df, labels)

            # Log ke MLflow
            mlflow.log_param("n_clusters", k)
            mlflow.log_metric("silhouette_score", sil)

            # Simpan Model per K
            model_filename = f"model_k_{k}.pkl"
            model_path = os.path.join(OUTPUT_DIR, model_filename)
            joblib.dump(model, model_path)
            
            mlflow.log_artifact(model_path)
            
            print(f"K={k} -> Silhouette Score: {sil:.4f}")

            # Update Best Model
            if sil > best_score:
                best_score = sil
                best_model = model
                best_k = k

    print(f"\n--- Tuning Selesai ---")
    print(f"Best K: {best_k} dengan Score: {best_score:.4f}")

    # Simpan Best Model
    best_model_path = os.path.join(OUTPUT_DIR, "best_model_clustering.pkl")
    joblib.dump(best_model, best_model_path)

    # Log Summary Run
    with mlflow.start_run(run_name="Best_Model_Summary"):
        mlflow.log_param("best_k", best_k)
        mlflow.log_metric("best_silhouette_score", best_score)
        mlflow.log_artifact(best_model_path)

    print(f"Semua artefak tersimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
