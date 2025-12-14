import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering Tuning without Preprocessing")
    parser.add_argument("--data_path", type=str, required=True, help="Path ke file CSV hasil preprocess")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Folder untuk simpan model")
    return parser.parse_args()

def main():
    args = parse_args()
    BASE_DIR = os.getcwd()
    OUTPUT_DIR = os.path.join(BASE_DIR, args.output_dir)
    MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MLRUNS_DIR, exist_ok=True)


    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")
    mlflow.sklearn.autolog(disable=True)
    mlflow.set_experiment("Clustering_Tuning")

    print(f"DEBUG: Membaca data dari {args.data_path}")

    # 3. Load Data
    if not os.path.exists(args.data_path):
        print(f"ERROR: File {args.data_path} tidak ditemukan!")
        sys.exit(1)

    df = pd.read_csv(args.data_path)
    
    print(f"Data shape: {df.shape}")

    candidate_k = [2, 3, 4, 5, 6, 7, 8]
    best_score = -1
    best_model = None
    best_k = None

    print("Mulai Tuning...")
    

    with mlflow.start_run(run_name="Tuning_Session_GitHub"):
        
        for k in candidate_k:
            with mlflow.start_run(run_name=f"k={k}", nested=True):
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(df)

                sil = silhouette_score(df, labels)

                # Log parameter & metric ke MLflow
                mlflow.log_param("n_clusters", k)
                mlflow.log_metric("silhouette_score", sil)

                model_filename = f"model_k_{k}.pkl"
                model_path = os.path.join(OUTPUT_DIR, model_filename)
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path)
                
                print(f"K={k} -> Silhouette Score: {sil:.4f}")

                # Cek Best Score
                if sil > best_score:
                    best_score = sil
                    best_model = model
                    best_k = k

        print(f"Best K: {best_k} dengan Score: {best_score:.4f}")

        best_model_path = os.path.join(OUTPUT_DIR, "best_model_clustering.pkl")
        joblib.dump(best_model, best_model_path)

        mlflow.log_param("best_k_found", best_k)
        mlflow.log_metric("best_silhouette_score", best_score)
        mlflow.log_artifact(best_model_path)
        
        print(f"Model terbaik tersimpan di: {best_model_path}")

if __name__ == "__main__":
    main()
