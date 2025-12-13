import mlflow
import mlflow.sklearn
import pandas as pd
import os
import joblib
import argparse
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser(description="Clustering Hyperparameter Tuning")
    parser.add_argument("--data_path", type=str, default="loan_clean.csv", help="Path dataset")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Folder output")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"DEBUG: Data Path: {args.data_path}")
    print(f"DEBUG: Output Dir: {args.output_dir}")

    # Matikan autolog 
    mlflow.sklearn.autolog(disable=True)

    if not os.path.exists(args.data_path):
        print(f"ERROR: File {args.data_path} tidak ditemukan.")
        sys.exit(1)

    df = pd.read_csv(args.data_path)
    
    # Preprocessing
    cols_to_drop = ['Age_Binned', 'Amount_Binned']
    existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
    X = df.drop(columns=existing_drop_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Tuning Loop (Mencari K Terbaik)
    candidate_k = [2, 3, 4, 5, 6, 7, 8]
    best_score = -1
    best_model = None
    best_k = None
    
    print("Mulai Hyperparameter Tuning...")

    for k in candidate_k:
        with mlflow.start_run(run_name=f"k={k}"):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X_pca)
            
            sil = silhouette_score(X_pca, labels)
            
            # Log metrics ke MLflow
            mlflow.log_param("n_clusters", k)
            mlflow.log_metric("silhouette_score", sil)
            
            print(f"  > k={k}, Score={sil:.4f}")

            if sil > best_score:
                best_score = sil
                best_model = model
                best_k = k

    print(f"\nTraining Selesai. Best k: {best_k} dengan Score: {best_score:.4f}")
    
    with mlflow.start_run(run_name="Best_Model_Final"):
        mlflow.log_param("best_k", best_k)
        mlflow.log_metric("final_silhouette_score", best_score)
        
        paths = {
            "model": os.path.join(args.output_dir, "model_clustering.pkl"),
            "scaler": os.path.join(args.output_dir, "scaler.pkl"),
            "pca": os.path.join(args.output_dir, "pca.pkl")
        }

        # Dump file ke folder artifacts
        joblib.dump(best_model, paths["model"])
        joblib.dump(scaler, paths["scaler"])
        joblib.dump(pca, paths["pca"])
        
        # Log artifact ke MLflow Cloud/Server
        for name, path in paths.items():
            mlflow.log_artifact(path)
            print(f"Artifact tersimpan: {path}")

    print("Semua proses selesai.")

if __name__ == "__main__":
    main()
