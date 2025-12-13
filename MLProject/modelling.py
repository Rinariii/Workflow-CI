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
from yellowbrick.cluster import KElbowVisualizer
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Clustering Model Training")
    parser.add_argument("--data_path", type=str, default="loan_clean.csv")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"DEBUG: Tracking URI saat ini: {mlflow.get_tracking_uri()}")
    mlflow.sklearn.autolog(disable=True)

    # Setup output
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Data
    if not os.path.exists(args.data_path):
        print(f"ERROR: File data tidak ditemukan di path: {args.data_path}")
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

    # Elbow
    print("Menghitung optimal K...")
    try:
        elbow_model = KMeans(random_state=42, n_init=10)
        elbow = KElbowVisualizer(elbow_model, k=(2, 10))
        elbow.fit(X_pca)
        elbow.finalize()
        optimal_k = elbow.elbow_value_
        print("Optimal k:", optimal_k)
    except Exception as e:
        print(f"Warning: Gagal visualisasi: {e}")
        optimal_k = 3

    # MLflow Run
    # Menggunakan active run yang dibuat oleh 'mlflow run' CLI
    with mlflow.start_run() as run:
        print(f"Sukses terhubung ke Run ID: {run.info.run_id}")
        
        model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = model.fit_predict(X_pca)
        
        silhouette = silhouette_score(X_pca, labels)
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_param("optimal_k", optimal_k)

        paths = {
            "model": os.path.join(args.output_dir, "model_clustering.pkl"),
            "scaler": os.path.join(args.output_dir, "scaler.pkl"),
            "pca": os.path.join(args.output_dir, "pca.pkl"),
            "elbow_plot": os.path.join(args.output_dir, "elbow_plot.png")
        }

        joblib.dump(model, paths["model"])
        joblib.dump(scaler, paths["scaler"])
        joblib.dump(pca, paths["pca"])
        
        try:
            if 'elbow' in locals():
                elbow.fig.savefig(paths["elbow_plot"])
                plt.close(elbow.fig)
        except:
            pass

        for name, path in paths.items():
            if os.path.exists(path):
                mlflow.log_artifact(path)

    print("Selesai.")

if __name__ == "__main__":
    main()
