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
    parser.add_argument("--data_path", type=str, default="loan_clean.csv", help="Lokasi file data CSV")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Folder untuk menyimpan output")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"DEBUG: Menggunakan data dari: {args.data_path}")
    print(f"DEBUG: Output akan disimpan di: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    mlruns_dir = os.path.abspath("mlruns")
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
    
    mlflow.sklearn.autolog(disable=True)
    experiment_name = "Clustering_Experiment"
    mlflow.set_experiment(experiment_name)

    if not os.path.exists(args.data_path):
        print(f"ERROR: File data tidak ditemukan di path: {args.data_path}")
        # Coba cek direktori saat ini untuk debugging
        print(f"Isi direktori saat ini ({os.getcwd()}): {os.listdir('.')}")
        sys.exit(1)

    df = pd.read_csv(args.data_path)
    
    cols_to_drop = ['Age_Binned', 'Amount_Binned']
    existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
    X = df.drop(columns=existing_drop_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print("Menghitung optimal K...")
    elbow_model = KMeans(random_state=42, n_init=10)
    elbow = KElbowVisualizer(elbow_model, k=(2, 10))
    elbow.fit(X_pca)
    elbow.finalize()
    optimal_k = elbow.elbow_value_
    print("Optimal k:", optimal_k)

    with mlflow.start_run(run_name="kmeans-manual-logging"):
        # Training dengan optimal K
        model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = model.fit_predict(X_pca)

        silhouette = silhouette_score(X_pca, labels)
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_param("optimal_k", optimal_k)

        # Definisi path penyimpanan artefak (menggunakan args.output_dir)
        paths = {
            "model": os.path.join(args.output_dir, "model_clustering.pkl"),
            "scaler": os.path.join(args.output_dir, "scaler.pkl"),
            "pca": os.path.join(args.output_dir, "pca.pkl"),
            "elbow_plot": os.path.join(args.output_dir, "elbow_plot.png")
        }

        # Simpan file lokal
        joblib.dump(model, paths["model"])
        joblib.dump(scaler, paths["scaler"])
        joblib.dump(pca, paths["pca"])
        
        # Simpan plot
        try:
            elbow.fig.savefig(paths["elbow_plot"])
            plt.close(elbow.fig)
        except Exception as e:
            print(f"Warning: Gagal menyimpan plot: {e}")

        # Log ke MLflow
        for name, path in paths.items():
            try:
                mlflow.log_artifact(path)
                print(f"Berhasil log artifact: {name}")
            except Exception as e:
                print(f"Gagal log artefak {name}: {e}")

        print("Model saved at:", paths["model"])
        print("Silhouette Score:", silhouette)

    print("Selesai. Artefak tersimpan di:", args.output_dir)

if __name__ == "__main__":
    main()
