import mlflow
import mlflow.sklearn
import pandas as pd
import os
import joblib
import argparse
import sys
import matplotlib.pyplot as plt
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
    mlflow.sklearn.autolog(disable=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"DEBUG: Data Path: {args.data_path}")
    print(f"DEBUG: Output Dir: {args.output_dir}")

    if not os.path.exists(args.data_path):
        print(f"ERROR: File {args.data_path} tidak ditemukan.")
        # Cek fallback path (siapa tahu dijalankan dari root)
        if os.path.exists(os.path.join("MLProject", args.data_path)):
             args.data_path = os.path.join("MLProject", args.data_path)
        else:
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

    # TUNING LOOP
    candidate_k = [2, 3, 4, 5, 6, 7, 8]
    best_score = -1
    best_model = None
    best_k = None
    
    inertia_scores = []
    silhouette_scores = []
    
    print("Mulai Hyperparameter Tuning (Lokal)...")

    # Kita cek apakah script ini berjalan di dalam Run yang sudah ada (dari CLI)
    # Jika ya, kita gunakan run tersebut sebagai Parent.
    active_run = mlflow.active_run()
    
    if active_run:
        print(f"Active Run ID (Parent): {active_run.info.run_id}")
    else:
        print("Warning: Tidak ada active run, membuat run baru.")
        mlflow.start_run(run_name="Manual_Run")

    for k in candidate_k:
        # Gunakan nested=True agar run ini menjadi 'anak' dari run utama
        with mlflow.start_run(run_name=f"Train_k={k}", nested=True):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X_pca)
            
            sil = silhouette_score(X_pca, labels)
            inertia = model.inertia_
            
            inertia_scores.append(inertia)
            silhouette_scores.append(sil)
            
            # Log metrics untuk child run
            mlflow.log_param("n_clusters", k)
            mlflow.log_metric("silhouette_score", sil)
            mlflow.log_metric("inertia", inertia)
            
            # Kita tidak perlu simpan model tiap K ke artefak utama agar tidak penuh,
            # Cukup simpan statistik di MLflow.

            if sil > best_score:
                best_score = sil
                best_model = model
                best_k = k
    
    print(f"Tuning Selesai. Best K: {best_k} dengan Score: {best_score:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
 
    ax1.plot(candidate_k, inertia_scores, marker='o', linestyle='--', color='blue')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Inertia')
    ax1.grid(True)
    
    
    ax2.plot(candidate_k, silhouette_scores, marker='o', linestyle='-', color='green')
    ax2.set_title('Silhouette Score')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Score')
    ax2.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    ax2.legend()
    ax2.grid(True)
    
    plot_path = os.path.join(args.output_dir, "evaluation_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")

    
    # Simpan Metrics Terbaik
    mlflow.log_param("best_k_final", best_k)
    mlflow.log_metric("best_silhouette_score", best_score)
    
    # Define Paths
    paths = {
        "model": os.path.join(args.output_dir, "model_clustering.pkl"),
        "scaler": os.path.join(args.output_dir, "scaler.pkl"), # Artefak 2
        "pca": os.path.join(args.output_dir, "pca.pkl"),       # Artefak 3
        "plot": plot_path                                      # Artefak 4
    }

    # Dump file lokal
    joblib.dump(best_model, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(pca, paths["pca"])
    
    # Log Artifacts ke MLflow (Local / GitHub Artifacts nantinya)
    for name, path in paths.items():
        mlflow.log_artifact(path)
        print(f"Artifact logged: {name}")

    print("Semua proses selesai.")

if __name__ == "__main__":
    main()
