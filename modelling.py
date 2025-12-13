import mlflow
import mlflow.sklearn
import pandas as pd
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from matplotlib import pyplot as plt

PROJECT_DIR = r"C:\Users\steve\Downloads\Eksperimen_SML_Steven Lie Wibowo\Membangun_Model"
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
MLRUNS_DIR = os.path.join(PROJECT_DIR, "mlruns")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")
mlflow.sklearn.autolog(disable=True)

experiment_name = "Clustering_Experiment"
mlflow.set_experiment(experiment_name)

df = pd.read_csv(
    r"C:\Users\steve\Downloads\Eksperimen_SML_Steven Lie Wibowo\preprocessing\loan_clean.csv"
)
X = df.drop(columns=['Age_Binned', 'Amount_Binned'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

elbow_model = KMeans(random_state=42, n_init=10)
elbow = KElbowVisualizer(elbow_model, k=(2, 10))
elbow.fit(X_pca)
elbow.finalize()
optimal_k = elbow.elbow_value_
print("Optimal k:", optimal_k)

with mlflow.start_run(run_name="kmeans-manual-logging"):
    model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = model.fit_predict(X_pca)

    silhouette = silhouette_score(X_pca, labels)
    mlflow.log_metric("silhouette_score", silhouette)

    paths = {
        "model": os.path.abspath(os.path.join(ARTIFACTS_DIR, "model_clustering.pkl")),
        "scaler": os.path.abspath(os.path.join(ARTIFACTS_DIR, "scaler.pkl")),
        "pca": os.path.abspath(os.path.join(ARTIFACTS_DIR, "pca.pkl")),
        "elbow_plot": os.path.abspath(os.path.join(ARTIFACTS_DIR, "elbow_plot.png"))
    }

    joblib.dump(model, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(pca, paths["pca"])
    elbow.fig.savefig(paths["elbow_plot"])
    plt.close(elbow.fig)

    for name, path in paths.items():
        try:
            mlflow.log_artifact(path)
        except Exception as e:
            print(f"Gagal log artefak {name}: {e}")

    print("Model saved at:", paths["model"])
    print("Silhouette Score:", silhouette)

print("Semua artefak tersimpan di:", ARTIFACTS_DIR)
print("MLflow tracking di:", MLRUNS_DIR)
