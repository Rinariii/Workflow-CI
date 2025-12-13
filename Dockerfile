FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install mlflow pandas numpy "scikit-learn==1.5.2" yellowbrick matplotlib joblib dagshub

# Copy seluruh folder repository ke dalam /app
COPY . /app

# Update Entrypoint ke lokasi baru script
# Script sekarang ada di dalam folder MLProject
ENTRYPOINT ["python", "MLProject/modelling.py"]

# Update default arguments agar path sesuai struktur di dalam container
CMD ["--data_path", "MLProject/loan_clean.csv", "--output_dir", "MLProject/artifacts"]
