FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir mlflow pandas numpy "scikit-learn==1.5.2" yellowbrick matplotlib joblib

COPY MLProject/ .

ENTRYPOINT ["python", "modelling.py"]
CMD ["--data_path", "loan_clean.csv", "--output_dir", "artifacts"]
