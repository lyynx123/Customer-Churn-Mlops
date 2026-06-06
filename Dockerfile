# Menggunakan pangkalan Python 3.12 yang resmi dan aman dari masalah pip lama
FROM python:3.12-slim

WORKDIR /app

# Install dependencies yang dibutuhkan model kamu
RUN pip install mlflow==2.19.0 pandas numpy scikit-learn

# Menyalin folder hasil run MLflow ke dalam Docker
COPY mlruns /app/mlruns

# Perintah default saat container dijalankan (opsional, untuk serving model)
CMD ["mlflow", "models", "serve", "-m", "mlruns/0/", "-h", "0.0.0.0", "-p", "8080"]
