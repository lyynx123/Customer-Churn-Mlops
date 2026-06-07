import os
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model():
    # --- SISTEM OTORISASI ADAPTIF (ANTI-ERROR GITHUB ACTIONS) ---
    # Jika berjalan di GitHub Actions, gunakan Environment Variables bawaan MLflow
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("Berjalan di GitHub Actions. Menggunakan token otomatis...")
        # MLflow akan otomatis membaca MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, dan MLFLOW_TRACKING_PASSWORD dari ci.yml
    else:
        print("Berjalan di Lokal. Menginisialisasi DagsHub secara interaktif...")
        dagshub.init(repo_owner='lyynx123', repo_name='Eksperimen_SML_Ahmad', mlflow=True)
    
    # Load Data bersih
    df = pd.read_csv('Customer_Churn_Dataset_preprocessing.csv')
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    mlflow.set_experiment("Telco_Churn_RandomForest")
    
    # Mengaktifkan Autolog sesuai Kriteria 2 Basic
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RF_Production_Autolog_Run"):
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        print("Model training selesai dengan Autolog.")

if __name__ == "__main__":
    train_model()
