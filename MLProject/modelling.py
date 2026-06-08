import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import mlflow.sklearn
import pandas as pembaca_tabel_churn
from sklearn.ensemble import RandomForestClassifier as AlgoritmaHutanKeputusan
from sklearn.model_selection import train_test_split as pembagi_data_acak

def proses_latih_murni_lokal():
    
    os.environ.pop("MLFLOW_RUN_ID", None)
    
    # Setel langsung ke cloud DagsHub agar server GitHub Actions bisa mengirim data
    mlflow.set_tracking_uri("https://dagshub.com/lyynx123/Workflow-CI.mlflow")
    mlflow.set_experiment("Eksperimen_Dasar_Zudin")
    
    mlflow.sklearn.autolog()
    
    dokumen_sumber = "Customer_Churn_Dataset_preprocessing.csv"
    matriks_pelanggan = pembaca_tabel_churn.read_csv(dokumen_sumber)
    
    fitur_x = matriks_pelanggan.drop(columns=['Churn'])
    target_y = matriks_pelanggan['Churn']
    
    x_latih, x_uji, y_latih, y_uji = pembagi_data_acak(
        fitur_x, target_y, test_size=0.20, random_state=42, stratify=target_y
    )
    
    with mlflow.start_run(run_name="Sesi_Autolog_Basic"):
        arsitektur_hutan = AlgoritmaHutanKeputusan(n_estimators=100, max_depth=8, random_state=42)
        arsitektur_hutan.fit(x_latih, y_latih)
        print("[SUKSES] Model basic berhasil direkam secara cloud!")

if __name__ == "__main__":
    proses_latih_murni_lokal()
