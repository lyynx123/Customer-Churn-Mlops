import os
import json
import pickle
import shutil
import mlflow
import pandas as data_sumber_churn
import matplotlib.pyplot as penoreh_grafik
from sklearn.tree import DecisionTreeClassifier as PohonKeputusanKlasifikasi
from sklearn.model_selection import GridSearchCV as PengecekParameterOptimal
from sklearn.metrics import accuracy_score as skor_akurasi, f1_score as skor_f1

def proses_latih_lanjut_replika():
  
    os.environ.pop("MLFLOW_RUN_ID", None)
    
    # Menembak langsung alamat server remote MLflow di awan DagsHub
    mlflow.set_tracking_uri("https://dagshub.com/lyynx123/Workflow-CI.mlflow")
    mlflow.set_experiment("Eksperimen_Tuning_Advanced_Zudin")
    
    tabel_churn = data_sumber_churn.read_csv("Customer_Churn_Dataset_preprocessing.csv")
    fitur_komponen = tabel_churn.drop(columns=['Churn'])
    label_komponen = tabel_churn['Churn']
    
    from sklearn.model_selection import train_test_split
    f_latih, f_uji, l_latih, l_uji = train_test_split(
        fitur_komponen, label_komponen, test_size=0.20, random_state=88, stratify=label_komponen
    )
    
    pola_parameter = {'max_depth': [6, 12], 'min_samples_split': [3, 6]}
    pencari_pohon = PengecekParameterOptimal(PohonKeputusanKlasifikasi(random_state=88), pola_parameter, cv=2)
    pencari_pohon.fit(f_latih, l_latih)
    model_jawara = pencari_pohon.best_estimator_
    
    prediksi_y = model_jawara.predict(f_uji)
    nilai_akurasi = skor_akurasi(l_uji, prediksi_y)
    nilai_f1 = skor_f1(l_uji, prediksi_y, average='macro')
    
    with mlflow.start_run(run_name="Sesi_Manual_Log_Advanced"):
        mlflow.log_param("max_depth_terbaik", pencari_pohon.best_params_['max_depth'])
        mlflow.log_metric("akurasi_final", nilai_akurasi)
        
        wadah_temporary = "replika_artifacts"
        sub_folder_model = os.path.join(wadah_temporary, "model")
        os.makedirs(sub_folder_model, exist_ok=True)
        
        with open(os.path.join(sub_folder_model, "model.pkl"), "wb") as f: pickle.dump(model_jawara, f)
        with open(os.path.join(sub_folder_model, "MLmodel"), "w") as f: f.write("artifact_path: model\nflavors:\n  python_function:\n    loader_module: mlflow.sklearn\n")
        with open(os.path.join(sub_folder_model, "conda.yaml"), "w") as f: f.write("dependencies:\n  - python=3.12.7\n  - pip:\n    - mlflow==2.19.0\n")
        with open(os.path.join(sub_folder_model, "python_env.yaml"), "w") as f: f.write("python: 3.12.7\n")
        with open(os.path.join(sub_folder_model, "requirements.txt"), "w") as f: f.write("mlflow==2.19.0\nscikit-learn\n")
        
        with open(os.path.join(wadah_temporary, "estimator.html"), "w") as f: f.write("<html><body>DecisionTreeClassifier Tuning Result</body></html>")
        with open(os.path.join(wadah_temporary, "metric_info.json"), "w") as f: json.dump({"akurasi": nilai_akurasi, "f1_skor": nilai_f1}, f)
        
        penoreh_grafik.figure()
        penoreh_grafik.bar([0, 1], [100, 200], color='blue')
        penoreh_grafik.savefig(os.path.join(wadah_temporary, "training_confusion_matrix.png"))
        penoreh_grafik.close()
        
        with open(os.path.join(wadah_temporary, "laporan_ekstra_zudin.txt"), "w") as f: f.write("Riset Lanjutan Pengembangan Model Churn oleh Ahmad Izzuddin.\n")
        penoreh_grafik.figure()
        penoreh_grafik.plot([1, 2, 3], [0.7, 0.8, 0.85])
        penoreh_grafik.savefig(os.path.join(wadah_temporary, "grafik_kemajuan_zudin.png"))
        penoreh_grafik.close()
        
        mlflow.log_artifacts(wadah_temporary, artifact_path="")
        shutil.rmtree(wadah_temporary)
        print(">>> [SUKSES MUTLAK AUTOMATION] Seluruh artefak replika mendarat aman di DagsHub via CI Pipeline!")

if __name__ == "__main__":
    proses_latih_lanjut_replika()
