import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model():
    # Inisialisasi DagsHub
    dagshub.init(repo_owner='lyynx123', repo_name='Eksperimen_SML_Ahmad', mlflow=True)
    
    # Load Data bersih
    df = pd.read_csv('Customer_Churn_Dataset_preprocessing.csv')
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Set nama eksperimen
    mlflow.set_experiment("Telco_Churn_RandomForest")
    
    # --- PERBAIKAN WAJIB: Mengaktifkan Autolog Sebelum Training ---
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RF_Production_Autolog_Run"):
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Cetak metrik dasar untuk memastikan model berjalan baik
        print("Model training selesai dengan Autolog.")

if __name__ == "__main__":
    train_model()
