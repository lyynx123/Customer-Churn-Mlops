import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_model():
    mlflow.set_tracking_uri("file:///home/runner/work/Workflow-CI/Workflow-CI/mlruns")
    
    # Karena file .csv sudah di halaman depan, panggil langsung namanya!
    df = pd.read_csv('Customer_Churn_Dataset_preprocessing.csv')
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mlflow.set_experiment("Telco_Churn_Production")
    
    with mlflow.start_run(run_name="CI_Automated_Run"):
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Model Automated Training via CI Berhasil!")

if __name__ == "__main__":
    train_model()
