# Customer Churn Prediction — End-to-End Machine Learning & MLOps Pipeline
# Customer Churn Prediction — Machine Learning & MLOps Pipeline

<p align="center">
  <b>End-to-End Machine Learning Workflow with MLflow, DagsHub, GitHub Actions CI/CD, Hyperparameter Tuning, and Docker</b>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.19.0-0194E2?logo=mlflow\&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn\&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=github-actions\&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker\&logoColor=white)

</p>

---

## 📌 Overview

**Workflow-CI** adalah proyek Machine Learning yang membangun pipeline **Customer Churn Prediction** sekaligus menerapkan konsep **MLOps** untuk mengotomatisasi proses eksperimen, training, hyperparameter tuning, experiment tracking, artifact management, dan continuous integration.

Proyek ini menggunakan **MLflow Project** sebagai struktur eksekusi machine learning, **DagsHub** sebagai remote MLflow tracking server, **GitHub Actions** untuk menjalankan pipeline secara otomatis, serta **Docker** sebagai dasar containerization dan model serving.

Pipeline utama:

```text
Customer Churn Dataset
        │
        ▼
Data Validation
        │
        ▼
MLflow Project
        │
        ├───────────────┐
        ▼               ▼
Baseline Model     Hyperparameter Tuning
        │               │
        └───────┬───────┘
                ▼
        Experiment Tracking
             MLflow
                │
                ▼
             DagsHub
                │
                ▼
          Model Artifacts
                │
                ▼
             Docker
                │
                ▼
       MLflow Model Serving
```

Repository ini dibuat untuk menunjukkan bagaimana model machine learning dapat dikembangkan menjadi workflow yang lebih terstruktur, reproducible, dan otomatis.

---

# 🎯 Project Objectives

Tujuan utama proyek:

* Membangun model klasifikasi untuk memprediksi customer churn.
* Membuat workflow machine learning menggunakan MLflow Project.
* Membandingkan baseline model dengan model hasil hyperparameter tuning.
* Melakukan experiment tracking menggunakan MLflow.
* Menyimpan parameter, metric, model, dan artifact eksperimen.
* Menggunakan DagsHub sebagai remote MLflow tracking server.
* Mengotomatisasi proses training melalui GitHub Actions.
* Melakukan validasi keberadaan dataset sebelum proses training.
* Menyiapkan environment menggunakan Conda/Python.
* Menyiapkan Docker image untuk model serving menggunakan MLflow.

---

# 🧠 Machine Learning Problem

Project ini menggunakan dataset **Customer Churn** dengan target:

```text
Churn
```

Target tersebut digunakan untuk membangun model klasifikasi yang dapat memprediksi apakah seorang pelanggan termasuk kategori churn atau tidak.

Dataset preprocessing disimpan pada:

```text
MLProject/Customer_Churn_Dataset_preprocessing.csv
```

Feature dan target dipisahkan dengan:

```python
X = data.drop(columns=["Churn"])
y = data["Churn"]
```

---

# 🤖 Machine Learning Models

Project memiliki dua workflow utama.

## 1. Baseline Model — Random Forest

Baseline menggunakan:

```text
RandomForestClassifier
```

dengan konfigurasi:

```text
n_estimators = 100
max_depth    = 8
random_state = 42
```

Dataset dibagi menjadi:

```text
Training Data : 80%
Testing Data  : 20%
Random State  : 42
Stratified    : Yes
```

Model kemudian dicatat menggunakan MLflow.

MLflow autologging digunakan untuk membantu mencatat informasi eksperimen secara otomatis.

---

## 2. Advanced Model — Decision Tree + Hyperparameter Tuning

Workflow kedua menggunakan:

```text
DecisionTreeClassifier
```

dengan `GridSearchCV`.

Parameter yang diuji:

```python
{
    "max_depth": [6, 12],
    "min_samples_split": [3, 6]
}
```

Cross-validation:

```text
CV = 2
```

Model terbaik kemudian dievaluasi menggunakan:

* Accuracy
* Macro F1-Score

Hasil eksperimen dan artifact dicatat secara manual menggunakan MLflow.

---

# 🧪 MLflow Project

Project menggunakan format **MLflow Project**.

File utama:

```text
MLProject/
├── MLproject
├── conda.yaml
├── Customer_Churn_Dataset_preprocessing.csv
├── modelling.py
└── modelling_tuning.py
```

File `MLproject` mendefinisikan dua entry point:

```text
main
tuning
```

dengan command:

```text
main   → python modelling.py
tuning → python modelling_tuning.py
```

---

# 🔬 Experiment Tracking

MLflow digunakan untuk melakukan experiment tracking.

Informasi yang dicatat meliputi:

```text
Parameters
Metrics
Model
Artifacts
Environment
```

Baseline experiment menggunakan:

```python
mlflow.sklearn.autolog()
```

Sedangkan advanced experiment melakukan manual logging terhadap parameter dan metric.

---

# ☁️ DagsHub Integration

MLflow pada project diarahkan ke remote tracking server DagsHub:

```text
https://dagshub.com/lyynx123/Workflow-CI.mlflow
```

Dengan pendekatan ini, hasil eksperimen tidak hanya tersimpan secara lokal tetapi dapat dikirim ke remote tracking server.

Eksperimen yang digunakan:

```text
Eksperimen_Dasar_Zudin
Eksperimen_Tuning_Advanced_Zudin
```

> 🔐 **Security:** Credential DagsHub tidak ditulis di README. Gunakan environment variables atau GitHub Secrets untuk menyimpan credential.

---

# ⚙️ Continuous Integration with GitHub Actions

Salah satu bagian utama repository adalah automated ML pipeline menggunakan **GitHub Actions**.

Workflow:

```text
.github/workflows/ci.yml
```

Pipeline dijalankan ketika terdapat:

```text
push
pull_request
```

pada branch:

```text
main
master
```

---

## 🔄 CI Pipeline Stages

Workflow memiliki beberapa tahap utama:

### Stage 1 — Checkout Repository

Mengambil source code dari repository.

```yaml
actions/checkout@v3
```

### Stage 2 — Setup Python

Environment menggunakan:

```text
Python 3.10
```

### Stage 3 — Upgrade pip

Memastikan package manager menggunakan versi terbaru.

### Stage 4 — Install ML Dependencies

Dependencies utama:

```text
MLflow 2.19.0
Scikit-learn
Pandas
Matplotlib
```

### Stage 5 — Dataset Validation

Pipeline memeriksa apakah dataset tersedia:

```text
MLProject/Customer_Churn_Dataset_preprocessing.csv
```

Jika dataset tidak ditemukan, workflow dihentikan.

### Stage 6 — Baseline Training

MLflow Project dijalankan menggunakan:

```bash
mlflow run . --env-manager local --entry-point main
```

### Stage 7 — Hyperparameter Tuning

Advanced experiment dijalankan menggunakan:

```bash
mlflow run . --env-manager local --entry-point tuning
```

---

# 🔐 Secrets Management

Credential untuk remote MLflow tracking tidak ditulis secara langsung di workflow.

GitHub Actions menggunakan:

```text
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

yang diambil dari:

```text
GitHub Secrets
```

Contoh:

```yaml
env:
  MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_TRACKING_USERNAME }}
  MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}
```

Pendekatan ini membantu mencegah credential ditulis secara langsung di source code.

---

# 📦 Model Artifacts

Advanced training workflow menghasilkan artifact seperti:

```text
replika_artifacts/
│
├── model/
│   ├── model.pkl
│   ├── MLmodel
│   ├── conda.yaml
│   ├── python_env.yaml
│   └── requirements.txt
│
├── estimator.html
├── metric_info.json
├── training_confusion_matrix.png
├── grafik_kemajuan_zudin.png
└── laporan_ekstra_zudin.txt
```

Artifact kemudian dicatat ke MLflow menggunakan:

```python
mlflow.log_artifacts(...)
```

---

# 🐳 Docker

Repository menyediakan:

```text
Dockerfile
```

yang menggunakan:

```text
python:3.12-slim
```

Dependencies utama yang dipasang:

```text
mlflow==2.19.0
pandas
numpy
scikit-learn
```

Dockerfile juga menyiapkan command untuk menjalankan MLflow model serving pada:

```text
0.0.0.0:8080
```

---

## Build Docker Image

```bash
docker build -t customer-churn-ml .
```

Run container:

```bash
docker run -p 8080:8080 customer-churn-ml
```

Model serving kemudian dapat diakses melalui:

```text
http://localhost:8080
```

> **Catatan:** Dockerfile saat ini mengharapkan model MLflow berada pada path `mlruns/0/`. Pastikan model artifact/MLflow run tersedia pada path tersebut sebelum menjalankan container.

---

# 📂 Repository Structure

```text
Workflow-CI/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── MLProject/
│   ├── Customer_Churn_Dataset_preprocessing.csv
│   ├── MLproject
│   ├── conda.yaml
│   ├── modelling.py
│   └── modelling_tuning.py
│
├── Membangun_model/
│   ├── Dataset/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── requirements.txt
│   └── screenshots/
│
├── Dockerfile
│
└── README.md
```

Struktur repository tersebut menunjukkan pemisahan antara workflow automation, MLProject, dan material pengembangan model.

---

# 🚀 Getting Started

## 1. Clone Repository

```bash
git clone https://github.com/lyynx123/Workflow-CI.git
cd Workflow-CI
```

## 2. Masuk ke MLProject

```bash
cd MLProject
```

## 3. Install Dependencies

```bash
pip install mlflow==2.19.0 scikit-learn pandas matplotlib
```

## 4. Jalankan Baseline Model

```bash
mlflow run . --env-manager local --entry-point main
```

## 5. Jalankan Hyperparameter Tuning

```bash
mlflow run . --env-manager local --entry-point tuning
```

---

# 🧪 Menjalankan Script Secara Langsung

Baseline:

```bash
python modelling.py
```

Advanced tuning:

```bash
python modelling_tuning.py
```

Namun untuk mereproduksi workflow MLflow, disarankan menggunakan:

```bash
mlflow run .
```

karena project telah mendefinisikan entry point melalui file `MLproject`.

---

# 📊 Evaluation

Model dievaluasi menggunakan:

### Accuracy

Mengukur proporsi prediksi yang benar terhadap keseluruhan data pengujian.

### Macro F1-Score

Digunakan untuk mengevaluasi performa klasifikasi dengan memberikan bobot yang sama terhadap setiap kelas.

Pada advanced experiment, metric dicatat melalui:

```python
mlflow.log_metric("akurasi_final", nilai_akurasi)
```

dan informasi F1-score juga disimpan dalam artifact:

```text
metric_info.json
```

> Nilai metric tidak ditulis sebagai angka tetap di README karena hasil dapat berubah ketika eksperimen dijalankan kembali.

---

# 🏗️ End-to-End Architecture

```text
                         ┌───────────────────────┐
                         │ Customer Churn Dataset│
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   Dataset Validation  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │      MLProject        │
                         └───────────┬───────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │                               │
                     ▼                               ▼
            ┌─────────────────┐             ┌─────────────────┐
            │ Baseline Model  │             │ Model Tuning    │
            │ Random Forest   │             │ Decision Tree   │
            └────────┬────────┘             └────────┬────────┘
                     │                               │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                           ┌────────────────────┐
                           │       MLflow       │
                           │ Experiment Tracking│
                           └─────────┬──────────┘
                                     │
                                     ▼
                           ┌────────────────────┐
                           │      DagsHub       │
                           │ Remote Tracking    │
                           └─────────┬──────────┘
                                     │
                                     ▼
                           ┌────────────────────┐
                           │ Model & Artifacts  │
                           └─────────┬──────────┘
                                     │
                                     ▼
                           ┌────────────────────┐
                           │       Docker       │
                           └─────────┬──────────┘
                                     │
                                     ▼
                           ┌────────────────────┐
                           │ MLflow Model Serve │
                           └────────────────────┘


       Git Push / Pull Request
                 │
                 ▼
       ┌─────────────────────┐
       │    GitHub Actions   │
       ├─────────────────────┤
       │ Environment Setup   │
       │ Dependency Install  │
       │ Dataset Validation  │
       │ Baseline Training   │
       │ Hyperparameter Tune │
       └─────────────────────┘
```

---

# 💻 Technology Stack

| Category             | Technology                   |
| -------------------- | ---------------------------- |
| Programming Language | Python                       |
| Data Processing      | Pandas                       |
| Machine Learning     | Scikit-learn                 |
| Experiment Tracking  | MLflow                       |
| Remote Tracking      | DagsHub                      |
| Automation           | GitHub Actions               |
| Containerization     | Docker                       |
| Visualization        | Matplotlib                   |
| Model                | Random Forest                |
| Model Tuning         | Decision Tree + GridSearchCV |

---

# 🎓 Key MLOps Concepts Implemented

Project ini mengimplementasikan beberapa konsep penting dalam MLOps:

* [x] MLProject
* [x] Reproducible ML workflow
* [x] Experiment tracking
* [x] MLflow autologging
* [x] Manual metric logging
* [x] Model artifact logging
* [x] Hyperparameter tuning
* [x] Remote experiment tracking
* [x] GitHub Actions automation
* [x] Dataset validation
* [x] Secrets management
* [x] Docker containerization
* [x] MLflow model serving

---

# 📸 Project Documentation

Folder `Membangun_model` berisi material pengembangan model dan dokumentasi eksperimen.

Dokumentasi tersebut dapat digunakan untuk melihat:

* proses training
* MLflow experiment
* model artifact
* DagsHub tracking
* hasil eksperimen

---

# 🔗 Project Links

### GitHub Repository

https://github.com/lyynx123/Workflow-CI

### GitHub Actions

https://github.com/lyynx123/Workflow-CI/actions

### ML Experiment Repository

https://github.com/lyynx123/Eksperimen_SML

### DagsHub

https://dagshub.com/lyynx123/Workflow-CI

---

# 👨‍💻 Author

**Ahmad Izzuddin Ulinnuha**

Machine Learning / MLOps Project

---

# 📚 Learning Outcomes

Melalui project ini, saya mempelajari dan mengimplementasikan:

1. Pengembangan model klasifikasi menggunakan Scikit-learn.
2. Baseline modeling menggunakan Random Forest.
3. Hyperparameter tuning menggunakan GridSearchCV.
4. Experiment tracking dengan MLflow.
5. Remote experiment tracking menggunakan DagsHub.
6. Model artifact management.
7. MLflow Project untuk reproducible execution.
8. Continuous Integration menggunakan GitHub Actions.
9. Dataset validation dalam automated workflow.
10. Secure credential management menggunakan GitHub Secrets.
11. Containerization menggunakan Docker.
12. Persiapan model untuk MLflow serving.

---

# ⭐ Project Summary

**Customer Churn Prediction — Machine Learning & MLOps Pipeline**

Project ini mendemonstrasikan bagaimana sebuah model machine learning dapat dikembangkan dari proses eksperimen hingga automated workflow:

```text
Develop
   ↓
Experiment
   ↓
Track
   ↓
Tune
   ↓
Validate
   ↓
Automate
   ↓
Package
   ↓
Serve
```

Fokus utama project:

> **Machine Learning + Experiment Tracking + Automation + Reproducibility + Model Deployment**
