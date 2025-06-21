# 🏥 MLOps Pipeline for Hospital Mortality Prediction (Azure Databricks)

An end-to-end MLOps pipeline for hospital mortality prediction using the MIMIC-III dataset, built and orchestrated on **Azure Databricks**. This project follows modular design principles, integrates CI/CD workflows, and includes synthetic data generation, model training, evaluation, and inference.

---

## 📌 Project Overview

| Component            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **Platform**          | Azure Databricks (Notebooks + Jobs)                                         |
| **CI/CD**             | Azure Pipelines (YAML-based workflows)                                      |
| **Data**              | MIMIC-III EHR Data (Synthetic version)                                      |
| **Model**             | Logistic Regression (with potential for extension: XGBoost, SVM, etc.)     |
| **Output**            | ROC AUC, model artifacts, inference outputs                                 |

---

## 🧱 Folder Structure

MLops_Project/
├── MLops/
│ ├── SRC/
│ │ ├── Training_Pipeline/
│ │ ├── Inference_Pipeline/
├── CI_CD/
│ ├── azure-pipelines.yml
│ ├── training_pipeline.yml
│ ├── inference_pipeline.yml


---

## 🔍 Key Features

- ✅ **Raw + Synthetic Data Handling**
- ✅ **Full EDA & KPI Visualizations**
- ✅ **Modular Preprocessing Pipeline**
- ✅ **Train/Test Splits & ROC Evaluation**
- ✅ **Model Saving & Inference Comparison**
- ✅ **CI/CD via Azure DevOps Pipelines**
- ✅ **Designed for Scalability on Databricks**

---

## 📦 Setup Instructions

### 🧰 Prerequisites

- Azure Databricks workspace
- Azure DevOps / GitHub linked to Databricks
- MIMIC-III dataset (or synthetic variant)
- Databricks cluster (Spark 3.0+ recommended)
- Python 3.8+

### 🔧 Clone the Repo

```bash
git clone https://github.com/<your-username>/MLOps_Hospital_Mortality.git
cd MLOps_Hospital_Mortality

⚙️ How to Run
🧪 1. Training Pipeline
Run main.py from Training_Pipeline/ in a Databricks notebook.

The pipeline performs:

Raw/Synthetic Data Loading

Preprocessing

Model Training & Evaluation

ROC Plot Saving

🚀 2. Inference Pipeline
Load the trained model.

Run main.py from Inference_Pipeline/.

Outputs:

Inference predictions

ROC comparison with training

Saved artifacts

🔁 CI/CD with Azure Pipelines
Connect your GitHub repo to Azure DevOps.

Use the YAML files in CI_CD/ to orchestrate:

training_pipeline.yml

inference_pipeline.yml

Trigger automatically on main branch push or PR.

🔍 Results Snapshot
Model: Logistic Regression

Metric: ROC AUC

Artifacts: ROC curves (.pkl), trained models (Spark format)

📈 Future Enhancements
 Add FastAPI for live inference serving

 Integrate MLflow for experiment tracking

 Add feature drift detection with Evidently

 Containerize with Dockerfile

 Add unit tests with pytest

📄 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Built by Sathya Rohith using Azure Databricks and MIMIC-III.
