# Databricks notebook source
pip install imblearn

# COMMAND ----------

# MAGIC %pip install azure-storage-blob

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from azure.storage.blob import BlobServiceClient
import pandas as pd
import joblib
import os
import uuid
import shutil

spark = SparkSession.builder.appName("MLOps_Modeling").getOrCreate()

def load_scaled_data(path):
    df = spark.read.format("delta").load(path)
    print(f"Loaded data with {df.count()} rows")
    return df

def convert_to_pandas(df):
    df_clean = df.filter(col("hospital_expire_flag").isNotNull())
    pd_df = df_clean.drop("scaled_features").drop("features_vector").toPandas()
    print(f" Converted to Pandas: {len(pd_df)} records")
    return pd_df

def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"After SMOTE — Class 0: {(y_res == 0).sum()}, Class 1: {(y_res == 1).sum()}")
    return X_res, y_res

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Model trained successfully")
    return model

def save_model(model, model_path):
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    joblib.dump(model, model_path)
    print(" Model saved to DBFS:", os.path.exists(model_path))

def save_model_to_blob(model, model_name, connection_string, container_name):
    model_file = f"{model_name}_{uuid.uuid4().hex[:8]}.pkl"
    local_path = f"/tmp/{model_file}"

    joblib.dump(model, local_path)

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    try:
        container_client.create_container()
    except:
        pass  

    blob_client = container_client.get_blob_client(model_file)
    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    os.remove(local_path)

    blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{model_file}"
    print(f"Model uploaded to Azure Blob Storage: {blob_url}")
    return blob_url

def predict_and_save(model, X_test, y_test, save_path):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    predictions_df = pd.DataFrame({
        'actual': y_test,
        'prediction': y_pred,
        'probability': y_prob
    })

    predictions_spark_df = spark.createDataFrame(predictions_df)
    predictions_spark_df.write.mode("overwrite").option("header", True).csv(save_path)

    print("Predictions saved to:", save_path)
    return predictions_df

def main():
    data_path = "dbfs:/mnt/rohith/MLops_Project/Training_Pipeline/06_clr_feature_scaling_df"
    model_path = "/dbfs/mnt/rohith/MLops_Project/model/logistic_model_smote.pkl"
    predictions_path = "dbfs:/mnt/rohith/MLops_Project/data/predictions_smote_csv"
    azure_connection = "DefaultEndpointsProtocol=https;AccountName=sathya;AccountKey=5ec/eBcXSe6Pgb8ptCuXk8j63lsHFOnpxA7dnKHw7rDtwYXQNm0UvIk2w4iXsrNc9oTtBS6i1XVT+AStvIqxoQ==;EndpointSuffix=core.windows.net"
    container_name = "rohith"

    df = load_scaled_data(data_path)
    pd_df = convert_to_pandas(df)

    feature_cols = ['log_los', 'log_age', 'log_min_vitals', 'log_max_vitals',
                    'mean_labs', 'log_min_labs', 'max_labs', 'num_comorbidities']
    X = pd_df[feature_cols]
    y = pd_df["hospital_expire_flag"]


    X_res, y_res = apply_smote(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

    model = train_model(X_train, y_train)

    save_model(model, model_path)

    save_model_to_blob(model, "logistic_model_smote", azure_connection, container_name)

    predictions_df = predict_and_save(model, X_test, y_test, predictions_path)

    print("Classification Report:\n", classification_report(y_test, predictions_df['prediction']))

if __name__ == "__main__":
    main()
