# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
import pickle
import os
from sklearn.metrics import roc_curve
import numpy as np

spark = SparkSession.builder.appName("Model_Training_Evaluation").getOrCreate()

def load_preprocessed_data(path):
    return spark.read.format("delta").load(path)

def train_logistic_regression(df, feature_col, label_col):
    lr = LogisticRegression(featuresCol=feature_col, labelCol=label_col)
    model = lr.fit(df)
    return model

def evaluate_model_auc(df, label_col):
    evaluator = BinaryClassificationEvaluator(labelCol=label_col)
    auc_score = evaluator.evaluate(df)
    print(f"Training AUC: {auc_score:.4f}")
    return auc_score

def save_model_spark(model, path):
    model.write().overwrite().save(path)

def save_roc_curve_data(df, label_col, path):
    df_temp = df.select(vector_to_array("probability").alias("probability_array"), label_col)
    pred_pd = df_temp.toPandas()

    y_true = pred_pd[label_col]
    y_score = pred_pd['probability_array'].apply(lambda x: float(x[1]))
    fpr, tpr, _ = roc_curve(y_true, y_score)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"fpr": fpr, "tpr": tpr}, f)

def check_nan_features(df, feature_col):
    return df.select(feature_col).rdd.map(lambda row: any(np.isnan(row[0].toArray()))).filter(lambda x: x).count()

if __name__ == "__main__":
    preprocessed_path = "dbfs:/mnt/rohith/MLops_Project/Inference_Pipeline/02_synthetic_preprocessed_df/"
    model_path = "/dbfs/mnt/rohith/MLops_Project/model/logistic_model_smote_20240615"
    roc_data_path = "/dbfs/mnt/rohith/MLops_Project/model/training_roc_curve.pkl"

    label_col = 'hospital_expire_flag'
    feature_col = 'scaledFeatures'

    df = load_preprocessed_data(preprocessed_path)

    df.groupBy(label_col).count().show()

    nan_count = check_nan_features(df, feature_col)
    print("Features with NaNs:", nan_count)
    if nan_count > 0:
        raise ValueError(f"Found {nan_count} NaN feature vectors. Check preprocessing!")

    model = train_logistic_regression(df, feature_col, label_col)

    predictions = model.transform(df)

    evaluate_model_auc(predictions, label_col)

    save_model_spark(model, model_path)

    save_roc_curve_data(predictions, label_col, roc_data_path)

    print("Model training and ROC data saving complete.")