# Databricks notebook source
from pyspark.sql import SparkSession
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

spark = SparkSession.builder.appName("ModelComparisonPipeline").getOrCreate()


def load_test_data():
    """Load test data from predictions CSV as Pandas DataFrame."""
    spark_df = spark.read.csv("/mnt/rohith/MLops_Project/data/predictions_smote_csv/*.csv", header=True, inferSchema=True)
    df = spark_df.toPandas()
    X_test = df.drop("actual", axis=1)
    y_test = df["actual"]
    return X_test, y_test


def load_models():
    """Load tuned models from DBFS."""
    models = {
        "Logistic Regression": joblib.load("/dbfs/mnt/rohith/MLops_Project/model/logistic_model_tuned.pkl"),
        "SVM": joblib.load("/dbfs/mnt/rohith/MLops_Project/model/svm_model_tuned.pkl"),
        "Linear Regression": joblib.load("/dbfs/mnt/rohith/MLops_Project/model/linear_regression_model.pkl"),
    }
    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate models and compile results."""
    results = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "AUC": [],
    }

    for name, model in models.items():
        y_pred = model.predict(X_test)

        if name == "Linear Regression":
            y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
            y_prob = model.predict(X_test)
        else:
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_val = roc_auc_score(y_test, y_prob)

        results["Model"].append(name)
        results["Accuracy"].append(acc)
        results["Precision"].append(prec)
        results["Recall"].append(rec)
        results["F1 Score"].append(f1)
        results["AUC"].append(auc_val)

    return pd.DataFrame(results)


def main():
    X_test, y_test = load_test_data()
    models = load_models()
    results_df = evaluate_models(models, X_test, y_test)
    print(results_df)


if __name__ == "__main__":
    main()