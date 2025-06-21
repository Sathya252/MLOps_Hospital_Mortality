# Databricks notebook source
from pyspark.sql import SparkSession
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_curve, auc
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

spark = SparkSession.builder.appName("ReportingPipeline").getOrCreate()


def load_predictions(prediction_path):
    """Load prediction CSV files from DBFS into a Pandas DataFrame."""
    spark_df = spark.read.csv(prediction_path, header=True, inferSchema=True)
    display(spark_df)
    df = spark_df.toPandas()
    return df


def evaluate_model_predictions(df):
    """Compute standard classification metrics from predictions."""
    actual_values = df['actual']
    predictions = df['prediction']

    accuracy = accuracy_score(actual_values, predictions)
    precision = precision_score(actual_values, predictions)
    recall = recall_score(actual_values, predictions)
    f1 = f1_score(actual_values, predictions)
    report = classification_report(actual_values, predictions)

    print(f"\n--- Evaluation Report ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(report)


def compare_baseline_models(df):
    """Train and evaluate baseline models for comparison."""
    X = df.drop('actual', axis=1)
    y = df['actual']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True, random_state=42)
    }

    model_results = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": []
    }

    plt.figure(figsize=(10, 8))

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if model_name == "Linear Regression":
            y_pred = [1 if i >= 0.5 else 0 for i in y_pred]

        model_results["Model"].append(model_name)
        model_results["Accuracy"].append(accuracy_score(y_test, y_pred))
        model_results["Precision"].append(precision_score(y_test, y_pred))
        model_results["Recall"].append(recall_score(y_test, y_pred))
        model_results["F1 Score"].append(f1_score(y_test, y_pred))

        if model_name == "Linear Regression":
            y_prob = model.predict(X_test)
        else:
            y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()

    results_df = pd.DataFrame(model_results)
    display(results_df)


def main():
    prediction_path = "/mnt/rohith/MLops_Project/data/predictions_smote_csv/*.csv"
    df = load_predictions(prediction_path)

    evaluate_model_predictions(df)

    compare_baseline_models(df)


if __name__ == "__main__":
    main()
