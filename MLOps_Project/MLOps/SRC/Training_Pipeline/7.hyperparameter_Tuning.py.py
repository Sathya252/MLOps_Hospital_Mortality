# Databricks notebook source
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, accuracy_score, roc_curve, auc
)
import pandas as pd
import joblib
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("HyperparameterTuningPipeline").getOrCreate()


def load_data():
    """Load prediction CSV files into a Pandas DataFrame and split into X, y."""
    spark_df = spark.read.csv("/mnt/rohith/MLops_Project/data/predictions_smote_csv/*.csv", header=True, inferSchema=True)
    df = spark_df.toPandas()
    X_data = df.drop("actual", axis=1)
    y_data = df["actual"]
    return X_data, y_data


def tune_logistic_regression(X_train, y_train):
    """Tune Logistic Regression hyperparameters."""
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print("Best Params for Logistic Regression:", grid.best_params_)
    print("Best Accuracy for Logistic Regression:", grid.best_score_)

    joblib.dump(grid.best_estimator_, "/dbfs/mnt/rohith/MLops_Project/model/logistic_model_tuned.pkl")
    return grid.best_estimator_


def tune_svm(X_train, y_train):
    """Tune SVM hyperparameters."""
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print("Best Params for SVM:", grid.best_params_)
    print("Best Accuracy for SVM:", grid.best_score_)

    joblib.dump(grid.best_estimator_, "/dbfs/mnt/rohith/MLops_Project/model/svm_model_tuned.pkl")
    return grid.best_estimator_


def train_linear_regression(X_train, y_train):
    """Train plain Linear Regression model (no tuning)."""
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, "/dbfs/mnt/rohith/MLops_Project/model/linear_regression_model.pkl")
    return lr_model


def evaluate_model(model, X_test, y_test):
    """Evaluate classification model performance."""
    y_pred = model.predict(X_test)

    if isinstance(model, LinearRegression):
        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]

    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    if isinstance(model, LinearRegression):
        y_prob = model.predict(X_test)
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc_curves(roc_results):
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(10, 8))

    for label, (fpr, tpr, roc_auc) in roc_results.items():
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    X_data, y_data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

    print("\nTuning Logistic Regression...")
    tuned_lr = tune_logistic_regression(X_train, y_train)

    print("\nTuning SVM...")
    tuned_svm = tune_svm(X_train, y_train)

    print("\nTraining Linear Regression...")
    linear_reg = train_linear_regression(X_train, y_train)

    print("\nEvaluating Logistic Regression...")
    fpr_lr, tpr_lr, auc_lr = evaluate_model(tuned_lr, X_test, y_test)

    print("\nEvaluating SVM...")
    fpr_svm, tpr_svm, auc_svm = evaluate_model(tuned_svm, X_test, y_test)

    print("\nEvaluating Linear Regression...")
    fpr_lin, tpr_lin, auc_lin = evaluate_model(linear_reg, X_test, y_test)

    roc_results = {
        "Logistic Regression": (fpr_lr, tpr_lr, auc_lr),
        "SVM": (fpr_svm, tpr_svm, auc_svm),
        "Linear Regression": (fpr_lin, tpr_lin, auc_lin)
    }
    plot_roc_curves(roc_results)


if __name__ == "__main__":
    main()
