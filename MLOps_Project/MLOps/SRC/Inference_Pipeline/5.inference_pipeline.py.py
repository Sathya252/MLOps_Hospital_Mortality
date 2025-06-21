# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def start_spark():
    return SparkSession.builder.appName("InferenceMetricsPipeline").getOrCreate()


def load_model(model_path):
    return LogisticRegressionModel.load(model_path)


def load_data(spark, data_path):
    return spark.read.format("delta").load(data_path)


def run_inference(model, df):
    return model.transform(df)


def compute_metrics(predictions, label_col):
    evaluator_acc = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
    evaluator_precision = MulticlassClassificationEvaluator(labelCol=label_col, metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(labelCol=label_col, metricName="weightedRecall")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label_col, metricName="f1")

    accuracy = evaluator_acc.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    return accuracy, precision, recall, f1


def load_training_roc(training_roc_path):
    with open(training_roc_path, "rb") as f:
        training_roc = pickle.load(f)
    return training_roc


def compute_inference_roc(predictions, label_col="hospital_expire_flag"):
    df_temp = predictions.select(vector_to_array("probability").alias("probability_array"), label_col)
    pred_pd = df_temp.toPandas()

    y_true = pred_pd[label_col]
    y_score = pred_pd['probability_array'].apply(lambda x: float(x[1]))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def plot_roc_curves(training_roc, inference_rocs):
    plt.figure(figsize=(8, 6))

    plt.plot(training_roc['fpr'], training_roc['tpr'],
             label=f"Training ROC (AUC = {auc(training_roc['fpr'], training_roc['tpr']):.3f})", color='green')

    for model_name, roc_data in inference_rocs.items():
        plt.plot(roc_data['fpr'], roc_data['tpr'],
                 label=f"{model_name} Inference ROC (AUC = {roc_data['auc']:.3f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # <- fixed line here
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Training vs Inference ROC Curves")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()



def main():
    spark = start_spark()
    inference_data_path = "dbfs:/mnt/rohith/MLops_Project/Inference_Pipeline/02_synthetic_preprocessed_df/"
    training_roc_path = "/dbfs/mnt/rohith/MLops_Project/model/training_roc_curve.pkl"
    label_col = "hospital_expire_flag"

    logistic_model_path = "/dbfs/mnt/rohith/MLops_Project/model/logistic_model_smote_20240615"

    df_infer = load_data(spark, inference_data_path)
    training_roc = load_training_roc(training_roc_path)

    metrics_summary = []
    inference_rocs = {}

    log_model = load_model(logistic_model_path)
    predictions_log = run_inference(log_model, df_infer)
    acc, prec, rec, f1 = compute_metrics(predictions_log, label_col)
    fpr_log, tpr_log, auc_log = compute_inference_roc(predictions_log, label_col)

    inference_rocs['Logistic Regression'] = {'fpr': fpr_log, 'tpr': tpr_log, 'auc': auc_log}
    metrics_summary.append(['Logistic Regression', acc, prec, rec, f1])

    metrics_df = pd.DataFrame(metrics_summary, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    print(metrics_df)

    plot_roc_curves(training_roc, inference_rocs)


if __name__ == "__main__":
    main()