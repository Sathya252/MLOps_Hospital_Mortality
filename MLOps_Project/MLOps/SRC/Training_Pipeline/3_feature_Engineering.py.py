# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.regression import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

def load_cleaned_data(path):
    return spark.read.format("delta").load(path)

def apply_log_transformation(df, skewed_cols):
    for col_name in skewed_cols:
        if f'log_{col_name}' not in df.columns:
            df = df.withColumn(f'log_{col_name}', log1p(col(col_name)))
    return df

def plot_distributions(df_pd, numerical_cols):
    for col_name in numerical_cols:
        plt.figure(figsize=(10, 3))
        sns.histplot(df_pd[col_name], kde=True)
        plt.title(f'Distribution: {col_name}')
        plt.show()

        plt.figure(figsize=(10, 2))
        sns.boxplot(x=df_pd[col_name], color='skyblue')
        plt.title(f'Boxplot: {col_name}')
        plt.show()

def correlation_matrix(df_pd, cols):
    corr = df_pd[cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
    return corr

def pairplot_visualization(df_pd, selected_cols):
    sns.pairplot(df_pd[selected_cols])
    plt.suptitle("Pairplot of Top Correlated Features", y=1.02)
    plt.show()

def skewness_report(df_pd, numerical_cols):
    skew_vals = df_pd[numerical_cols].skew()
    print("\nSkewness Report:\n", skew_vals)

def calculate_vif(df, features_list):
    vif_dict = {}
    for col_name in features_list:
        other_features = [f for f in features_list if f != col_name]
        assembler = VectorAssembler(inputCols=other_features, outputCol="features")
        assembled_df = assembler.transform(df).select(col_name, "features")
        lr = LinearRegression(featuresCol="features", labelCol=col_name)
        lr_model = lr.fit(assembled_df)
        r_squared = lr_model.summary.r2
        vif = 1 / (1 - r_squared) if r_squared < 1 else float("inf")
        vif_dict[col_name] = vif
    print("\nVIF Results:")
    for feature, vif in vif_dict.items():
        print(f"{feature}: {vif:.3f}")
    return vif_dict

def main():
    input_path = "dbfs:/mnt/rohith/MLops_Project/Training_Pipeline/04_clr_transformed_df"
    output_path = "dbfs:/mnt/rohith/MLops_Project/Training_Pipeline/05_clr_feature_engineered_df/"

    synthetic_df = load_cleaned_data(input_path)

    numerical_cols = ['los', 'age', 'mean_vitals', 'min_vitals', 'max_vitals',
                      'mean_labs', 'min_labs', 'max_labs', 'num_comorbidities']

    skewed_cols = ['los', 'age', 'max_vitals', 'min_labs']

    synthetic_df = apply_log_transformation(synthetic_df, skewed_cols)

    log_cols = [f'log_{c}' for c in skewed_cols]
    all_cols = numerical_cols + log_cols

    synthetic_pd = synthetic_df.select([col(c) for c in all_cols]).toPandas()

    plot_distributions(synthetic_pd, numerical_cols)

    corr_df = correlation_matrix(synthetic_pd, numerical_cols)

    pairplot_features = ['log_max_vitals', 'max_labs', 'mean_labs']
    pairplot_visualization(synthetic_pd, pairplot_features)

    skewness_report(synthetic_pd, numerical_cols)

    vif_features = ['log_los', 'log_age', 'mean_vitals', 'min_vitals',
                    'log_max_vitals', 'mean_labs', 'log_min_labs', 'max_labs', 'num_comorbidities']
    vif_result = calculate_vif(synthetic_df, vif_features)

    synthetic_df.write.format("delta").mode("overwrite").save(output_path)
    print(" Rows written:", spark.read.format("delta").load(output_path).count())

if __name__ == "__main__":
    main()
