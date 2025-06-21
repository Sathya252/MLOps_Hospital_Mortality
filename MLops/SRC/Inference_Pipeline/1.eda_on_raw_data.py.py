# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, mean
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

spark = SparkSession.builder.appName("MIMIC_Inference_EDA").getOrCreate()


def load_raw_data(path):
    return spark.read.format("delta").load(path)


def null_counts(df):
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show(truncate=False)

def categorize_columns_by_distinct_count(df, id_columns, categorical_string_cols, threshold=10):
    numeric_types = ['double', 'float', 'int', 'bigint']
    numerical_cols = [c for c, t in df.dtypes if t in numeric_types and c not in id_columns]

    binary_cols, categorical_numeric_cols, continuous_cols = [], [], []

    for col_name in numerical_cols:
        distinct_count = df.select(col_name).distinct().count()
        if distinct_count == 2:
            binary_cols.append(col_name)
        elif distinct_count <= threshold:
            categorical_numeric_cols.append(col_name)
        else:
            continuous_cols.append(col_name)

    return binary_cols, categorical_numeric_cols, continuous_cols, categorical_string_cols

def eda_numeric_distribution(df, numerical_cols):
    pd_df = df.select(numerical_cols).toPandas()
    for col_name in numerical_cols:
        sns.histplot(pd_df[col_name], kde=True)
        plt.title(f"Distribution of {col_name}")
        plt.xlabel(col_name)
        plt.ylabel("Count")
        plt.show()

def eda_boxplots(df, numerical_cols):
    pd_df = df.select(numerical_cols).toPandas()
    for col_name in numerical_cols:
        sns.boxplot(x=pd_df[col_name])
        plt.title(f"Boxplot of {col_name}")
        plt.show()

def eda_category_counts(df, categorical_cols):
    for col_name in categorical_cols:
        print(f"Value counts for {col_name}")
        df.groupBy(col_name).count().orderBy("count", ascending=False).show(truncate=False)

def eda_correlation_heatmap(df, numerical_cols):
    pd_df = df.select(numerical_cols).toPandas()
    corr = pd_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def aggregate_kpis(df):
    print("KPI Aggregation:")
    df.groupBy("hospital_expire_flag").agg(
        mean("age").alias("avg_age"),
        mean("los").alias("avg_los"),
        mean("mean_vitals").alias("avg_vitals"),
        mean("num_comorbidities").alias("avg_comorbidities")
    ).show()

def generate_synthetic_data(raw_path, save_path, num_records=2500):
    raw_df = spark.read.format("delta").load(raw_path)
    raw_pd = raw_df.toPandas()

    synthetic_data = {}
    for col_name in raw_pd.columns:
        if pd.api.types.is_numeric_dtype(raw_pd[col_name]):
            synthetic_data[col_name] = np.random.normal(raw_pd[col_name].mean(), raw_pd[col_name].std(), num_records)
        else:
            synthetic_data[col_name] = np.random.choice(raw_pd[col_name].dropna().unique(), num_records)

    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df['hospital_expire_flag'] = [1]*(num_records//2) + [0]*(num_records//2)
    synthetic_sdf = spark.createDataFrame(synthetic_df)
    synthetic_sdf.write.mode("overwrite").format("delta").save(save_path)

    print(f"Synthetic data of {num_records} rows saved to {save_path}")

def main():
    raw_path = "dbfs:/mnt/rohith/MLops_Project/Training_Pipeline/02_merge_total_tables_df/"
    save_path = "dbfs:/mnt/rohith/MLops_Project/Inference_Pipeline/01_synthetic_test_df/"
    
    df = load_raw_data(raw_path)
    null_counts(df)

    id_columns = ['subject_id', 'hadm_id', 'icustay_id']
    categorical_string_cols = ['admission_type', 'admission_location', 'discharge_location', 'insurance',
                               'language', 'religion', 'marital_status', 'ethnicity', 'diagnosis',
                               'gender', 'dbsource', 'first_careunit', 'last_careunit']

    binary_cols, categorical_numeric_cols, continuous_cols, categorical_cols = categorize_columns_by_distinct_count(
        df, id_columns, categorical_string_cols)

    print("Binary Columns:", binary_cols)
    print("Categorical Numeric Columns:", categorical_numeric_cols)
    print("Continuous Columns:", continuous_cols)
    print("Categorical String Columns:", categorical_cols)

    eda_numeric_distribution(df, continuous_cols)
    eda_boxplots(df, continuous_cols)
    eda_category_counts(df, categorical_cols)
    eda_correlation_heatmap(df, continuous_cols)
    aggregate_kpis(df)

    generate_synthetic_data(raw_path, save_path)

if __name__ == "__main__":
    main()
