# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

spark = SparkSession.builder.appName("MIMIC_Synthetic_Data_Inference").getOrCreate()

def load_raw_data(path):
    return spark.read.format("delta").load(path)

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

def generate_synthetic_data(df, num_records):
    return df.orderBy(rand()).limit(num_records)

def eda_numeric_distribution(df, numerical_cols):
    pd_df = df.select(numerical_cols).toPandas()
    for col_name in numerical_cols:
        sns.histplot(pd_df[col_name], kde=True)
        plt.title(f"Synthetic Distribution of {col_name}")
        plt.xlabel(col_name)
        plt.ylabel("Count")
        plt.show()


def eda_category_counts(df, categorical_cols):
    for col_name in categorical_cols:
        print(f"Category counts for {col_name}:")
        df.groupBy(col_name).count().orderBy("count", ascending=False).show(truncate=False)

def save_synthetic_data(df, path):
    df.write.mode("overwrite").format("delta").save(path)
    print(f"Synthetic data saved at: {path}")

def run_synthetic_data_generator(raw_path, synthetic_save_path, num_records):
    df = load_raw_data(raw_path)

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

    print(f"Generating {num_records} synthetic records.")
    synthetic_df = generate_synthetic_data(df, num_records)

    print("Running EDA on synthetic data...")
    eda_numeric_distribution(synthetic_df, continuous_cols)
    eda_category_counts(synthetic_df, categorical_cols)

    print("Saving synthetic data")
    save_synthetic_data(synthetic_df, synthetic_save_path)

if __name__ == "__main__":
    raw_path = "dbfs:/mnt/rohith/MLops_Project/Training_Pipeline/02_merge_total_tables_df/"
    synthetic_save_path = "dbfs:/mnt/rohith/MLops_Project/Inference_Pipeline/01_synthetic_table_df/"
    num_records = 2500

    run_synthetic_data_generator(raw_path, synthetic_save_path, num_records)