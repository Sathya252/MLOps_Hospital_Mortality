# Databricks notebook source
from pyspark.sql.functions import col, mean, when, desc, monotonically_increasing_id, rand, skewness
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_aggregated_data():
    core_df = spark.read.format("delta").load("dbfs:/mnt/rohith/MLops_Project/Training_Pipeline/02_merge_total_tables_df/")
    return core_df

def generate_synthetic_data(df, num_records):
    factor = (num_records // df.count()) + 1
    duplicated_df = df
    for _ in range(factor - 1):
        duplicated_df = duplicated_df.union(df)
    synthetic_df = duplicated_df.withColumn("unique_id", monotonically_increasing_id()) \
                                .orderBy(rand()).limit(num_records).drop("unique_id")
    print("Synthetic records generated:", synthetic_df.count())
    return synthetic_df

def column_categorization(df):
    schema_fields = df.schema.fields
    id_cols = ['subject_id', 'hadm_id', 'icustay_id']
    num_cols = [f.name for f in schema_fields if isinstance(f.dataType, (DoubleType,)) and f.name not in id_cols]
    cat_cols = [f.name for f in schema_fields if isinstance(f.dataType, StringType)]
    return num_cols, cat_cols

def handle_nulls(df, numerical_cols, categorical_cols):
    for col_name in numerical_cols:
        mean_value = df.select(mean(col_name)).collect()[0][0]
        df = df.withColumn(col_name, when(col(col_name).isNull(), mean_value).otherwise(col(col_name)))
    for col_name in categorical_cols:
        mode_df = df.groupBy(col_name).count().orderBy(desc("count"))
        mode = mode_df.filter(col(col_name).isNotNull()).first()[col_name] if mode_df.count() else "Unknown"
        df = df.withColumn(col_name, when(col(col_name).isNull(), mode).otherwise(col(col_name)))
    return df

def plot_distributions(df, numerical_cols):
    eda_pd = df.select(numerical_cols).toPandas()
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(eda_pd[col], kde=True, color='skyblue')
        plt.title(f"Distribution of {col}")
        plt.grid(True)
        plt.show()

def plot_outliers(df, numerical_cols):
    eda_pd = df.select(numerical_cols).toPandas()
    for col in numerical_cols:
        plt.figure(figsize=(8, 2))
        sns.boxplot(x=eda_pd[col], color='lightcoral')
        plt.title(f"Boxplot of {col}")
        plt.grid(True)
        plt.show()

def plot_correlation_matrix(df, numerical_cols):
    eda_pd = df.select(numerical_cols).toPandas()
    corr = eda_pd.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

def plot_pairplot(df, numerical_cols):
    eda_pd = df.select(numerical_cols).toPandas()
    sns.pairplot(eda_pd)
    plt.show()

def compute_vif(df, features):
    for col_name in features:
        df = df.withColumn(col_name, col(col_name).cast("double"))
    df = df.na.drop(subset=features)
    assembler = VectorAssembler(inputCols=features, outputCol="features_vector")
    assembled_df = assembler.transform(df).select("features_vector")
    corr_matrix = Correlation.corr(assembled_df, "features_vector", "pearson").head()[0].toArray()
    vif_values = {features[i]: 1 / (1 - corr_matrix[i, i]) for i in range(len(features))}
    print("VIF values:", vif_values)
    return vif_values

def check_skewness(df, numerical_cols):
    for col_name in numerical_cols:
        skew_val = df.select(skewness(col_name)).collect()[0][0]
        print(f"Skewness of {col_name}: {skew_val}")

def iqr_capping(df, numerical_cols):
    for col_name in numerical_cols:
        q1, q3 = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df = df.withColumn(col_name, when(col(col_name) < lower, lower)
                                      .when(col(col_name) > upper, upper)
                                      .otherwise(col(col_name)))
    return df


def main():
    core_df = load_aggregated_data()

    synthetic_df = generate_synthetic_data(core_df, 5000)

    numerical_cols, categorical_cols = column_categorization(synthetic_df)
    print("Numerical Columns:", numerical_cols)
    print("Categorical Columns:", categorical_cols)

    synthetic_df = handle_nulls(synthetic_df, numerical_cols, categorical_cols)
    output_path = "/mnt/rohith/MLops_Project/Training_Pipeline/03_generate_synthetic_df/"
    synthetic_df.write.format("delta").mode("overwrite").save(output_path)

    synthetic_df_reloaded = spark.read.format("delta").load(output_path)
    print("Rows written after handling nulls:", synthetic_df_reloaded.count())

    plot_distributions(synthetic_df_reloaded, numerical_cols)
    plot_outliers(synthetic_df_reloaded, numerical_cols)
    plot_correlation_matrix(synthetic_df_reloaded, numerical_cols)
    plot_pairplot(synthetic_df_reloaded, numerical_cols)

    compute_vif(synthetic_df_reloaded, numerical_cols)
    check_skewness(synthetic_df_reloaded, numerical_cols)

    synthetic_clr_df = iqr_capping(synthetic_df_reloaded, numerical_cols)
    capped_output_path = "/mnt/rohith/MLops_Project/Training_Pipeline/04_clr_transformed_df/"
    synthetic_clr_df.write.format("delta").mode("overwrite").save(capped_output_path)

    final_df = spark.read.format("delta").load(capped_output_path)
    print("Rows written after IQR capping:", final_df.count())

if __name__ == "__main__":
    main()
