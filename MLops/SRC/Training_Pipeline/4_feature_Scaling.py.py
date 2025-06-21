# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, log1p
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName("FeatureScalingPipeline").getOrCreate()

def load_data(path):
    df = spark.read.format("delta").load(path)
    print(f" Initial row count: {df.count()}")
    return df

def apply_log_transformations(df, log_cols):
    for c in log_cols:
        if f'log_{c}' not in df.columns:
            df = df.withColumn(f'log_{c}', log1p(col(c)))
    print(" Log transformations applied.")
    return df

def replace_nulls(df, columns):
    for c in columns:
        df = df.withColumn(c, when(col(c).isNull(), 0).otherwise(col(c)))
    print(" Null replacement done.")
    return df

def assemble_features(df, feature_cols):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vector")
    df = assembler.transform(df)
    assembled_count = df.filter(col("features_vector").isNotNull()).count()
    print(f" Assembled features_vector — Non-null rows: {assembled_count}")
    if assembled_count == 0:
        raise ValueError(" No valid feature vectors found after assembling. Pipeline halted.")
    return df

def apply_scaling(df):
    scaler = StandardScaler(inputCol="features_vector", outputCol="scaled_features", withMean=True, withStd=True)
    scaler_model = scaler.fit(df)
    scaled_df = scaler_model.transform(df)
    scaled_count = scaled_df.count()
    print(f" Scaling done. Total rows after scaling: {scaled_count}")
    if scaled_count == 0:
        raise ValueError(" No rows found after scaling. Pipeline halted.")
    return scaled_df

def fit_linear_model(df):
    lr = LinearRegression(featuresCol="scaled_features", labelCol="log_los")
    lr_model = lr.fit(df)
    print(f" Linear Regression model fitted. R²: {lr_model.summary.r2:.4f}")

def save_data(df, output_path):
    df.write.format("delta").mode("overwrite").save(output_path)
    row_count = spark.read.format("delta").load(output_path).count()
    print(f" Data saved. Rows written: {row_count}")

def main():
    input_path = "dbfs:/mnt/rohith/MLops_Project/Training_Pipeline/05_clr_feature_engineered_df"
    output_path = "dbfs:/mnt/rohith/MLops_Project/Training_Pipeline/06_clr_feature_scaling_df/"

    log_transform_cols = ['los', 'age', 'min_vitals', 'max_vitals', 'mean_vitals', 'min_labs']
    feature_cols = ['log_los', 'log_age', 'log_mean_vitals', 'log_min_vitals', 'log_max_vitals',
                    'mean_labs', 'log_min_labs', 'max_labs', 'num_comorbidities']

    df = load_data(input_path)
    df = apply_log_transformations(df, log_transform_cols)
    df = replace_nulls(df, feature_cols)
    df = assemble_features(df, feature_cols)
    scaled_df = apply_scaling(df)
    fit_linear_model(scaled_df)
    save_data(scaled_df, output_path)

if __name__ == "__main__":
    main()