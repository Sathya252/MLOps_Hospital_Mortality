# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, mean as _mean
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("Preprocessing_Synthetic_Data").getOrCreate()

def load_synthetic_data(path):
    return spark.read.format("delta").load(path)

def impute_mean(df, continuous_cols):
    for c in continuous_cols:
        mean_val = df.select(_mean(col(c))).collect()[0][0]
        df = df.fillna({c: mean_val})
    return df

def impute_mode(df, categorical_cols):
    for c in categorical_cols:
        mode_val = df.groupBy(c).count().orderBy('count', ascending=False).first()[0]
        df = df.fillna({c: mode_val})
    return df

def assemble_features(df, feature_cols, output_col="features"):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol=output_col, handleInvalid="keep")
    return assembler.transform(df)

def scale_features(df, input_col="features", output_col="scaledFeatures"):
    scaler = StandardScaler(inputCol=input_col, outputCol=output_col, withMean=True, withStd=True)
    scaler_model = scaler.fit(df)
    return scaler_model.transform(df)

def save_preprocessed_data(df, path):
    df.write.format("delta").mode("overwrite").save(path)

def run_preprocessing_pipeline(synthetic_path, preprocessed_path, feature_cols, continuous_cols, categorical_numeric_cols):
    df = load_synthetic_data(synthetic_path)
    df = impute_mean(df, continuous_cols)
    df = impute_mode(df, categorical_numeric_cols)
    df = assemble_features(df, feature_cols)
    df = scale_features(df)
    save_preprocessed_data(df, preprocessed_path)

if __name__ == "__main__":
    synthetic_path = "dbfs:/mnt/rohith/MLops_Project/Inference_Pipeline/01_synthetic_table_df/"
    preprocessed_path = "dbfs:/mnt/rohith/MLops_Project/Inference_Pipeline/02_synthetic_preprocessed_df/"

    feature_cols = ['los', 'age', 'mean_vitals', 'min_vitals', 'max_vitals',
                    'mean_labs', 'min_labs', 'max_labs', 'num_comorbidities']

    continuous_cols = ['los', 'age', 'mean_vitals', 'min_vitals', 'max_vitals',
                       'mean_labs', 'min_labs', 'max_labs']

    categorical_numeric_cols = ['num_comorbidities']

    run_preprocessing_pipeline(synthetic_path, preprocessed_path, feature_cols, continuous_cols, categorical_numeric_cols)
