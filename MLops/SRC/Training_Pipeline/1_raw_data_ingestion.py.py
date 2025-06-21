# Databricks notebook source
from pyspark.sql.functions import col, round, months_between, when

def load_core_tables():
    admissions_df = spark.read.csv("/mnt/rohith/MIMIC3_Datasets/ADMISSIONS.csv", header=True, inferSchema=True)
    patients_df  = spark.read.csv("/mnt/rohith/MIMIC3_Datasets/PATIENTS.csv", header=True, inferSchema=True)
    icustays_df  = spark.read.csv("/mnt/rohith/MIMIC3_Datasets/ICUSTAYS.csv", header=True, inferSchema=True)
    return admissions_df, patients_df, icustays_df

def merge_core_tables(admissions_df, patients_df, icustays_df):
    adm_pat_df = admissions_df.join(patients_df, on="subject_id", how="inner")
    merged_df = adm_pat_df.join(icustays_df, on=["subject_id", "hadm_id"], how="inner").drop("row_id")
    merged_df = merged_df.withColumn("age", round(months_between(col("admittime"), col("dob"))/12, 2))
    return merged_df

def main():
    admissions_df, patients_df, icustays_df = load_core_tables()
    merged_df = merge_core_tables(admissions_df, patients_df, icustays_df)
    merged_df.write.format("delta").mode("overwrite").save("dbfs:/mnt/rohith/MLops_Project/Training_Pipeline/01_merge_core_tables_df/")
    # merged_df = spark.read.format("delta").load("dbfs:/mnt/rohith/MLops_Project/Training_Pipeline/01_merge_core_tables_df/")
    print(f"Rows: {merged_df.count()}, Columns: {len(merged_df.columns)}")

if __name__ == "__main__":
    main()