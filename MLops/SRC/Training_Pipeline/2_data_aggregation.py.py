# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, min, max, countDistinct, desc, when, round
from pyspark.sql.types import LongType

def load_raw_tables():
    chartevents_df = spark.read.csv("/mnt/rohith/MIMIC3_Datasets/CHARTEVENTS.csv", header=True, inferSchema=True)
    labevents_df = spark.read.csv("/mnt/rohith/MIMIC3_Datasets/LABEVENTS.csv", header=True, inferSchema=True)
    diagnoses_df = spark.read.csv("/mnt/rohith/MIMIC3_Datasets/DIAGNOSES_ICD.csv", header=True, inferSchema=True)
    d_icd_diagnoses_df = spark.read.csv("/mnt/rohith/MIMIC3_Datasets/D_ICD_DIAGNOSES.csv", header=True, inferSchema=True)
    d_items_df = spark.read.csv("/mnt/rohith/MIMIC3_Datasets/D_ITEMS.csv", header=True, inferSchema=True)
    return chartevents_df, labevents_df, diagnoses_df, d_icd_diagnoses_df, d_items_df

def aggregate_chartevents(chartevents_df, d_items_df):
    vital_item_ids = [220045, 220210, 220277, 220048, 211]
    filtered_df = chartevents_df.filter(chartevents_df.itemid.isin(vital_item_ids))
    agg_df = filtered_df.groupBy("subject_id", "hadm_id", "icustay_id").agg(
        mean("valuenum").alias("mean_vitals"),
        min("valuenum").alias("min_vitals"),
        max("valuenum").alias("max_vitals")
    )
    return agg_df

def aggregate_labevents(labevents_df, d_items_df):
    top_lab_items_df = labevents_df.groupBy("itemid").count().orderBy(desc("count")).limit(5)
    top_lab_itemids = [row['itemid'] for row in top_lab_items_df.collect()]
    filtered_df = labevents_df.filter(labevents_df.itemid.isin(top_lab_itemids))
    agg_df = filtered_df.groupBy("subject_id", "hadm_id").agg(
        mean("valuenum").alias("mean_labs"),
        min("valuenum").alias("min_labs"),
        max("valuenum").alias("max_labs")
    )
    return agg_df

def compute_comorbidities(diagnoses_df, d_icd_diagnoses_df):
    diagnoses_full_df = diagnoses_df.join(d_icd_diagnoses_df, on="icd9_code", how="left")
    comorbidity_df = diagnoses_full_df.groupBy("subject_id", "hadm_id").agg(
        countDistinct("icd9_code").alias("num_comorbidities")
    )
    return comorbidity_df

def merge_all_tables(merge_core_df, chartevents_agg_df, labevents_agg_df, comorbidity_df):
    merged_df = merge_core_df.join(chartevents_agg_df, on=["subject_id", "hadm_id", "icustay_id"], how="left")\
                             .join(labevents_agg_df, on=["subject_id", "hadm_id"], how="left")\
                             .join(comorbidity_df, on=["subject_id", "hadm_id"], how="left")
    return merged_df

def main():
    merge_core_df = spark.read.format("delta").load("dbfs:/mnt/rohith/MLops_Project/Training_Pipeline/01_merge_core_tables_df/")
    
    chartevents_df, labevents_df, diagnoses_df, d_icd_diagnoses_df, d_items_df = load_raw_tables()
    chartevents_agg_df = aggregate_chartevents(chartevents_df, d_items_df)
    labevents_agg_df = aggregate_labevents(labevents_df, d_items_df)
    comorbidity_df = compute_comorbidities(diagnoses_df, d_icd_diagnoses_df)
    
    final_df = merge_all_tables(merge_core_df, chartevents_agg_df, labevents_agg_df, comorbidity_df)
    final_df.write.format("delta").mode("overwrite").save("dbfs:/mnt/rohith/MLops_Project/Training_Pipeline/02_merge_total_tables_df/")

    print(f"Rows: {final_df.count()}, Columns: {len(final_df.columns)}")

if __name__ == "__main__":
    main()
