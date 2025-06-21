# Databricks notebook source
display(dbutils.fs.ls("dbfs:/mnt/rohith/MLops_Project/Training_Pipeline/"))

# COMMAND ----------

import subprocess
import os

def run_script(script_path):
    """Run a Python script using subprocess."""
    print(f"Running: {script_path}")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error in: {script_path}")
        print(result.stderr)
    else:
        print(f"Completed: {script_path}")
        print(result.stdout)

def main():
    base_path = "/Users/rohith.challa@apexon.com/ML_Project/MLops/SRC/Training_Pipeline/"

    pipeline_steps = [
        "raw_data_ingestion.py",
        "data_aggregation.py",
        "data_preprocessing.py",
        "feature_engineering.py",
        "feature_scaling.py",
        "modeling.py",
        "reporting.py",
        "hyperparameter_tuning.py",
        "model_comparison.py"
    ]

    print("Starting MLOps Training Pipeline...\n")

    for step in pipeline_steps:
        full_path = os.path.join(base_path, step)
        run_script(full_path)

    print("\n All pipeline steps completed successfully.")

if __name__ == "__main__":
    main()
