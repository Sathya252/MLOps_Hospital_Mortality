# Databricks notebook source
import os
import subprocess

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

base_path = "/Workspace/Users/rohith.challa@apexon.com/ML_Project/MLops/SRC/Training_Pipeline/"

def run_pipeline_step(script_path):
    print(f"Running: {script_path}")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}:\n{result.stderr}")
    else:
        print(f"Finished {script_path} successfully.\n")

if __name__ == "__main__":
    print("Starting MLOps Training Pipeline...\n")

    for step in pipeline_steps:
        full_script_path = os.path.join(base_path, step)
        run_pipeline_step(full_script_path)

    print("All pipeline steps completed.")
