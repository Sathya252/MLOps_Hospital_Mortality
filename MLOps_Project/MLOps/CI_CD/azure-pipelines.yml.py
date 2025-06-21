# Databricks notebook source
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

variables:
  DATABRICKS_HOST: 'https://<your-databricks-workspace-url>'
  DATABRICKS_TOKEN: '$(DATABRICKS_TOKEN)'  # store this in Azure Pipeline secrets

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.9'

- script: |
    pip install databricks-cli
  displayName: 'Install Databricks CLI'

- script: |
    databricks configure --token <<EOF
    $(DATABRICKS_HOST)
    $(DATABRICKS_TOKEN)
    EOF
  displayName: 'Configure Databricks CLI'

- script: |
    databricks fs cp ./inference_pipeline.py dbfs:/mnt/rohith/MLops_Project/Inference_Pipeline/inference_pipeline.py --overwrite
  displayName: 'Upload Inference Pipeline Script'

- script: |
    databricks jobs run-now --job-id <your-job-id>
  displayName: 'Trigger Databricks Inference Job'
