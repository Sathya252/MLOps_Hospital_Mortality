# Databricks notebook source
name: MLOps Training Pipeline

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  train_pipeline:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install databricks-cli

    - name: Configure Databricks CLI using Env Vars
      env:
        DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
        DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
      run: |
        databricks configure --token --host $DATABRICKS_HOST --token $DATABRICKS_TOKEN

    - name: Trigger Databricks Training Job
      env:
        DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
        DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
      run: |
        databricks jobs run-now --job-id <your-job-id>
