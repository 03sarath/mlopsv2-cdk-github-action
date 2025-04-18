import argparse
import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_prepared_data():
    """Load the prepared datasets from the source URLs"""
    train_url = "https://psitron.s3.ap-southeast-1.amazonaws.com/dataset/data/train.csv"
    test_url = "https://psitron.s3.ap-southeast-1.amazonaws.com/dataset/data/test.csv"
    validation_url = "https://psitron.s3.ap-southeast-1.amazonaws.com/dataset/data/validation.csv"
    
    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)
    validation_df = pd.read_csv(validation_url)
    
    return train_df, test_df, validation_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=False)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    
    print("Loading prepared datasets...")
    train_df, test_df, validation_df = load_prepared_data()
    
    # Create output directories
    os.makedirs(f"{base_dir}/train", exist_ok=True)
    os.makedirs(f"{base_dir}/test", exist_ok=True)
    os.makedirs(f"{base_dir}/validation", exist_ok=True)
    
    print("Saving datasets to S3...")
    train_df.to_csv(f"{base_dir}/train/train.csv", index=False, header=False)
    test_df.to_csv(f"{base_dir}/test/test.csv", index=False, header=False)
    validation_df.to_csv(f"{base_dir}/validation/validation.csv", index=False, header=False)