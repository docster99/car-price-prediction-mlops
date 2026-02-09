import os
import argparse
import logging
import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")
    return parser.parse_args()

def main(args):
    '''Read, preprocess, split, and save datasets'''
    # Reading Data
    df = pd.read_csv(args.raw_data)

    # Step 1: Label encoding
    label_encoder = LabelEncoder()
    # Note: Ensure 'Segment' matches your dataset column name
    df['Segment'] = label_encoder.fit_transform(df['Segment'])

    # Step 2: Split the dataset
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Step 3: Save datasets (Required for Azure ML outputs)
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "data.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "data.csv"), index=False)

    # Step 4: Log metrics
    mlflow.log_metric("train_size", len(train_df))
    mlflow.log_metric("test_size", len(test_df))

if __name__ == "__main__":
    mlflow.start_run()

    args = parse_args()

    # This satisfies the "lines" printing from your template
    print(f"Raw data path: {args.raw_data}")
    print(f"Train dataset output path: {args.train_data}")
    print(f"Test dataset path: {args.test_data}")
    print(f"Test-train ratio: {args.test_train_ratio}")
    
    main(args)

    mlflow.end_run()