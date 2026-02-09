# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains a Random Forest Regressor and logs results to MLflow.
"""

import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("train")
    
    # Aligning with your pipeline component definition
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--n_estimators", type=int, default=100, help='Number of trees in the forest')
    parser.add_argument("--max_depth", type=int, default=None, help='Maximum depth of the tree')
    parser.add_argument("--model_output", type=str, help="Path of output model")
    
    return parser.parse_args()

def select_first_file(path):
    """
    Selects the first CSV file in a folder. 
    Azure ML mounts data as directories, so we must pick the file inside.
    """
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not files:
        raise RuntimeError(f"No CSV files found in {path}")
    return os.path.join(path, files[0])

def main(args):
    '''Read data, train model, and save artifacts'''

    # Loading datasets
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    # Identify target column (must match data_prep.py output)
    target_col = "price"

    # Separate features and target
    y_train = train_df[target_col].values
    X_train = train_df.drop(target_col, axis=1).values
    y_test = test_df[target_col].values
    X_test = test_df.drop(target_col, axis=1).values

    # Initialize and train Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth, 
        random_state=42
    )
    model.fit(X_train, y_train)

    # Log hyperparameters to MLflow
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Evaluation
    ypred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, ypred_test)
    print(f'Mean Squared Error on test set: {mse:.2f}')
    mlflow.log_metric("MSE", float(mse))

    # Output and save the model
    # This creates the 'MLmodel' metadata folder Azure ML requires
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

if __name__ == "__main__":
    # Start the MLflow Experiment run
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    # Audit Trail: Prints paths and params to the Azure ML 'std_log.txt'
    print("--- Training Step Audit Log ---")
    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)
    print("-------------------------------")
    
    # Execute training
    main(args)

    # End the MLflow run
    mlflow.end_run()

