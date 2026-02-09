# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model and outputs registration metadata.
"""

import argparse
import mlflow
import os 
import json
import logging

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser()
    # Aligning with template argument names
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    
    args, _ = parser.parse_known_args()
    return args

def main(args):
    '''Registers the model and writes details to JSON'''

    print(f"Registering model: {args.model_name}")

    # Step 1 & 2: We use the URI approach to bypass the 404/400 errors 
    # encountered with the newer MLflow client in Azure.
    model_uri = f"file://{os.path.abspath(args.model_path)}"
    print(f"Model URI: {model_uri}")

    # Step 3: Register the model and retrieve the version info
    model_details = mlflow.register_model(
        model_uri=model_uri,
        name=args.model_name
    )

    # Step 4: Write model registration details into a JSON file
    # This is often required for downstream deployment steps in MLOps pipelines
    print(f"Writing registration info to {args.model_info_output_path}")
    os.makedirs(args.model_info_output_path, exist_ok=True)
    
    model_info = {
        "model_name": args.model_name,
        "model_version": model_details.version,
        "model_uri": model_uri
    }

    output_file = os.path.join(args.model_info_output_path, "model_info.json")
    with open(output_file, "w") as f:
        json.dump(model_info, f)
    
    print(f"Successfully registered version {model_details.version}")

if __name__ == "__main__":
    # Start MLflow run to track the registration step
    mlflow.start_run()
    
    args = parse_args()
    
    # Audit Log to match template style
    print("--- Registration Step Audit Log ---")
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]
    for line in lines:
        print(line)
    print("-----------------------------------")

    main(args)

    mlflow.end_run()