import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Trading Strategy with Neural Network Integration')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'download'], default='test',
                        help='Mode: train, test, or download data')
    parser.add_argument('--data', type=str, help='Path to data CSV file')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str, default='2010-01-01', help='Start date for data download')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date for data download')
    parser.add_argument('--train-end-date', type=str, default='2020-12-31', help='End date for training data')
    parser.add_argument('--model-path', type=str, default='models/nn_model_weights.weights.h5',
                       help='Path to model weights')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    args = parser.parse_args()
    
    # Add the project root to Python's path
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Import modules
    from src.python.data_processor import download_data, save_data, split_data
    
    if args.mode == 'download':
        print(f"Downloading data for {args.ticker} from {args.start_date} to {args.end_date}")
        data = download_data(args.ticker, args.start_date, args.end_date)
        
        # Split and save data
        train_data, test_data = split_data(data, args.train_end_date)
        save_data(train_data, f"data/{args.ticker}_training.csv")
        save_data(test_data, f"data/{args.ticker}_testing.csv")
        
    elif args.mode == 'train':
        if not args.data:
            print("Error: --data argument is required for training mode")
            return
            
        # Import and run training directly instead of using subprocess
        from src.python.train import train_model
        train_model(args.data, args.epochs, args.model_path)
        
    elif args.mode == 'test':
        if not args.data:
            print("Error: --data argument is required for testing mode")
            return
            
        # Import and run testing directly instead of using subprocess
        from src.python.test import test_model
        test_model(args.data, args.model_path)
    
if __name__ == "__main__":
    main()
