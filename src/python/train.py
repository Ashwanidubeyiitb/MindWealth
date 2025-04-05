import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.python.data_processor import (
    load_data, prepare_data, prepare_features, prepare_labels, generate_balanced_labels
)
from src.python.model import TradingModel
import os
import sys
from trading_strategies import MACDStrategy, RSIStrategy, SupertrendStrategy

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_sequences(data, seq_length=60):
    """Create sequences for LSTM model"""
    # Select only features we want to use
    feature_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 
                   'returns', 'log_returns', 'volatility', 'volume_ma', 'volume_std']
    
    # Extract features and target
    X_data = data[feature_cols].copy()
    y_data = data['target'].values
    
    # Normalize features if not already normalized
    for col in X_data.columns:
        if X_data[col].std() > 10:  # Only normalize if not already normalized
            X_data[col] = (X_data[col] - X_data[col].mean()) / (X_data[col].std() + 1e-8)
    
    X, y = [], []
    for i in range(seq_length, len(data)):
        # Get sequence of features
        X.append(X_data.iloc[i-seq_length:i].values)
        # Get target and ensure it's an integer
        y.append(int(y_data[i]))
    
    return np.array(X), np.array(y, dtype=np.int32)

def train_model(data_path, epochs, model_path):
    # Load data
    print(f"Loading data from {data_path}")
    data = load_data(data_path)
    
    print("\nData Info:")
    print(data.info())
    print("\nFirst few rows:")
    print(data.head())
    
    # Prepare data for C++ strategies
    ohlc_data = prepare_data(data)
    
    # Run strategies
    print("\nRunning MACD strategy...")
    macd_strategy = MACDStrategy()
    macd_result = macd_strategy.runStrategy(ohlc_data)
    print(f"MACD signals length: {len(macd_result.signals)}")
    print(f"First few MACD signals: {[s.action for s in macd_result.signals[:5]]}")
    
    print("\nRunning RSI strategy...")
    rsi_strategy = RSIStrategy()
    rsi_result = rsi_strategy.runStrategy(ohlc_data)
    print(f"RSI signals length: {len(rsi_result.signals)}")
    print(f"First few RSI signals: {[s.action for s in rsi_result.signals[:5]]}")
    
    print("\nRunning Supertrend strategy...")
    supertrend_strategy = SupertrendStrategy()
    supertrend_result = supertrend_strategy.runStrategy(ohlc_data)
    print(f"Supertrend signals length: {len(supertrend_result.signals)}")
    print(f"First few Supertrend signals: {[s.action for s in supertrend_result.signals[:5]]}")
    
    # Prepare features
    print("Preparing features...")
    features = prepare_features(data, macd_result.signals, rsi_result.signals, supertrend_result.signals)
    
    # Create combined labels (simple majority voting)
    print("Preparing labels...")
    prepare_labels(features)
    
    # Drop the signal columns as they directly leak the target
    features = features.drop(['macd_signal', 'rsi_signal', 'supertrend_signal'], axis=1)
    
    # If we only have one class, force balance
    if len(features['target'].unique()) == 1:
        print("Only one class detected. Forcing balanced labels...")
        features = generate_balanced_labels(features)
    
    # Print label distribution
    print("\nLabel distribution:")
    print(features['target'].value_counts())
    
    # Create sequences
    print("\nCreating sequences...")
    X, y = create_sequences(features)
    print(f"Sequence shape: {X.shape}, Target shape: {y.shape}")
    print("Target value range:", np.min(y), "to", np.max(y))
    print("Unique target values:", np.unique(y))
    
    # Split data
    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Create and train model
    print("Building and training model...")
    model = TradingModel(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = model.train(X_train, y_train, X_val, y_val, epochs=epochs)
    
    # Evaluate model
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
    
    print(f"Model training complete. Model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model-path', default='models/nn_model_weights.h5')
    args = parser.parse_args()
    
    train_model(args.data, args.epochs, args.model_path)

if __name__ == "__main__":
    main()
