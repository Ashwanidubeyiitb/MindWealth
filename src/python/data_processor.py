import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
from trading_strategies import OHLC, Action  # Add Action to imports

def download_data(ticker="AAPL", start_date="2010-01-01", end_date="2024-12-31", interval="1d"):
    """Download stock data using yfinance"""
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    data.reset_index(inplace=True)
    return data

def prepare_data(df):
    """Convert pandas DataFrame to list of OHLC objects for C++ strategies"""
    ohlc_data = []
    
    # Make sure we're using numeric values
    for _, row in df.iterrows():
        candle = OHLC()
        candle.open = float(row['Open'])
        candle.high = float(row['High'])
        candle.low = float(row['Low'])
        candle.close = float(row['Close'])
        candle.volume = float(row['Volume'])
        ohlc_data.append(candle)
    
    return ohlc_data

def save_data(df, filepath):
    """Save DataFrame to CSV file"""
    df.to_csv(filepath)

def load_data(filepath):
    """Load data from CSV file"""
    # Read CSV file
    data = pd.read_csv(filepath)
    
    # Remove the ticker symbol row (second row)
    data = data.drop(data.index[0])
    
    # Drop any non-numeric rows and reset index
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[pd.to_numeric(data['Open'], errors='coerce').notna()]
    
    # Convert Date column if it exists
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    
    # Make sure all price and volume columns are numeric
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Drop any rows with NaN values and reset index
    data = data.dropna(subset=numeric_columns)
    data = data.reset_index()
    
    return data

def split_data(df, train_end_date="2020-12-31"):
    """Split data into training and testing sets"""
    train_data = df[df.index <= train_end_date].copy()
    test_data = df[df.index > train_end_date].copy()
    return train_data, test_data

def prepare_features(df, macd_signals, rsi_signals, supertrend_signals):
    """Prepare feature DataFrame with technical indicators"""
    features = df.copy()
    
    # Convert signals to numpy arrays of the correct length
    signal_length = len(features)
    macd_array = np.zeros(signal_length)
    rsi_array = np.zeros(signal_length)
    supertrend_array = np.zeros(signal_length)
    
    # Fill in the signals
    for i, signal in enumerate(macd_signals):
        if i >= signal_length: break
        macd_array[i] = 1 if signal.action == Action.BUY else (-1 if signal.action == Action.SELL else 0)
    
    for i, signal in enumerate(rsi_signals):
        if i >= signal_length: break
        rsi_array[i] = 1 if signal.action == Action.BUY else (-1 if signal.action == Action.SELL else 0)
    
    for i, signal in enumerate(supertrend_signals):
        if i >= signal_length: break
        supertrend_array[i] = 1 if signal.action == Action.BUY else (-1 if signal.action == Action.SELL else 0)
    
    # Add strategy signals
    features['macd_signal'] = macd_array
    features['rsi_signal'] = rsi_array
    features['supertrend_signal'] = supertrend_array
    
    # Add and normalize price features
    features['returns'] = features['Close'].pct_change()
    features['log_returns'] = np.log(features['Close']/features['Close'].shift(1))
    features['volatility'] = features['returns'].rolling(window=20).std()
    
    # Normalize price columns
    price_columns = ['Open', 'High', 'Low', 'Close']
    for col in price_columns:
        mean = features[col].mean()
        std = features[col].std()
        features[col] = (features[col] - mean) / std
    
    # Normalize volume
    features['Volume'] = (features['Volume'] - features['Volume'].mean()) / features['Volume'].std()
    features['volume_ma'] = features['Volume'].rolling(window=20).mean()
    features['volume_std'] = features['Volume'].rolling(window=20).std()
    
    # Drop any NaN values
    features = features.dropna()
    
    return features

def prepare_labels(features):
    """Prepare target labels using a more aggressive approach to create buy/sell signals"""
    # Initialize target column with integer type
    features['target'] = 1  # Default to hold (class 1)
    
    # Use even more aggressive thresholds for signal generation
    buy_threshold = 0.2  # More aggressive buy threshold (was 0.4)
    sell_threshold = -0.2  # More aggressive sell threshold (was -0.4)
    
    # Add volatility-based signals
    features['volatility_z'] = (features['volatility'] - features['volatility'].mean()) / features['volatility'].std()
    features['vol_signal'] = 0
    features.loc[features['volatility_z'] > 1.5, 'vol_signal'] = -0.5  # High volatility - potential sell
    features.loc[features['volatility_z'] < -1.0, 'vol_signal'] = 0.3  # Low volatility - potential buy
    
    # Add trend detection (comparing current price to moving average)
    features['price_ma'] = features['Close'].rolling(window=20).mean()
    features['trend_signal'] = 0
    features.loc[features['Close'] > features['price_ma'] * 1.05, 'trend_signal'] = 0.4  # Strong uptrend - buy
    features.loc[features['Close'] < features['price_ma'] * 0.95, 'trend_signal'] = -0.4  # Strong downtrend - sell
    
    for i in range(len(features)):
        if i < 20:  # Skip the first few rows due to rolling calculations
            continue
            
        # Get all signals
        macd_signal = features.iloc[i]['macd_signal']
        rsi_signal = features.iloc[i]['rsi_signal']
        supertrend_signal = features.iloc[i]['supertrend_signal']
        vol_signal = features.iloc[i]['vol_signal']
        trend_signal = features.iloc[i]['trend_signal']
        
        # Weight the signals (all strategies get equal weight)
        signals = [
            macd_signal * 0.25,
            rsi_signal * 0.15,
            supertrend_signal * 0.25,
            vol_signal * 0.15,
            trend_signal * 0.20
        ]
        
        # Remove zero signals for better signal clarity
        non_zero_signals = [s for s in signals if s != 0]
        
        if non_zero_signals:
            avg_signal = sum(non_zero_signals) / len(non_zero_signals)
            if avg_signal > buy_threshold:
                features.loc[i, 'target'] = 2  # Buy
            elif avg_signal < sell_threshold:
                features.loc[i, 'target'] = 0  # Sell
    
    # Clean up the additional columns
    features = features.drop(['volatility_z', 'vol_signal', 'price_ma', 'trend_signal'], axis=1)
    
    # Ensure target is integer type
    features['target'] = features['target'].astype(np.int32)
    
    # Reset index to ensure continuous indices
    features = features.reset_index(drop=True)
    
    return features

def generate_balanced_labels(features):
    """Manually create balanced labels for better model training"""
    # Reset the target column
    features['target'] = 1  # Default to hold
    
    # Use actual indices from the DataFrame
    valid_indices = features.index.tolist()
    n = len(valid_indices)
    
    # Ensure we have some examples of each class
    buy_indices = np.random.choice(valid_indices, size=int(n*0.2), replace=False)
    sell_indices_pool = [idx for idx in valid_indices if idx not in buy_indices]
    sell_indices = np.random.choice(sell_indices_pool, size=int(n*0.2), replace=False)
    
    # Set the target values
    features.loc[buy_indices, 'target'] = 2  # Buy
    features.loc[sell_indices, 'target'] = 0  # Sell
    
    # Ensure target is integer type
    features['target'] = features['target'].astype(np.int32)
    
    return features