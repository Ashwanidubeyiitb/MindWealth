import argparse
import numpy as np
from src.python.data_processor import load_data, prepare_data, prepare_features
from src.python.model import TradingModel
from src.python.train import create_sequences
from trading_strategies import MACDStrategy, RSIStrategy, SupertrendStrategy
import os
import sys
from trading_strategies import Action, TradeSignal

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_performance(signals, data):
    """Calculate performance metrics for a strategy"""
    if not signals:
        return 0.0, 0.0, 0
    
    # Count number of trades (buy-sell pairs)
    num_trades = len(signals) // 2
    
    # Calculate success rate and per-trade return
    profitable_trades = 0
    total_return = 0.0
    
    for i in range(0, len(signals) - 1, 2):
        if i + 1 >= len(signals):
            break
            
        # Verify we have a buy-sell pair
        if signals[i].action != Action.BUY or signals[i+1].action != Action.SELL:
            continue
            
        buy_price = signals[i].price
        sell_price = signals[i+1].price
        
        if sell_price > buy_price:
            profitable_trades += 1
            
        trade_return = (sell_price - buy_price) / buy_price * 100.0
        total_return += trade_return
    
    success_rate = (profitable_trades / num_trades * 100.0) if num_trades > 0 else 0.0
    per_trade_return = (total_return / num_trades) if num_trades > 0 else 0.0
    
    return success_rate, per_trade_return, num_trades

def evaluate_signals(signals, data):
    """Evaluate the performance of trading signals"""
    if not signals:
        return 0, 0, 0
    
    # Ensure proper alternating buy/sell signals
    cleaned_signals = []
    expected_action = Action.BUY  # We should start with a buy
    
    for signal in signals:
        if signal.action == expected_action:
            cleaned_signals.append(signal)
            # Toggle expected action
            expected_action = Action.SELL if expected_action == Action.BUY else Action.BUY
    
    # Make sure we end with a sell
    if cleaned_signals and cleaned_signals[-1].action == Action.BUY:
        # If we're still in a position at the end, close it
        last_signal = TradeSignal()
        last_signal.action = Action.SELL
        last_signal.date = data.iloc[-1]['Date'].strftime('%Y-%m-%d')
        last_signal.price = data.iloc[-1]['Close']
        cleaned_signals.append(last_signal)
    
    # Calculate performance
    total_return = 0
    successful_trades = 0
    trades = []
    
    for i in range(0, len(cleaned_signals) - 1, 2):
        if i + 1 < len(cleaned_signals):
            buy_price = cleaned_signals[i].price
            sell_price = cleaned_signals[i + 1].price
            trade_return = (sell_price - buy_price) / buy_price * 100
            total_return += trade_return
            trades.append(trade_return)
            if trade_return > 0:
                successful_trades += 1
    
    num_trades = len(trades)
    success_rate = (successful_trades / num_trades * 100) if num_trades > 0 else 0
    avg_return = total_return / num_trades if num_trades > 0 else 0
    
    return success_rate, avg_return, num_trades

def test_model(data_path, model_path):
    print(f"Loading test data from {data_path}")
    data = load_data(data_path)
    
    # Prepare data for C++ strategies
    ohlc_data = prepare_data(data)
    
    # Run strategies
    print("\nRunning MACD strategy...")
    macd_strategy = MACDStrategy()
    macd_result = macd_strategy.runStrategy(ohlc_data)
    
    print("\nRunning RSI strategy...")
    rsi_strategy = RSIStrategy()
    rsi_result = rsi_strategy.runStrategy(ohlc_data)
    
    print("\nRunning Supertrend strategy...")
    supertrend_strategy = SupertrendStrategy()
    supertrend_result = supertrend_strategy.runStrategy(ohlc_data)
    
    # Prepare features
    print("Preparing features...")
    features = prepare_features(data, macd_result.signals, rsi_result.signals, supertrend_result.signals)
    
    # Create target labels (same as in train.py)
    print("Preparing labels...")
    for i in range(len(data)):
        macd_signal = features.iloc[i]['macd_signal']
        rsi_signal = features.iloc[i]['rsi_signal']
        supertrend_signal = features.iloc[i]['supertrend_signal']
        
        # Majority voting
        signals = [s for s in [macd_signal, rsi_signal, supertrend_signal] if s != 0]
        if signals:
            avg_signal = sum(signals) / len(signals)
            if avg_signal > 0.3:
                features.loc[i, 'target'] = 2  # Buy
            elif avg_signal < -0.3:
                features.loc[i, 'target'] = 0  # Sell
            else:
                features.loc[i, 'target'] = 1  # Hold
        else:
            features.loc[i, 'target'] = 1  # Hold
    
    # Drop the signal columns as they directly leak the target
    features = features.drop(['macd_signal', 'rsi_signal', 'supertrend_signal'], axis=1)
    
    # Create sequences
    print("Creating sequences...")
    print("Feature columns:", features.columns.tolist())  # Debug print
    X_test, y_test = create_sequences(features)
    print(f"Sequence shape: {X_test.shape}, Target shape: {y_test.shape}")  # Debug print
    
    # Load and evaluate model
    print("Loading model...")
    model = TradingModel(input_shape=(X_test.shape[1], X_test.shape[2]), model_path=model_path)
    
    if not model.load_weights():
        print("Error: Could not load model weights. Make sure you have trained the model first.")
        return
    
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Convert predictions back to trading signals
    signals = model.generate_signals(data.iloc[len(data)-len(predicted_classes):], predicted_classes)
    
    # Calculate and print performance metrics
    success_rate, per_trade_return, num_trades = calculate_performance(signals, data)
    print("\nNeural Network Performance:")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Return per Trade: {per_trade_return:.2f}%")
    print(f"Number of Trades: {num_trades}")
    
    # Print trading signals
    print("\nTrading Signals:")
    for signal in signals:
        action = "BUY" if signal.action == Action.BUY else "SELL"
        print(f"Date: {signal.date}, Action: {action}, Price: {signal.price:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model-path', default='models/nn_model_weights.keras')
    args = parser.parse_args()
    
    test_model(args.data, args.model_path)

if __name__ == "__main__":
    main()
