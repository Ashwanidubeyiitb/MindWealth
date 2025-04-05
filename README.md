# Neural Network Trading Strategy

A machine learning-based trading system that combines traditional technical indicators with a neural network to generate buy and sell signals for financial markets.

## Project Overview

This project implements a trading strategy that uses a Long Short-Term Memory (LSTM) neural network to learn patterns from traditional technical indicators (MACD, RSI, and Supertrend). The system:

1. Processes historical price data
2. Generates signals using multiple technical strategies
3. Trains a neural network to recognize profitable patterns
4. Produces BUY/SELL signals with timing optimization

## Project Structure

 
├── data/
│   ├── AAPL_testing.csv    # Testing dataset
│   └── AAPL_training.csv   # Training dataset
├── models/                 # Saved model weights
├── src/
│   ├── cpp/
│   │   └── bindings.cpp    # C++ bindings for technical strategies
│   ├── python/
│   │   ├── data_processor.py  # Data processing and feature engineering
│   │   ├── model.py           # Neural network model definition
│   │   ├── test.py            # Testing and evaluation
│   │   └── train.py           # Model training
│   └── main.py             # Main entry point
├── setup.py                # Package setup
└── requirements.txt        # Project dependencies
 

## Installation

1. Clone the repository:
    
   git clone https://github.com/yourusername/neural-trading-strategy.git
   cd neural-trading-strategy
    

2. Install dependencies:
    
   pip install -r requirements.txt
    

3. Build the C++ extensions:
    
   pip install -e .
    

## Usage

### Download Historical Data

 
python src/main.py --mode download --ticker AAPL --start-date 2010-01-01 --end-date 2024-12-31 --train-end-date 2020-12-31
 

### Train the Model

# Quick training option (20 epochs)
python src/main.py --mode train --data data/AAPL_training.csv --epochs 20

# Extended training option (early stopping will determine the optimal number of epochs)
python src/main.py --mode train --data data/AAPL_training.csv --epochs 100

### Test the Strategy

 
python src/main.py --mode test --data data/AAPL_testing.csv
 

## Performance Results

### Standard Model (20 epochs)
- **Success Rate: 60.00%**
- **Average Return per Trade: 9.76%**
- **Number of Trades: 10**
//More active trading with good success rate

### Extended Training Model (100 epochs, early stopped)
- **Success Rate: 50.00%**
- **Average Return per Trade: 0.13%**
- **Number of Trades: 6**
- **Test Accuracy: 56.67%** (compared to ~10% for the standard model)
//Moderate trading frequency but lower returns

### Highly Selective Model (100 epochs, early stopped at epoch 11)
- **Success Rate: 100.00%**
- **Average Return per Trade: 54.68%**
- **Number of Trades: 1**
- **Test Accuracy: 84.53%**
//Very selective, high conviction, longer-term trades


This model demonstrates a "quality over quantity" approach, making very few trades but with high conviction. The single trade captured a significant market uptrend from March 2023 to December 2024.

## Recommended Approach

After evaluating the three models, the **Standard Model (20 epochs)** is recommended for practical trading purposes. Here's why:

- **Balanced Trading Frequency**: Provides a reasonable number of trading opportunities (10 trades)
- **Strong Profitability**: 9.76% average return per trade is excellent for real-world trading
- **Reliable Success Rate**: 60% win rate offers a good balance of risk and reward
- **Statistical Significance**: With 10 trades, results are more reliable than the highly selective model
- **Practical Implementation**: Suitable for traders who want regular opportunities without sacrificing quality

While the Highly Selective Model shows an impressive return, its single-trade approach is too limited for most trading applications. The Standard Model offers the best compromise between opportunity frequency and profitability.

### Sample Trading Signals

 
Date: 2021-03-31, Action: BUY, Price: 119.49
Date: 2021-05-14, Action: SELL, Price: 124.89
Date: 2021-05-24, Action: BUY, Price: 124.55
Date: 2022-01-27, Action: SELL, Price: 156.48
...
Date: 2023-04-10, Action: BUY, Price: 160.41
Date: 2024-12-30, Action: SELL, Price: 251.92
 

## Technical Details

### Feature Engineering

The system generates features including:
- Price data (OHLC + Volume)
- Price returns and log returns
- Volatility measures
- Volume analysis
- Technical indicator signals (MACD, RSI, Supertrend)

### Model Architecture

The neural network uses:
- LSTM layer to process sequential data
- Regularization to prevent overfitting
- Three-class output (Buy, Hold, Sell)
- Class weighting to handle imbalanced data

### Signal Generation

The signals go through multiple processing steps:
1. Raw model predictions (0=Sell, 1=Hold, 2=Buy)
2. Position state tracking (in/out of market)
3. Minimum hold period enforcement
4. Proper BUY/SELL alternation


### Individual Strategy Descriptions

- **MACD (Moving Average Convergence Divergence)**: Uses the relationship between two moving averages of a price to identify momentum changes, trend direction, and potential reversals.

- **RSI (Relative Strength Index)**: Measures the speed and magnitude of price movements to identify overbought or oversold conditions. Generates signals when RSI crosses specific thresholds.

- **Supertrend**: Combines ATR (Average True Range) with basic price action to identify trend direction. Provides clear buy and sell signals when price crosses the Supertrend line.

These strategies provide complementary signals that the neural network learns to integrate for improved trading decisions.

## Future Improvements

Potential enhancements to the system:
- More sophisticated feature engineering
- Advanced neural network architectures
- Multi-asset portfolio management
- Risk-adjusted position sizing
- Transaction cost modeling
- Hyperparameter optimization

## License

This project is licensed under the MIT License - see the LICENSE file for details.
