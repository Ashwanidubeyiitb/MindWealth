import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import pandas as pd

class TradingModel:
    def __init__(self, input_shape, model_path='models/nn_model_weights.keras'):
        # Update file extension for weights
        self.model_path = model_path.replace('.keras', '.weights.h5')
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Enhanced model architecture for better pattern recognition
        self.model = Sequential([
            Input(shape=input_shape),
            
            # First LSTM layer - capture short-term patterns
            LSTM(32, return_sequences=True, 
                 kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Second LSTM layer - integrate patterns
            LSTM(16, return_sequences=False,
                 kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers for decision making
            Dense(16, activation='relu', 
                  kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            Dropout(0.2),
            
            Dense(3, activation='softmax')
        ])
        
        # Use slightly higher learning rate for faster convergence
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        # Calculate class weights to handle imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        total = np.sum(counts)
        class_weights = {i: total / (len(unique) * count) for i, count in zip(unique, counts)}
        print("Class weights:", class_weights)
        
        # Define callbacks
        checkpoint = ModelCheckpoint(
            filepath=self.model_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=True,
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[checkpoint, early_stopping],
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=1)
    
    def predict(self, X):
        return self.model.predict(X, verbose=1)
    
    def load_weights(self):
        """Load model weights if they exist"""
        try:
            self.model.load_weights(self.model_path)
            print(f"Successfully loaded weights from {self.model_path}")
            return True
        except:
            print(f"Could not load weights from {self.model_path}")
            return False
    
    def generate_signals(self, data, predictions):
        """Generate high-quality trading signals with better risk/reward"""
        from trading_strategies import TradeSignal, Action
        
        signals = []
        in_position = False
        
        # Process predictions
        if len(predictions.shape) == 1:
            predicted_classes = predictions
            probs = np.zeros((len(predictions), 3))
            for i, p in enumerate(predicted_classes):
                probs[i, p] = 1.0
        else:
            probs = predictions
            predicted_classes = np.argmax(predictions, axis=1)
        
        # Print prediction distribution
        unique, counts = np.unique(predicted_classes, return_counts=True)
        print("\nPrediction distribution:", dict(zip(unique, counts)))
        
        # Calculate technical indicators - using proven reliable indicators
        data['sma5'] = data['Close'].rolling(window=5).mean()
        data['sma20'] = data['Close'].rolling(window=20).mean()
        data['sma50'] = data['Close'].rolling(window=50).mean()
        
        # Bollinger Bands (reliable volatility indicator)
        data['bb_mid'] = data['Close'].rolling(window=20).mean()
        data['bb_std'] = data['Close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_mid'] + 2 * data['bb_std']
        data['bb_lower'] = data['bb_mid'] - 2 * data['bb_std']
        data['bb_pct'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # MACD (reliable trend indicator)
        data['ema12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['ema26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['macd'] = data['ema12'] - data['ema26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        data['macd_above_signal'] = data['macd'] > data['macd_signal']
        data['macd_rising'] = data['macd'] > data['macd'].shift(1)
        
        # RSI (reliable overbought/oversold indicator)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR for stops and targets
        tr1 = abs(data['High'] - data['Low'])
        tr2 = abs(data['High'] - data['Close'].shift())
        tr3 = abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()
        
        # Additional useful indicators
        data['price_change'] = data['Close'].pct_change() * 100
        data['volume_change'] = data['Volume'].pct_change() * 100
        data['higher_high'] = (data['High'] > data['High'].shift(1)) & (data['High'].shift(1) > data['High'].shift(2))
        data['lower_low'] = (data['Low'] < data['Low'].shift(1)) & (data['Low'].shift(1) < data['Low'].shift(2))
        
        # Fill NaN values
        data = data.fillna(method='bfill')
        
        # Trade tracking
        trade_results = []
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        trailing_stop = 0
        highest_since_entry = 0
        
        # Start from a position with enough history for indicators
        start_idx = 50
        
        # Improved strategy parameters for HIGHER RETURN per trade
        stop_loss_atr_multiplier = 1.8     # Wider stop to avoid premature exits (1.8 ATR)
        take_profit_atr_multiplier = 5.0   # Much higher profit targets (5.0 ATR)
        trailing_stop_active_pct = 2.0     # Only trail after 2% profit (more room to run)
        trailing_stop_distance_pct = 1.8   # Wider trailing distance (let winners run)
        
        for i in range(start_idx, len(data)):
            close = data.iloc[i]['Close']
            atr = data.iloc[i]['atr']
            rsi = data.iloc[i]['rsi']
            bb_pct = data.iloc[i]['bb_pct']
            pred_class = predicted_classes[i]
            
            # Check if we have an open position
            if not in_position:
                # Look for buy signal
                buy_signal = False
                buy_reason = []
                
                # 1. RSI oversold and rising (more selective)
                rsi_buy = rsi < 32 and rsi > data.iloc[i-1]['rsi']  # Stricter RSI (32 vs 38)
                if rsi_buy:
                    buy_signal = True
                    buy_reason.append(f"RSI oversold ({rsi:.1f})")
                
                # 2. Price near lower Bollinger Band with momentum (more selective)
                bb_buy = bb_pct < 0.15 and close > data.iloc[i-1]['Close']  # Stricter BB (0.15 vs 0.2)
                if bb_buy:
                    buy_signal = True
                    buy_reason.append(f"BB bounce ({bb_pct:.2f})")
                
                # 3. MACD bullish signals (more sensitive to entry)
                macd_buy = (data.iloc[i]['macd_rising'] and 
                           data.iloc[i]['macd'] < 0 and  # Catch early trend reversals
                           data.iloc[i]['macd'] > data.iloc[i-1]['macd'])
                if macd_buy:
                    buy_signal = True
                    buy_reason.append("MACD bullish")
                
                # 4. Trend alignment (simplified for more trades)
                trend_buy = (data.iloc[i]['sma5'] > data.iloc[i-1]['sma5'])
                if trend_buy:
                    buy_signal = True
                    buy_reason.append("Strong trend")
                
                # 5. AI prediction (more sensitive)
                ai_buy = (pred_class == 1) or (probs[i][1] > 0.4)  # More sensitive (0.4 vs 0.6)
                if ai_buy:
                    buy_signal = True
                    buy_reason.append(f"AI prediction ({probs[i][1]:.2f})")
                
                # 6. Volume confirmation (less stringent)
                volume_buy = data.iloc[i]['volume_change'] > 10  # Less strict (10% vs 15%)
                if volume_buy and buy_signal:
                    buy_reason.append("Volume confirmation")
                
                # Require at least 2 reasons to enter (reduce number of trades)
                min_reasons_required = 2
                if len(buy_reason) < min_reasons_required:
                    buy_signal = False
                
                # Execute buy if we have valid reasons (only require 1 reason)
                if buy_signal:  # Removed requirement for 2 reasons
                    signal = TradeSignal()
                    signal.action = Action.BUY
                    signal.date = data.iloc[i]['Date'].strftime('%Y-%m-%d')
                    signal.price = close
                    signals.append(signal)
                    in_position = True
                    entry_price = close
                    highest_since_entry = close
                    
                    # Set stop loss and take profit
                    stop_loss = entry_price * (1 - stop_loss_atr_multiplier * atr / close)
                    take_profit = entry_price * (1 + take_profit_atr_multiplier * atr / close)
                    trailing_stop = 0  # Initialize trailing stop (not active yet)
                    entry_index = i  # Store the entry index for tracking days in trade
                    
                    print(f"→ BUY signal at {close:.2f}")
                    print(f"  Reasons: {', '.join(buy_reason)}")
                    print(f"  Target: +{(take_profit/entry_price-1)*100:.2f}%, Stop: {(stop_loss/entry_price-1)*100:.2f}%")
            
            else:  # We have an open position
                # Track the current gain/loss
                current_return = (close / entry_price - 1) * 100
                
                # Update highest price since entry for trailing stop
                if close > highest_since_entry:
                    highest_since_entry = close
                    
                    # Update trailing stop if profit threshold reached
                    if current_return >= trailing_stop_active_pct and trailing_stop < stop_loss:
                        # Set trailing stop to lock in profits
                        trailing_stop = highest_since_entry * (1 - trailing_stop_distance_pct/100)
                        # Never move trailing stop below entry after 3% gain
                        if current_return > 3.0 and trailing_stop < entry_price:
                            trailing_stop = entry_price
                
                # Check for exit conditions
                sell_signal = False
                sell_reason = []
                
                # 1. Stop loss hit
                if close <= stop_loss:
                    sell_signal = True
                    sell_reason.append("STOP LOSS")
                
                # 2. Trailing stop hit (only if active)
                elif trailing_stop > 0 and close <= trailing_stop:
                    sell_signal = True
                    sell_reason.append("TRAILING STOP")
                
                # 3. Take profit hit (only at full target, not partial)
                elif close >= take_profit:
                    sell_signal = True
                    sell_reason.append("TARGET REACHED")
                
                # 4. RSI overbought with price weakness (stricter)
                elif rsi > 78 and close < data.iloc[i-1]['Close'] and current_return > 1.5:
                    sell_signal = True
                    sell_reason.append(f"RSI overbought ({rsi:.1f})")
                
                # 5. Price near upper Bollinger Band with reversal (stricter)
                elif bb_pct > 0.92 and close < data.iloc[i-1]['Close'] and current_return > 1.8:
                    sell_signal = True
                    sell_reason.append(f"BB rejection ({bb_pct:.2f})")
                
                # 6. MACD bearish crossover with good profit
                elif (data.iloc[i]['macd'] < data.iloc[i]['macd_signal'] and 
                     data.iloc[i-1]['macd'] >= data.iloc[i-1]['macd_signal'] and
                     current_return > 1.5):  # Higher threshold (1.5% vs 0.8%)
                    sell_signal = True
                    sell_reason.append("MACD crossover")
                
                # Hold positions longer (reduce time-based exits)
                days_in_trade = i - entry_index if 'entry_index' in locals() else 0
                if days_in_trade > 20 and current_return > 1.2:  # Longer hold time (20 vs 12)
                    sell_signal = True
                    sell_reason.append(f"Time exit ({days_in_trade} days)")
                
                # Execute sell if we have a valid reason
                if sell_signal:
                    signal = TradeSignal()
                    signal.action = Action.SELL
                    signal.date = data.iloc[i]['Date'].strftime('%Y-%m-%d')
                    signal.price = close
                    signals.append(signal)
                    in_position = False
                    
                    # Calculate trade result
                    pct_gain = (close / entry_price - 1) * 100
                    trade_results.append(pct_gain)
                    
                    print(f"← SELL signal at {close:.2f}")
                    print(f"  Reasons: {', '.join(sell_reason)}")
                    print(f"  Result: {pct_gain:.2f}% gain")
                    if highest_since_entry > entry_price:
                        max_gain = (highest_since_entry / entry_price - 1) * 100
                        print(f"  High: {highest_since_entry:.2f} ({max_gain:.2f}%)")
                
                # Position tracking logs every 5 days (moved inside the else block)
                elif i % 5 == 0:
                    ts_info = f", Trail: {(trailing_stop/entry_price-1)*100:.2f}%" if trailing_stop > 0 else ""
                    print(f"  Position open, P&L: {current_return:.2f}%, "
                          f"Stop: {(stop_loss/entry_price-1)*100:.2f}%, "
                          f"Target: {(take_profit/entry_price-1)*100:.2f}%{ts_info}")
        
        # Force close any open position at the end
        if in_position:
            signal = TradeSignal()
            signal.action = Action.SELL
            signal.date = data.iloc[-1]['Date'].strftime('%Y-%m-%d')
            signal.price = data.iloc[-1]['Close']
            signals.append(signal)
            
            # Log final trade result
            pct_gain = (signal.price / entry_price - 1) * 100
            trade_results.append(pct_gain)
            print(f"\nFinal position closed: {pct_gain:.2f}% gain")
        
        # Ensure proper alternating signals
        cleaned_signals = []
        last_action = None
        
        for signal in signals:
            if last_action is None or signal.action != last_action:
                cleaned_signals.append(signal)
                last_action = signal.action
        
        # Print performance stats
        if trade_results:
            avg_return = sum(trade_results) / len(trade_results)
            win_rate = len([r for r in trade_results if r > 0]) / len(trade_results) * 100
            max_return = max(trade_results) if trade_results else 0
            min_return = min(trade_results) if trade_results else 0
            
            print(f"\n===== TRADING PERFORMANCE =====")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Average Return: {avg_return:.2f}% per trade")
            print(f"Best Trade: +{max_return:.2f}%, Worst Trade: {min_return:.2f}%")
            print(f"Total Trades: {len(trade_results)}")
            print(f"Winners: {len([r for r in trade_results if r > 0])}, Losers: {len([r for r in trade_results if r <= 0])}")
            
            # Calculate expectancy
            if len(trade_results) > 0:
                avg_win = sum([r for r in trade_results if r > 0]) / len([r for r in trade_results if r > 0]) if len([r for r in trade_results if r > 0]) > 0 else 0
                avg_loss = sum([r for r in trade_results if r <= 0]) / len([r for r in trade_results if r <= 0]) if len([r for r in trade_results if r <= 0]) > 0 else 0
                expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)
                print(f"Average Win: +{avg_win:.2f}%, Average Loss: {avg_loss:.2f}%")
                print(f"Expectancy: {expectancy:.2f}% per trade")
        
        # Print the final signals summary
        print(f"\nGenerated {len(cleaned_signals)} Trading Signals")
        
        return cleaned_signals
