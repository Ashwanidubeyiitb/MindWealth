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
        """Generate high-quality trading signals with balanced metrics"""
        from trading_strategies import TradeSignal, Action
        
        signals = []
        in_position = False
        min_hold_period = 0
        days_since_last_trade = min_hold_period
        
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
        
        # Calculate technical indicators - optimized for quality trades
        data['sma5'] = data['Close'].rolling(window=5).mean()
        data['sma10'] = data['Close'].rolling(window=10).mean() 
        data['sma20'] = data['Close'].rolling(window=20).mean()
        data['sma50'] = data['Close'].rolling(window=50).mean()
        
        # Exponential moving averages with signal detection
        data['ema9'] = data['Close'].ewm(span=9, adjust=False).mean()
        data['ema21'] = data['Close'].ewm(span=21, adjust=False).mean()
        data['golden_cross'] = (data['ema9'] > data['ema21']) & (data['ema9'].shift(1) <= data['ema21'].shift(1))
        data['death_cross'] = (data['ema9'] < data['ema21']) & (data['ema9'].shift(1) >= data['ema21'].shift(1))
        
        # Momentum indicators with trend identification
        data['roc1'] = data['Close'].pct_change(periods=1) * 100
        data['roc3'] = data['Close'].pct_change(periods=3) * 100
        data['roc_trend'] = data['roc3'].rolling(window=3).mean()
        data['momentum_up'] = data['roc_trend'] > 0.5
        data['momentum_down'] = data['roc_trend'] < -0.5
        
        # Volume indicators with spikes
        data['volume_ma20'] = data['Volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_ma20']
        data['volume_spike'] = data['volume_ratio'] > 1.5
        
        # RSI indicator with zones
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_ma3'] = data['rsi'].rolling(window=3).mean()
        data['rsi_oversold'] = data['rsi'] < 35  # More selective (35 instead of 40)
        data['rsi_overbought'] = data['rsi'] > 70  # More selective (70 instead of 65)
        
        # Bollinger Bands with volatility analysis
        data['bb_mid'] = data['Close'].rolling(window=20).mean()
        data['bb_std'] = data['Close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_mid'] + 2 * data['bb_std']
        data['bb_lower'] = data['bb_mid'] - 2 * data['bb_std']
        data['bb_pct'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_mid'] * 100
        data['bb_squeeze'] = data['bb_width'] < data['bb_width'].rolling(window=50).mean() * 0.85
        data['bb_expanding'] = data['bb_width'] > data['bb_width'].shift(1)
        
        # ATR for volatility-based stops and targets
        tr1 = abs(data['High'] - data['Low'])
        tr2 = abs(data['High'] - data['Close'].shift())
        tr3 = abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()
        data['atr_pct'] = data['atr'] / data['Close'] * 100
        
        # Trend strength and reversals
        data['trend_strength'] = abs(data['Close'] - data['sma50']) / data['sma50'] * 100
        data['reversal_down'] = (data['Close'] < data['Close'].shift(1)) & (data['Close'].shift(1) < data['Close'].shift(2)) & (data['Close'].shift(2) < data['Close'].shift(3))
        data['reversal_up'] = (data['Close'] > data['Close'].shift(1)) & (data['Close'].shift(1) > data['Close'].shift(2)) & (data['Close'].shift(2) > data['Close'].shift(3))
        
        # Additional indicators
        data['price_above_ma'] = data['Close'] > data['sma10']
        data['volume_surge'] = data['Volume'] > data['volume_ma20'] * 1.3
        data['momentum_shift'] = (data['roc1'] > 0) & (data['roc1'].shift(1) < 0)
        
        # Fill NaN values
        data = data.fillna(method='bfill')
        
        # Quality tracking
        buy_signals_quality = []
        sell_signals_quality = []
        trade_results = []
        
        # Balanced parameters - optimized for quality over quantity
        quality_threshold_buy = 30       # More selective entries
        quality_threshold_sell = 25      # More selective exits
        
        # Maximum hold period - increased for more patience
        max_hold_days = 25               # Longer max hold period
        
        # Better profit targets and stops
        base_take_profit_pct = 0.052     # 5.2% target (higher)
        min_stop_loss_pct = 0.015        # 1.5% stop loss (tight)
        trailing_stop_factor = 0.70      # 70% of ATR (less tight)
        
        # Position tracking
        entry_price = 0
        profit_target = 0 
        stop_level = 0
        highest_since_entry = 0
        entry_date_idx = 0
        
        # Performance tracking
        consecutive_wins = 0
        consecutive_losses = 0
        
        # Start from later index to allow for indicator warmup
        start_idx = 40
        
        # Track days in any trade
        trade_duration = 0
        
        # Generate signals
        if len(data) > start_idx:
            for i in range(start_idx, len(data)):
                days_since_last_trade += 1
                
                # Extract data for current bar
                close = data.iloc[i]['Close']
                rsi = data.iloc[i]['rsi']
                bb_pct = data.iloc[i]['bb_pct']
                atr = data.iloc[i]['atr']
                
                # Check if we have a valid prediction for this bar
                pred_idx = i - start_idx
                if pred_idx < len(predicted_classes):
                    pred_class = predicted_classes[pred_idx]
                    pred_probs = probs[pred_idx] if len(probs) > 0 else np.array([0, 0, 0])
                    
                    buy_quality = 0
                    sell_quality = 0
                    buy_reason = []
                    sell_reason = []
                    
                    # If not in a position, look for buy signals
                    if not in_position and days_since_last_trade >= min_hold_period:
                        # Neural network prediction (0-25 points)
                        if pred_class == 1:  # Buy signal
                            buy_quality += 25 
                            buy_reason.append(f"NN prediction ({pred_probs[1]:.2f})")
                        elif pred_class == 2:  # Strong buy
                            buy_quality += 25
                            buy_reason.append(f"NN strong signal ({pred_probs[2]:.2f})")
                        
                        # RSI oversold (0-20 points)
                        if data.iloc[i]['rsi_oversold'] and rsi > data.iloc[i-1]['rsi']:
                            buy_quality += 20
                            buy_reason.append(f"RSI oversold ({rsi:.1f})")
                        
                        # Bollinger Band bounce (0-20 points)
                        if bb_pct < 0.05 and close > data.iloc[i-1]['Close']:
                            buy_quality += 20
                            buy_reason.append(f"BB bounce ({bb_pct:.2f})")
                        
                        # Golden cross (0-25 points)
                        if data.iloc[i]['golden_cross']:
                            buy_quality += 25
                            buy_reason.append("Golden cross")
                        
                        # Strong momentum (0-15 points)
                        if data.iloc[i]['momentum_up']:
                            buy_quality += 15
                            buy_reason.append("Strong momentum")
                        
                        # Price above key MA (0-10 points)
                        if data.iloc[i]['price_above_ma']:
                            buy_quality += 10
                            buy_reason.append("Above MA")
                        
                        # Trend strength (0-15 points)
                        if data.iloc[i]['trend_strength'] > 1.5:
                            # Scale points based on trend strength
                            trend_points = min(data.iloc[i]['trend_strength'] * 3, 15)
                            buy_quality += trend_points
                            buy_reason.append(f"Trend strength ({data.iloc[i]['trend_strength']:.1f}%)")
                        
                        # Reversal pattern (0-15 points)
                        if data.iloc[i]['reversal_up']:
                            buy_quality += 15
                            buy_reason.append("Reversal pattern")
                        
                        # Volume confirmation (0-15 points)
                        if data.iloc[i]['volume_surge']:
                            buy_quality += 15
                            buy_reason.append("Volume surge")
                        
                        # Adapt after losses (0-20 points)
                        if consecutive_losses > 1:
                            loss_adjustment = min(consecutive_losses * 7, 20)
                            buy_quality += loss_adjustment
                            buy_reason.append(f"Loss adaptation (+{loss_adjustment})")
                        
                        # Generate buy signal if quality threshold met
                        if buy_quality >= quality_threshold_buy:
                            signal = TradeSignal()
                            signal.action = Action.BUY
                            signal.date = data.iloc[i]['Date'].strftime('%Y-%m-%d')
                            signal.price = close
                            signals.append(signal)
                            
                            # Update state
                            in_position = True
                            entry_price = close
                            entry_date_idx = i
                            highest_since_entry = close
                            days_since_last_trade = 0
                            
                            # Calculate stop and targets
                            atr_value = data.iloc[i]['atr']
                            volatility_factor = min(1.0, data.iloc[i]['atr_pct'] / 2)  # Cap at 1.0
                            profit_target = entry_price * (1 + base_take_profit_pct * volatility_factor)
                            stop_level = entry_price * (1 - min_stop_loss_pct)
                            
                            buy_signals_quality.append(buy_quality)
                            
                            print(f"→ BUY signal at {close:.2f} (quality: {buy_quality:.1f}/100)")
                            print(f"  Reasons: {', '.join(buy_reason)}")
                            print(f"  Target: +{base_take_profit_pct*100:.1f}%, Stop: -{min_stop_loss_pct*100:.1f}%")
                    
                    # If in a position, look for exit conditions
                    elif in_position:
                        # Update tracking variables
                        highest_since_entry = max(highest_since_entry, close)
                        days_in_trade = i - entry_date_idx
                        
                        # Check if stop loss hit or maximum hold time reached
                        if close <= stop_level:
                            signal = TradeSignal()
                            signal.action = Action.SELL
                            signal.date = data.iloc[i]['Date'].strftime('%Y-%m-%d')
                            signal.price = close
                            signals.append(signal)
                            in_position = False
                            days_since_last_trade = 0
                            sell_signals_quality.append(100)  # Maximum quality for stop loss
                            
                            # Calculate and log trade result
                            pct_gain = (close / entry_price - 1) * 100
                            trade_results.append(pct_gain)
                            days_held = i - entry_date_idx
                            
                            # Update consecutive win/loss tracking
                            consecutive_losses += 1
                            consecutive_wins = 0
                            
                            print(f"← SELL signal at {close:.2f} (STOP LOSS HIT)")
                            print(f"  Result: {pct_gain:.2f}% loss, Held: {days_held} days")
                        
                        # Force exit after maximum hold period
                        elif days_in_trade >= max_hold_days:
                            signal = TradeSignal()
                            signal.action = Action.SELL
                            signal.date = data.iloc[i]['Date'].strftime('%Y-%m-%d')
                            signal.price = close
                            signals.append(signal)
                            in_position = False
                            days_since_last_trade = 0
                            sell_signals_quality.append(100)  # Maximum quality for max hold
                            
                            # Calculate and log trade result
                            pct_gain = (close / entry_price - 1) * 100
                            trade_results.append(pct_gain)
                            days_held = i - entry_date_idx
                            
                            # Update consecutive win/loss tracking
                            if pct_gain > 0:
                                consecutive_wins += 1
                                consecutive_losses = 0
                            else:
                                consecutive_losses += 1
                                consecutive_wins = 0
                            
                            print(f"← SELL signal at {close:.2f} (MAX HOLD PERIOD)")
                            print(f"  Result: {pct_gain:.2f}% gain, Held: {days_held} days (max)")
                        
                        # Check if profit target hit
                        elif close >= profit_target:
                            signal = TradeSignal()
                            signal.action = Action.SELL
                            signal.date = data.iloc[i]['Date'].strftime('%Y-%m-%d')
                            signal.price = close
                            signals.append(signal)
                            in_position = False
                            days_since_last_trade = 0
                            sell_signals_quality.append(100)  # Maximum quality for profit target
                            
                            # Calculate and log trade result
                            pct_gain = (close / entry_price - 1) * 100
                            trade_results.append(pct_gain)
                            days_held = i - entry_date_idx
                            
                            # Update consecutive win/loss tracking
                            consecutive_wins += 1
                            consecutive_losses = 0
                            
                            print(f"← SELL signal at {close:.2f} (TARGET REACHED)")
                            print(f"  Result: {pct_gain:.2f}% gain, Held: {days_held} days")
                        
                        # Check for trailing stop (after 2.5% profit)
                        elif (highest_since_entry / entry_price > 1.025 and 
                              close <= highest_since_entry * (1 - trailing_stop_factor * atr / highest_since_entry)):
                            signal = TradeSignal()
                            signal.action = Action.SELL
                            signal.date = data.iloc[i]['Date'].strftime('%Y-%m-%d')
                            signal.price = close
                            signals.append(signal)
                            in_position = False
                            days_since_last_trade = 0
                            sell_signals_quality.append(100)  # Maximum quality for trailing stop
                            
                            # Calculate and log trade result
                            pct_gain = (close / entry_price - 1) * 100
                            trade_results.append(pct_gain)
                            days_held = i - entry_date_idx
                            
                            # Update consecutive win/loss tracking
                            if pct_gain > 0:
                                consecutive_wins += 1
                                consecutive_losses = 0
                            else:
                                consecutive_losses += 1
                                consecutive_wins = 0
                            
                            print(f"← SELL signal at {close:.2f} (TRAILING STOP)")
                            print(f"  Result: {pct_gain:.2f}% gain, Held: {days_held} days")
                            print(f"  High: {highest_since_entry:.2f} ({(highest_since_entry/entry_price-1)*100:.2f}%)")
                        
                        # Consider other exit signals
                        else:
                            current_profit_pct = (close / entry_price - 1) * 100
                            
                            # 1. Death cross (0-25 points)
                            if data.iloc[i]['death_cross']:
                                sell_quality += 25
                                sell_reason.append("Death cross")
                            
                            # 2. RSI overbought reversal (0-20 points)
                            if data.iloc[i]['rsi_overbought'] and rsi < data.iloc[i-1]['rsi']:
                                sell_quality += 20
                                sell_reason.append(f"RSI reversal ({rsi:.1f})")
                            
                            # 3. Momentum exhaustion (0-20 points)
                            if (data.iloc[i]['roc1'] < 0 and 
                                data.iloc[i-1]['roc1'] < 0 and
                                data.iloc[i-2]['roc1'] < 0):  # Now require 3 days negative
                                sell_quality += 20
                                sell_reason.append("Momentum exhaustion")
                            
                            # 4. Moving average breakdown (0-15 points)
                            if close < data.iloc[i]['sma10'] and current_profit_pct > 1.0:
                                sell_quality += 15
                                sell_reason.append("MA breakdown")
                            
                            # 5. Upper Bollinger Band rejection (0-15 points)
                            if bb_pct > 0.95 and close < data.iloc[i-1]['Close']:
                                sell_quality += 15
                                sell_reason.append(f"BB rejection ({bb_pct:.2f})")
                            
                            # 6. Profit protection - more significant profits
                            if current_profit_pct > 3.0:
                                profit_protection = min(current_profit_pct * 5, 25)
                                sell_quality += profit_protection
                                sell_reason.append(f"Profit protection (+{current_profit_pct:.1f}%)")
                            
                            # 7. Time-based exit - only with profit
                            if days_in_trade > 12 and current_profit_pct > 1.5:
                                time_points = min(days_in_trade, 20)
                                sell_quality += time_points
                                sell_reason.append(f"Time exit ({days_in_trade} days)")
                            
                            # 8. Major trend reversal (0-20 points)
                            if data.iloc[i]['reversal_down'] and current_profit_pct > 1.0:
                                sell_quality += 20
                                sell_reason.append("Reversal pattern")
                            
                            # Generate sell signal if quality threshold met
                            if sell_quality >= quality_threshold_sell:
                                signal = TradeSignal()
                                signal.action = Action.SELL
                                signal.date = data.iloc[i]['Date'].strftime('%Y-%m-%d')
                                signal.price = close
                                signals.append(signal)
                                in_position = False
                                days_since_last_trade = 0
                                sell_signals_quality.append(sell_quality)
                                
                                # Calculate and log trade result
                                pct_gain = (close / entry_price - 1) * 100
                                trade_results.append(pct_gain)
                                days_held = i - entry_date_idx
                                
                                # Update consecutive win/loss tracking
                                if pct_gain > 0:
                                    consecutive_wins += 1
                                    consecutive_losses = 0
                                else:
                                    consecutive_losses += 1
                                    consecutive_wins = 0
                                
                                print(f"← SELL signal at {close:.2f} (quality: {sell_quality:.1f}/100)")
                                print(f"  Reasons: {', '.join(sell_reason)}")
                                print(f"  Result: {pct_gain:.2f}% gain, Held: {days_held} days")
                
                # Position tracking logs - only show every 5 days to reduce clutter
                if in_position and entry_price > 0 and i % 5 == 0:
                    current_gain = (close / entry_price - 1) * 100
                    days_held = i - entry_date_idx
                    print(f"  Position open: Day {days_held}, P&L: {current_gain:.2f}%, "
                          f"Stop: {(stop_level/entry_price-1)*100:.2f}%")
        
        # Force close any open position at the end
        if in_position and entry_price > 0:
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
