import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

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
        """Generate optimized trading signals balancing frequency and quality"""
        from trading_strategies import TradeSignal, Action
        
        signals = []
        in_position = False
        min_hold_period = 1
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
        
        # Calculate technical indicators
        data['sma5'] = data['Close'].rolling(window=5).mean()
        data['sma20'] = data['Close'].rolling(window=20).mean()
        data['sma50'] = data['Close'].rolling(window=50).mean()
        
        # Price action indicators
        data['roc1'] = data['Close'].pct_change(periods=1) * 100
        data['roc5'] = data['Close'].pct_change(periods=5) * 100
        data['roc20'] = data['Close'].pct_change(periods=20) * 100
        
        # Bollinger Bands (20-day, 2 standard deviations)
        data['bb_mid'] = data['Close'].rolling(window=20).mean()
        data['bb_std'] = data['Close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_mid'] + 2 * data['bb_std']
        data['bb_lower'] = data['bb_mid'] - 2 * data['bb_std']
        data['bb_pct'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Enhanced trend and volatility metrics
        data['volatility'] = data['Close'].pct_change().rolling(window=10).std() * 100
        data['trend_strength'] = (data['Close'] - data['sma50']) / data['sma50'] * 100
        data['momentum_quality'] = data['roc5'].rolling(window=5).mean()
        
        # Volume trend
        data['volume_ma'] = data['Volume'].rolling(window=20).mean()
        data['volume_trend'] = data['Volume'] / data['volume_ma']
        
        # Fill NaN values
        data = data.fillna(method='bfill')
        
        # Quality tracking
        buy_signals_quality = []
        sell_signals_quality = []
        
        # Slightly more aggressive parameters for more trades
        quality_threshold = 45  # Lowered from 50 for more trades
        take_profit_pct = 0.03  # 3% take profit (was 2%)
        stop_loss_pct = 0.015   # 1.5% stop loss (unchanged)
        
        # Track profit targets for active trades
        profit_target = 0
        stop_level = 0
        
        # Generate signals with optimized balance
        for i in range(20, len(data)):  # Start earlier (20 vs 50) for more opportunities
            if i >= len(probs):
                break
            
            if days_since_last_trade < min_hold_period:
                days_since_last_trade += 1
                continue
            
            # Extract indicators
            close = data.iloc[i]['Close']
            sma5 = data.iloc[i]['sma5']
            sma20 = data.iloc[i]['sma20']
            sma50 = data.iloc[i]['sma50']
            roc1 = data.iloc[i]['roc1']
            roc5 = data.iloc[i]['roc5']
            roc20 = data.iloc[i]['roc20']
            vol = data.iloc[i]['volatility']
            trend = data.iloc[i]['trend_strength']
            vol_trend = data.iloc[i]['volume_trend']
            bb_pct = data.iloc[i]['bb_pct']
            
            # Model probabilities
            buy_prob = probs[i, 2]
            sell_prob = probs[i, 0]
            
            # BUY signal quality scoring
            if not in_position:
                buy_quality = 0
                
                # Model prediction (0-40 points)
                buy_quality += min(buy_prob * 100, 40)
                
                # Trend strength (0-20 points)
                if trend > 0:
                    buy_quality += min(trend * 2, 20)
                
                # Momentum analysis (0-15 points)
                if roc5 > 0:
                    buy_quality += min(roc5 * 3, 15)
                
                # Moving average alignment (0-15 points)
                if sma5 > sma20 and sma20 > sma50:
                    buy_quality += 15
                elif sma5 > sma20:
                    buy_quality += 8
                
                # Bollinger Band position (0-10 points)
                if bb_pct < 0.3:  # Near lower band - value opportunity
                    buy_quality += 10
                elif bb_pct > 0.7 and roc5 > 0:  # Strong momentum breakout
                    buy_quality += 5
                
                # Setup for larger move (0-10 points)
                if roc20 > 5 and roc5 < 2:  # Strong trend with recent consolidation
                    buy_quality += 10
                
                # Volume confirmation (0-10 points)
                if vol_trend > 1.2:  # Above average volume
                    buy_quality += 10
                
                # Generate buy signal if quality sufficient
                if buy_quality >= quality_threshold:
                    signal = TradeSignal()
                    signal.action = Action.BUY
                    signal.date = data.iloc[i]['Date'].strftime('%Y-%m-%d')
                    signal.price = close
                    signals.append(signal)
                    in_position = True
                    days_since_last_trade = 0
                    buy_signals_quality.append(buy_quality)
                    
                    # Set profit target and stop loss levels
                    profit_target = close * (1 + take_profit_pct)
                    stop_level = close * (1 - stop_loss_pct)
            
            # SELL signal quality scoring
            elif in_position:
                sell_quality = 0
                
                # Check auto-exit conditions first
                # Take profit - automatic sell with high quality
                if close >= profit_target:
                    sell_quality = 100
                
                # Stop loss - automatic sell with high quality
                elif close <= stop_level:
                    sell_quality = 100
                
                else:
                    # Model prediction (0-40 points)
                    sell_quality += min(sell_prob * 100, 40)
                    
                    # Trend reversal (0-20 points)
                    if trend < 0:
                        sell_quality += min(abs(trend) * 2, 20)
                    
                    # Momentum deterioration (0-15 points)
                    if roc5 < 0:
                        sell_quality += min(abs(roc5) * 3, 15)
                    
                    # Moving average bearish alignment (0-15 points)
                    if sma5 < sma20 and sma20 < sma50:
                        sell_quality += 15
                    elif sma5 < sma20:
                        sell_quality += 8
                    
                    # Bollinger Band position (0-10 points)
                    if bb_pct > 0.7:  # Near upper band - fading strength
                        sell_quality += 10
                    
                    # Significant price extension (0-10 points)
                    if signals and (close / signals[-1].price - 1) * 100 > 2.5:
                        sell_quality += 10
                
                # Generate sell signal if quality sufficient
                if sell_quality >= quality_threshold:
                    signal = TradeSignal()
                    signal.action = Action.SELL
                    signal.date = data.iloc[i]['Date'].strftime('%Y-%m-%d')
                    signal.price = close
                    signals.append(signal)
                    in_position = False
                    days_since_last_trade = 0
                    sell_signals_quality.append(sell_quality)
                    
                    # Calculate and log trade result
                    if len(signals) >= 2:
                        last_buy = signals[-2].price
                        pct_gain = (close / last_buy - 1) * 100
                        print(f"Trade result: {pct_gain:.2f}% gain")
        
        # Force close any open position at the end
        if in_position and len(signals) > 0:
            signal = TradeSignal()
            signal.action = Action.SELL
            signal.date = data.iloc[-1]['Date'].strftime('%Y-%m-%d')
            signal.price = data.iloc[-1]['Close']
            signals.append(signal)
            
            # Log final trade result
            if len(signals) >= 2:
                last_buy = signals[-2].price
                pct_gain = (signal.price / last_buy - 1) * 100
                print(f"Final trade result: {pct_gain:.2f}% gain")
        
        # Ensure proper alternating signals
        cleaned_signals = []
        last_action = None
        
        for signal in signals:
            if last_action is None or signal.action != last_action:
                cleaned_signals.append(signal)
                last_action = signal.action
        
        # Print signal quality stats
        if buy_signals_quality:
            avg_buy_quality = sum(buy_signals_quality) / len(buy_signals_quality)
            print(f"Average BUY signal quality: {avg_buy_quality:.2f}/100")
        if sell_signals_quality:
            avg_sell_quality = sum(sell_signals_quality) / len(sell_signals_quality)
            print(f"Average SELL signal quality: {avg_sell_quality:.2f}/100")
        
        # Print the final signals summary
        print(f"\nGenerated {len(cleaned_signals)} Trading Signals:")
        for signal in cleaned_signals:
            print(f"Date: {signal.date}, Action: {signal.action}, Price: {signal.price}")
        
        return cleaned_signals
