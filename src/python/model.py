import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
from trading_strategies import TradeSignal, Action

class TradingModel:
    def __init__(self, input_shape, model_path=None):
        self.input_shape = input_shape
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            self.model = self._build_model()
    
    def _build_model(self):
        """Build and compile the neural network model"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(128, input_shape=self.input_shape, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(64, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Dense(16, activation='relu'))
        
        # Output layer (3 classes: buy, sell, hold)
        model.add(Dense(3, activation='softmax'))
        
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(y_train + 1, num_classes=3)
        y_val_cat = tf.keras.utils.to_categorical(y_val + 1, num_classes=3)
        
        # Create callbacks
        checkpoint = ModelCheckpoint(
            self.model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping]
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        # Convert from one-hot encoding to class labels (-1, 0, 1)
        return np.argmax(predictions, axis=1) - 1
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_cat = tf.keras.utils.to_categorical(y + 1, num_classes=3)
        loss, accuracy = self.model.evaluate(X, y_cat)
        return loss, accuracy
    
    def generate_signals(self, data, predictions):
        """Generate NN signals using predictions with high-frequency tuning."""
        # Create a copy of the data to avoid SettingWithCopyWarning
        data_copy = data.copy()

        # --- Ensure necessary indicators are calculated ---
        indicators_needed = ['atr', 'sma5', 'sma20', 'rsi']
        for indicator in indicators_needed:
             if indicator not in data_copy.columns:
                 print(f"Calculating missing indicator: {indicator}")
                 try:
                     if indicator == 'atr':
                         high_low = data_copy['High'] - data_copy['Low']
                         high_close = (data_copy['High'] - data_copy['Close'].shift()).abs()
                         low_close = (data_copy['Low'] - data_copy['Close'].shift()).abs()
                         tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                         data_copy.loc[:, 'atr'] = tr.rolling(window=14).mean()
                         data_copy['atr'] = data_copy['atr'].bfill().fillna(data_copy['Close'] * 0.015)
                     elif indicator == 'sma5':
                         data_copy.loc[:, 'sma5'] = data_copy['Close'].rolling(window=5).mean().bfill()
                     elif indicator == 'sma20':
                         data_copy.loc[:, 'sma20'] = data_copy['Close'].rolling(window=20).mean().bfill()
                     elif indicator == 'rsi':
                         # Assuming calculate_rsi function exists or implement basic RSI here
                         delta = data_copy['Close'].diff()
                         gain = delta.where(delta > 0, 0).fillna(0)
                         loss = -delta.where(delta < 0, 0).fillna(0)
                         avg_gain = gain.rolling(window=14).mean()
                         avg_loss = loss.rolling(window=14).mean()
                         rs = avg_gain / avg_loss.replace(0, 1e-10)
                         data_copy.loc[:, 'rsi'] = (100 - (100 / (1 + rs))).bfill().fillna(50)
                 except Exception as e:
                     print(f"Error calculating {indicator}: {e}. Using fallback.")
                     if indicator == 'atr': data_copy.loc[:, 'atr'] = data_copy['Close'] * 0.015
                     else: data_copy.loc[:, indicator] = data_copy['Close'] # Simple fallback
        # --- End Indicator Calculation ---

        signals = []
        in_position = False
        entry_price = 0.0
        entry_idx = -1
        last_exit_idx = -2 # Allow immediate first trade

        # --- Hyperparameters for High Frequency & Better Return ---
        buy_pred_threshold = 1  # Require explicit BUY prediction from model
        sell_pred_threshold = -1 # Require explicit SELL prediction from model
        
        atr_tp_multiplier = 2.5 # Target 2.5x ATR profit
        atr_sl_multiplier = 1.2 # Stop loss at 1.2x ATR
        
        min_hold_days = 1       # Minimum days to hold
        max_hold_days = 6       # Maximum days to hold
        cooldown_period = 1     # Only 1 day cooldown after exit
        # --- End Hyperparameters ---

        # Adjust index for predictions (assuming predictions align with data starting from seq_length)
        seq_length = self.input_shape[0] # Get sequence length from model input shape
        prediction_offset = seq_length

        print(f"NN Signal Generation: Using prediction offset {prediction_offset}")

        for i in range(prediction_offset, len(data_copy)):
            # Get prediction for the *current* day i (prediction index i - offset)
            pred_idx = i - prediction_offset
            if pred_idx < 0 or pred_idx >= len(predictions):
                # print(f"Skipping index {i}: Prediction index {pred_idx} out of bounds (len={len(predictions)})")
                continue # Skip if no prediction available for this day

            pred = predictions[pred_idx]
            current_price = data_copy.iloc[i]['Close']
            current_atr = data_copy.iloc[i]['atr']
            current_date_str = data_copy.iloc[i]['Date'].strftime('%Y-%m-%d')

            # Handle potential NaN ATR
            if pd.isna(current_atr) or current_atr <= 0:
                current_atr = current_price * 0.015 # Fallback

            # --- Entry Logic ---
            if not in_position and i > last_exit_idx + cooldown_period:
                # Condition 1: Model predicts BUY
                model_buy = (pred == buy_pred_threshold)
                # Condition 2: Basic momentum/trend filter (optional, can be tuned)
                momentum_ok = data_copy.iloc[i]['sma5'] >= data_copy.iloc[i]['sma20'] * 0.99 # Price near or above short MA vs long MA
                rsi_ok = data_copy.iloc[i]['rsi'] < 75 # Not extremely overbought

                if model_buy and momentum_ok and rsi_ok:
                    signal = TradeSignal()
                    signal.action = Action.BUY
                    signal.date = current_date_str
                    signal.price = current_price
                    signals.append(signal)

                    in_position = True
                    entry_price = current_price
                    entry_idx = i
                    print(f"NN BUY -> {signal.date} @ ${signal.price:.2f} (Pred: {pred}, RSI: {data_copy.iloc[i]['rsi']:.1f})")
            # --- End Entry Logic ---

            # --- Exit Logic ---
            elif in_position:
                days_held = i - entry_idx
                current_gain_pct = (current_price - entry_price) / entry_price * 100
                
                # Define TP/SL prices based on entry ATR
                entry_atr = data_copy.iloc[entry_idx]['atr']
                if pd.isna(entry_atr) or entry_atr <= 0: entry_atr = entry_price * 0.015
                take_profit_price = entry_price + (entry_atr * atr_tp_multiplier)
                stop_loss_price = entry_price - (entry_atr * atr_sl_multiplier)

                # Exit Conditions (Order matters)
                exit_signal = False
                exit_reason = ""

                # 1. Stop Loss Hit
                if current_price <= stop_loss_price:
                    exit_signal = True
                    exit_reason = f"STOP LOSS ({current_gain_pct:.2f}%)"
                # 2. Take Profit Hit
                elif current_price >= take_profit_price:
                    exit_signal = True
                    exit_reason = f"TAKE PROFIT ({current_gain_pct:.2f}%)"
                # 3. Model Predicts SELL (after min hold)
                elif pred == sell_pred_threshold and days_held >= min_hold_days:
                    exit_signal = True
                    exit_reason = f"MODEL SELL ({current_gain_pct:.2f}%)"
                # 4. Max Hold Days Reached
                elif days_held >= max_hold_days:
                    exit_signal = True
                    exit_reason = f"MAX HOLD ({current_gain_pct:.2f}%)"

                if exit_signal:
                    signal = TradeSignal()
                    signal.action = Action.SELL
                    signal.date = current_date_str
                    signal.price = current_price
                    signals.append(signal)

                    in_position = False
                    last_exit_idx = i
                    print(f"NN SELL <- {signal.date} @ ${signal.price:.2f} ({exit_reason})")
            # --- End Exit Logic ---

        # If still in position at the end, force close
        if in_position:
             final_idx = len(data_copy) - 1
             final_price = data_copy.iloc[final_idx]['Close']
             final_date_str = data_copy.iloc[final_idx]['Date'].strftime('%Y-%m-%d')
             current_gain_pct = (final_price - entry_price) / entry_price * 100

             signal = TradeSignal()
             signal.action = Action.SELL
             signal.date = final_date_str
             signal.price = final_price
             signals.append(signal)
             print(f"NN SELL <- {signal.date} @ ${signal.price:.2f} (End of Data, Gain: {current_gain_pct:.2f}%)")


        print(f"Generated {len(signals)//2} neural network trades using prediction-based logic.")
        return signals
