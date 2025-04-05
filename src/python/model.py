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
        
        # Simpler model architecture (similar to original)
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(16, return_sequences=False,
                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(0.2),
            
            Dense(8, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(0.2),
            
            Dense(3, activation='softmax')
        ])
        
        # Use a lower learning rate for stability
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
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
        """Generate trade signals from model predictions with balanced approach"""
        from trading_strategies import TradeSignal, Action
        
        signals = []
        in_position = False  # Initialize as not in position
        min_hold_period = 2  # Short but reasonable minimum hold period
        days_since_last_trade = min_hold_period  # Start ready to trade
        
        # If predictions is already class indices, no need to take argmax
        if len(predictions.shape) == 1:
            predicted_classes = predictions
        else:
            # If we have probability distributions, get the class with highest probability
            predicted_classes = np.argmax(predictions, axis=1)
        
        # Print prediction distribution to understand what's happening
        unique, counts = np.unique(predicted_classes, return_counts=True)
        print("\nPrediction distribution:", dict(zip(unique, counts)))
        
        # Moderately aggressive thresholds for signal generation
        buy_threshold = 0.25  # Moderately low threshold
        sell_threshold = 0.25  # Moderately low threshold
        
        # Track consecutive days with same prediction to filter noise
        buy_count = 0
        sell_count = 0
        
        # Start with looking for a BUY signal first
        for i, pred in enumerate(predicted_classes):
            if days_since_last_trade < min_hold_period:
                days_since_last_trade += 1
                continue
            
            # Check for strong buy signals
            if pred == 2 and not in_position:
                buy_count += 1
                sell_count = 0
                # Generate signal on strong buy prediction or consecutive buy days
                if buy_count >= 1 or (len(predictions.shape) > 1 and predictions[i][2] > buy_threshold):
                    signal = TradeSignal()
                    signal.action = Action.BUY
                    signal.date = data.iloc[i]['Date'].strftime('%Y-%m-%d')
                    signal.price = data.iloc[i]['Close']
                    signals.append(signal)
                    in_position = True
                    days_since_last_trade = 0
                    buy_count = 0
            
            # Check for strong sell signals
            elif pred == 0 and in_position:
                sell_count += 1
                buy_count = 0
                # Generate signal on strong sell prediction or consecutive sell days
                if sell_count >= 1 or (len(predictions.shape) > 1 and predictions[i][0] > sell_threshold):
                    signal = TradeSignal()
                    signal.action = Action.SELL
                    signal.date = data.iloc[i]['Date'].strftime('%Y-%m-%d')
                    signal.price = data.iloc[i]['Close']
                    signals.append(signal)
                    in_position = False
                    days_since_last_trade = 0
                    sell_count = 0
            else:
                # Reset counters when we don't have a consistent prediction
                buy_count = 0
                sell_count = 0
        
        # Force close any open position at the end
        if in_position:
            signal = TradeSignal()
            signal.action = Action.SELL
            signal.date = data.iloc[-1]['Date'].strftime('%Y-%m-%d')
            signal.price = data.iloc[-1]['Close']
            signals.append(signal)
        
        # Make sure we have proper alternating BUY/SELL signals
        cleaned_signals = []
        last_action = None
        
        for signal in signals:
            if last_action is None or signal.action != last_action:
                cleaned_signals.append(signal)
                last_action = signal.action
        
        # Print the final signals for debugging
        print("\nGenerated Signals:")
        for signal in cleaned_signals:
            print(f"Date: {signal.date}, Action: {signal.action}, Price: {signal.price}")
        
        return cleaned_signals
