import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.regularizers import l2
import talib
from datetime import datetime, timedelta
import pytz
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import TimeSeriesSplit
import os
import joblib

# Simulation dates
current_date = datetime.now()
training_start = current_date - timedelta(days=60)
training_end = current_date - timedelta(days=10)
validation_end = current_date - timedelta(days=1)

# Calculate the shift needed for historical data
days_to_shift = (datetime(2024, 8, 30).date() - current_date.date()).days

def get_data(start_date, end_date):
    # Use current date if end_date is in the future
    real_current_date = datetime.now().date()
    if end_date.date() > real_current_date:
        end_date = real_current_date
    
    # Ensure start_date is not in the future and is within 50 days of end_date
    if start_date.date() > real_current_date:
        start_date = end_date - timedelta(days=50)
    elif (end_date - start_date).days > 50:
        start_date = end_date - timedelta(days=50)

    tickers = ['ES=F']
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval='5m')
            if not data.empty:
                print(f"Successfully downloaded data for {ticker}", data, len(data))
                
                # Convert index to EST
                data.index = data.index.tz_convert('America/New_York')

                print("Historical data1:", data, len(data))
                # Filter for market hours (9:30 AM to 5:00 PM EST)
                data = data.between_time('09:30', '17:00')
                # Simulate future dates by shifting the index
                data.index = data.index + timedelta(days=days_to_shift)
                print("Historical data4:", data, len(data))
                
                return data
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")
    
    raise ValueError("Unable to download data for any of the attempted tickers")

class QuantNeuralNet:
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.model = self._build_model()
        self.scaler = StandardScaler()

    def _build_model(self):
        reg = l2(0.2)
        
        price_input = Input(shape=(self.lookback, 5))
        x1 = LSTM(16, kernel_regularizer=reg, recurrent_regularizer=reg)(price_input)

        tech_input = Input(shape=(self.lookback, 7))  # Increased to 7 for Lorentzian
        x2 = LSTM(16, kernel_regularizer=reg, recurrent_regularizer=reg)(tech_input)

        merged = Concatenate()([x1, x2])
        merged = Dense(16, activation='relu', kernel_regularizer=reg)(merged)
        merged = Dropout(0.5)(merged)
        output = Dense(3, activation='softmax')(merged)

        model = Model(inputs=[price_input, tech_input], outputs=output)
        
        def custom_loss(y_true, y_pred):
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
            return categorical_crossentropy(y_true, y_pred)
        
        optimizer = Adam(learning_rate=0.00000315, clipvalue=0.5)
        
        model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])
        return model

    def lorentzian_classification(self, close, high, low, lookback=20):
        tr = np.maximum(high - low, 
                        np.maximum(np.abs(high - close.shift(1)), 
                                   np.abs(low - close.shift(1))))
        
        dist = np.log(1 + tr)
        
        gauss_weights = np.exp(-np.arange(lookback)**2 / (2 * (lookback/4)**2))
        gauss_weights /= gauss_weights.sum()
        
        lorentzian = pd.Series(np.convolve(dist, gauss_weights, mode='valid'), 
                               index=close.index[lookback-1:])
        
        lorentzian = 100 * (lorentzian - lorentzian.rolling(window=lookback).min()) / \
                     (lorentzian.rolling(window=lookback).max() - lorentzian.rolling(window=lookback).min())
        
        return lorentzian

    def prepare_data(self, data):
        data = data.copy()

        price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        data['MA5'] = talib.SMA(data['Close'], timeperiod=5)
        data['MA20'] = talib.SMA(data['Close'], timeperiod=20)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data['MACD'], data['MACD_Signal'], _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = talib.BBANDS(data['Close'], timeperiod=20)
        data['Lorentzian'] = self.lorentzian_classification(data['Close'], data['High'], data['Low'])
        
        tech_features = ['MA5', 'MA20', 'RSI', 'MACD', 'ATR', 'BB_Middle', 'Lorentzian']

        data = data.fillna(method='ffill').fillna(method='bfill')

        X_price = data[price_features].values
        X_tech = data[tech_features].values
        
        X_price_scaled = self.scaler.fit_transform(X_price)
        X_tech_scaled = self.scaler.fit_transform(X_tech)
        
        X_price_sequences = []
        X_tech_sequences = []
        for i in range(len(X_price_scaled) - self.lookback):
            X_price_sequences.append(X_price_scaled[i:i+self.lookback])
            X_tech_sequences.append(X_tech_scaled[i:i+self.lookback])

        future_returns = data['Close'].pct_change(5).shift(-5)
        y = np.where(future_returns > 0.01, 2, np.where(future_returns < -0.01, 0, 1))[self.lookback:]
        y = tf.keras.utils.to_categorical(y, num_classes=3)

        return [np.array(X_price_sequences), np.array(X_tech_sequences)], y

    def train(self, train_data, validation_data):
        X_train, y_train = self.prepare_data(train_data)
        X_val, y_val = self.prepare_data(validation_data)

        early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=300,
            batch_size=64,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history

    def cross_validate(self, data, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        histories = []
        for train_index, val_index in tscv.split(data):
            train_data = data.iloc[train_index]
            val_data = data.iloc[val_index]
            model = self._build_model()
            X_train, y_train = self.prepare_data(train_data)
            X_val, y_val = self.prepare_data(val_data)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
                verbose=0
            )
            histories.append(history)
        return histories

    def predict(self, data):
        X, _ = self.prepare_data(data)
        return self.model.predict(X)

    def paper_trade(self, live_data):
        live_data = live_data.copy()
        
        predictions = self.predict(live_data)
        
        live_data = live_data.iloc[self.lookback:]
        predictions = predictions[-len(live_data):]

        
        live_data['sell_prob'] = predictions[:, 0]
        live_data['hold_prob'] = predictions[:, 1]
        live_data['buy_prob'] = predictions[:, 2]
        
        buy_threshold = 0.4
        sell_threshold = 0.4
        
        position = 0
        trades = []
        for i, row in live_data.iterrows():
            current_price = row['Close']
            buy_prob = row['buy_prob']
            sell_prob = row['sell_prob']
            
            print(f"Time: {i}, Price: {current_price:.2f}, Buy Prob: {buy_prob:.2f}, Sell Prob: {sell_prob:.2f}, Position: {position}")
            
            if buy_prob > buy_threshold and position <= 0:
                position = 1
                trades.append(('BUY', i, current_price))
                print(f"BUY signal generated at {i}")
            elif sell_prob > sell_threshold and position >= 0:
                position = -1
                trades.append(('SELL', i, current_price))
                print(f"SELL signal generated at {i}")
        
        print("\nPrediction distribution:")
        print(live_data[['sell_prob', 'hold_prob', 'buy_prob']].mean())
        
        print(f"\nTotal trading opportunities: {len(trades)}")
        
        return trades

    def save_model(self, model_path, scaler_path):
        """Save the trained model and scaler."""
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

    @classmethod
    def load_model(cls, model_path, scaler_path):
        """Load a trained model and scaler."""
        instance = cls()
        instance.model = load_model(model_path, custom_objects={'custom_loss': instance.custom_loss})
        instance.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")
        return instance

    @staticmethod
    def custom_loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
        return categorical_crossentropy(y_true, y_pred)

if __name__ == "__main__":
    try:
        historical_data = get_data(training_start, validation_end)
        
        if len(historical_data) < 100:
            print(f"Error: Insufficient data for analysis. Data points: {len(historical_data)}")
            exit(1)

        # Split data into train, validation, and test sets
        train_data = historical_data.iloc[:int(len(historical_data)*0.6)]
        val_data = historical_data.iloc[int(len(historical_data)*0.6):int(len(historical_data)*0.8)]
        test_data = historical_data.iloc[int(len(historical_data)*0.8):]

        model = QuantNeuralNet()
        
        # Perform cross-validation
        cv_histories = model.cross_validate(historical_data.iloc[:int(len(historical_data)*0.8)])
        
        # Train the final model
        history = model.train(train_data, val_data)

        # Evaluate on test set
        X_test, y_test = model.prepare_data(test_data)
        test_loss, test_accuracy = model.model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Plot training history
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Save the model and scaler
        model_path = 'quant_neural_net_model.h5'
        scaler_path = 'quant_neural_net_scaler.joblib'
        model.save_model(model_path, scaler_path)

        # Simulating paper trading
        simulation_start = datetime(2024, 8, 25)
        simulation_end = datetime(2024, 8, 29)
        simulation_data = get_data(simulation_start, simulation_end)

        print(f"\nSimulating paper trading from {simulation_start} to {simulation_end}:")
        simulated_trades = model.paper_trade(simulation_data)
        for trade in simulated_trades:
            print(f"{trade[0]} at {trade[1]} - Price: {trade[2]:.2f}")

        # Example of loading the model
        loaded_model = QuantNeuralNet.load_model(model_path, scaler_path)
        # You can now use loaded_model for predictions or further training

    except ValueError as e:
        print(f"Error: {e}") 
        print("Unable to proceed due to data retrieval issues.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)