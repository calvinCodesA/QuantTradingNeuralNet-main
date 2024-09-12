import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz
from quantneuralnet import QuantNeuralNet
import time

def fetch_live_data(symbol, interval='5m', period='1d'):
    """Fetch the most recent data for the given symbol."""
    data = yf.download(symbol, interval=interval, period=period)
    data.index = data.index.tz_convert('America/New_York')
    return data.between_time('09:30', '16:00')

def simulate_trade(model, data):
    """Simulate trading based on model predictions."""
    predictions = model.predict(data)
    latest_prediction = predictions[-1]
    
    buy_threshold = 0.4
    sell_threshold = 0.4
    
    if latest_prediction[2] > buy_threshold:
        return 'BUY'
    elif latest_prediction[0] > sell_threshold:
        return 'SELL'
    else:
        return 'HOLD'

def main():
    # Load the trained model
    model_path = 'quant_neural_net_model.h5'
    scaler_path = 'quant_neural_net_scaler.joblib'
    model = QuantNeuralNet.load_model(model_path, scaler_path)

    symbol = 'ES=F'  # E-mini S&P 500 futures
    position = 0
    entry_price = 0

    while True:
        now = datetime.now(pytz.timezone('America/New_York'))
        
        # Check if it's a weekday and within market hours
        if now.weekday() < 5 and 9 <= now.hour < 16:
            try:
                # Fetch the most recent data
                live_data = fetch_live_data(symbol)
                
                if not live_data.empty:
                    current_price = live_data['Close'].iloc[-1]
                    
                    # Make a trading decision
                    decision = simulate_trade(model, live_data)
                    
                    print(f"Time: {now}, Price: {current_price:.2f}, Decision: {decision}")
                    
                    # Execute the trade
                    if decision == 'BUY' and position <= 0:
                        if position < 0:
                            print(f"Closing short position. P&L: {entry_price - current_price:.2f}")
                        position = 1
                        entry_price = current_price
                        print(f"Opening long position at {entry_price:.2f}")
                    elif decision == 'SELL' and position >= 0:
                        if position > 0:
                            print(f"Closing long position. P&L: {current_price - entry_price:.2f}")
                        position = -1
                        entry_price = current_price
                        print(f"Opening short position at {entry_price:.2f}")
                
                # Wait for 5 minutes before the next iteration
                time.sleep(300)
            
            except Exception as e:
                print(f"An error occurred: {e}")
                time.sleep(60)  # Wait for 1 minute before retrying
        else:
            # Outside of trading hours, wait for 1 hour
            print("Outside of trading hours. Waiting...")
            time.sleep(3600)

if __name__ == "__main__":
    main()