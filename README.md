# Trading E-mini S&P 500 Futures using a Quantitative Neural Network (Includes Backtesting and Live Trading Code Implementation) 

Training Phase:
https://github.com/user-attachments/assets/0a2971c4-161c-4be0-887d-4daa4ab6e00a

Results:
<img width="1196" alt="results" src="https://github.com/user-attachments/assets/4e2039ab-cd1d-4c94-a487-5f352e50e5f7">

## Overview
This Python script implements a quantitative trading system using a neural network model specifically trained on recent E-mini S&P 500 futures (ES) data. It performs backtesting on historical financial data, simulates trades, and provides performance metrics and visualizations tailored for ES futures trading.

## Key Components

### 1. Data Retrieval
- Uses `yfinance` to download historical price data for the E-mini S&P 500 futures (ES=F).
- Adjusts data to simulate future dates for testing purposes.

### 2. QuantNeuralNet Class
- Implements a neural network model for price prediction, specifically trained on ES futures data.
- Uses LSTM layers to process price and technical indicator data relevant to ES futures trading.
- Includes methods for data preparation, model training, featuring engineering involving several indicators (including Lorentzian Classification) and prediction optimized for ES futures.

### 3. Backtesting Logic
- Simulates trading based on the model's predictions for ES futures.
- Implements a paper trading function to generate buy/sell signals tailored to ES futures market characteristics.


### 4. Performance Metrics Calculation
- Calculates various trading performance metrics including:
  - Total return
  - Average win/loss

### 5. Visualization
- Plots the curve of the asset with buy and sell points marked, specific to ES futures trading.
- Displays performance metrics in the context of ES futures market behavior.


## Detailed Process Flow

1. **User Input**
   - Prompts for initial investment amount (minimum investment amount of $150,000) and backtest duration (max 50 days) for ES futures trading simulation.

2. **Model Loading**
   - Loads a pre-trained neural network model and scaler, specifically optimized for ES futures data.

3. **Data Retrieval and Preprocessing**
   - Downloads historical ES futures data for the specified period.
   - Applies necessary data transformations and scaling, tailored to ES futures characteristics.

4. **Backtesting**
   - Runs the loaded model on historical ES futures data.
   - Generates buy/sell signals based on model predictions, considering ES futures market dynamics.
   - Simulates trades and calculates equity changes in the context of ES futures trading.

https://github.com/user-attachments/assets/f1d0b072-bb99-4845-8565-4b37043fbfb1


5. **Performance Analysis**
   - Calculates various performance metrics from the simulated trades, relevant to ES futures trading strategies.


6. **Visualization**
   - Plots the equity curve specific to ES futures trading.
   - Marks buy and sell points on the curve, reflecting ES futures market behavior.
   - Displays key performance metrics in the context of ES futures trading.
     
![ES](https://github.com/user-attachments/assets/f752b26f-c4a7-4a09-8a2d-45cabda00460)

## Key Functions

- `get_data(start_date, end_date)`: Retrieves historical ES futures price data.
    The numpy and Pandas functionalities help with the following
1. **Pandas DataFrame Creation*: The yf.download() function returns a Pandas DataFrame. This DataFrame is the primary data structure used to store and manipulate the financial data.
2. **DataFrame Manipulation*:
        data.empty: This Pandas method checks if the DataFrame is empty.
        data.index: Accesses the index of the DataFrame, which in this case is a DatetimeIndex.
3. **Time Zone Conversion*:
          data.index = data.index.tz_convert('America/New_York'): This Pandas operation converts the time zone of the index to Easter Standard Time (EST).
4. **Time-based Filtering*:
          data = data.between_time('09:30', '17:00'): This Pandas method filters the DataFrame to include only rows with times between 9:30 AM   and 5:00 PM, effectively keeping only market hours data.
5. **Index Manipulation*:
          data.index = data.index + timedelta(days=days_to_shift): This operation shifts the dates in the index by a specified number of days.   It's using Pandas' ability to perform vectorized operations on DatetimeIndex objects.
6. **Data Inspection*:
          print(data, len(data)): This prints the DataFrame and its length. The len() function works with Pandas DataFrames to return the        number of rows.
- `QuantNeuralNet.paper_trade(live_data)`: Simulates trading based on model predictions for ES futures.
- `calculate_returns_and_metrics(trades, data, initial_investment)`: Computes performance metrics for ES futures trading.

## Usage
1. Ensure all required libraries are installed.
2. Run the script: `python3 backtestNeuralNet.py`
3. Enter the initial investment amount and backtest duration when prompted.
4. Review the output metrics and visualizations specific to ES futures trading.

## Notes
- The script uses a pre-trained model specifically optimized for ES futures data. Ensure `quant_neural_net_model.h5` and `quant_neural_net_scaler.joblib` are present in the working directory.
- Adjust the `buy_threshold` and `sell_threshold` in the `paper_trade` method to fine-tune trading signals for ES futures.
- The script simulates future dates for testing purposes. Adjust the `days_to_shift` variable for different future projections in the context of ES futures trading.

## Caution
This script was developed by Calvin Sowah for educational and research purposes only. It is specifically designed to simulate E-mini S&P 500 futures trading. Real trading involves significant risks, including the potential for substantial losses. Past performance does not guarantee future results. Always consult with a qualified financial advisor before engaging in real trading activities. Feel free to reach out to me with any questions or concerns regarding running this codebase: caksowah@gmail.com
