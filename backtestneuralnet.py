import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from quantneuralnet import QuantNeuralNet, get_data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tabulate import tabulate
import math
import matplotlib.patches as patches
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib.gridspec import GridSpec

def get_historical_data(symbol, start_date, end_date):
    """Fetch historical data for the given symbol and date range."""
    data = yf.download(symbol, start=start_date, end=end_date, interval='5m')
    data.index = data.index.tz_convert('America/New_York')
    return data.between_time('09:30', '16:00')

def simulate_trades(model, data, num_contracts):
    """Simulate trading based on model predictions."""
    predictions = model.predict(data)
    
    buy_threshold = 0.4
    sell_threshold = 0.4
    
    trades = []
    position = 0
    entry_price = 0
    entry_time = None
    
    for i, pred in enumerate(predictions):
        current_price = data['Close'].iloc[i + model.lookback]
        current_time = data.index[i + model.lookback]
        
        if pred[2] > buy_threshold and position <= 0:
            if position < 0:
                trades.append(('CLOSE_SHORT', current_time, current_price, entry_time, pred, num_contracts))
            trades.append(('BUY', current_time, current_price, None, pred, num_contracts))
            position = num_contracts
            entry_price = current_price
            entry_time = current_time
        elif pred[0] > sell_threshold and position >= 0:
            if position > 0:
                trades.append(('CLOSE_LONG', current_time, current_price, entry_time, pred, num_contracts))
            trades.append(('SELL', current_time, current_price, None, pred, num_contracts))
            position = -num_contracts
            entry_price = current_price
            entry_time = current_time
    
    # Close any open position at the end
    if position != 0:
        action = 'CLOSE_LONG' if position > 0 else 'CLOSE_SHORT'
        trades.append((action, data.index[-1], data['Close'].iloc[-1], entry_time, predictions[-1], abs(position)))
    
    return trades

def calculate_returns_and_metrics(trades, data, initial_investment, contract_size):
    """Calculate returns and various performance metrics based on the trades."""
    position = 0
    entry_price = 0
    entry_time = None
    returns = []
    hold_times = []
    equity_curve = [initial_investment]
    equity_dates = [data.index[0]]
    current_equity = initial_investment
    
    trade_details = []
    
    buy_points = []
    sell_points = []
    
    for trade in trades:
        action, timestamp, price, trade_entry_time, prediction, num_contracts = trade
        
        if action in ['BUY', 'SELL']:
            if position != 0:
                trade_return = (price - entry_price) * position * contract_size
                returns.append(trade_return / current_equity)
                hold_times.append((timestamp - entry_time).total_seconds() / 3600)  # in hours
                current_equity += trade_return
                equity_curve.append(current_equity)
                equity_dates.append(timestamp)
                
                trade_details.append({
                    'Entry Time': entry_time,
                    'Exit Time': timestamp,
                    'Entry Price': f"{entry_price:.2f}",
                    'Exit Price': f"{price:.2f}",
                    'Action': 'CLOSE_LONG' if position > 0 else 'CLOSE_SHORT',
                    'Contracts': abs(position),
                    'Return': f"{trade_return / current_equity:.2%}",
                    'Sell Signal': f"{prediction[0]:.2f}",
                    'Hold Signal': f"{prediction[1]:.2f}",
                    'Buy Signal': f"{prediction[2]:.2f}"
                })
            else:
                # If there was no position, add a flat point to the equity curve
                equity_curve.append(current_equity)
                equity_dates.append(timestamp)
            
            position = num_contracts if action == 'BUY' else -num_contracts
            entry_price = price
            entry_time = timestamp
            
            trade_details.append({
                'Entry Time': timestamp,
                'Exit Time': '-',
                'Entry Price': f"{price:.2f}",
                'Exit Price': '-',
                'Action': action,
                'Contracts': num_contracts,
                'Return': '-',
                'Sell Signal': f"{prediction[0]:.2f}",
                'Hold Signal': f"{prediction[1]:.2f}",
                'Buy Signal': f"{prediction[2]:.2f}"
            })
        elif action in ['CLOSE_LONG', 'CLOSE_SHORT']:
            trade_return = (price - entry_price) * position * contract_size
            returns.append(trade_return / current_equity)
            hold_times.append((timestamp - trade_entry_time).total_seconds() / 3600)  # in hours
            current_equity += trade_return
            equity_curve.append(current_equity)
            equity_dates.append(timestamp)  
            
            trade_details.append({
                'Entry Time': trade_entry_time,
                'Exit Time': timestamp,
                'Entry Price': f"{entry_price:.2f}",
                'Exit Price': f"{price:.2f}",
                'Action': action,
                'Contracts': abs(position),
                'Return': f"{trade_return / current_equity:.2%}",
                'Sell Signal': f"{prediction[0]:.2f}",
                'Hold Signal': f"{prediction[1]:.2f}",
                'Buy Signal': f"{prediction[2]:.2f}"
            })
            
            position = 0
        
        if action == 'BUY':
            buy_points.append((timestamp, current_equity))
        elif action == 'SELL':
            sell_points.append((timestamp, current_equity))
    
    total_return = (equity_curve[-1] - initial_investment) / initial_investment
    avg_win = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
    avg_loss = abs(np.mean([r for r in returns if r < 0])) if any(r < 0 for r in returns) else 0
    avg_hold_time = np.mean(hold_times) if hold_times else 0
    
    return {
        'Initial Investment': initial_investment,
        'Final Portfolio Value': equity_curve[-1],
        'Total Return': total_return,
        'Average Win': avg_win,
        'Average Loss': avg_loss,
        'Average Hold Time (hours)': avg_hold_time,
        'Number of Trades': len(trades),
        'Equity Curve': equity_curve,
        'Equity Dates': equity_dates,
        'Trade Details': trade_details,
        'Buy Points': buy_points,
        'Sell Points': sell_points
    }

def create_card_annotation(ax, text, xy, width=1.6, height=0.8, alpha=0.8, fc='white', ec='gray', text_color='black'):
    # Create a rectangle (card)
    card = patches.Rectangle(xy, width, height, fill=True, facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=2, zorder=9)
    ax.add_patch(card)
    
    # Add text to the card
    text_obj = ax.text(xy[0] + width/2, xy[1] + height/2, text,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=13,  # Increased font size by 60%
            fontweight='bold',  # Made text bold
            color=text_color,  # Added text color parameter
            wrap=True,
            zorder=10)  # Ensure text is on top
    
    return card, text_obj

def main():
    # Get user inputs with validation
    while True:
        try:
            initial_investment = float(input("Enter the amount you want to invest (minimum $150,000): $").replace(',', ''))
            assert initial_investment >= 150000, "Investment amount must be at least $150,000"
            break
        except ValueError:
            print("Please enter a valid number for the investment amount. Must be a float/decimal.")
        except AssertionError as e:
            print(e)

    while True:
        try:
            days_ago = int(input("Enter the number of days ago to start the backtest (Must be positive and 50 at most): "))
            assert 1 < days_ago <= 50, "Number of days must be greater than 1 and 50 at most"
            break
        except ValueError:
            print("Please enter a valid integer for the number of days.")
        except AssertionError as e:
            print(e)

    # Calculate number of contracts based on initial investment
    num_contracts = math.floor(initial_investment / 50000)
    assert num_contracts >= 3, "Investment amount must allow for at least 3 contracts"
    print(f"Trading {num_contracts} contract(s) based on the initial investment.")

    # Load the trained model
    model_path = 'quant_neural_net_model.h5'
    scaler_path = 'quant_neural_net_scaler.joblib'
    model = QuantNeuralNet.load_model(model_path, scaler_path)

    symbol = 'ES=F'  # E-mini S&P 500 futures
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)
    
    # Fetch historical data
    data = get_historical_data(symbol, start_date, end_date)
    
    # Simulate trades
    trades = simulate_trades(model, data, num_contracts)
    
    # Calculate returns and metrics
    contract_size = 50  # E-mini S&P 500 futures have a contract size of $50 times the index
    metrics = calculate_returns_and_metrics(trades, data, initial_investment, contract_size)
    
    # Create the figure and axes
    fig = plt.figure(figsize=(16, 12))  # Adjusted figure size
    gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Plot ES price as a line chart
    ax1.plot(data.index, data['Close'], label='ES Price', color='blue')
    ax1.set_ylabel('ES Price')
    ax1.set_title('Trading /ES Futures (A Quantitative Approach by Calvin Sowah)')

    # Initialize empty lists for legend
    buy_scatter = sell_scatter = close_long_scatter = close_short_scatter = None

    # Plot buy and sell points on the ES chart
    for trade in metrics['Trade Details']:
        if trade['Action'] in ['BUY', 'SELL', 'CLOSE_LONG', 'CLOSE_SHORT']:
            date = trade['Entry Time'] if trade['Action'] in ['BUY', 'SELL'] else trade['Exit Time']
            price = float(trade['Entry Price'] if trade['Action'] in ['BUY', 'SELL'] else trade['Exit Price'])
            
            if isinstance(date, str) and date == '-':
                continue
            
            if isinstance(date, str):
                try:
                    date = pd.to_datetime(date)
                except ValueError:
                    print(f"Skipping invalid date: {date}")
                    continue
            
            color = {
                'BUY': 'lightgreen',
                'SELL': 'lightcoral',
                'CLOSE_LONG': 'palegreen',
                'CLOSE_SHORT': 'lightsalmon'
            }[trade['Action']]
            
            marker = {
                'BUY': '^',
                'SELL': 'v',
                'CLOSE_LONG': 'o',
                'CLOSE_SHORT': 'o'
            }[trade['Action']]
            
            scatter = ax1.scatter(date, price, color=color, marker=marker, s=100, zorder=5)
            
            # Store scatter plot for legend
            if trade['Action'] == 'BUY' and buy_scatter is None:
                buy_scatter = scatter
            elif trade['Action'] == 'SELL' and sell_scatter is None:
                sell_scatter = scatter
            elif trade['Action'] == 'CLOSE_LONG' and close_long_scatter is None:
                close_long_scatter = scatter
            elif trade['Action'] == 'CLOSE_SHORT' and close_short_scatter is None:
                close_short_scatter = scatter

    # Add legend
    legend_elements = [
        ax1.plot([], [], color='blue', label='ES Price')[0],
        buy_scatter,
        sell_scatter,
        close_long_scatter,
        close_short_scatter
    ]
    legend_labels = ['ES Price', 'Buy', 'Sell', 'Close Long', 'Close Short']
    ax1.legend(legend_elements, legend_labels, loc='upper right')

    # Format x-axis for ES chart
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.set_xlabel('Date')

    # Create PnL report
    pnl_report = f"""
    PnL Report:
    Initial Investment: ${metrics['Initial Investment']:,.2f}
    Final Portfolio Value: ${metrics['Final Portfolio Value']:,.2f}
    Total Return: {metrics['Total Return']:.2%}
    Average Win: {metrics['Average Win']:.2%}
    Average Loss: {metrics['Average Loss']:.2%}
    Number of Trades: {metrics['Number of Trades']}
    """

    # Create trades taken report
    trades_list = "Trades Taken:\n\n"
    trade_count = 0
    for i in range(0, len(metrics['Trade Details']), 2):
        if i + 1 < len(metrics['Trade Details']):
            entry = metrics['Trade Details'][i]
            exit = metrics['Trade Details'][i + 1]
            trade_count += 1
            trades_list += f"{trade_count}. {entry['Action']} {entry['Contracts']} contract(s) at ${entry['Entry Price']} on {entry['Entry Time']}\n"
            trades_list += f"   {exit['Action']} at ${exit['Exit Price']} on {exit['Exit Time']} (Return: {exit['Return']})\n\n"
        
        if trade_count == 5:
            trades_list += "...\n"  # Indicate there might be more trades
            break

    # Add PnL report to the left subplot
    ax2.axis('off')
    ax2.text(0, 1, pnl_report, verticalalignment='top', fontsize=10, family='monospace')

    # Add Trades Taken report to the right subplot
    ax3.axis('off')
    ax3.text(0, 1, trades_list, verticalalignment='top', fontsize=10, family='monospace')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    # Print metrics
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        if metric not in ['Equity Curve', 'Trade Details']:
            if isinstance(value, float):
                print(f"{metric}: {value:.2f}")
            else:
                print(f"{metric}: {value}")
    
    # Print trade details
    print("\nTrade Details:")
    print(tabulate(metrics['Trade Details'], headers="keys", tablefmt="grid"))

if __name__ == "__main__":
    main()
