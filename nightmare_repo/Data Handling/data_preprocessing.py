import pandas as pd
from sklearn.preprocessing import StandardScaler
import talib as ta

def preprocess_data(filename):
    """
    Preprocesses historical market data for machine learning.

    Args:
    filename (str): the name of the CSV file containing the historical data.

    Returns:
    pandas.DataFrame: the preprocessed data.
    """
    # Load data from CSV file
    data = pd.read_csv(filename, parse_dates=True, index_col=0)

    # Fill any missing values
    data.fillna(method='ffill', inplace=True)

    # Normalize the data
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    # Save the preprocessed data to a new CSV file
    data.to_csv('preprocessed_' + filename)

    return data

def generate_indicators(data):
    """
    Generates technical indicators from market data.

    Args:
    data (pandas.DataFrame): The market data.

    Returns:
    pandas.DataFrame: The market data with additional columns for the technical indicators.
    """
    # Generate simple moving averages
    data['SMA_10'] = ta.SMA(data['Close'], timeperiod=10)
    data['SMA_50'] = ta.SMA(data['Close'], timeperiod=50)

    # Generate RSI
    data['RSI'] = ta.RSI(data['Close'], timeperiod=14)

    # Generate Bollinger Bands
    upper, middle, lower = ta.BBANDS(data['Close'], timeperiod=20)
    data['BB_upper'] = upper
    data['BB_middle'] = middle
    data['BB_lower'] = lower

    # Generate MACD
    macd, macdsignal, macdhist = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = macd
    data['MACD_signal'] = macdsignal
    data['MACD_hist'] = macdhist

    return data


if __name__ == "__main__":
    # Preprocess the data
    preprocess_data('russell_2000_daily.csv')