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

    # Calculate Bollinger Band %b
    data['BB_%b'] = (data['Close'] - lower) / (upper - lower)

    # Calculate Bollinger Band Width
    data['BB_BandWidth'] = (upper - lower) / middle


    # Generate MACD
    macd, macdsignal, macdhist = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = macd
    data['MACD_signal'] = macdsignal
    data['MACD_hist'] = macdhist
    
    # Parabolic SAR
    data['PSAR'] = ta.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)

    # True Range
    data['TRANGE'] = ta.TRANGE(data['High'], data['Low'], data['Close'])

    # Weighted Moving Average
    data['WMA'] = ta.WMA(data['Close'], timeperiod=30)
    
    # Exponential Moving Average (EMA)
    data['EMA'] = ta.EMA(data['Close'], timeperiod=30)
    
    # Aroon Oscillator
    aroondown, aroonup = ta.AROON(data['High'], data['Low'], timeperiod=14)
    data['Aroon_Oscillator'] = aroonup - aroondown

    # Average True Range (ATR)
    data['ATR'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    
    # Accumulation Distribution
    data['AD'] = ta.AD(data['High'], data['Low'], data['Close'], data['Volume'])

    # ADX
    data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    
    # Ichimoku Cloud
    data = data_utils.calculate_ichimoku(data)
    
    # Add ATR Trailing Stops to DataFrame
    data['ATR_Trailing_Stops'] = calculate_atr_trailing_stops(data['High'], data['Low'], data['Close'])
   
    # Add Linear Regression and Linear Regression Indicator to DataFrame
    data = calculate_linear_regression(data)
    
    # Chande Momentum Oscillator
    data['CMO'] = ta.CMO(data['Close'], timeperiod=14)
    
    # Detrended Price Oscillator
    data['DPO'] = ta.DPO(data['Close'], timeperiod=20)
    
    # Chande Momentum Oscillator
    data['CMO'] = ta.CMO(data['Close'], timeperiod=14)
    
    # Detrended Price Oscillator
    data['DPO'] = ta.DPO(data['Close'], timeperiod=20)
    
    # Directional Movement Index
    data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['MINUS_DI'] = ta.MINUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['PLUS_DI'] = ta.PLUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)
    
    # Momentum Indicator
    data['MOM'] = ta.MOM(data['Close'], timeperiod=10)
    
    # TRIX Indicator
    data['TRIX'] = ta.TRIX(data['Close'], timeperiod=30)
    
    # Ultimate Oscillator
    data['UO'] = ta.ULTOSC(data['High'], data['Low'], data['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # Slow Stochastic Oscillator
    slowk, slowd = ta.STOCH(data['High'], data['Low'], data['Close'], 
                        fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    
    data['slowk'] = slowk
    data['slowd'] = slowd
    
    # Smoothed Rate of Change
    roc = ta.ROC(data['Close'], timeperiod=11)
    data['SROC'] = ta.WMA(roc, timeperiod=3)
    
    # Stochastic Oscillator
    fastk, fastd = ta.STOCHF(data['High'], data['Low'], data['Close'], 
                         fastk_period=5, fastd_period=3, fastd_matype=0)
    data['fastk'] = fastk
    data['fastd'] = fastd
    
    # Rate of Change
    data['ROC'] = ta.ROC(data['Close'], timeperiod=10)
    
    # Stochastic RSI
    data['STOCH_RSI'] = ta.STOCHRSI(data['Close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    
    # Twiggs Smoothed Momentum
    roc = ta.ROC(data['Close'], timeperiod=10)
    data['Twiggs_Smoothed_Momentum'] = ta.EMA(roc, timeperiod=10)
    
    # Williams %R
    data['Williams_%R'] = ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)

    # Choppiness Index (manually calculated)
    highest = data['High'].rolling(window=14).max()
    lowest = data['Low'].rolling(window=14).min()

    ATR = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

    log_sum = (np.log10(ATR / (highest - lowest))).rolling(window=14).sum()

    data['Choppiness Index'] = 100 * np.exp(-4.6 * log_sum)
    
    # Ease of Movement
    data['EOM'] = ta.EOM(data['High'], data['Low'], data['Close'], volume=data['Volume'], timeperiod=14)

    # Mass Index
    data['Mass Index'] = ta.MA(data['High'] - data['Low'], timeperiod=9).rolling(window=25).sum() / ta.MA(ta.MA(data['High'] - data['Low'], timeperiod=9), timeperiod=9)

    # Twiggs Volatility (manually calculated)
    data['Twiggs Volatility'] = ta.MA(np.log10(data['High'] / data['Low']), timeperiod=30)

    # Volatility
    data['Volatility'] = data['Close'].rolling(window=14).std()
    
    # Volatility Ratio (manually calculated)
    TR = ta.TRANGE(data['High'], data['Low'], data['Close'])
    ATR = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['Volatility Ratio'] = TR / ATR
    
    # Money Flow Index
    data['MFI'] = ta.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14)

    # On Balance Volume
    data['OBV'] = ta.OBV(data['Close'], data['Volume'])
    
    # Volume Oscillator (5 days and 20 days)
    vol_5 = data['Volume'].rolling(window=5).mean()
    vol_20 = data['Volume'].rolling(window=20).mean()
    data['Vol_Oscillator'] = vol_5 - vol_20
    
    # Williams Accumulation Distribution
    data['Williams_AD'] = ta.AD(data['High'], data['Low'], data['Close'], data['Volume'])
    
    # Commodity Channel Index
    data['CCI'] = ta.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
    
    # Price Volume Trend
    data['PVT'] = (data['Volume'] * (data['Close'].pct_change())).cumsum()
    
    # Twiggs Money Flow
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    prev_typical_price = typical_price.shift(1)
    money_flow = typical_price - prev_typical_price
    positive_money_flow = money_flow.copy()
    positive_money_flow[positive_money_flow < 0] = 0
    negative_money_flow = -1 * money_flow.copy()
    negative_money_flow[negative_money_flow < 0] = 0
    data['TMF'] = ta.SMA(positive_money_flow, 21) - ta.SMA(negative_money_flow, 21)
    
    return data

if __name__ == "__main__":
    # Preprocess the data
    preprocess_data('russell_2000_daily.csv')