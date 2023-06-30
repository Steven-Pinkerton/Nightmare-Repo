def calculate_ichimoku(data):
    high_prices = data['High']
    close_prices = data['Close']
    low_prices = data['Low']

    # Tenkan-sen (Conversion Line)
    nine_period_high = high_prices.rolling(window=9).max()
    nine_period_low = low_prices.rolling(window=9).min()
    data['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    # Kijun-sen (Base Line)
    twenty_six_period_high = high_prices.rolling(window=26).max()
    twenty_six_period_low = low_prices.rolling(window=26).min()
    data['kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2

    # Senkou Span A (Leading Span A)
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)

    # Senkou Span B (Leading Span B)
    fifty_two_period_high = high_prices.rolling(window=52).max()
    fifty_two_period_low = low_prices.rolling(window=52).min()
    data['senkou_span_b'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)

    # Chikou Span (Lagging Span)
    data['chikou_span'] = close_prices.shift(-26)

    return data

def calculate_donchian_channels(data, n=20):
    """
    Calculates the Donchian Channels for a given period.

    Args:
    data (pandas.DataFrame): The market data.
    n (int): The period for calculating the channels.

    Returns:
    pandas.DataFrame: The market data with additional columns for the Donchian Channels.
    """
    # Calculate the upper channel (highest high in last N periods)
    data['Donchian Channel High'] = data['High'].rolling(n).max()

    # Calculate the lower channel (lowest low in last N periods)
    data['Donchian Channel Low'] = data['Low'].rolling(n).min()

    return data

def calculate_keltner_channels(data, n=20):
    """
    Calculates the Keltner Channels for a given period.

    Args:
    data (pandas.DataFrame): The market data.
    n (int): The period for calculating the channels.

    Returns:
    pandas.DataFrame: The market data with additional columns for the Keltner Channels.
    """
    # Calculate the middle line (exponential moving average)
    data['Keltner Channel Mid'] = data['Close'].ewm(span=n).mean()

    # Calculate the Average True Range (ATR)
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close']).abs()
    low_close = (data['Low'] - data['Close']).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(n).mean()

    # Calculate the upper and lower channels
    data['Keltner Channel High'] = data['Keltner Channel Mid'] + 2 * atr
    data['Keltner Channel Low'] = data['Keltner Channel Mid'] - 2 * atr

    return data

def calculate_atr_bands(data, n=20):   
    """
    Calculates the ATR Bands for a given period.

    Args:
    data (pandas.DataFrame): The market data.
    n (int): The period for calculating the bands.

    Returns:
    pandas.DataFrame: The market data with additional columns for the ATR Bands.
    """
    # Calculate the moving average
    data['ATR Band Mid'] = data['Close'].rolling(n).mean()

    # Calculate the Average True Range (ATR)
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close']).abs()
    low_close = (data['Low'] - data['Close']).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(n).mean()

    # Calculate the bands
    data['ATR Band High'] = data['ATR Band Mid'] + atr
    data['ATR Band Low'] = data['ATR Band Mid'] - atr

    return data

def calculate_elder_ray_index(data, n=13):
    """
    Calculates the Elder Ray Index.

    Args:
    data (pandas.DataFrame): The market data.
    n (int): The period for calculating the EMA.

    Returns:
    pandas.DataFrame: The market data with additional columns for the Elder Ray Index.
    """
    ema = data['Close'].ewm(span=n).mean()

    data['Bull Power'] = data['High'] - ema
    data['Bear Power'] = data['Low'] - ema

    return data

def calculate_hull_moving_average(data, n=9):
    """
    Calculates the Hull Moving Average.

    Args:
    data (pandas.DataFrame): The market data.
    n (int): The period for calculating the HMA.

    Returns:
    pandas.DataFrame: The market data with additional columns for the HMA.
    """
    import numpy as np
    wma_half_length = int(n / 2)
    sqrt_length = int(np.sqrt(n))

    wma_half = data['Close'].rolling(window=wma_half_length).mean()
    wma_full = data['Close'].rolling(window=n).mean()

    hull_moving_average = pd.Series(np.sqrt(n) * (wma_half * 2 - wma_full)).rolling(window=sqrt_length).mean()

    data['HMA'] = hull_moving_average

    return data

def calculate_rainbow_moving_averages(data, periods=[5, 10, 15, 20, 30, 40, 50, 100]):
    """
    Calculates Rainbow 3D Moving Averages.

    Args:
    data (pandas.DataFrame): The market data.
    periods (list of int): The periods for calculating the moving averages.

    Returns:
    pandas.DataFrame: The market data with additional columns for the Rainbow 3D Moving Averages.
    """
    for i in periods:
        data[f'SMA_{i}'] = data['Close'].rolling(window=i).mean()

    return data

def calculate_chaikin_money_flow(data, n=20):
    """
    Calculates the Chaikin Money Flow.

    Args:
    data (pandas.DataFrame): The market data.
    n (int): The period for calculating the CMF.

    Returns:
    pandas.DataFrame: The market data with additional columns for the CMF.
    """
    clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    clv = clv.fillna(0.0) # filling NaNs with 0
    cmf = clv * data['Volume']
    cmf = cmf.rolling(window=n).sum() / data['Volume'].rolling(window=n).sum()
    data['CMF'] = cmf

    return data

def calculate_chaikin_oscillator(data):
    """
    Calculates the Chaikin Oscillator.

    Args:
    data (pandas.DataFrame): The market data.

    Returns:
    pandas.DataFrame: The market data with additional columns for the Chaikin Oscillator.
    """
    adl = calculate_adl(data)
    ema_3 = adl.ewm(span=3, adjust=False).mean()
    ema_10 = adl.ewm(span=10, adjust=False).mean()

    data['Chaikin_Oscillator'] = ema_3 - ema_10

    return data

def calculate_chaikin_volatility(data, n=10):
    """
    Calculates the Chaikin Volatility.

    Args:
    data (pandas.DataFrame): The market data.
    n (int): The period for calculating the Chaikin Volatility.

    Returns:
    pandas.DataFrame: The market data with additional columns for the Chaikin Volatility.
    """
    hl_diff = data['High'] - data['Low']
    chaikin_vol = hl_diff.ewm(span=n).mean() / hl_diff.shift(n).ewm(span=n).mean() - 1
    data['Chaikin_Volatility'] = chaikin_vol

    return data

def calculate_standard_deviation_channels(data, n=20):
    """
    Calculates the standard deviation channels.

    Args:
    data (pandas.DataFrame): The market data.
    n (int): The period for calculating the standard deviation channels.

    Returns:
    pandas.DataFrame: The market data with additional columns for the standard deviation channels.
    """
    data['SMA'] = data['Close'].rolling(window=n).mean()
    data['SD'] = data['Close'].rolling(window=n).std()
    data['Upper_Channel'] = data['SMA'] + (2 * data['SD'])
    data['Lower_Channel'] = data['SMA'] - (2 * data['SD'])

    return data

def calculate_wilder_moving_average(data, n=14):
    """
    Calculates the Wilder's Moving Average.

    Args:
    data (pandas.DataFrame): The market data.
    n (int): The period for calculating the Wilder's Moving Average.

    Returns:
    pandas.DataFrame: The market data with additional columns for the Wilder's Moving Average.
    """
    data['WMA'] = data['Close'].ewm(alpha=1/n, adjust=False).mean()

    return data

def calculate_twiggs_momentum_oscillator(data, n=11):
    """
    Calculates the Twiggs Momentum Oscillator.

    Args:
    data (pandas.DataFrame): The market data.
    n (int): The period for calculating the Twiggs Momentum Oscillator.

    Returns:
    pandas.DataFrame: The market data with additional columns for the Twiggs Momentum Oscillator.
    """
    # Rate of Change
    data['ROC'] = data['Close'].pct_change(n) * 100
    # Exponential Moving Average of ROC
    data['EMA_ROC'] = data['ROC'].ewm(span=n, adjust=False).mean()
    # Twiggs Momentum Oscillator
    data['TMO'] = data['EMA_ROC'].ewm(span=n, adjust=False).mean()

    return data

def calculate_twiggs_trend_index(data, n=21):
    """
    Calculates the Twiggs Trend Index.

    Args:
    data (pandas.DataFrame): The market data.
    n (int): The period for calculating the Twiggs Trend Index.

    Returns:
    pandas.DataFrame: The market data with additional columns for the Twiggs Trend Index.
    """
    data['Close_EMA'] = data['Close'].ewm(span=n, adjust=False).mean()
    data['TTI'] = (data['Close'] - data['Close_EMA']) / data['Close_EMA']

    return data

def calculate_atr_trailing_stops(high, low, close, atr_period=14, multiplier=3):
    """
    Calculate Average True Range (ATR) Trailing Stops.

    Args:
        high (pandas.Series): Series of 'high' prices
        low (pandas.Series): Series of 'low' prices
        close (pandas.Series): Series of 'close' prices
        atr_period (int): The period to consider for the ATR calculation
        multiplier (int): The multiplier to use for the ATR Stops

    Returns:
        pandas.Series: The ATR Trailing Stops values
    """
    atr = ta.ATR(high, low, close, timeperiod=atr_period)
    atr_trailing_stop = pd.Series(index=close.index)

    for i in range(len(close)):
        if close[i] > atr_trailing_stop[i - 1]:
            atr_trailing_stop[i] = max(atr_trailing_stop[i - 1], close[i] - multiplier * atr[i])
        else:
            atr_trailing_stop[i] = min(atr_trailing_stop[i - 1], close[i] + multiplier * atr[i])

    return atr_trailing_stop

def calculate_linear_regression(data, window=14):
    """
    Calculate linear regression line and linear regression indicator.

    Args:
        data (pandas.DataFrame): The market data.
        window (int): The window period for calculating the linear regression.

    Returns:
        pandas.DataFrame: The market data with additional columns for linear regression and indicator.
    """
    # Calculate the linear regression line for the closing prices over the window period
    data['LR_Slope'] = data['Close'].rolling(window=window).apply(lambda x: np.polyfit(np.arange(window), x, 1)[0])
    data['LR_Intercept'] = data['Close'].rolling(window=window).apply(lambda x: np.polyfit(np.arange(window), x, 1)[1])

    # Calculate the linear regression line
    data['LR'] = data['LR_Slope'] * np.arange(window) + data['LR_Intercept']

    # Calculate the linear regression indicator
    data['LR_Indicator'] = data['Close'] - data['LR']

    return data

def calculate_coppock(data, short_roc_period=11, long_roc_period=14, wma_period=10):
    """
    Calculate the Coppock Curve.

    Args:
        data (pandas.DataFrame): The market data.
        short_roc_period (int): The period for the shorter Rate-of-Change.
        long_roc_period (int): The period for the longer Rate-of-Change.
        wma_period (int): The period for the Weighted Moving Average.

    Returns:
        pandas.DataFrame: The market data with additional column for the Coppock Curve.
    """
    roc_short = data['Close'].pct_change(periods=short_roc_period)
    roc_long = data['Close'].pct_change(periods=long_roc_period)
    data['Coppock'] = (roc_short + roc_long).rolling(window=wma_period).mean()

    return data

def calculate_coppock(data, short_roc_period=11, long_roc_period=14, wma_period=10):
    """
    Calculate the Coppock Curve.

    Args:
        data (pandas.DataFrame): The market data.
        short_roc_period (int): The period for the shorter Rate-of-Change.
        long_roc_period (int): The period for the longer Rate-of-Change.
        wma_period (int): The period for the Weighted Moving Average.

    Returns:
        pandas.DataFrame: The market data with additional column for the Coppock Curve.
    """
    roc_short = data['Close'].pct_change(periods=short_roc_period)
    roc_long = data['Close'].pct_change(periods=long_roc_period)
    data['Coppock'] = (roc_short + roc_long).rolling(window=wma_period).mean()

    return data

def calculate_kst(data, rc1=10, rc2=15, rc3=20, rc4=30, sma1=10, sma2=10, sma3=10, sma4=15):
    """
    Calculate the KST Indicator.

    Args:
        data (pandas.DataFrame): The market data.
        rc1, rc2, rc3, rc4: The periods for the Rate-of-Change.
        sma1, sma2, sma3, sma4: The periods for the Simple Moving Average.

    Returns:
        pandas.DataFrame: The market data with additional column for the KST Indicator.
    """
    rcma1 = data['Close'].pct_change(periods=rc1).rolling(window=sma1).mean()
    rcma2 = data['Close'].pct_change(periods=rc2).rolling(window=sma2).mean()
    rcma3 = data['Close'].pct_change(periods=rc3).rolling(window=sma3).mean()
    rcma4 = data['Close'].pct_change(periods=rc4).rolling(window=sma4).mean()

    data['KST'] = rcma1 + 2 * rcma2 + 3 * rcma3 + 4 * rcma4

    return data

def calculate_force_index(data, period=13):
    """
    Calculate the Force Index.

    Args:
        data (pandas.DataFrame): The market data.
        period (int): The period for calculating the Force Index.

    Returns:
        pandas.DataFrame: The market data with additional column for the Force Index.
    """
    data['Force_Index'] = data['Close'].diff(period) * data['Volume']
    data['Force_Index'] = data['Force_Index'].rolling(window=period).mean()

    return data

