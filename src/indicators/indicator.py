import math

import talib
import numpy as np
import pandas as pd


def calculate_wilder_rsi(prices, period=14):
    """
    Вычисляет RSI по методу Уайлдера (Wilder's RSI).
    
    В отличие от стандартного RSI, который использует простое скользящее среднее,
    метод Уайлдера использует модифицированное экспоненциальное сглаживание:
    - Первое значение: простое среднее
    - Последующие: previous_average * (period-1)/period + current_value/period
    
    Параметры:
        prices (pd.Series или np.array): массив цен закрытия
        period (int): период для RSI (по умолчанию 14)
    
    Возвращает:
        pd.Series: значения RSI по Уайлдеру
    """
    if len(prices) < period + 1:
        return pd.Series([np.nan] * len(prices))
    
    prices = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
    
    # Вычисляем изменения цен
    deltas = prices.diff()
    
    # Разделяем на положительные и отрицательные изменения
    gains = deltas.where(deltas > 0, 0)
    losses = (-deltas).where(deltas < 0, 0)
    
    # Инициализируем массивы для средних значений
    avg_gains = np.zeros(len(prices))
    avg_losses = np.zeros(len(prices))
    
    # Заполняем первыми NaN значениями
    avg_gains[:period] = np.nan
    avg_losses[:period] = np.nan
    
    # Первое значение - простое среднее за период
    if len(gains[1:period+1]) > 0:
        avg_gains[period] = gains[1:period+1].mean()
        avg_losses[period] = losses[1:period+1].mean()
    
    # Последующие значения по формуле Уайлдера
    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains.iloc[i]) / period
        avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses.iloc[i]) / period
    
    # Вычисляем RSI
    rs = avg_gains / (avg_losses + 1e-10)  # Избегаем деления на ноль
    rsi = 100 - (100 / (1 + rs))
    
    return pd.Series(rsi, index=prices.index)


def calculate_indicators(df, ema_period=2, sma_period=4, lengthEMA=2, lengthSMA=4, atr_length=42, atr_multiplier=3):
    # EMA и SMA для сигналов
    #df['ema_signal'] = talib.EMA(df['close'], timeperiod=ema_period)
    if ema_period == 1:
        df['ema_signal'] = df['close']
    else:
        df['ema_signal'] = talib.EMA(df['close'], timeperiod=ema_period)

    df['sma_signal'] = talib.EMA(df['close'], timeperiod=sma_period)
    #df['sma_signal'] = alma(df['close'], window_size=sma_period, offset=0.85, sigma=6)

    # EMA и SMA для отклонений
    df['ema_dev'] = talib.EMA(df['close'], timeperiod=lengthEMA)
    df['sma_dev'] = talib.SMA(df['close'], timeperiod=lengthSMA)

    # Отклонения
    df['ema_deviation'] = df['close'] - df['ema_dev']
    df['sma_deviation'] = df['close'] - df['sma_dev']

    # ATR для стоп-лосса
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_length)
    df['atr_stop'] = df['atr'] * atr_multiplier

    return df

def generate_signals(df):
    """
    Генерирует торговые сигналы на основе пересечений EMA и SMA,
    а также дивергенций.

    Параметры:
    df (pd.DataFrame): Датафрейм с колонками 'ema_signal', 'sma_signal',
                       'close', 'high', 'low', 'atr_stop', и другими необходимыми.

    Возвращает:
    pd.DataFrame: Датафрейм с новой колонкой 'signal', содержащей сигналы.
    """
    # Инициализация колонки 'signal' значениями None
    df['signal'] = None

    # Определение пересечений EMA и SMA для BUY и SELL сигналов
    df['buy_cross'] = (df['ema_signal'] > df['sma_signal']) & (df['ema_signal'].shift(1) <= df['sma_signal'].shift(1))


    df['sell_cross'] = (df['ema_signal'] < df['sma_signal']) & (df['ema_signal'].shift(1) >= df['sma_signal'].shift(1))

    # Создание сигналов на основе обнаруженных пересечений
    df['signal'] = np.where(df['buy_cross'], 'BUY',
                        np.where(df['sell_cross'], 'SELL', None))

    # Удаление временных колонок
    df.drop(['buy_cross', 'sell_cross'], axis=1, inplace=True)

    return df


def calculate_volatility_indicator(df, period=36):
    """
    Вычисляет волатильность (%) по заданному периоду на основе самых высоких и низких цен.

    Параметры:
    - df: DataFrame с колонками 'high', 'low' и 'close'.
    - period: Период расчёта волатильности.

    Возвращает:
    - DataFrame с новой колонкой 'volatility_percent'.
    """
    # Расчёт максимума и минимума за заданный период
    df['highest_value'] = df['high'].rolling(window=period, min_periods=1).max()
    df['lowest_value'] = df['low'].rolling(window=period, min_periods=1).min()

    # Разница между максимумом и минимумом за период
    df['range_value'] = df['highest_value'] - df['lowest_value']

    # Волатильность в процентах относительно текущей цены закрытия
    df['volatility_percent'] = (df['range_value'] / df['close']) * 100.0

    return df

def alma(data, window_size, offset=0.85, sigma=6):
    """
    Calculate the Arnaud Legoux Moving Average (ALMA).

    Parameters:
        data (list or np.array): Input price data.
        window_size (int): The size of the moving window.
        offset (float): Bias parameter, typically between 0 and 1 (default: 0.85).
        sigma (float): Controls the sensitivity of the weights (default: 6).

    Returns:
        np.array: ALMA values.
    """
    if len(data) < window_size:
        raise ValueError("Data length must be greater than or equal to the window size.")

    alma_values = []
    m = offset * (window_size - 1)  # Mean
    s = window_size / sigma  # Standard deviation

    # Precompute weights с правильным возведением в степень
    weights = np.exp(-((np.arange(window_size) - m)**2) / (2 * s**2))
    weights /= np.sum(weights)  # Normalize weights

    # Apply weights to calculate ALMA
    for i in range(window_size - 1, len(data)):
        window = data[i - window_size + 1 : i + 1]
        alma_values.append(np.dot(window, weights))

    # Prepend NaNs for the initial period where ALMA cannot be calculated
    alma_values = [np.nan] * (window_size - 1) + alma_values

    return np.array(alma_values)

def calculate_vix_stochastic(df, vix_period=8, k_period=6, d_period=5, k_smoothing=3):
    """
    Вычисляет CF BTC VIX Stochastic для заданного DataFrame.

    Параметры:
    - df: DataFrame с колонками 'high', 'low', 'close'.
    - vix_period: Период расчёта VIX (используется для high/low).
    - k_period: Период для расчёта %K (стохастик).
    - d_period: Период для расчёта %D (сигнал стохастика).
    - k_smoothing: Сглаживание %K.

    Возвращает:
    - DataFrame с новыми колонками 'stochK' и 'stochD'.
    """
    # 1. Расчёт волатильности (VIX)
    df['highest_value'] = df['high'].rolling(window=vix_period, min_periods=1).max()
    df['lowest_value'] = df['low'].rolling(window=vix_period, min_periods=1).min()
    df['range_value'] = df['highest_value'] - df['lowest_value']
    df['volatility_percent'] = (df['range_value'] / df['close']) * 100.0

    # 2. Стохастик по волатильности
    lowest_vol = df['volatility_percent'].rolling(window=k_period, min_periods=1).min()
    highest_vol = df['volatility_percent'].rolling(window=k_period, min_periods=1).max()

    # Защита от деления на ноль, если highest_vol == lowest_vol
    denominator = highest_vol - lowest_vol
    df['stoch_raw'] = np.where(denominator != 0,
                               (df['volatility_percent'] - lowest_vol) / denominator * 100.0,
                               0.0)

    # Сглаживание %K
    df['stochK'] = df['stoch_raw'].rolling(window=k_smoothing, min_periods=1).mean()

    # Вычисление %D как SMA от %K
    df['stochD'] = df['stochK'].rolling(window=d_period, min_periods=1).mean()

    # Опционально: очистить промежуточные колонки, если они больше не нужны
    df.drop(columns=['highest_value', 'lowest_value', 'range_value', 'stoch_raw'], inplace=True)

    return df

def calculate_bollinger_signals(df, length=20, mult=2.0, level=50):
    """
    Вычисляет сигналы BUY и SELL при пересечении позиции Bollinger Bands уровня level.

    Параметры:
      df      : DataFrame с колонками 'close', 'high', 'low'
      length  : период SMA и стандартного отклонения (Bollinger Band Length)
      mult    : множитель для стандартного отклонения (Multiplier)
      level   : уровень для генерации сигналов (по умолчанию 50)

    Возвращает:
      df с добавленными столбцами 'positionBetweenBands' и 'signal', где
      'signal' принимает значения 'BUY' или 'SELL' при пересечении уровня.
    """
    # Вычисление основных параметров Bollinger Bands
    basis = talib.SMA(df['close'], timeperiod=length)
    dev = mult * talib.STDDEV(df['close'], timeperiod=length, nbdev=1)
    upper = basis + dev
    lower = basis - dev

    # Вычисление позиции между полосами в процентах
    src = df['close']
    positionBetweenBands = 100 * (src - lower) / (upper - lower)

    # Сдвиг для определения пересечений
    previous = positionBetweenBands.shift(1)

    # Генерация сигналов при пересечении уровня 50
    buy_signals = (previous < level) & (positionBetweenBands >= level)
    sell_signals = (previous > level) & (positionBetweenBands <= level)

    # Добавление результатов в DataFrame
    df['positionBetweenBands'] = positionBetweenBands
    df.loc[buy_signals, 'Bol'] = 'BUY'
    df.loc[sell_signals, 'Bol'] = 'SELL'

    return df

def compute_gma(window, sigma_val, gama_length):
    """
    Вычисляет GMA для заданного окна avpchange (длина окна = gama_length),
    используя веса на основе гауссовой функции с параметром sigma_val.
    """
    sum_weights = 0.0
    weighted_sum = 0.0
    for i in range(gama_length):
        weight = math.exp(-(((i - (gama_length - 1)) / (2 * sigma_val)) ** 2) / 2)
        sub_window = window[:i+1]
        value = np.max(sub_window) + np.min(sub_window)
        weighted_sum += value * weight
        sum_weights += weight
    if sum_weights != 0:
        return (weighted_sum / sum_weights) / 2
    else:
        return np.nan

def calculate_alma_smoothed_gma(df, smooth=3, lookback=48, offset=0.85, sigma1=7,
                                gama_length=2, adaptive=True, volatilityPeriod=12, sigma_input=1.0):
    """
    Рассчитывает ALMA для процентного изменения цены (pchange) и на его основе GMA.
    Для каждого бара (начиная с gama_length-го) рассчитывается GMA по последним gama_length значениям avpchange.
    Результат сглаживается с помощью EMA (период 7) и генерируется сигнал на основе пересечения avpchange и gma_smoothed.
    """
    # Расчет pchange: процентное изменение цены с заданным сглаживанием
    pchange = df['close'].diff(smooth) / df['close'] * 100.0
    pchange = pchange.fillna(0)

    # Расчет ALMA для pchange (результат имеет ту же длину, что и df)
    avpchange = alma(pchange.values, window_size=lookback, offset=offset, sigma=sigma1)
    df['avpchange'] = avpchange

    # Определяем sigma для GMA
    if adaptive:
        sigma_val = df['close'].rolling(window=volatilityPeriod, min_periods=1).std().iloc[-1]
    else:
        sigma_val = sigma_input

    # Рассчитаем GMA для каждого бара, начиная с индекса (gama_length - 1)
    gma_values = [np.nan] * (gama_length - 1)
    avpchange_series = pd.Series(avpchange)
    for i in range(gama_length - 1, len(avpchange_series)):
        window = avpchange_series.iloc[i - gama_length + 1: i + 1].values
        gma_val = compute_gma(window, sigma_val, gama_length)
        gma_values.append(gma_val)

    # Сохраняем рассчитанные GMA в DataFrame (для каждого бара, где возможно вычисление)
    df['gma'] = gma_values
    # Дополнительное сглаживание через EMA с периодом 7
    df['gma_smoothed'] = df['gma'].ewm(span=7, adjust=False).mean()

    # Генерация сигнала: если текущее значение avpchange >= gma_smoothed, сигнал BUY, иначе SELL
    df['signal_gma'] = np.where(df['avpchange'] >= df['gma_smoothed'], 'BUY', 'SELL')

    # Для отладки можно добавить вывод текущих значений (опционально)
    # print("avpchange:", df['avpchange'].tail(5))
    # print("gma_smoothed:", df['gma_smoothed'].tail(5))

    return df

def crossover(series1, series2):
    """Возвращает Series булевых значений: True, если произошло пересечение вверх."""
    return (series1.shift(1) < series2.shift(1)) & (series1 >= series2)

def crossunder(series1, series2):
    """Возвращает Series булевых значений: True, если произошло пересечение вниз."""
    return (series1.shift(1) > series2.shift(1)) & (series1 <= series2)

def calculate_ago(df,
                  smooth=3,
                  loopback=32,
                  offset=0.85,
                  sigma1=7,
                  rsi_period=14,
                  momentum_length=9,
                  gma_length=7,
                  ema_length=21,
                  adaptive=True,
                  volatilityPeriod=20,
                  sigma_input=1.0):
    """
    Рассчитывает индикатор CF AGO BTC 1day.

    Параметры:
      df              : DataFrame с колонками 'open', 'high', 'low', 'close'
      smooth          : Период сглаживания для pchange (default 1)
      length1         : Lookback для ALMA (default 24)
      offset          : Offset для ALMA (default 0.85)
      sigma1          : Sigma для ALMA (default 7)
      rsi_period      : Период для расчёта RSI (default 14)
      momentum_length : Период для расчёта Chande Momentum (default 9)
      gma_length      : MAGMA GMA Length, период для расчёта WMA от avpchange (default 7)
      ema_length      : Период для EMA, используемой в новой логике (default 21)
      adaptive        : Флаг адаптивного расчёта sigma для GAMA (default True)
      volatilityPeriod : Период для расчёта стандартного отклонения (default 20)
      sigma_input     : Альтернативное значение sigma, если adaptive=False

    Возвращает:
      DataFrame с новыми колонками:
         'avpchange'    – ALMA от pchange,
         'rsi'          – RSI,
         'chandeMO'     – Chande Momentum,
         'gma'          – Gaussian Adaptive Moving Average (WMA от avpchange),
         'ema21'        – EMA(21) от gma,
         'finalSignal'  – Итоговый сигнал ('BUY' или 'SELL')
    """
    # 1. Источник: используем hlcc4, если нет другого (hlcc4 = (high + low + close + close)/4)
    df['hlcc4'] = (df['high'] + df['low'] + df['close'] + df['close']) / 4
    src = df['hlcc4']

    # 2. ALMA Smoothing: pchange и avpchange
    pchange = src.diff(smooth) / src * 100.0
    pchange = pchange.fillna(0)
    avpchange = alma(pchange.values, window_size=loopback, offset=offset, sigma=sigma1)
    df['avpchange'] = avpchange

    # 3. RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)
    df['rsiL'] = df['rsi'] > df['rsi'].shift(1)
    df['rsiS'] = df['rsi'] < df['rsi'].shift(1)

    # 4. Chande Momentum
    momm = df['close'].diff()  # разница цены между барами
    m1 = momm.clip(lower=0)
    m2 = (-momm).clip(lower=0)
    sm1 = m1.rolling(window=momentum_length, min_periods=1).sum()
    sm2 = m2.rolling(window=momentum_length, min_periods=1).sum()
    df['chandeMO'] = 100 * (sm1 - sm2) / (sm1 + sm2 + 1e-10)
    df['cL'] = df['chandeMO'] > df['chandeMO'].shift(1)
    df['cS'] = df['chandeMO'] < df['chandeMO'].shift(1)

    # 5. GAMA: рассчитываем WMA от avpchange с периодом gma_length
    df['gma'] = talib.WMA(df['avpchange'], timeperiod=gma_length)
    # Определяем цвет GMA: green, если avpchange >= gma, иначе red.
    df['gmaColor'] = np.where(df['avpchange'] >= df['gma'], 'green', 'red')

    # 6. Базовая логика сигналов MAGMA: пересечение avpchange и gma с подтверждением RSI и Chande Momentum
    buySignal_base = crossover(df['avpchange'], df['gma']) & df['rsiL'] & df['cL']
    sellSignal_base = crossunder(df['avpchange'], df['gma']) & df['rsiS'] & df['cS']

    # 7. Новая логика: пересечение GMA с EMA(21)
    ema = talib.EMA(df['gma'], timeperiod=ema_length)
    ema = ema.fillna(0)  # Заполняем NaN нулями
    df['gma'] = df['gma'].fillna(0)
    df['delta'] = (df['gma'] - ema)
    # 7. Новая логика: пересечение GMA с EMA(21)
    df['ema21'] = talib.EMA(df['gma'], timeperiod=ema_length)
    buySignal_new = crossover(df['gma'], df['ema21']) & (df['gmaColor'] == 'green')
    sellSignal_new = crossunder(df['gma'], df['ema21']) & (df['gmaColor'] == 'red')

    # 8. Итоговый сигнал:
    df['finalSignal'] = np.where(buySignal_new, 'BUY',
                                 np.where(sellSignal_new, 'SELL', None))

    return df


def wma(series, window):
    """
    Взвешенное скользящее среднее (WMA) с линейными весами.
    series: Pandas Series с исходными данными.
    window: Размер окна.
    """
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hma(series, length):
    """
    Классическая формула Hull Moving Average (HMA):
        HMA = WMA( 2 * WMA(series, length/2) - WMA(series, length), sqrt(length) )

    Параметры:
      series: Pandas Series с исходными данными.
      length: Период расчёта HMA.

    Возвращает:
      Pandas Series с значениями HMA.
    """
    half_length = max(int(round(length / 2)), 1)
    sqrt_length = max(int(round(math.sqrt(length))), 1)
    wma_half = wma(series, half_length)
    wma_full = wma(series, length)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_length)


def hma3(series, length):
    """
    Альтернативная формула Hull Moving Average (HMA3), аналог Pine‑Script функции f_hma3:
      p = length / 2,
      HMA3 = WMA( 3 * WMA(series, p/3) - WMA(series, p/2) - WMA(series, p), p )

    Параметры:
      series: Pandas Series с исходными данными.
      length: Период расчёта HMA3.

    Возвращает:
      Pandas Series с значениями HMA3.
    """
    p = max(int(round(length / 2)), 1)
    p_div3 = max(int(round(p / 3)), 1)
    p_div2 = max(int(round(p / 2)), 1)
    wma1 = wma(series, p_div3)
    wma2 = wma(series, p_div2)
    wma3_val = wma(series, p)
    inner = 3 * wma1 - wma2 - wma3_val
    return wma(inner, p)

def get_nma(series, length1, length2, ma_type):
    """
    NMA = (1 + α) * MA1 - α * MA2,
    где MA1 = MA(series, length1),
          MA2 = MA(MA1, length2),
          α = (λ*(length1-1))/(length1-λ) и λ = length1/length2.
    """
    lam = length1 / length2
    alpha = lam * (length1 - 1) / (length1 - lam)
    if ma_type.upper() == "EMA":
        ma1 = talib.EMA(series, timeperiod=length1)
        ma2 = talib.EMA(ma1, timeperiod=length2)
    elif ma_type.upper() == "SMA":
        ma1 = talib.SMA(series, timeperiod=length1)
        ma2 = talib.SMA(ma1, timeperiod=length2)
    elif ma_type.upper() == "VWMA":
        # Если требуется, можно добавить реализацию VWMA
        raise NotImplementedError("VWMA не реализована")
    else:  # WMA
        ma1 = talib.WMA(series, timeperiod=length1)
        ma2 = talib.WMA(ma1, timeperiod=length2)
    return (1 + alpha) * ma1 - alpha * ma2

def kahlman_filter(series, gain):
    """
    Реализует рекурсивный фильтр Кальмана по аналогии с Pine Script.
    series: Pandas Series с исходными значениями (например, Hull MA).
    gain: параметр gain (например, 10000).

    Возвращает:
      Pandas Series с отфильтрованными значениями.
    """
    # Инициализируем первую точку фильтра равной первому значению series,
    # а скорость (velo) – 0.
    kf = [series.iloc[0]]
    velo = [0.0]
    for i in range(1, len(series)):
        prev_kf = kf[-1]
        prev_velo = velo[-1]
        current = series.iloc[i]
        dk = current - prev_kf
        smooth = prev_kf + dk * math.sqrt((gain / 10000) * 2)
        new_velo = prev_velo + (gain / 10000) * dk
        new_kf = smooth + new_velo
        kf.append(new_kf)
        velo.append(new_velo)
    return pd.Series(kf, index=series.index)
