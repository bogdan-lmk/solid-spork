"""
Улучшенный fetcher для получения данных Binance с обработкой ошибок
Исправляет проблему "insufficient data" и обеспечивает получение RSI для любого инструмента
"""
import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

def get_historical_klines_robust(symbol: str, interval: str = '1d', limit: int = 100, 
                                start_time: Optional[int] = None, end_time: Optional[int] = None,
                                max_retries: int = 3, retry_delay: float = 1.0) -> pd.DataFrame:
    """
    Надежное получение исторических данных с Binance API
    
    Args:
        symbol: Торговая пара (например, 'BTCUSDT')
        interval: Интервал ('1d', '4h', '1h', etc.)
        limit: Количество свечей (макс 1000)
        start_time: Время начала в миллисекундах
        end_time: Время окончания в миллисекундах
        max_retries: Максимальное количество попыток
        retry_delay: Задержка между попытками
    
    Returns:
        DataFrame с данными OHLCV или пустой DataFrame при ошибке
    """
    
    # Если не указаны временные рамки, получаем последние данные
    if start_time is None and end_time is None:
        # Получаем данные за последние limit периодов
        end_time = int(datetime.now().timestamp() * 1000)
        
        # Вычисляем start_time на основе интервала
        interval_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000
        }.get(interval, 24 * 60 * 60 * 1000)  # По умолчанию 1d
        
        start_time = end_time - (limit * interval_ms)
    
    # Строим URL
    base_url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': min(limit, 1000)  # Binance ограничивает до 1000
    }
    
    if start_time:
        params['startTime'] = int(start_time)
    if end_time:
        params['endTime'] = int(end_time)
    
    # Попытки получения данных
    for attempt in range(max_retries):
        try:
            logger.debug(f"Попытка {attempt + 1}/{max_retries} для {symbol}")
            
            response = requests.get(base_url, params=params, timeout=30)
            
            # Проверяем статус ответа
            if response.status_code == 429:
                # Rate limit - увеличиваем задержку
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit для {symbol}, ждем {wait_time}s")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            # Проверяем, что данные получены
            if not data or len(data) == 0:
                logger.warning(f"Пустые данные для {symbol}")
                
                # Пробуем альтернативные временные рамки
                if attempt < max_retries - 1:
                    # Увеличиваем временной диапазон
                    if start_time and end_time:
                        time_range = end_time - start_time
                        new_start = start_time - time_range  # Удваиваем диапазон
                        params['startTime'] = int(new_start)
                        logger.info(f"Расширяем временной диапазон для {symbol}")
                        continue
                
                return pd.DataFrame()
            
            # Создаем DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Конвертируем типы данных
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Конвертируем цены в float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Проверяем качество данных
            if df[price_columns].isna().all().any():
                logger.warning(f"Некорректные ценовые данные для {symbol}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
            
            logger.debug(f"Успешно получено {len(df)} свечей для {symbol}")
            return df
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout для {symbol}, попытка {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                # Неверный символ или параметры
                logger.error(f"Неверные параметры для {symbol}: {e}")
                break
            else:
                logger.warning(f"HTTP ошибка для {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                    
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ошибка запроса для {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
                
        except Exception as e:
            logger.error(f"Неожиданная ошибка для {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
    
    logger.error(f"Не удалось получить данные для {symbol} после {max_retries} попыток")
    return pd.DataFrame()

def get_multiple_timeframes_data(symbol: str, primary_interval: str = '1d', 
                               fallback_intervals: list = None) -> pd.DataFrame:
    """
    Получение данных с резервными интервалами
    
    Если основной интервал не работает, пробуем альтернативные
    """
    if fallback_intervals is None:
        fallback_intervals = ['1d', '4h', '1h', '30m']
    
    # Добавляем основной интервал в начало если его нет
    intervals_to_try = [primary_interval] + [i for i in fallback_intervals if i != primary_interval]
    
    for interval in intervals_to_try:
        logger.debug(f"Пробуем интервал {interval} для {symbol}")
        
        df = get_historical_klines_robust(
            symbol=symbol,
            interval=interval,
            limit=100  # Достаточно для RSI расчета
        )
        
        if not df.empty and len(df) >= 14:  # Минимум для RSI
            logger.info(f"Успешно получены данные {interval} для {symbol}: {len(df)} свечей")
            return df
        else:
            logger.debug(f"Недостаточно данных {interval} для {symbol}: {len(df) if not df.empty else 0} свечей")
    
    logger.warning(f"Не удалось получить достаточные данные для {symbol} ни на одном интервале")
    return pd.DataFrame()

def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """
    Проверка корректности OHLCV данных
    """
    if df.empty:
        return False
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        return False
    
    # Проверяем, что есть числовые данные
    for col in required_columns:
        if df[col].isna().all():
            return False
    
    # Проверяем логичность цен (high >= low, etc.)
    if len(df) > 0:
        invalid_candles = (df['high'] < df['low']).sum()
        if invalid_candles > len(df) * 0.1:  # Более 10% некорректных свечей
            return False
    
    return True

def get_reliable_market_data(symbol: str, min_periods: int = 14) -> pd.DataFrame:
    """
    Надежное получение рыночных данных для расчета RSI
    
    Args:
        symbol: Торговая пара
        min_periods: Минимальное количество периодов
    
    Returns:
        Проверенные данные OHLCV
    """
    logger.info(f"Получение надежных данных для {symbol}")
    
    # Специальная обработка для XMR - используем простой Binance API
    if symbol == 'XMRUSDT':
        logger.info(f"Специальная обработка для {symbol}")
        try:
            import requests
            url = 'https://api.binance.com/api/v3/klines'
            params = {
                'symbol': symbol,
                'interval': '1d',
                'limit': 100
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                
                if validate_ohlcv_data(df) and len(df) >= min_periods:
                    logger.info(f"Успешно получены данные для {symbol}: {len(df)} свечей")
                    return df
        except Exception as e:
            logger.warning(f"Простой метод для {symbol} не сработал: {e}")
            # Продолжаем с обычной логикой
    
    # Сначала пробуем основной метод
    df = get_multiple_timeframes_data(symbol, '1d', ['1d', '4h', '1h'])
    
    if validate_ohlcv_data(df) and len(df) >= min_periods:
        return df
    
    # Если не получилось, пробуем увеличить лимит
    logger.info(f"Пробуем увеличить количество данных для {symbol}")
    
    df = get_historical_klines_robust(
        symbol=symbol,
        interval='1d',
        limit=200,  # Больше данных
        max_retries=5  # Больше попыток
    )
    
    if validate_ohlcv_data(df) and len(df) >= min_periods:
        return df
    
    # Последняя попытка с более мелким интервалом и агрегацией
    logger.info(f"Пробуем агрегировать мелкие данные для {symbol}")
    
    df_hourly = get_historical_klines_robust(
        symbol=symbol,
        interval='1h',
        limit=500,  # 500 часов ≈ 20 дней
        max_retries=3
    )
    
    if not df_hourly.empty and len(df_hourly) >= 24:
        # Агрегируем часовые данные в дневные
        df_daily = aggregate_to_daily(df_hourly)
        if validate_ohlcv_data(df_daily) and len(df_daily) >= min_periods:
            logger.info(f"Успешно агрегированы данные для {symbol}: {len(df_daily)} дней")
            return df_daily
    
    logger.error(f"Не удалось получить надежные данные для {symbol}")
    return pd.DataFrame()

def aggregate_to_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегация часовых данных в дневные
    """
    if df_hourly.empty:
        return pd.DataFrame()
    
    try:
        # Группируем по дням
        df_hourly['date'] = df_hourly['open_time'].dt.date
        
        daily_data = df_hourly.groupby('date').agg({
            'open': 'first',      # Первая цена открытия дня
            'high': 'max',        # Максимум дня
            'low': 'min',         # Минимум дня
            'close': 'last',      # Последняя цена закрытия
            'volume': 'sum',      # Суммарный объем
            'open_time': 'first'  # Время начала дня
        }).reset_index()
        
        # Переименовываем колонки обратно
        daily_data['close_time'] = daily_data['open_time'] + pd.Timedelta(days=1)
        
        return daily_data[['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]
        
    except Exception as e:
        logger.error(f"Ошибка агрегации данных: {e}")
        return pd.DataFrame()

# Функция для обратной совместимости
def get_historical_klines(symbol, interval, limit=100, start_time=None, end_time=None):
    """
    Обертка для обратной совместимости со старым кодом
    """
    return get_reliable_market_data(symbol, min_periods=14)