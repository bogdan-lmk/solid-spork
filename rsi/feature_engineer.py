"""
Создание признаков для RSI предиктора
"""
import pandas as pd
import numpy as np
import talib
import logging
from typing import List

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Класс для создания признаков"""
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> pd.DataFrame:
        """Валидация входных данных"""
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")
        
        # Проверка на валидные значения
        for col in required_columns:
            if df[col].isna().any():
                logger.warning(f"Найдены NaN в колонке {col}, заполняем методом forward fill")
                df[col] = df[col].ffill()  # Исправлено для новых версий pandas
        
        # Если volume отсутствует, создаем заглушку
        if 'volume' not in df.columns:
            logger.info("Volume отсутствует, создаем синтетический объем")
            df['volume'] = np.random.randint(1000000, 10000000, len(df))
            
        return df.copy()
    
    @staticmethod
    def create_rsi_features(df: pd.DataFrame, periods: List[int] = [14]) -> pd.DataFrame:
        """Создание RSI и связанных признаков"""
        result = df.copy()
        
        for period in periods:
            rsi_col = f'rsi_{period}' if period != 14 else 'rsi'
            
            # Проверяем наличие достаточного количества данных
            if len(df) < period * 2:
                logger.warning(f"Недостаточно данных для RSI с периодом {period}. Пропускаем.")
                continue
                
            result[rsi_col] = talib.RSI(df['close'].astype(float), timeperiod=period)
            
            # Производные RSI
            if not result[rsi_col].isna().all():
                result[f'{rsi_col}_sma_5'] = talib.SMA(result[rsi_col], timeperiod=5)
                result[f'{rsi_col}_ema_5'] = talib.EMA(result[rsi_col], timeperiod=5)
                result[f'{rsi_col}_change'] = result[rsi_col].diff()
                result[f'{rsi_col}_velocity'] = result[f'{rsi_col}_change'].diff()
                result[f'{rsi_col}_volatility'] = result[rsi_col].rolling(window=10).std()
                
                # Лаговые признаки
                for lag in [1, 2, 3, 5]:
                    result[f'{rsi_col}_lag_{lag}'] = result[rsi_col].shift(lag)
        
        return result
    
    @staticmethod
    def create_oscillator_features(df: pd.DataFrame) -> pd.DataFrame:
        """Создание осцилляторных индикаторов"""
        result = df.copy()
        
        # Проверяем наличие достаточного количества данных
        if len(df) < 14:
            logger.warning("Недостаточно данных для создания осцилляторов")
            return result
        
        try:
            # Stochastic Oscillator
            result['stoch_k'], result['stoch_d'] = talib.STOCH(
                df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), 
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            result['stoch_divergence'] = result['stoch_k'] - result['stoch_d']
            
            # Williams %R
            result['williams_r'] = talib.WILLR(
                df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), 
                timeperiod=14
            )
            
            # CCI
            result['cci'] = talib.CCI(
                df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), 
                timeperiod=14
            )
            
            # MFI
            result['mfi'] = talib.MFI(
                df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), 
                df['volume'].astype(float), timeperiod=14
            )
            
            # Ultimate Oscillator
            result['ultimate_osc'] = talib.ULTOSC(
                df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
            )
            
            # Momentum индикаторы
            result['momentum'] = talib.MOM(df['close'].astype(float), timeperiod=10)
            result['roc'] = talib.ROC(df['close'].astype(float), timeperiod=10)
            
        except Exception as e:
            logger.warning(f"Ошибка при создании осцилляторов: {e}")
        
        return result
    
    @staticmethod
    def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """Создание трендовых индикаторов"""
        result = df.copy()
        
        # Проверяем наличие достаточного количества данных
        if len(df) < 26:
            logger.warning("Недостаточно данных для создания трендовых индикаторов")
            return result
        
        try:
            # MACD
            result['macd'], result['macd_signal'], result['macd_hist'] = talib.MACD(
                df['close'].astype(float), fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # Избегаем деления на ноль
            close_prices = df['close'].astype(float)
            result['macd_normalized'] = np.where(
                close_prices != 0, 
                result['macd'] / close_prices * 100, 
                0
            )
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].astype(float), timeperiod=20)
            
            # Избегаем деления на ноль в Bollinger Bands
            bb_range = bb_upper - bb_lower
            result['bb_percent_b'] = np.where(
                bb_range != 0,
                (df['close'] - bb_lower) / bb_range,
                0.5  # средняя позиция если диапазон = 0
            )
            result['bb_width'] = np.where(
                bb_middle != 0,
                bb_range / bb_middle,
                0
            )
            
            # ADX
            result['adx'] = talib.ADX(
                df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), 
                timeperiod=14
            )
            
        except Exception as e:
            logger.warning(f"Ошибка при создании трендовых индикаторов: {e}")
        
        return result
    
    @classmethod
    def create_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Создание всех признаков"""
        try:
            # Валидация
            df = cls.validate_data(df)
            
            # Проверяем минимальное количество данных
            if len(df) < 50:
                logger.warning(f"Мало данных для качественного анализа: {len(df)} строк. Рекомендуется минимум 50.")
            
            # Создание признаков
            df = cls.create_rsi_features(df)
            df = cls.create_oscillator_features(df)
            df = cls.create_trend_features(df)
            
            # Целевая переменная
            if 'rsi' in df.columns:
                df['rsi_next'] = df['rsi'].shift(-1)
            else:
                logger.error("RSI не был создан! Проверьте данные.")
                raise ValueError("Не удалось создать RSI индикатор")
            
            # Удаляем строки с бесконечными значениями
            df = df.replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"Создано признаков: {len(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при создании признаков: {e}")
            raise