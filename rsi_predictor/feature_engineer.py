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
                df[col] = df[col].fillna(method='ffill')
        
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
            result[rsi_col] = talib.RSI(df['close'], timeperiod=period)
            
            # Производные RSI
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
        
        # Stochastic Oscillator
        result['stoch_k'], result['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'], 
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        result['stoch_divergence'] = result['stoch_k'] - result['stoch_d']
        
        # Williams %R
        result['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # CCI
        result['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # MFI
        result['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # Ultimate Oscillator
        result['ultimate_osc'] = talib.ULTOSC(df['high'], df['low'], df['close'])
        
        # Momentum индикаторы
        result['momentum'] = talib.MOM(df['close'], timeperiod=10)
        result['roc'] = talib.ROC(df['close'], timeperiod=10)
        
        return result
    
    @staticmethod
    def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """Создание трендовых индикаторов"""
        result = df.copy()
        
        # MACD
        result['macd'], result['macd_signal'], result['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        result['macd_normalized'] = result['macd'] / df['close'] * 100
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20)
        result['bb_percent_b'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        result['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # ADX
        result['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        return result
    
    @classmethod
    def create_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Создание всех признаков"""
        # Валидация
        df = cls.validate_data(df)
        
        # Создание признаков
        df = cls.create_rsi_features(df)
        df = cls.create_oscillator_features(df)
        df = cls.create_trend_features(df)
        
        # Целевая переменная
        df['rsi_next'] = df['rsi'].shift(-1)
        
        return df