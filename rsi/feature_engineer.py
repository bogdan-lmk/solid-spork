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
        """Валидация входных данных (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")
        
        df = df.copy()
        
        # ИСПРАВЛЕНИЕ: Конвертируем числовые колонки перед проверкой на NaN
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                try:
                    # Конвертируем в строку и заменяем запятые на точки
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                    
                    # Конвертируем в числовой тип
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                except Exception as e:
                    logger.warning(f"Ошибка конвертации колонки {col}: {e}")
        
        # Проверка на валидные значения ПОСЛЕ конвертации
        for col in required_columns:
            if df[col].isna().any():
                logger.warning(f"Найдены NaN в колонке {col}, заполняем методом forward fill")
                df[col] = df[col].ffill()
        
        # Если volume отсутствует, создаем заглушку
        if 'volume' not in df.columns:
            logger.info("Volume отсутствует, создаем синтетический объем")
            df['volume'] = np.random.randint(1000000, 10000000, len(df))
        elif df['volume'].isna().any():
            # Конвертируем volume аналогично
            try:
                if df['volume'].dtype == 'object':
                    df['volume'] = df['volume'].astype(str).str.replace(',', '.', regex=False).str.strip()
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                df['volume'] = df['volume'].ffill()
            except Exception as e:
                logger.warning(f"Ошибка обработки volume: {e}")
                df['volume'] = np.random.randint(1000000, 10000000, len(df))
            
        return df
    
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
        
    @staticmethod
    def create_rsi_correlated_features(df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков, коррелирующих с RSI"""
        result = df.copy()
        
        if len(df) < 14:
            logger.warning("Недостаточно данных для RSI-коррелирующих индикаторов")
            return result
        
        try:
            # 1. Стохастический осциллятор (высокая корреляция с RSI)
            result['stoch_k'], result['stoch_d'] = talib.STOCH(
                df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), 
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            result['stoch_divergence'] = result['stoch_k'] - result['stoch_d']
            result['stoch_momentum'] = result['stoch_k'].diff()
            
            # 2. Williams %R (перевернутый стохастик)
            result['williams_r'] = talib.WILLR(
                df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), 
                timeperiod=14
            )
            # Нормализуем Williams %R к шкале 0-100 как RSI
            result['williams_r_normalized'] = (result['williams_r'] + 100)
            
            # 3. CCI (Commodity Channel Index) - улучшенная версия
            result['cci'] = talib.CCI(
                df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), 
                timeperiod=14
            )
            # Нормализуем CCI к шкале RSI (0-100)
            result['cci_normalized'] = np.clip((result['cci'] + 200) / 4, 0, 100)
            
            # 4. MACD с акцентом на корреляцию с RSI
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].astype(float), fastperiod=12, slowperiod=26, signalperiod=9
            )
            result['macd'] = macd
            result['macd_signal'] = macd_signal
            result['macd_hist'] = macd_hist
            
            # Создаем MACD-производные, коррелирующие с RSI
            result['macd_momentum'] = macd.diff()
            result['macd_oscillator'] = np.where(
                df['close'].astype(float) != 0,
                (macd / df['close'].astype(float)) * 1000,  # Нормализация
                0
            )
            
            # 5. Bollinger Bands %B (позиция между полосами)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].astype(float), timeperiod=20)
            bb_range = bb_upper - bb_lower
            result['bb_percent_b'] = np.where(
                bb_range != 0,
                (df['close'].astype(float) - bb_lower) / bb_range,
                0.5
            )
            # Конвертируем в шкалу 0-100 как RSI
            result['bb_percent_b_scaled'] = result['bb_percent_b'] * 100
            result['bb_width'] = np.where(
                bb_middle != 0,
                bb_range / bb_middle,
                0
            )
            
            # 6. Дополнительные RSI-коррелирующие признаки
            
            # Momentum (сильно коррелирует с RSI)
            result['momentum'] = talib.MOM(df['close'].astype(float), timeperiod=14)
            result['momentum_normalized'] = (result['momentum'] / df['close'].astype(float)) * 100
            
            # ROC (Rate of Change) - также коррелирует с RSI
            result['roc'] = talib.ROC(df['close'].astype(float), timeperiod=14)
            
            # MFI (Money Flow Index) - "RSI с объемом"
            result['mfi'] = talib.MFI(
                df['high'].astype(float), df['low'].astype(float), 
                df['close'].astype(float), df['volume'].astype(float), 
                timeperiod=14
            )
            
            # 7. Комбинированные RSI-подобные индикаторы
            
            # Среднее значение всех осцилляторов (сильный предиктор RSI)
            oscillators = ['stoch_k', 'williams_r_normalized', 'cci_normalized', 'bb_percent_b_scaled']
            available_oscillators = [col for col in oscillators if col in result.columns and not result[col].isna().all()]
            
            if available_oscillators:
                result['oscillators_mean'] = result[available_oscillators].mean(axis=1)
                result['oscillators_std'] = result[available_oscillators].std(axis=1)
            
            # Дивергенция между ценой и осцилляторами
            price_change = df['close'].astype(float).pct_change(5)  # 5-периодное изменение цены
            if 'stoch_k' in result.columns:
                stoch_change = result['stoch_k'].diff(5)
                result['price_stoch_divergence'] = price_change - (stoch_change / 100)
            
            logger.info("Созданы RSI-коррелирующие признаки")
            
        except Exception as e:
            logger.warning(f"Ошибка при создании RSI-коррелирующих признаков: {e}")
        
        return result
    
    @classmethod
    def create_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Создание всех признаков с акцентом на RSI-коррелирующие индикаторы"""
        try:
            # Валидация (ВКЛЮЧАЕТ КОНВЕРТАЦИЮ)
            df = cls.validate_data(df)
            
            # Проверяем минимальное количество данных
            if len(df) < 50:
                logger.warning(f"Мало данных для качественного анализа: {len(df)} строк. Рекомендуется минимум 50.")
            
            # 1. Создание базовых RSI признаков
            df = cls.create_rsi_features(df)
            
            # 2. НОВОЕ: Создание RSI-коррелирующих признаков (приоритет)
            df = cls.create_rsi_correlated_features(df)
            
            # 3. Дополнительные осцилляторы (если еще не созданы)
            df = cls.create_oscillator_features(df)
            
            # 4. Трендовые индикаторы
            df = cls.create_trend_features(df)
            
            # Целевая переменная
            if 'rsi' in df.columns:
                df['rsi_next'] = df['rsi'].shift(-1)
            else:
                logger.error("RSI не был создан! Проверьте данные.")
                raise ValueError("Не удалось создать RSI индикатор")
            
            # Удаляем строки с бесконечными значениями
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Подсчитываем RSI-коррелирующие признаки для статистики
            rsi_correlated_features = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                                     ['stoch', 'williams', 'cci', 'macd', 'bb_percent', 'mfi', 'oscillators', 'momentum', 'roc'])]
            
            logger.info(f"Создано признаков: {len(df.columns)}")
            logger.info(f"RSI-коррелирующих признаков: {len(rsi_correlated_features)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при создании признаков: {e}")
            raise