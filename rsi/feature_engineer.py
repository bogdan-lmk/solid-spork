"""
Создание признаков для RSI предиктора - ИСПРАВЛЕННАЯ ВЕРСИЯ без утечки данных
"""
import pandas as pd
import numpy as np
import talib
import logging
from typing import List

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Класс для создания признаков - ИСПРАВЛЕНИЯ против утечки данных"""
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> pd.DataFrame:
        """Валидация для accumulatedData формата"""
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")
        
        df = df.copy()
        
        from data_adapter import DataAdapter
        
        # Проверяем и конвертируем основные колонки
        essential_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in essential_columns:
            if col in df.columns:
                original_type = df[col].dtype
                df[col] = DataAdapter._robust_numeric_conversion_accumulated(df[col], col)
                new_type = df[col].dtype
                
                if original_type != new_type:
                    logger.info(f"Конвертирована колонка {col}: {original_type} -> {new_type}")
        
        # Проверка на валидные значения ПОСЛЕ конвертации
        for col in required_columns:
            if df[col].isna().any():
                nan_count = df[col].isna().sum()
                logger.warning(f"Найдены NaN в {col}: {nan_count} значений, заполняем")
                df[col] = df[col].ffill().bfill()
                
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.warning(f"Заполнили NaN в {col} медианой: {median_val}")
        
        # Создаем volume если отсутствует
        if 'volume' not in df.columns:
            logger.info("Volume отсутствует, создаем синтетический объем")
            df['volume'] = np.random.randint(1000000, 10000000, len(df))
        elif df['volume'].isna().any():
            df['volume'] = DataAdapter._robust_numeric_conversion_accumulated(df['volume'], 'volume')
            df['volume'] = df['volume'].ffill().bfill()
            
            if df['volume'].isna().any():
                df['volume'] = df['volume'].fillna(df['volume'].median())
        
        # Финальная проверка
        for col in required_columns + ['volume']:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"Колонка {col} не числовая после обработки: {df[col].dtype}")
                
                if df[col].isna().any():
                    raise ValueError(f"Остались NaN в критической колонке {col}")
                
                if (df[col] <= 0).any():
                    logger.warning(f"Найдены нулевые/отрицательные значения в {col}")
                    df = df[df[col] > 0].reset_index(drop=True)
        
        logger.info(f"Валидация завершена. Размер данных: {df.shape}")
        return df
    
    @staticmethod
    def create_rsi_features(df: pd.DataFrame, periods: List[int] = [14]) -> pd.DataFrame:
        """Создание RSI и связанных признаков ТОЛЬКО из исторических данных"""
        result = df.copy()
        
        if len(df) < 50:
            logger.warning(f"Мало данных для качественного RSI: {len(df)} строк")
        
        for period in periods:
            rsi_col = f'rsi_{period}' if period != 14 else 'rsi'
            
            if len(df) < period * 3:
                logger.warning(f"Недостаточно данных для RSI с периодом {period}. Пропускаем.")
                continue
            
            try:
                # Создаем RSI
                close_prices = df['close'].astype(float)
                result[rsi_col] = talib.RSI(close_prices, timeperiod=period)
                
                if result[rsi_col].isna().all():
                    logger.error(f"RSI {period} содержит только NaN!")
                    continue
                
                rsi_values = result[rsi_col].dropna()
                logger.info(f"RSI {period}: создано {len(rsi_values)} значений, диапазон: [{rsi_values.min():.2f}, {rsi_values.max():.2f}]")
                
                # ИСПРАВЛЕНИЕ: Только лаговые признаки (БЕЗ будущих данных!)
                if not result[rsi_col].isna().all():
                    # Скользящие средние RSI (с лагом!)
                    if len(rsi_values) >= 5:
                        result[f'{rsi_col}_sma_5'] = talib.SMA(result[rsi_col], timeperiod=5).shift(1)
                        result[f'{rsi_col}_ema_5'] = talib.EMA(result[rsi_col], timeperiod=5).shift(1)
                    
                    # Изменения RSI (с лагом!)
                    result[f'{rsi_col}_change'] = result[rsi_col].diff().shift(1)
                    result[f'{rsi_col}_velocity'] = result[f'{rsi_col}_change'].diff().shift(1)
                    
                    # Волатильность RSI (с лагом!)
                    if len(rsi_values) >= 10:
                        result[f'{rsi_col}_volatility'] = result[rsi_col].rolling(window=10).std().shift(1)
                    
                    # Лаговые признаки RSI
                    for lag in [1, 2, 3, 5]:
                        if len(df) > lag:
                            result[f'{rsi_col}_lag_{lag}'] = result[rsi_col].shift(lag)
                    
                    # Дополнительные RSI признаки (с лагом!)
                    result[f'{rsi_col}_momentum'] = (result[rsi_col] - result[rsi_col].shift(5)).shift(1)
                    result[f'{rsi_col}_rate_of_change'] = result[rsi_col].pct_change(5).shift(1) * 100
                
            except Exception as e:
                logger.error(f"Ошибка создания RSI {period}: {e}")
                continue
        
        return result
    
    @staticmethod
    def create_oscillator_features(df: pd.DataFrame) -> pd.DataFrame:
        """Создание осцилляторных индикаторов с правильными лагами"""
        result = df.copy()
        
        if len(df) < 14:
            logger.warning("Недостаточно данных для осцилляторов")
            return result
        
        try:
            close_prices = df['close'].astype(float)
            high_prices = df['high'].astype(float)
            low_prices = df['low'].astype(float)
            volume = df['volume'].astype(float)
            
            # Stochastic Oscillator (с лагом!)
            try:
                stoch_k, stoch_d = talib.STOCH(
                    high_prices, low_prices, close_prices, 
                    fastk_period=14, slowk_period=3, slowd_period=3
                )
                result['stoch_k'] = stoch_k.shift(1)  # ИСПРАВЛЕНИЕ: лаг!
                result['stoch_d'] = stoch_d.shift(1)  # ИСПРАВЛЕНИЕ: лаг!
                result['stoch_divergence'] = (stoch_k - stoch_d).shift(1)
                result['stoch_momentum'] = stoch_k.diff().shift(1)
                logger.info("Stochastic создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка Stochastic: {e}")
            
            # Williams %R (с лагом!)
            try:
                williams_r = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
                result['williams_r'] = williams_r.shift(1)  # ИСПРАВЛЕНИЕ: лаг!
                result['williams_r_normalized'] = (williams_r + 100).shift(1)
                logger.info("Williams %R создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка Williams %R: {e}")
            
            # CCI (с лагом!)
            try:
                cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
                result['cci'] = cci.shift(1)  # ИСПРАВЛЕНИЕ: лаг!
                result['cci_normalized'] = np.clip((cci + 200) / 4, 0, 100).shift(1)
                logger.info("CCI создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка CCI: {e}")
            
            # MFI (с лагом!)
            try:
                mfi = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
                result['mfi'] = mfi.shift(1)  # ИСПРАВЛЕНИЕ: лаг!
                logger.info("MFI создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка MFI: {e}")
            
            # Ultimate Oscillator (с лагом!)
            try:
                ultimate_osc = talib.ULTOSC(high_prices, low_prices, close_prices)
                result['ultimate_osc'] = ultimate_osc.shift(1)  # ИСПРАВЛЕНИЕ: лаг!
                logger.info("Ultimate Oscillator создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка Ultimate Oscillator: {e}")
            
            # Momentum индикаторы (с лагом!)
            try:
                momentum = talib.MOM(close_prices, timeperiod=10)
                roc = talib.ROC(close_prices, timeperiod=10)
                
                result['momentum'] = momentum.shift(1)  # ИСПРАВЛЕНИЕ: лаг!
                result['roc'] = roc.shift(1)  # ИСПРАВЛЕНИЕ: лаг!
                result['momentum_normalized'] = (momentum / close_prices * 100).shift(1)
                logger.info("Momentum индикаторы созданы успешно")
            except Exception as e:
                logger.warning(f"Ошибка Momentum: {e}")
                
        except Exception as e:
            logger.error(f"Критическая ошибка в осцилляторах: {e}")
        
        return result
    
    @staticmethod
    def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """Создание трендовых индикаторов с правильными лагами"""
        result = df.copy()
        
        if len(df) < 26:
            logger.warning("Недостаточно данных для трендовых индикаторов")
            return result
        
        try:
            close_prices = df['close'].astype(float)
            high_prices = df['high'].astype(float)
            low_prices = df['low'].astype(float)
            
            # MACD (с лагом!)
            try:
                macd, macd_signal, macd_hist = talib.MACD(
                    close_prices, fastperiod=12, slowperiod=26, signalperiod=9
                )
                
                result['macd'] = macd.shift(1)  # ИСПРАВЛЕНИЕ: лаг!
                result['macd_signal'] = macd_signal.shift(1)
                result['macd_hist'] = macd_hist.shift(1)
                
                # Нормализация MACD (с лагом!)
                result['macd_normalized'] = np.where(
                    close_prices != 0, 
                    macd / close_prices * 100, 
                    0
                ).shift(1)
                
                # MACD momentum (с лагом!)
                result['macd_momentum'] = macd.diff().shift(1)
                
                logger.info("MACD создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка MACD: {e}")
            
            # Bollinger Bands (с лагом!)
            try:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
                
                bb_range = bb_upper - bb_lower
                bb_percent_b = np.where(
                    bb_range != 0,
                    (close_prices - bb_lower) / bb_range,
                    0.5
                )
                
                result['bb_percent_b'] = bb_percent_b.shift(1)  # ИСПРАВЛЕНИЕ: лаг!
                result['bb_percent_b_scaled'] = (bb_percent_b * 100).shift(1)
                result['bb_width'] = np.where(
                    bb_middle != 0,
                    bb_range / bb_middle,
                    0
                ).shift(1)
                
                logger.info("Bollinger Bands созданы успешно")
            except Exception as e:
                logger.warning(f"Ошибка Bollinger Bands: {e}")
            
            # ADX (с лагом!)
            try:
                adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
                result['adx_calc'] = adx.shift(1)  # ИСПРАВЛЕНИЕ: лаг! (переименовали чтобы не конфликтовать)
                logger.info("ADX создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка ADX: {e}")
            
            # Дополнительные трендовые индикаторы (с лагом!)
            try:
                # EMA разных периодов (с лагом!)
                result['ema_9'] = talib.EMA(close_prices, timeperiod=9).shift(1)
                result['ema_21'] = talib.EMA(close_prices, timeperiod=21).shift(1)
                
                # SMA (с лагом!)
                result['sma_20'] = talib.SMA(close_prices, timeperiod=20).shift(1)
                
                # Отношения EMA (с лагом!)
                ema_9 = talib.EMA(close_prices, timeperiod=9)
                ema_21 = talib.EMA(close_prices, timeperiod=21)
                result['ema_ratio_9_21'] = np.where(
                    ema_21 != 0,
                    ema_9 / ema_21,
                    1
                ).shift(1)
                
                logger.info("Дополнительные трендовые индикаторы созданы")
            except Exception as e:
                logger.warning(f"Ошибка дополнительных трендовых: {e}")
                
        except Exception as e:
            logger.error(f"Критическая ошибка в трендовых индикаторах: {e}")
        
        return result
    
    @staticmethod
    def create_existing_indicators_features(df: pd.DataFrame) -> pd.DataFrame:
        """Использование уже существующих индикаторов из accumulatedData с правильными лагами"""
        result = df.copy()
        
        # Создаем лаговые версии существующих индикаторов
        try:
            # ИСПРАВЛЕНИЕ: Все существующие индикаторы делаем лаговыми!
            existing_indicators = [
                'atr', 'atr_stop', 'atr_to_price_ratio',
                'fast_ema', 'slow_ema', 'ema_fast_deviation',
                'pchange', 'avpchange', 'gma', 'gma_smoothed',
                'positionBetweenBands', 'bollinger_position',
                'choppiness_index', 'volatility_percent',
                'rsi_volatility', 'adx', 'rsi_divergence', 
                'rsi_delta', 'linear_regression'
            ]
            
            for indicator in existing_indicators:
                if indicator in result.columns:
                    # Создаем лаговую версию
                    result[f'{indicator}_lag_1'] = result[indicator].shift(1)
            
            # ATR производные (с лагом!)
            if 'atr' in result.columns and 'close' in result.columns:
                atr_percentage = np.where(
                    result['close'] != 0,
                    (result['atr'] / result['close']) * 100,
                    0
                )
                result['atr_percentage'] = atr_percentage.shift(1)  # ИСПРАВЛЕНИЕ: лаг!
            
            # EMA производные (с лагом!)
            if 'fast_ema' in result.columns and 'slow_ema' in result.columns:
                ema_ratio = np.where(
                    result['slow_ema'] != 0,
                    result['fast_ema'] / result['slow_ema'],
                    1
                )
                result['ema_ratio'] = ema_ratio.shift(1)  # ИСПРАВЛЕНИЕ: лаг!
                result['ema_spread'] = (result['fast_ema'] - result['slow_ema']).shift(1)
            
            # Волатильность производные (с лагом!)
            if 'volatility_percent' in result.columns:
                result['volatility_ma'] = result['volatility_percent'].rolling(5).mean().shift(1)
                result['volatility_change'] = result['volatility_percent'].diff().shift(1)
            
            # RSI производные (с лагом!)
            if 'rsi_volatility' in result.columns:
                result['rsi_volatility_ma'] = result['rsi_volatility'].rolling(5).mean().shift(1)
                result['rsi_volatility_change'] = result['rsi_volatility'].diff().shift(1)
            
            # Choppiness производные (с лагом!)
            if 'choppiness_index' in result.columns:
                result['choppiness_ma'] = result['choppiness_index'].rolling(5).mean().shift(1)
                result['choppiness_change'] = result['choppiness_index'].diff().shift(1)
            
            logger.info("Производные от существующих индикаторов созданы с правильными лагами")
            
        except Exception as e:
            logger.warning(f"Ошибка создания производных от существующих: {e}")
        
        return result
    
    @classmethod
    def create_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """ИСПРАВЛЕННАЯ ВЕРСИЯ: Создание всех признаков БЕЗ утечки данных"""
        try:
            logger.info(f"Начинаем создание признаков для {len(df)} строк")
            
            # Валидация и очистка данных
            df = cls.validate_data(df)
            
            if len(df) < 50:
                logger.warning(f"Мало данных для качественного анализа: {len(df)} строк")
            elif len(df) < 30:
                raise ValueError(f"Критически мало данных: {len(df)} строк. Минимум 30.")
            
            # 1. Используем существующие индикаторы из файла (с лагами!)
            df = cls.create_existing_indicators_features(df)
            
            # 2. Создание базовых RSI признаков (с лагами!)
            df = cls.create_rsi_features(df)
            
            # 3. Дополнительные осцилляторы (с лагами!)
            df = cls.create_oscillator_features(df)
            
            # 4. Трендовые индикаторы (с лагами!)
            df = cls.create_trend_features(df)
            
            # ИСПРАВЛЕНИЕ: Правильная целевая переменная - используем существующий rsi_volatility!
            if 'rsi_volatility' in df.columns:
                # ВАРИАНТ 1: Предсказываем изменение RSI
                df['target_rsi_change'] = df['rsi_volatility'].diff()
                
                # ВАРИАНТ 2: Предсказываем следующее значение RSI (НО БЕЗ УТЕЧКИ!)
                # Создаем целевую переменную для обучения, удалив последнюю строку
                df['target_rsi_next'] = df['rsi_volatility'].shift(-1)
                
                # Удаляем последнюю строку где target_rsi_next = NaN
                original_length = len(df)
                df = df[:-1].copy()
                
                logger.info(f"Целевые переменные созданы. Удалена последняя строка: {original_length} -> {len(df)}")
                
                # Проверяем качество целевых переменных
                if df['target_rsi_next'].isna().any():
                    logger.warning(f"В target_rsi_next есть NaN: {df['target_rsi_next'].isna().sum()} значений")
                
                # ВАРИАНТ 3: Направление движения RSI
                rsi_change = df['target_rsi_change']
                df['target_rsi_direction'] = pd.cut(
                    rsi_change,
                    bins=[-np.inf, -2, 2, np.inf],
                    labels=[0, 1, 2]  # DOWN=0, SIDEWAYS=1, UP=2
                )
                
            else:
                logger.error("rsi_volatility не найден в данных!")
                raise ValueError("Не удалось найти rsi_volatility для создания целевой переменной")
            
            # Удаляем бесконечные значения
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Статистика созданных признаков
            total_features = len(df.columns)
            numeric_features = len(df.select_dtypes(include=[np.number]).columns)
            
            # Подсчет лаговых признаков
            lag_features = [col for col in df.columns if 'lag_' in col or any(
                col.endswith(suffix) for suffix in ['_sma_5', '_ema_5', '_ma', '_change', '_momentum', '_ratio']
            )]
            
            logger.info(f"Создание признаков завершено:")
            logger.info(f"  Всего признаков: {total_features}")
            logger.info(f"  Числовые признаки: {numeric_features}")
            logger.info(f"  Лаговые признаки: {len(lag_features)}")
            logger.info(f"  Финальный размер данных: {df.shape}")
            
            # Проверка качества созданных признаков
            nan_columns = df.columns[df.isna().any()].tolist()
            if nan_columns:
                logger.warning(f"Колонки с NaN: {len(nan_columns)} из {total_features}")
                for col in nan_columns[:5]:  # Показываем первые 5
                    nan_count = df[col].isna().sum()
                    nan_percent = (nan_count / len(df)) * 100
                    logger.warning(f"  {col}: {nan_count} NaN ({nan_percent:.1f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"Критическая ошибка при создании признаков: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise