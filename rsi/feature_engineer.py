"""
Создание признаков для RSI предиктора (ИСПРАВЛЕННАЯ ВЕРСИЯ ПОД accumulatedData)
"""
import pandas as pd
import numpy as np
import talib
import logging
from typing import List

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Класс для создания признаков (КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ)"""
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> pd.DataFrame:
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Валидация для accumulatedData формата"""
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")
        
        df = df.copy()
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем специализированный DataAdapter
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
                
                # Если все еще есть NaN (в начале/конце), заполняем медианой
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
        """Создание RSI и связанных признаков"""
        result = df.copy()
        
        if len(df) < 50:
            logger.warning(f"Мало данных для качественного RSI: {len(df)} строк")
        
        for period in periods:
            rsi_col = f'rsi_{period}' if period != 14 else 'rsi'
            
            # Проверяем достаточность данных
            if len(df) < period * 3:
                logger.warning(f"Недостаточно данных для RSI с периодом {period}. Пропускаем.")
                continue
            
            try:
                # Создаем RSI
                close_prices = df['close'].astype(float)
                result[rsi_col] = talib.RSI(close_prices, timeperiod=period)
                
                # Проверяем результат
                if result[rsi_col].isna().all():
                    logger.error(f"RSI {period} содержит только NaN!")
                    continue
                
                rsi_values = result[rsi_col].dropna()
                logger.info(f"RSI {period}: создано {len(rsi_values)} значений, диапазон: [{rsi_values.min():.2f}, {rsi_values.max():.2f}]")
                
                # Производные RSI (только если RSI успешно создан)
                if not result[rsi_col].isna().all():
                    # Скользящие средние RSI
                    if len(rsi_values) >= 5:
                        result[f'{rsi_col}_sma_5'] = talib.SMA(result[rsi_col], timeperiod=5)
                        result[f'{rsi_col}_ema_5'] = talib.EMA(result[rsi_col], timeperiod=5)
                    
                    # Изменения RSI
                    result[f'{rsi_col}_change'] = result[rsi_col].diff()
                    result[f'{rsi_col}_velocity'] = result[f'{rsi_col}_change'].diff()
                    
                    # Волатильность RSI
                    if len(rsi_values) >= 10:
                        result[f'{rsi_col}_volatility'] = result[rsi_col].rolling(window=10).std()
                    
                    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Лаговые признаки (НЕ используем будущие данные!)
                    for lag in [1, 2, 3, 5]:
                        if len(df) > lag:
                            result[f'{rsi_col}_lag_{lag}'] = result[rsi_col].shift(lag)
                    
                    # Дополнительные RSI признаки
                    result[f'{rsi_col}_momentum'] = result[rsi_col] - result[rsi_col].shift(5)
                    result[f'{rsi_col}_rate_of_change'] = result[rsi_col].pct_change(5) * 100
                
            except Exception as e:
                logger.error(f"Ошибка создания RSI {period}: {e}")
                continue
        
        return result
    
    @staticmethod
    def create_oscillator_features(df: pd.DataFrame) -> pd.DataFrame:
        """Создание осцилляторных индикаторов"""
        result = df.copy()
        
        if len(df) < 14:
            logger.warning("Недостаточно данных для осцилляторов")
            return result
        
        try:
            close_prices = df['close'].astype(float)
            high_prices = df['high'].astype(float)
            low_prices = df['low'].astype(float)
            volume = df['volume'].astype(float)
            
            # Stochastic Oscillator
            try:
                result['stoch_k'], result['stoch_d'] = talib.STOCH(
                    high_prices, low_prices, close_prices, 
                    fastk_period=14, slowk_period=3, slowd_period=3
                )
                result['stoch_divergence'] = result['stoch_k'] - result['stoch_d']
                result['stoch_momentum'] = result['stoch_k'].diff()
                logger.info("Stochastic создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка Stochastic: {e}")
            
            # Williams %R
            try:
                result['williams_r'] = talib.WILLR(
                    high_prices, low_prices, close_prices, timeperiod=14
                )
                # Нормализация к шкале 0-100
                result['williams_r_normalized'] = (result['williams_r'] + 100)
                logger.info("Williams %R создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка Williams %R: {e}")
            
            # CCI (Commodity Channel Index)
            try:
                result['cci'] = talib.CCI(
                    high_prices, low_prices, close_prices, timeperiod=14
                )
                # Нормализация к шкале RSI (0-100)
                result['cci_normalized'] = np.clip((result['cci'] + 200) / 4, 0, 100)
                logger.info("CCI создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка CCI: {e}")
            
            # MFI (Money Flow Index)
            try:
                result['mfi'] = talib.MFI(
                    high_prices, low_prices, close_prices, volume, timeperiod=14
                )
                logger.info("MFI создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка MFI: {e}")
            
            # Ultimate Oscillator
            try:
                result['ultimate_osc'] = talib.ULTOSC(
                    high_prices, low_prices, close_prices
                )
                logger.info("Ultimate Oscillator создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка Ultimate Oscillator: {e}")
            
            # Momentum индикаторы
            try:
                result['momentum'] = talib.MOM(close_prices, timeperiod=10)
                result['roc'] = talib.ROC(close_prices, timeperiod=10)
                
                # Нормализация momentum
                result['momentum_normalized'] = (result['momentum'] / close_prices) * 100
                logger.info("Momentum индикаторы созданы успешно")
            except Exception as e:
                logger.warning(f"Ошибка Momentum: {e}")
                
        except Exception as e:
            logger.error(f"Критическая ошибка в осцилляторах: {e}")
        
        return result
    
    @staticmethod
    def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """Создание трендовых индикаторов"""
        result = df.copy()
        
        if len(df) < 26:
            logger.warning("Недостаточно данных для трендовых индикаторов")
            return result
        
        try:
            close_prices = df['close'].astype(float)
            high_prices = df['high'].astype(float)
            low_prices = df['low'].astype(float)
            
            # MACD
            try:
                result['macd'], result['macd_signal'], result['macd_hist'] = talib.MACD(
                    close_prices, fastperiod=12, slowperiod=26, signalperiod=9
                )
                
                # Нормализация MACD (избегаем деления на ноль)
                result['macd_normalized'] = np.where(
                    close_prices != 0, 
                    result['macd'] / close_prices * 100, 
                    0
                )
                
                # MACD momentum для корреляции с RSI
                result['macd_momentum'] = result['macd'].diff()
                
                logger.info("MACD создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка MACD: {e}")
            
            # Bollinger Bands
            try:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
                
                # Избегаем деления на ноль
                bb_range = bb_upper - bb_lower
                result['bb_percent_b'] = np.where(
                    bb_range != 0,
                    (close_prices - bb_lower) / bb_range,
                    0.5  # средняя позиция если диапазон = 0
                )
                
                # Масштабирование к шкале RSI
                result['bb_percent_b_scaled'] = result['bb_percent_b'] * 100
                
                result['bb_width'] = np.where(
                    bb_middle != 0,
                    bb_range / bb_middle,
                    0
                )
                
                logger.info("Bollinger Bands созданы успешно")
            except Exception as e:
                logger.warning(f"Ошибка Bollinger Bands: {e}")
            
            # ADX (Average Directional Index)
            try:
                result['adx'] = talib.ADX(
                    high_prices, low_prices, close_prices, timeperiod=14
                )
                logger.info("ADX создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка ADX: {e}")
            
            # Дополнительные трендовые индикаторы
            try:
                # EMA разных периодов
                result['ema_9'] = talib.EMA(close_prices, timeperiod=9)
                result['ema_21'] = talib.EMA(close_prices, timeperiod=21)
                
                # SMA
                result['sma_20'] = talib.SMA(close_prices, timeperiod=20)
                
                # Отношения EMA
                result['ema_ratio_9_21'] = np.where(
                    result['ema_21'] != 0,
                    result['ema_9'] / result['ema_21'],
                    1
                )
                
                logger.info("Дополнительные трендовые индикаторы созданы")
            except Exception as e:
                logger.warning(f"Ошибка дополнительных трендовых: {e}")
                
        except Exception as e:
            logger.error(f"Критическая ошибка в трендовых индикаторах: {e}")
        
        return result
    
    @staticmethod
    def create_rsi_correlated_features(df: pd.DataFrame) -> pd.DataFrame:
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: RSI-коррелирующие признаки без утечки данных"""
        result = df.copy()
        
        if len(df) < 14:
            logger.warning("Недостаточно данных для RSI-коррелирующих индикаторов")
            return result
        
        try:
            close_prices = df['close'].astype(float)
            high_prices = df['high'].astype(float)
            low_prices = df['low'].astype(float)
            volume = df['volume'].astype(float)
            
            # 1. Stochastic (высокая корреляция с RSI)
            try:
                stoch_k, stoch_d = talib.STOCH(
                    high_prices, low_prices, close_prices, 
                    fastk_period=14, slowk_period=3, slowd_period=3
                )
                result['stoch_k'] = stoch_k
                result['stoch_d'] = stoch_d
                result['stoch_divergence'] = stoch_k - stoch_d
                result['stoch_momentum'] = stoch_k.diff()
                
                logger.info("Stochastic для RSI корреляции создан")
            except Exception as e:
                logger.warning(f"Ошибка Stochastic RSI: {e}")
            
            # 2. Williams %R нормализованный
            try:
                williams_r = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
                result['williams_r'] = williams_r
                result['williams_r_normalized'] = (williams_r + 100)  # Шкала 0-100
                
                logger.info("Williams %R нормализованный создан")
            except Exception as e:
                logger.warning(f"Ошибка Williams %R RSI: {e}")
            
            # 3. CCI нормализованный
            try:
                cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
                result['cci'] = cci
                result['cci_normalized'] = np.clip((cci + 200) / 4, 0, 100)
                
                logger.info("CCI нормализованный создан")
            except Exception as e:
                logger.warning(f"Ошибка CCI RSI: {e}")
            
            # 4. MACD momentum (корреляция с RSI изменениями)
            try:
                macd, _, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
                result['macd_momentum'] = macd.diff()
                
                # MACD oscillator (нормализованный)
                result['macd_oscillator'] = np.where(
                    close_prices != 0,
                    (macd / close_prices) * 1000,  # Масштабирование
                    0
                )
                
                logger.info("MACD momentum создан")
            except Exception as e:
                logger.warning(f"Ошибка MACD momentum: {e}")
            
            # 5. Bollinger %B масштабированный
            try:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
                bb_range = bb_upper - bb_lower
                
                bb_percent_b = np.where(
                    bb_range != 0,
                    (close_prices - bb_lower) / bb_range,
                    0.5
                )
                
                result['bb_percent_b'] = bb_percent_b
                result['bb_percent_b_scaled'] = bb_percent_b * 100  # Шкала как RSI
                
                # Ширина полос (волатильность)
                result['bb_width'] = np.where(
                    bb_middle != 0,
                    bb_range / bb_middle,
                    0
                )
                
                logger.info("Bollinger %B масштабированный создан")
            except Exception as e:
                logger.warning(f"Ошибка Bollinger %B: {e}")
            
            # 6. Momentum нормализованный
            try:
                momentum = talib.MOM(close_prices, timeperiod=14)
                result['momentum'] = momentum
                result['momentum_normalized'] = np.where(
                    close_prices != 0,
                    (momentum / close_prices) * 100,
                    0
                )
                
                logger.info("Momentum нормализованный создан")
            except Exception as e:
                logger.warning(f"Ошибка Momentum нормализованный: {e}")
            
            # 7. ROC (Rate of Change)
            try:
                roc = talib.ROC(close_prices, timeperiod=14)
                result['roc'] = roc
                
                logger.info("ROC создан")
            except Exception as e:
                logger.warning(f"Ошибка ROC: {e}")
            
            # 8. MFI (Money Flow Index) - "RSI с объемом"
            try:
                mfi = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
                result['mfi'] = mfi
                
                logger.info("MFI создан")
            except Exception as e:
                logger.warning(f"Ошибка MFI: {e}")
            
            # 9. Комбинированные индикаторы
            try:
                # Среднее значение основных осцилляторов
                oscillator_columns = []
                
                if 'stoch_k' in result.columns:
                    oscillator_columns.append('stoch_k')
                if 'williams_r_normalized' in result.columns:
                    oscillator_columns.append('williams_r_normalized')
                if 'cci_normalized' in result.columns:
                    oscillator_columns.append('cci_normalized')
                if 'bb_percent_b_scaled' in result.columns:
                    oscillator_columns.append('bb_percent_b_scaled')
                if 'mfi' in result.columns:
                    oscillator_columns.append('mfi')
                
                if len(oscillator_columns) >= 2:
                    result['oscillators_mean'] = result[oscillator_columns].mean(axis=1)
                    result['oscillators_std'] = result[oscillator_columns].std(axis=1)
                    
                    logger.info(f"Комбинированные осцилляторы созданы из {len(oscillator_columns)} индикаторов")
                
            except Exception as e:
                logger.warning(f"Ошибка комбинированных осцилляторов: {e}")
            
            # 10. Дивергенция между ценой и осцилляторами
            try:
                if len(df) > 5:
                    price_change = close_prices.pct_change(5)  # 5-периодное изменение цены
                    
                    if 'stoch_k' in result.columns:
                        stoch_change = result['stoch_k'].diff(5)
                        result['price_stoch_divergence'] = price_change - (stoch_change / 100)
                    
                    if 'williams_r_normalized' in result.columns:
                        williams_change = result['williams_r_normalized'].diff(5)
                        result['price_williams_divergence'] = price_change - (williams_change / 100)
                
                logger.info("Дивергенции созданы")
            except Exception as e:
                logger.warning(f"Ошибка дивергенций: {e}")
            
            logger.info("RSI-коррелирующие признаки созданы успешно")
            
        except Exception as e:
            logger.error(f"Критическая ошибка RSI-коррелирующих признаков: {e}")
        
        return result
    
    @staticmethod
    def create_existing_indicators_features(df: pd.DataFrame) -> pd.DataFrame:
        """Использование уже существующих индикаторов из accumulatedData"""
        result = df.copy()
        
        # Список уже существующих индикаторов в файле
        existing_indicators = [
            'atr', 'atr_stop', 'atr_to_price_ratio',
            'fast_ema', 'slow_ema', 'ema_fast_deviation',
            'pchange', 'avpchange', 'gma', 'gma_smoothed',
            'positionBetweenBands', 'bollinger_position',
            'choppiness_index', 'volatility_percent',
            'rsi_volatility', 'adx', 'rsi_divergence', 
            'rsi_delta', 'linear_regression'
        ]
        
        # Создаем производные признаки из существующих
        try:
            # ATR производные
            if 'atr' in result.columns and 'close' in result.columns:
                result['atr_percentage'] = np.where(
                    result['close'] != 0,
                    (result['atr'] / result['close']) * 100,
                    0
                )
            
            # EMA производные
            if 'fast_ema' in result.columns and 'slow_ema' in result.columns:
                result['ema_ratio'] = np.where(
                    result['slow_ema'] != 0,
                    result['fast_ema'] / result['slow_ema'],
                    1
                )
                result['ema_spread'] = result['fast_ema'] - result['slow_ema']
            
            # Волатильность производные
            if 'volatility_percent' in result.columns:
                result['volatility_ma'] = result['volatility_percent'].rolling(5).mean()
                result['volatility_change'] = result['volatility_percent'].diff()
            
            # RSI производные
            if 'rsi_volatility' in result.columns:
                result['rsi_volatility_ma'] = result['rsi_volatility'].rolling(5).mean()
            
            # Choppiness производные
            if 'choppiness_index' in result.columns:
                result['choppiness_ma'] = result['choppiness_index'].rolling(5).mean()
                result['choppiness_change'] = result['choppiness_index'].diff()
            
            logger.info("Производные от существующих индикаторов созданы")
            
        except Exception as e:
            logger.warning(f"Ошибка создания производных от существующих: {e}")
        
        return result
    
    @classmethod
    def create_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Создание всех признаков БЕЗ утечки данных"""
        try:
            logger.info(f"Начинаем создание признаков для {len(df)} строк")
            
            # Валидация и очистка данных
            df = cls.validate_data(df)
            
            if len(df) < 50:
                logger.warning(f"Мало данных для качественного анализа: {len(df)} строк")
            elif len(df) < 30:
                raise ValueError(f"Критически мало данных: {len(df)} строк. Минимум 30.")
            
            # 1. Используем существующие индикаторы из файла
            df = cls.create_existing_indicators_features(df)
            
            # 2. Создание базовых RSI признаков
            df = cls.create_rsi_features(df)
            
            # 3. RSI-коррелирующие признаки (приоритет!)
            df = cls.create_rsi_correlated_features(df)
            
            # 4. Дополнительные осцилляторы
            df = cls.create_oscillator_features(df)
            
            # 5. Трендовые индикаторы
            df = cls.create_trend_features(df)
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильная целевая переменная
            if 'rsi' in df.columns:
                # ВАЖНО: Создаем целевую переменную для обучения
                # НО она будет использоваться только при обучении, НЕ при предсказании!
                df['rsi_next'] = df['rsi'].shift(-1)
                
                # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Удаляем последнюю строку где rsi_next = NaN
                # Это гарантирует отсутствие утечки данных из будущего
                original_length = len(df)
                df = df[:-1].copy()  # Убираем последнюю строку
                
                logger.info(f"Целевая переменная создана. Удалена последняя строка: {original_length} -> {len(df)}")
                
                # Проверяем, что целевая переменная создана корректно
                if df['rsi_next'].isna().any():
                    logger.warning(f"В rsi_next есть NaN: {df['rsi_next'].isna().sum()} значений")
                
            else:
                logger.error("RSI не был создан!")
                raise ValueError("Не удалось создать RSI индикатор")
            
            # Удаляем бесконечные значения
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Статистика созданных признаков
            total_features = len(df.columns)
            numeric_features = len(df.select_dtypes(include=[np.number]).columns)
            
            # Подсчет RSI-коррелирующих признаков
            rsi_correlated = [col for col in df.columns if any(keyword in col.lower() 
                            for keyword in ['stoch', 'williams', 'cci', 'macd', 'bb_percent', 
                                          'mfi', 'oscillators', 'momentum', 'roc', 'rsi_'])]
            
            logger.info(f"Создание признаков завершено:")
            logger.info(f"  Всего признаков: {total_features}")
            logger.info(f"  Числовые признаки: {numeric_features}")
            logger.info(f"  RSI-коррелирующие: {len(rsi_correlated)}")
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