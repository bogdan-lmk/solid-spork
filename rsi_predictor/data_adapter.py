"""
Адаптер для работы с различными форматами CSV данных
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataAdapter:
    """Адаптер для работы с различными форматами CSV данных"""
    
    @staticmethod
    def detect_format(df: pd.DataFrame) -> str:
        """Определение формата данных"""
        columns = set(df.columns.str.lower())
        
        if {'open', 'high', 'low', 'close'}.issubset(columns):
            return 'ohlcv'
        elif {'choppiness_index', 'volatility_percent', 'rsi_delta'}.issubset(columns):
            return 'indicators_only'
        elif 'close' in columns:
            return 'price_only'
        else:
            return 'unknown'
    
    @staticmethod
    def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
        """Загрузка CSV с автоопределением разделителей"""
        try:
            # Пробуем разные варианты
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(filepath, sep=sep, **kwargs)
                    if len(df.columns) > 1:  # Успешное разделение
                        logger.info(f"CSV загружен с разделителем '{sep}', колонок: {len(df.columns)}")
                        return df
                except:
                    continue
            
            # Если ничего не сработало, используем запятую по умолчанию
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"CSV загружен с разделителем по умолчанию, колонок: {len(df.columns)}")
            return df
            
        except Exception as e:
            raise ValueError(f"Не удалось загрузить CSV файл: {e}")
    
    @staticmethod
    def clean_accumulated_data(df: pd.DataFrame) -> pd.DataFrame:
        """Очистка данных из accumulatedData файлов"""
        df = df.copy()
        
        # Конвертация строковых колонок в числовые
        numeric_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'atr', 'atr_stop', 'atr_to_price_ratio',
            'fast_ema', 'slow_ema', 'ema_fast_deviation',
            'pchange', 'avpchange', 'gma', 'gma_smoothed',
            'positionBetweenBands', 'bollinger_position',
            'choppiness_index', 'volatility_percent',
            'rsi_volatility', 'adx', 'rsi_delta', 'linear_regression'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    # Заменяем запятые на точки (если есть)
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.replace(',', '.')
                    
                    # Конвертируем в числовой тип
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                except Exception as e:
                    logger.warning(f"Не удалось конвертировать колонку {col}: {e}")
        
        # Обработка временных меток
        if 'open_time' in df.columns:
            try:
                df['open_time'] = pd.to_datetime(df['open_time'])
                df = df.sort_values('open_time').reset_index(drop=True)
            except:
                logger.warning("Не удалось обработать временные метки")
        
        # Удаление строк с критичными NaN (в OHLC)
        critical_columns = ['open', 'high', 'low', 'close']
        before_rows = len(df)
        df = df.dropna(subset=critical_columns)
        after_rows = len(df)
        
        if before_rows != after_rows:
            logger.info(f"Удалено {before_rows - after_rows} строк с NaN в OHLC данных")
        
        # Заполнение остальных NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    @staticmethod
    def adapt_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Адаптация данных к формату OHLCV"""
        df = df.copy()
        format_type = DataAdapter.detect_format(df)
        
        if format_type == 'ohlcv':
            # Проверяем, не являются ли данные из accumulatedData
            if 'open_time' in df.columns and 'atr' in df.columns:
                logger.info("Обнаружены данные accumulatedData - применяем специальную очистку")
                df = DataAdapter.clean_accumulated_data(df)
            else:
                # Стандартная обработка OHLCV
                required_cols = ['open', 'high', 'low', 'close']
                df = df.rename(columns={col: col.lower() for col in df.columns})
                
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    raise ValueError(f"Отсутствуют колонки: {missing}")
                    
        elif format_type == 'price_only':
            # Только цена закрытия - создаем OHLC
            if 'close' not in df.columns:
                raise ValueError("Не найдена колонка 'close'")
            
            df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
            df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(df)))
            df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(df)))
            
            logger.info("Создан синтетический OHLC из цены закрытия")
            
        elif format_type == 'indicators_only':
            raise ValueError("Данные содержат только индикаторы без цен. Нужны OHLCV данные.")
        
        else:
            raise ValueError(f"Неизвестный формат данных. Колонки: {list(df.columns)}")
        
        # Добавляем volume если отсутствует
        if 'volume' not in df.columns:
            df['volume'] = np.random.randint(1000000, 10000000, len(df))
            logger.info("Добавлен синтетический объем торгов")
        
        return df