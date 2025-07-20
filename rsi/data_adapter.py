"""
Адаптер для работы с различными форматами CSV данных (исправленная версия)
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
                except Exception:
                    continue
            
            # Если ничего не сработало, используем запятую по умолчанию
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"CSV загружен с разделителем по умолчанию, колонок: {len(df.columns)}")
            return df
            
        except Exception as e:
            raise ValueError(f"Не удалось загрузить CSV файл: {e}")
    
    @staticmethod
    def clean_accumulated_data(df: pd.DataFrame) -> pd.DataFrame:
        """Очистка данных из accumulatedData файлов (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
        df = df.copy()
        
        # ИСПРАВЛЕНИЕ: Сначала обрабатываем временные колонки отдельно
        time_columns = ['open_time', 'close_time']
        for col in time_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"Временная колонка {col} успешно обработана")
                except Exception as e:
                    logger.warning(f"Не удалось обработать временную колонку {col}: {e}")
        
        # Определяем только числовые колонки (ИСКЛЮЧАЯ временные)
        numeric_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'atr', 'atr_stop', 'atr_to_price_ratio',
            'fast_ema', 'slow_ema', 'ema_fast_deviation',
            'pchange', 'avpchange', 'gma', 'gma_smoothed',
            'positionBetweenBands', 'bollinger_position',
            'choppiness_index', 'volatility_percent',
            'rsi_volatility', 'adx', 'rsi_divergence', 'rsi_delta', 'linear_regression'
        ]
        
        # ИСПРАВЛЕНИЕ: Конвертируем только числовые колонки, исключая временные
        for col in numeric_columns:
            if col in df.columns and col not in time_columns:  # Исключаем временные колонки!
                try:
                    # Заменяем запятые на точки (если есть)
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.replace(',', '.')
                    
                    # Конвертируем в числовой тип
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.debug(f"Колонка {col} конвертирована в числовой тип")
                    
                except Exception as e:
                    logger.warning(f"Не удалось конвертировать колонку {col}: {e}")
        
        # Сортировка по времени (если возможно)
        if 'open_time' in df.columns and not df['open_time'].isna().all():
            try:
                df = df.sort_values('open_time').reset_index(drop=True)
                logger.info("Данные отсортированы по open_time")
            except Exception as e:
                logger.warning(f"Не удалось отсортировать по времени: {e}")
        
        # Удаление строк с критичными NaN (в OHLC)
        critical_columns = ['open', 'high', 'low', 'close']
        available_critical = [col for col in critical_columns if col in df.columns]
        
        if available_critical:
            before_rows = len(df)
            df = df.dropna(subset=available_critical)
            after_rows = len(df)
            
            if before_rows != after_rows:
                logger.info(f"Удалено {before_rows - after_rows} строк с NaN в OHLC данных")
        
        # Заполнение остальных NaN в числовых колонках
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        # Проверка результата
        logger.info(f"Обработка завершена. Итоговый размер: {df.shape}")
        logger.info(f"Числовые колонки: {len(df.select_dtypes(include=[np.number]).columns)}")
        logger.info(f"Временные колонки: {len(df.select_dtypes(include=['datetime64']).columns)}")
        
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
    
    @staticmethod
    def debug_dataframe(df: pd.DataFrame, name: str = "DataFrame"):
        """Отладочная информация о DataFrame"""
        logger.info(f"\n🔍 DEBUG: {name}")
        logger.info(f"Размер: {df.shape}")
        logger.info(f"Типы данных:")
        
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].count()
            null_count = len(df) - non_null
            
            # Показываем примеры значений для object колонок
            if dtype == 'object' and non_null > 0:
                sample_values = df[col].dropna().head(2).tolist()
                logger.info(f"  {col}: {dtype} (non-null: {non_null}, null: {null_count}) примеры: {sample_values}")
            else:
                logger.info(f"  {col}: {dtype} (non-null: {non_null}, null: {null_count})")
        
        # Выявляем потенциальные проблемы
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            logger.warning(f"⚠️ Найдены object колонки: {list(object_cols)}")
            
        # Проверяем на смешанные типы в колонках
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_types = set(type(x).__name__ for x in df[col].dropna().head(10))
                if len(unique_types) > 1:
                    logger.warning(f"⚠️ Смешанные типы в {col}: {unique_types}")