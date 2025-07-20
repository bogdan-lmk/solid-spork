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
        
        # ГЛАВНОЕ ИСПРАВЛЕНИЕ: Конвертируем только числовые колонки, исключая временные
        for col in numeric_columns:
            if col in df.columns and col not in time_columns:  # Исключаем временные колонки!
                try:
                    # Конвертируем в строку и заменяем запятые на точки
                    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                    
                    # Убираем пробелы и другие символы
                    df[col] = df[col].str.strip()
                    
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

def validate_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Валидация OHLCV данных"""
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")
    
    # Проверка логических соотношений OHLC
    df_clean = df.copy()
    
    # High должен быть >= max(open, close)
    invalid_high = df_clean['high'] < df_clean[['open', 'close']].max(axis=1)
    if invalid_high.any():
        logger.warning(f"Найдено {invalid_high.sum()} строк с некорректным high")
        df_clean.loc[invalid_high, 'high'] = df_clean.loc[invalid_high, ['open', 'close']].max(axis=1)
    
    # Low должен быть <= min(open, close)
    invalid_low = df_clean['low'] > df_clean[['open', 'close']].min(axis=1)
    if invalid_low.any():
        logger.warning(f"Найдено {invalid_low.sum()} строк с некорректным low")
        df_clean.loc[invalid_low, 'low'] = df_clean.loc[invalid_low, ['open', 'close']].min(axis=1)
    
    # Удаление строк с отрицательными ценами
    negative_prices = (df_clean[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
    if negative_prices.any():
        logger.warning(f"Удалено {negative_prices.sum()} строк с отрицательными ценами")
        df_clean = df_clean[~negative_prices]
    
    return df_clean

def convert_numeric_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Конвертация колонок в числовой формат с обработкой различных форматов"""
    df_result = df.copy()
    
    if columns is None:
        # Определяем потенциально числовые колонки
        columns = [col for col in df.columns if col not in ['open_time', 'close_time']]
    
    for col in columns:
        if col in df_result.columns:
            try:
                # Если уже числовая, пропускаем
                if pd.api.types.is_numeric_dtype(df_result[col]):
                    continue
                
                # Конвертируем в строку и очищаем
                series = df_result[col].astype(str)
                
                # Заменяем запятые на точки (европейский формат)
                series = series.str.replace(',', '.', regex=False)
                
                # Удаляем пробелы
                series = series.str.strip()
                
                # Удаляем символы валют и процентов
                series = series.str.replace(r'[€$%]', '', regex=True)
                
                # Обрабатываем 'nan', 'null', пустые строки
                series = series.replace(['nan', 'null', 'NaN', 'NULL', ''], np.nan)
                
                # Конвертируем в числовой формат
                df_result[col] = pd.to_numeric(series, errors='coerce')
                
                logger.debug(f"Колонка {col} успешно конвертирована в числовой формат")
                
            except Exception as e:
                logger.warning(f"Не удалось конвертировать колонку {col}: {e}")
    
    return df_result

def detect_decimal_separator(df: pd.DataFrame, sample_columns: list = None) -> str:
    """Определение десятичного разделителя в данных"""
    if sample_columns is None:
        sample_columns = ['open', 'high', 'low', 'close', 'volume']
    
    comma_count = 0
    dot_count = 0
    
    for col in sample_columns:
        if col in df.columns:
            # Берем первые 100 не-null значений
            sample_data = df[col].dropna().head(100).astype(str)
            
            for value in sample_data:
                if ',' in value and '.' in value:
                    # Если есть и запятая и точка, вероятно запятая - тысячи, точка - десятичная
                    dot_count += 1
                elif ',' in value:
                    comma_count += 1
                elif '.' in value:
                    dot_count += 1
    
    if comma_count > dot_count:
        logger.info("Обнаружен европейский формат чисел (запятая как десятичный разделитель)")
        return ','
    else:
        logger.info("Обнаружен американский формат чисел (точка как десятичный разделитель)")
        return '.'

def clean_data_auto(df: pd.DataFrame) -> pd.DataFrame:
    """Автоматическая очистка данных с определением формата"""
    logger.info("Начинаем автоматическую очистку данных")
    
    df_clean = df.copy()
    
    # Определяем десятичный разделитель
    decimal_sep = detect_decimal_separator(df_clean)
    
    # Конвертируем числовые колонки
    if decimal_sep == ',':
        # Европейский формат - заменяем запятые на точки
        df_clean = convert_numeric_columns(df_clean)
    else:
        # Американский формат - конвертируем как есть
        df_clean = convert_numeric_columns(df_clean)
    
    # Обрабатываем временные колонки
    time_columns = ['open_time', 'close_time', 'timestamp', 'date', 'time']
    for col in time_columns:
        if col in df_clean.columns:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                logger.info(f"Временная колонка {col} обработана")
            except Exception as e:
                logger.warning(f"Не удалось обработать временную колонку {col}: {e}")
    
    # Валидация OHLCV если это торговые данные
    if all(col in df_clean.columns for col in ['open', 'high', 'low', 'close']):
        try:
            df_clean = validate_ohlcv_data(df_clean)
            logger.info("OHLCV данные валидированы")
        except Exception as e:
            logger.warning(f"Ошибка валидации OHLCV: {e}")
    
    # Финальная очистка
    initial_rows = len(df_clean)
    
    # Удаляем полностью пустые строки
    df_clean = df_clean.dropna(how='all')
    
    # Заполняем NaN в числовых колонках
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df_clean[numeric_cols] = df_clean[numeric_cols].ffill().bfill()
    
    final_rows = len(df_clean)
    
    if initial_rows != final_rows:
        logger.info(f"Удалено {initial_rows - final_rows} строк в процессе очистки")
    
    logger.info(f"Автоматическая очистка завершена. Итоговый размер: {df_clean.shape}")
    
    return df_clean