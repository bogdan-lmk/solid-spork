"""
Адаптер для работы с различными форматами CSV данных (ИСПРАВЛЕННАЯ ВЕРСИЯ ПОД accumulatedData)
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

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
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(filepath, sep=sep, **kwargs)
                if len(df.columns) > 1:
                    logger.info(f"CSV загружен с разделителем '{sep}', колонок: {len(df.columns)}")
                    return df
            except Exception:
                continue
        
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"CSV загружен с разделителем по умолчанию, колонок: {len(df.columns)}")
        return df
    
    @staticmethod
    def _robust_numeric_conversion_accumulated(series: pd.Series, column_name: str) -> pd.Series:
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Специализированная конвертация для accumulatedData"""
        
        # Если уже числовое, возвращаем как есть
        if pd.api.types.is_numeric_dtype(series):
            logger.debug(f"Колонка {column_name} уже числовая")
            return series
        
        # Особая обработка boolean колонок (rsi_divergence)
        if series.dtype == 'bool' or column_name in ['rsi_divergence']:
            logger.info(f"Конвертация boolean колонки {column_name}")
            return series.astype(int)
        
        # Обработка строковых колонок
        if series.dtype == 'object':
            try:
                # Попытка 1: Прямая конвертация
                converted = pd.to_numeric(series, errors='coerce')
                success_rate = 1 - converted.isna().sum() / len(series)
                
                if success_rate >= 0.8:
                    logger.info(f"Конвертация {column_name}: {success_rate:.1%} (прямая)")
                    return converted
                    
            except Exception:
                pass
            
            try:
                # Попытка 2: Очистка строк (для accumulatedData специфично)
                cleaned = series.astype(str)
                
                # Замена запятых на точки (европейский формат)
                cleaned = cleaned.str.replace(',', '.', regex=False)
                
                # Убираем пробелы
                cleaned = cleaned.str.strip()
                
                # Убираем символы валют и процентов
                cleaned = cleaned.str.replace(r'[€$%₽£¥]', '', regex=True)
                
                # Обработка специальных значений в accumulatedData
                cleaned = cleaned.replace(['nan', 'null', 'NaN', 'NULL', '', 'None', 'none'], np.nan)
                
                # Обработка научной нотации (если есть)
                cleaned = cleaned.str.replace(r'([+-]?\d+\.?\d*)[eE]([+-]?\d+)', 
                                            lambda m: str(float(m.group(0))), regex=True)
                
                converted = pd.to_numeric(cleaned, errors='coerce')
                success_rate = 1 - converted.isna().sum() / len(series)
                
                if success_rate >= 0.8:
                    logger.info(f"Конвертация {column_name}: {success_rate:.1%} (после очистки)")
                    return converted
                elif success_rate >= 0.5:
                    logger.warning(f"Конвертация {column_name}: {success_rate:.1%} (частичная)")
                    return converted
                    
            except Exception as e:
                logger.warning(f"Ошибка очистки {column_name}: {e}")
            
            try:
                # Попытка 3: Агрессивная очистка - извлекаем только числа
                numeric_pattern = r'([+-]?\d+\.?\d*)'
                extracted = series.astype(str).str.extract(numeric_pattern, expand=False)
                converted = pd.to_numeric(extracted, errors='coerce')
                success_rate = 1 - converted.isna().sum() / len(series)
                
                if success_rate >= 0.5:
                    logger.warning(f"Агрессивная конвертация {column_name}: {success_rate:.1%}")
                    return converted
                    
            except Exception as e:
                logger.warning(f"Ошибка агрессивной очистки {column_name}: {e}")
        
        # Если ничего не помогло
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось конвертировать {column_name}")
        logger.error(f"Тип данных: {series.dtype}, Примеры значений: {series.head(3).tolist()}")
        
        # Возвращаем хотя бы что-то
        return pd.to_numeric(series, errors='coerce')
    
    @staticmethod
    def _convert_datetime_accumulated(series: pd.Series, column_name: str) -> pd.Series:
        """Специальная конвертация datetime для accumulatedData"""
        try:
            # Попытка 1: Стандартная конвертация
            converted = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
            success_rate = 1 - converted.isna().sum() / len(series)
            
            if success_rate > 0.8:
                logger.info(f"Datetime {column_name}: {success_rate:.1%}")
                return converted
            
            # Попытка 2: Unix timestamp (если числа большие)
            if series.dtype in ['int64', 'float64'] or all(str(x).isdigit() for x in series.dropna().head(10)):
                try:
                    # Проверяем, похоже ли на Unix timestamp
                    test_values = pd.to_numeric(series.dropna().head(10), errors='coerce')
                    if test_values.min() > 1000000000:  # После 2001 года
                        converted = pd.to_datetime(series, unit='s', errors='coerce')
                        success_rate = 1 - converted.isna().sum() / len(series)
                        if success_rate > 0.8:
                            logger.info(f"Unix timestamp {column_name}: {success_rate:.1%}")
                            return converted
                except Exception:
                    pass
            
            # Попытка 3: Попробуем разные форматы
            formats_to_try = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%d/%m/%Y %H:%M:%S',
                '%d/%m/%Y',
                '%m/%d/%Y %H:%M:%S',
                '%m/%d/%Y'
            ]
            
            for fmt in formats_to_try:
                try:
                    converted = pd.to_datetime(series, format=fmt, errors='coerce')
                    success_rate = 1 - converted.isna().sum() / len(series)
                    if success_rate > 0.8:
                        logger.info(f"Datetime {column_name} с форматом {fmt}: {success_rate:.1%}")
                        return converted
                except Exception:
                    continue
            
            # Если ничего не помогло, возвращаем исходное с предупреждением
            logger.warning(f"Низкий успех datetime конвертации {column_name}: {success_rate:.1%}")
            return pd.to_datetime(series, errors='coerce')
            
        except Exception as e:
            logger.error(f"Ошибка конвертации datetime {column_name}: {e}")
            return pd.to_datetime(series, errors='coerce')
    
    @staticmethod
    def clean_accumulated_data(df: pd.DataFrame) -> pd.DataFrame:
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Очистка данных accumulatedData формата"""
        df = df.copy()
        
        logger.info("Обнаружены данные accumulatedData")
        logger.info(f"Исходный размер данных: {df.shape}")
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Обработка всех колонок по типам
        
        # 1. Временные колонки
        time_columns = ['open_time', 'close_time']
        for col in time_columns:
            if col in df.columns:
                df[col] = DataAdapter._convert_datetime_accumulated(df[col], col)
        
        # 2. Основные OHLCV колонки (критически важные)
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_columns:
            if col in df.columns:
                df[col] = DataAdapter._robust_numeric_conversion_accumulated(df[col], col)
        
        # 3. Технические индикаторы (большинство колонок)
        indicator_columns = [
            'atr', 'atr_stop', 'atr_to_price_ratio',
            'fast_ema', 'slow_ema', 'ema_fast_deviation',
            'pchange', 'avpchange', 'gma', 'gma_smoothed',
            'positionBetweenBands', 'bollinger_position',
            'choppiness_index', 'volatility_percent',
            'rsi_volatility', 'adx', 'rsi_delta', 'linear_regression'
        ]
        
        for col in indicator_columns:
            if col in df.columns:
                df[col] = DataAdapter._robust_numeric_conversion_accumulated(df[col], col)
        
        # 4. Специальные колонки
        if 'rsi_divergence' in df.columns:
            # Boolean колонка
            df['rsi_divergence'] = DataAdapter._robust_numeric_conversion_accumulated(df['rsi_divergence'], 'rsi_divergence')
        
        # Сигнальные колонки (могут быть строковыми)
        signal_columns = ['ema_cross_signal', 'signal_gma', 'Bol']
        for col in signal_columns:
            if col in df.columns:
                # Для сигнальных колонок делаем label encoding
                if df[col].dtype == 'object':
                    try:
                        # Простое кодирование уникальных значений
                        unique_vals = df[col].dropna().unique()
                        if len(unique_vals) <= 10:  # Если мало уникальных значений
                            df[col] = pd.Categorical(df[col]).codes
                            df[col] = df[col].replace(-1, np.nan)  # -1 это NaN в codes
                            logger.info(f"Label encoding для {col}: {len(unique_vals)} значений")
                        else:
                            # Слишком много значений, пробуем числовую конвертацию
                            df[col] = DataAdapter._robust_numeric_conversion_accumulated(df[col], col)
                    except Exception as e:
                        logger.warning(f"Ошибка обработки сигнальной колонки {col}: {e}")
                        df[col] = pd.Categorical(df[col]).codes
        
        # 5. Сортировка по времени
        if 'open_time' in df.columns and not df['open_time'].isna().all():
            try:
                df = df.sort_values('open_time').reset_index(drop=True)
                logger.info("Данные отсортированы по open_time")
            except Exception as e:
                logger.warning(f"Не удалось отсортировать по времени: {e}")
        
        # 6. Валидация OHLC данных
        ohlc_available = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
        if len(ohlc_available) >= 4:
            try:
                # Проверка логических соотношений OHLC
                before_fixes = len(df)
                
                # Удаляем строки с некорректными ценами
                invalid_prices = (
                    (df['open'] <= 0) | (df['high'] <= 0) | 
                    (df['low'] <= 0) | (df['close'] <= 0) |
                    (df['high'] < df['low']) |
                    (df['high'] < df[['open', 'close']].max(axis=1)) |
                    (df['low'] > df[['open', 'close']].min(axis=1))
                )
                
                if invalid_prices.any():
                    logger.warning(f"Найдено некорректных OHLC строк: {invalid_prices.sum()}")
                    df = df[~invalid_prices].reset_index(drop=True)
                    logger.info(f"Удалено строк: {before_fixes - len(df)}")
                
            except Exception as e:
                logger.warning(f"Ошибка валидации OHLC: {e}")
        
        # 7. Обработка выбросов в ценах
        if 'close' in df.columns and len(df) > 1:
            try:
                # Находим экстремальные изменения цены
                price_changes = df['close'].pct_change().abs()
                extreme_threshold = 0.20  # 20% изменение за период
                extreme_changes = price_changes > extreme_threshold
                
                if extreme_changes.sum() > 0:
                    logger.warning(f"Найдено экстремальных изменений цены: {extreme_changes.sum()}")
                    
                    # Сглаживаем выбросы вместо удаления
                    extreme_indices = df[extreme_changes].index
                    for idx in extreme_indices:
                        if idx > 0 and idx < len(df) - 1:
                            df.loc[idx, 'close'] = (df.loc[idx-1, 'close'] + df.loc[idx+1, 'close']) / 2
                            
            except Exception as e:
                logger.warning(f"Ошибка обработки выбросов: {e}")
        
        # 8. Удаление дубликатов по времени
        if 'open_time' in df.columns:
            try:
                initial_rows = len(df)
                df = df.drop_duplicates(subset=['open_time'], keep='first')
                removed_duplicates = initial_rows - len(df)
                
                if removed_duplicates > 0:
                    logger.info(f"Удалено дубликатов по времени: {removed_duplicates}")
                    
            except Exception as e:
                logger.warning(f"Ошибка удаления дубликатов: {e}")
        
        # 9. Финальная обработка NaN
        # Для числовых колонок используем forward fill
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].ffill()
            
            # Если остались NaN в начале, используем backward fill
            df[numeric_cols] = df[numeric_cols].bfill()
        
        # 10. Финальные проверки
        final_numeric_cols = df.select_dtypes(include=[np.number]).columns
        final_object_cols = df.select_dtypes(include=['object']).columns
        
        logger.info(f"Обработка завершена. Итоговый размер: {df.shape}")
        logger.info(f"Числовые колонки: {len(final_numeric_cols)}")
        logger.info(f"Object колонки: {len(final_object_cols)}")
        
        if len(final_object_cols) > 0:
            logger.warning(f"Остались object колонки: {list(final_object_cols)}")
            for col in final_object_cols:
                if col not in ['open_time', 'close_time']:  # Эти могут быть datetime
                    logger.warning(f"  {col}: {df[col].dtype}, примеры: {df[col].dropna().head(3).tolist()}")
        
        return df
    
    @staticmethod
    def _identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Классификация колонок по типам для accumulatedData"""
        datetime_columns = ['open_time', 'close_time']
        
        numeric_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'atr', 'atr_stop', 'atr_to_price_ratio',
            'fast_ema', 'slow_ema', 'ema_fast_deviation',
            'pchange', 'avpchange', 'gma', 'gma_smoothed',
            'positionBetweenBands', 'bollinger_position',
            'choppiness_index', 'volatility_percent',
            'rsi_volatility', 'adx', 'rsi_delta', 'linear_regression',
            'rsi_divergence'  # boolean -> numeric
        ]
        
        categorical_columns = ['ema_cross_signal', 'signal_gma', 'Bol']
        
        # Фильтруем только существующие колонки
        datetime_cols = [col for col in datetime_columns if col in df.columns]
        numeric_cols = [col for col in numeric_columns if col in df.columns]
        categorical_cols = [col for col in categorical_columns if col in df.columns]
        
        # Остальные колонки
        all_classified = set(datetime_cols + numeric_cols + categorical_cols)
        other_cols = [col for col in df.columns if col not in all_classified]
        
        return {
            'datetime': datetime_cols,
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'other': other_cols
        }
    
    @staticmethod
    def adapt_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Адаптация данных к формату OHLCV (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
        result = df.copy()
        format_type = DataAdapter.detect_format(result)
        
        if format_type == 'ohlcv':
            # Это наш случай - accumulatedData уже содержит OHLCV
            result = DataAdapter.clean_accumulated_data(result)
        elif format_type == 'price_only':
            if 'close' not in result.columns:
                raise ValueError("Не найдена колонка 'close'")
            
            # Создание синтетического OHLC
            result['open'] = result['close'].shift(1).fillna(result['close'].iloc[0])
            
            volatility = result['close'].pct_change().rolling(20).std().fillna(0.01)
            
            result['high'] = result[['open', 'close']].max(axis=1) * (1 + volatility * 0.5)
            result['low'] = result[['open', 'close']].min(axis=1) * (1 - volatility * 0.5)
            
            logger.info("Создан синтетический OHLC")
            
        elif format_type == 'indicators_only':
            raise ValueError("Нужны OHLCV данные")
        else:
            raise ValueError(f"Неизвестный формат: {list(result.columns)}")
        
        # Добавляем volume если отсутствует
        if 'volume' not in result.columns:
            if 'close' in result.columns:
                price_change = result['close'].pct_change().abs().fillna(0)
                base_volume = 5000000
                result['volume'] = (base_volume * (1 + price_change * 10)).astype(int)
            else:
                result['volume'] = np.random.randint(1000000, 10000000, len(result))
            logger.info("Добавлен синтетический volume")
        
        return result
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, any]:
        """Комплексная проверка качества данных для accumulatedData"""
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'datetime_columns': [],
            'numeric_columns': [],
            'missing_data': {},
            'data_issues': [],
            'quality_score': 0.0
        }
        
        # Анализ типов колонок
        column_types = DataAdapter._identify_column_types(df)
        report['datetime_columns'] = column_types['datetime']
        report['numeric_columns'] = column_types['numeric']
        
        # Анализ пропущенных данных
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                report['missing_data'][col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(df) * 100)
                }
        
        # Проверка качества OHLC данных
        ohlc_columns = ['open', 'high', 'low', 'close']
        available_ohlc = [col for col in ohlc_columns if col in df.columns]
        
        if len(available_ohlc) == 4:
            try:
                # Проверка логических соотношений
                invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
                invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
                
                if invalid_high > 0:
                    report['data_issues'].append(f"Некорректных High значений: {invalid_high}")
                if invalid_low > 0:
                    report['data_issues'].append(f"Некорректных Low значений: {invalid_low}")
            except Exception:
                pass
        
        # Проверка временных данных
        if 'open_time' in df.columns:
            try:
                time_series = pd.to_datetime(df['open_time'], errors='coerce').dropna()
                if len(time_series) > 1:
                    # Проверка на дубликаты времени
                    duplicates = time_series.duplicated().sum()
                    if duplicates > 0:
                        report['data_issues'].append(f"Дубликатов по времени: {duplicates}")
            except Exception:
                pass
        
        # Расчет общего балла качества
        quality_factors = []
        
        # Фактор полноты данных
        total_missing = sum(info['count'] for info in report['missing_data'].values())
        completeness = 1 - (total_missing / (len(df) * len(df.columns)))
        quality_factors.append(completeness * 0.4)
        
        # Фактор корректности структуры
        structure_score = 1.0 if len(available_ohlc) >= 4 else 0.5
        quality_factors.append(structure_score * 0.3)
        
        # Фактор отсутствия критических проблем
        critical_issues = len([issue for issue in report['data_issues'] if 'Некорректных' in issue])
        issues_score = max(0, 1 - critical_issues * 0.1)
        quality_factors.append(issues_score * 0.3)
        
        report['quality_score'] = sum(quality_factors)
        
        return report