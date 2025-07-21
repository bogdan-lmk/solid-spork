"""
Адаптер для работы с различными форматами CSV данных
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
    def _identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Классификация колонок по типам"""
        datetime_patterns = ['time', 'date', 'timestamp']
        numeric_patterns = ['open', 'high', 'low', 'close', 'volume', 'price', 'rsi', 'atr', 
                           'ema', 'sma', 'macd', 'bollinger', 'stoch', 'adx', 'cci', 'mfi']
        
        datetime_cols = []
        numeric_cols = []
        other_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            if any(pattern in col_lower for pattern in datetime_patterns):
                datetime_cols.append(col)
            elif any(pattern in col_lower for pattern in numeric_patterns):
                numeric_cols.append(col)
            else:
                # Пытаемся определить по содержимому
                sample_data = df[col].dropna().head(100).astype(str)
                
                # Проверка на datetime
                datetime_like = any(
                    len(val) > 8 and ('-' in val or '/' in val or ':' in val)
                    for val in sample_data
                )
                
                if datetime_like:
                    datetime_cols.append(col)
                else:
                    # Проверка на числовые данные
                    numeric_count = 0
                    for val in sample_data:
                        cleaned_val = val.replace(',', '.').replace(' ', '').strip()
                        try:
                            float(cleaned_val)
                            numeric_count += 1
                        except ValueError:
                            pass
                    
                    if numeric_count / len(sample_data) > 0.8:
                        numeric_cols.append(col)
                    else:
                        other_cols.append(col)
        
        return {
            'datetime': datetime_cols,
            'numeric': numeric_cols,
            'other': other_cols
        }
    
    @staticmethod
    def _convert_datetime_columns(df: pd.DataFrame, datetime_cols: List[str]) -> pd.DataFrame:
        """Конвертация datetime колонок"""
        result = df.copy()
        
        for col in datetime_cols:
            if col in result.columns:
                try:
                    # Пытаемся разные форматы
                    if result[col].dtype == 'int64':
                        # Unix timestamp
                        result[col] = pd.to_datetime(result[col], unit='s', errors='coerce')
                    else:
                        result[col] = pd.to_datetime(result[col], errors='coerce')
                    
                    success_rate = 1 - result[col].isna().sum() / len(result)
                    if success_rate < 0.5:
                        logger.warning(f"Низкий успех конвертации datetime для {col}: {success_rate:.1%}")
                    else:
                        logger.info(f"Datetime колонка {col} обработана успешно: {success_rate:.1%}")
                        
                except Exception as e:
                    logger.error(f"Ошибка конвертации datetime колонки {col}: {e}")
        
        return result
    
    @staticmethod
    def _convert_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Конвертация числовых колонок"""
        result = df.copy()
        
        for col in numeric_cols:
            if col in result.columns:
                try:
                    if pd.api.types.is_numeric_dtype(result[col]):
                        continue
                    
                    # Преобразование в строку и очистка
                    series = result[col].astype(str)
                    series = series.str.replace(',', '.', regex=False)
                    series = series.str.strip()
                    series = series.str.replace(r'[€$%\s]', '', regex=True)
                    series = series.replace(['nan', 'null', 'NaN', 'NULL', '', 'None'], np.nan)
                    
                    # Конвертация в числовой формат
                    result[col] = pd.to_numeric(series, errors='coerce')
                    
                    success_rate = 1 - result[col].isna().sum() / len(result)
                    if success_rate < 0.8:
                        logger.warning(f"Низкий успех конвертации numeric для {col}: {success_rate:.1%}")
                    else:
                        logger.debug(f"Numeric колонка {col} обработана: {success_rate:.1%}")
                        
                except Exception as e:
                    logger.error(f"Ошибка конвертации numeric колонки {col}: {e}")
        
        return result
    
    @staticmethod
    def clean_accumulated_data(df: pd.DataFrame) -> pd.DataFrame:
        """Очистка данных из accumulatedData файлов (УЛУЧШЕННАЯ ВЕРСИЯ)"""
        df = df.copy()
        
        logger.info("Обнаружены данные accumulatedData")
        
        # Обрабатываем временные колонки
        time_columns = ['open_time', 'close_time']
        for col in time_columns:
            if col in df.columns:
                try:
                    # Пробуем разные форматы datetime
                    df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    success_rate = (1 - df[col].isna().sum() / len(df)) * 100
                    
                    if success_rate > 50:
                        logger.info(f"Datetime колонка {col} обработана успешно: {success_rate:.1f}%")
                    else:
                        logger.warning(f"Низкий успех конвертации datetime для {col}: {success_rate:.1f}%")
                        
                except Exception as e:
                    logger.warning(f"Не удалось обработать временную колонку {col}: {e}")
        
        # Определяем числовые колонки
        numeric_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'atr', 'atr_stop', 'atr_to_price_ratio',
            'fast_ema', 'slow_ema', 'ema_fast_deviation',
            'pchange', 'avpchange', 'gma', 'gma_smoothed',
            'positionBetweenBands', 'bollinger_position',
            'choppiness_index', 'volatility_percent',
            'rsi_volatility', 'adx', 'rsi_divergence', 'rsi_delta', 'linear_regression'
        ]
        
        # Конвертируем числовые колонки
        for col in numeric_columns:
            if col in df.columns and col not in time_columns:
                try:
                    # Сначала пробуем конвертировать как datetime (для некоторых странных случаев)
                    if df[col].dtype == 'object':
                        temp_datetime = pd.to_datetime(df[col], errors='coerce')
                        datetime_success_rate = (1 - temp_datetime.isna().sum() / len(df)) * 100
                        
                        if datetime_success_rate > 50:
                            logger.info(f"Datetime колонка {col} обработана успешно: {datetime_success_rate:.1f}%")
                            df[col] = temp_datetime
                            continue
                        elif datetime_success_rate > 0:
                            logger.warning(f"Низкий успех конвертации datetime для {col}: {datetime_success_rate:.1f}%")
                    
                    # Если не datetime, то конвертируем как число
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        numeric_success_rate = (1 - df[col].isna().sum() / len(df)) * 100
                        if numeric_success_rate < 50:
                            logger.warning(f"Низкий успех конвертации numeric для {col}: {numeric_success_rate:.1f}%")
                            
                except Exception as e:
                    logger.warning(f"Не удалось конвертировать колонку {col}: {e}")
        
        # Сортировка по времени
        if 'open_time' in df.columns and not df['open_time'].isna().all():
            try:
                df = df.sort_values('open_time').reset_index(drop=True)
                logger.info("Данные отсортированы по open_time")
            except Exception as e:
                logger.warning(f"Не удалось отсортировать по времени: {e}")
        
        # НОВОЕ: Удаление экстремальных изменений цены (выбросов)
        if 'close' in df.columns and df['close'].dtype in ['float64', 'int64']:
            try:
                # Вычисляем процентные изменения
                price_changes = df['close'].pct_change().abs()
                
                # Определяем выбросы (изменения больше 20% за период)
                extreme_threshold = 0.20  # 20%
                extreme_changes = price_changes > extreme_threshold
                
                if extreme_changes.sum() > 0:
                    logger.warning(f"Найдено экстремальных изменений цены: {extreme_changes.sum()}")
                    
                    # Опционально: можно удалить или сгладить выбросы
                    # df = df[~extreme_changes]  # Удаление
                    # Или сглаживание:
                    extreme_indices = df[extreme_changes].index
                    for idx in extreme_indices:
                        if idx > 0 and idx < len(df) - 1:
                            # Заменяем выброс средним от соседних значений
                            df.loc[idx, 'close'] = (df.loc[idx-1, 'close'] + df.loc[idx+1, 'close']) / 2
                            
            except Exception as e:
                logger.warning(f"Ошибка при обработке экстремальных изменений: {e}")
        
        # НОВОЕ: Удаление дубликатов по времени
        if 'open_time' in df.columns:
            try:
                initial_rows = len(df)
                df = df.drop_duplicates(subset=['open_time'], keep='first')
                removed_duplicates = initial_rows - len(df)
                
                if removed_duplicates > 0:
                    logger.info(f"Удалено дубликатов по времени: {removed_duplicates}")
                    
            except Exception as e:
                logger.warning(f"Ошибка при удалении дубликатов: {e}")
        
        # Удаление строк с критичными NaN
        critical_columns = ['open', 'high', 'low', 'close']
        available_critical = [col for col in critical_columns if col in df.columns]
        
        if available_critical:
            before_rows = len(df)
            df = df.dropna(subset=available_critical)
            after_rows = len(df)
            
            if before_rows != after_rows:
                logger.info(f"Удалено строк с NaN в OHLC данных: {before_rows - after_rows}")
        
        # Заполнение остальных NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        logger.info(f"Обработка завершена. Итоговый размер: {df.shape}")
        
        return df
    
    @staticmethod
    def _validate_ohlc_data(df: pd.DataFrame) -> pd.DataFrame:
        """Валидация OHLC данных"""
        result = df.copy()
        ohlc_columns = ['open', 'high', 'low', 'close']
        available_ohlc = [col for col in ohlc_columns if col in result.columns]
        
        if len(available_ohlc) < 4:
            return result
        
        # Проверка логических соотношений
        errors_fixed = 0
        
        # High должен быть >= max(open, close)
        max_oc = result[['open', 'close']].max(axis=1)
        invalid_high = result['high'] < max_oc
        if invalid_high.any():
            result.loc[invalid_high, 'high'] = max_oc[invalid_high]
            errors_fixed += invalid_high.sum()
        
        # Low должен быть <= min(open, close)
        min_oc = result[['open', 'close']].min(axis=1)
        invalid_low = result['low'] > min_oc
        if invalid_low.any():
            result.loc[invalid_low, 'low'] = min_oc[invalid_low]
            errors_fixed += invalid_low.sum()
        
        # Удаление строк с отрицательными или нулевыми ценами
        negative_prices = (result[available_ohlc] <= 0).any(axis=1)
        if negative_prices.any():
            logger.warning(f"Удалено строк с некорректными ценами: {negative_prices.sum()}")
            result = result[~negative_prices]
        
        # Проверка на экстремальные выбросы (изменение более 50% за период)
        if len(result) > 1:
            price_changes = result['close'].pct_change().abs()
            extreme_changes = price_changes > 0.5
            if extreme_changes.any():
                logger.warning(f"Найдено экстремальных изменений цены: {extreme_changes.sum()}")
                # Можно удалить или сгладить
        
        if errors_fixed > 0:
            logger.info(f"Исправлено OHLC ошибок: {errors_fixed}")
        
        return result
    
    @staticmethod
    def adapt_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Адаптация данных к формату OHLCV"""
        result = df.copy()
        format_type = DataAdapter.detect_format(result)
        
        if format_type == 'ohlcv':
            if 'open_time' in result.columns and 'atr' in result.columns:
                logger.info("Обнаружены данные accumulatedData")
                result = DataAdapter.clean_accumulated_data(result)
            else:
                result = DataAdapter.clean_accumulated_data(result)
                
        elif format_type == 'price_only':
            if 'close' not in result.columns:
                raise ValueError("Не найдена колонка 'close'")
            
            # Создание синтетического OHLC
            result['open'] = result['close'].shift(1).fillna(result['close'].iloc[0])
            
            # Более реалистичное создание High/Low
            volatility = result['close'].pct_change().rolling(20).std().fillna(0.01)
            
            result['high'] = result[['open', 'close']].max(axis=1) * (1 + volatility * 0.5)
            result['low'] = result[['open', 'close']].min(axis=1) * (1 - volatility * 0.5)
            
            logger.info("Создан синтетический OHLC из цены закрытия")
            
        elif format_type == 'indicators_only':
            raise ValueError("Данные содержат только индикаторы без цен. Нужны OHLCV данные.")
        
        else:
            raise ValueError(f"Неизвестный формат данных. Колонки: {list(result.columns)}")
        
        # Добавляем volume если отсутствует или содержит NaN
        if 'volume' not in result.columns:
            logger.info("Колонка volume отсутствует, заполняем нулями")
            result['volume'] = 0
        else:
            if result['volume'].isna().any():
                median_volume = result['volume'].median()
                fill_value = 0 if np.isnan(median_volume) else median_volume
                result['volume'] = result['volume'].fillna(fill_value)
                logger.info("Заполнены пропущенные значения volume")
        
        return result
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, any]:
        """Комплексная проверка качества данных"""
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
            # Проверка логических соотношений
            invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
            invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
            
            if invalid_high > 0:
                report['data_issues'].append(f"Некорректных High значений: {invalid_high}")
            if invalid_low > 0:
                report['data_issues'].append(f"Некорректных Low значений: {invalid_low}")
        
        # Проверка временных данных
        if report['datetime_columns']:
            time_col = report['datetime_columns'][0]
            if time_col in df.columns:
                time_series = df[time_col].dropna()
                if len(time_series) > 1:
                    # Проверка на дубликаты времени
                    duplicates = time_series.duplicated().sum()
                    if duplicates > 0:
                        report['data_issues'].append(f"Дубликатов по времени: {duplicates}")
                    
                    # Проверка на пропуски во времени
                    time_diff = time_series.diff().dropna()
                    if len(time_diff) > 0:
                        median_interval = time_diff.median()
                        large_gaps = (time_diff > median_interval * 3).sum()
                        if large_gaps > 0:
                            report['data_issues'].append(f"Больших пропусков времени: {large_gaps}")
        
        # Расчет общего балла качества
        quality_factors = []
        
        # Фактор полноты данных
        completeness = 1 - (sum(info['count'] for info in report['missing_data'].values()) / (len(df) * len(df.columns)))
        quality_factors.append(completeness * 0.4)
        
        # Фактор корректности структуры
        structure_score = 1.0 if len(available_ohlc) >= 3 else 0.5
        quality_factors.append(structure_score * 0.3)
        
        # Фактор отсутствия критических проблем
        critical_issues = len([issue for issue in report['data_issues'] if 'Некорректных' in issue])
        issues_score = max(0, 1 - critical_issues * 0.1)
        quality_factors.append(issues_score * 0.3)
        
        report['quality_score'] = sum(quality_factors)
        
        return report