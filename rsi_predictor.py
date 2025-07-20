import pandas as pd
import numpy as np
import talib
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

# ONNX экспорт
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from onnxmltools import convert_catboost, convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX библиотеки не найдены. Установите: pip install onnx onnxmltools skl2onnx")

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Конфигурация модели"""
    model_type: str = 'catboost'
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    
    # CatBoost параметры
    catboost_params: Dict = None
    
    # XGBoost параметры  
    xgboost_params: Dict = None
    
    def __post_init__(self):
        if self.catboost_params is None:
            self.catboost_params = {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'random_seed': self.random_state,
                'verbose': False,
                'early_stopping_rounds': 50
            }
            
        if self.xgboost_params is None:
            self.xgboost_params = {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'random_state': self.random_state,
                'early_stopping_rounds': 50
            }

@dataclass
class PredictionResult:
    """Результат предсказания"""
    predicted_rsi: float
    current_rsi: float
    confidence: float
    change: float
    prediction_date: pd.Timestamp
    
    def __str__(self):
        return f"RSI: {self.current_rsi:.2f} → {self.predicted_rsi:.2f} ({self.change:+.2f})"

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

class FeatureEngineer:
    """Класс для создания признаков (вынесен отдельно для переиспользования)"""
    
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
        result['adx_talib'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
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
    """Класс для создания признаков (вынесен отдельно для переиспользования)"""
    
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

class ModelEvaluator:
    """Класс для оценки качества модели"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Расчет метрик качества"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # RSI-специфичные метрики
        errors = np.abs(y_true - y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        accuracy_1 = np.mean(errors <= 1) * 100
        accuracy_2 = np.mean(errors <= 2) * 100
        accuracy_5 = np.mean(errors <= 5) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'mape': mape,
            'accuracy_1': accuracy_1,
            'accuracy_2': accuracy_2,
            'accuracy_5': accuracy_5
        }
    
    @staticmethod
    def print_evaluation(train_metrics: Dict, test_metrics: Dict):
        """Красивый вывод метрик"""
        print("\n" + "="*60)
        print("📊 РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
        print("="*60)
        
        print(f"\n📈 ОСНОВНЫЕ МЕТРИКИ:")
        print(f"{'Метрика':<15} {'Train':<12} {'Test':<12} {'Разность':<12}")
        print("-" * 55)
        print(f"{'MAE':<15} {train_metrics['mae']:<12.4f} {test_metrics['mae']:<12.4f} {abs(train_metrics['mae'] - test_metrics['mae']):<12.4f}")
        print(f"{'MSE':<15} {train_metrics['mse']:<12.4f} {test_metrics['mse']:<12.4f} {abs(train_metrics['mse'] - test_metrics['mse']):<12.4f}")
        print(f"{'R²':<15} {train_metrics['r2']:<12.4f} {test_metrics['r2']:<12.4f} {abs(train_metrics['r2'] - test_metrics['r2']):<12.4f}")
        
        print(f"\n🎯 ТОЧНОСТЬ ПРЕДСКАЗАНИЯ RSI (Test):")
        print(f"MAPE: {test_metrics['mape']:.2f}%")
        print(f"Точность ±1 пункт:  {test_metrics['accuracy_1']:.1f}%")
        print(f"Точность ±2 пункта: {test_metrics['accuracy_2']:.1f}%")
        print(f"Точность ±5 пунктов: {test_metrics['accuracy_5']:.1f}%")
        
        # Оценка переобучения
        overfitting_score = abs(train_metrics['r2'] - test_metrics['r2'])
        if overfitting_score < 0.05:
            print(f"✅ Переобучение: Низкое ({overfitting_score:.3f})")
        elif overfitting_score < 0.1:
            print(f"⚠️  Переобучение: Среднее ({overfitting_score:.3f})")
        else:
            print(f"❌ Переобучение: Высокое ({overfitting_score:.3f})")

class RSIPredictor:
    """Улучшенный предиктор RSI с интеграцией в проект"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = RobustScaler()  # Более устойчив к выбросам
        self.feature_names: List[str] = []
        self.is_trained = False
        
    def _get_model(self):
        """Создание модели согласно конфигурации"""
        if self.config.model_type == 'catboost':
            return CatBoostRegressor(**self.config.catboost_params)
        elif self.config.model_type == 'xgboost':
            return XGBRegressor(**self.config.xgboost_params)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.config.model_type}")
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Подготовка признаков"""
        # Исключаем целевую переменную и базовые данные
        exclude_columns = {
            'open', 'high', 'low', 'close', 'volume', 'rsi_next', 'rsi'
        }
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Удаляем строки с NaN
        df_clean = df.dropna()
        
        if len(df_clean) == 0:
            raise ValueError("После удаления NaN не осталось данных для обучения")
        
        X = df_clean[feature_columns]
        y = df_clean['rsi_next']
        
        self.feature_names = feature_columns
        return X, y
    
    def train(self, df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Обучение модели с кросс-валидацией
        
        Args:
            df: DataFrame с OHLCV данными
            save_path: Путь для сохранения модели
            
        Returns:
            Dict с метриками качества
        """
        logger.info("Начинаем обучение RSI предиктора...")
        
        # Создание признаков
        df_features = FeatureEngineer.create_all_features(df)
        logger.info(f"Создано признаков: {len([col for col in df_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']])}")
        
        # Подготовка данных
        X, y = self._prepare_features(df_features)
        logger.info(f"Размер данных: {X.shape}, целевая переменная: {y.shape}")
        
        # Временное разделение
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Масштабирование
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_names,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=self.feature_names,
            index=X_test.index
        )
        
        # Обучение модели
        self.model = self._get_model()
        
        if self.config.model_type == 'catboost':
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=(X_test_scaled, y_test),
                use_best_model=True
            )
        else:
            self.model.fit(X_train_scaled, y_train)
        
        # Оценка качества
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_metrics = ModelEvaluator.calculate_metrics(y_train, y_pred_train)
        test_metrics = ModelEvaluator.calculate_metrics(y_test, y_pred_test)
        
        ModelEvaluator.print_evaluation(train_metrics, test_metrics)
        
        # Кросс-валидация
        cv_scores = self._cross_validate(X_train_scaled, y_train)
        logger.info(f"CV R² score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.is_trained = True
        
        # Сохранение модели
        if save_path:
            self.save(save_path)
        
        return test_metrics
    
    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Кросс-валидация для временных рядов"""
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
            y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self._get_model()
            model.fit(X_cv_train, y_cv_train)
            
            y_cv_pred = model.predict(X_cv_val)
            score = r2_score(y_cv_val, y_cv_pred)
            cv_scores.append(score)
        
        return np.array(cv_scores)
    
    def predict(self, df: pd.DataFrame, return_confidence: bool = False) -> Union[float, PredictionResult]:
        """
        Предсказание RSI для следующего периода
        
        Args:
            df: DataFrame с данными
            return_confidence: Возвращать ли детальный результат
            
        Returns:
            float или PredictionResult
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена! Используйте метод train()")
        
        # Создание признаков
        df_features = FeatureEngineer.create_all_features(df)
        
        # Подготовка данных
        X, _ = self._prepare_features(df_features)
        
        if len(X) == 0:
            raise ValueError("Недостаточно данных для предсказания")
        
        # Берем последнюю доступную строку
        X_last = X.iloc[[-1]]
        X_last_scaled = self.scaler.transform(X_last)
        
        # Предсказание
        prediction = self.model.predict(X_last_scaled)[0]
        prediction = np.clip(prediction, 0, 100)  # Ограничиваем RSI
        
        if not return_confidence:
            return prediction
        
        # Создаем детальный результат
        current_rsi = df_features['rsi'].iloc[-1]
        change = prediction - current_rsi
        
        # Простая оценка уверенности (можно улучшить)
        confidence = min(95.0, max(50.0, 100 - abs(change) * 5))
        
        return PredictionResult(
            predicted_rsi=prediction,
            current_rsi=current_rsi,
            confidence=confidence,
            change=change,
            prediction_date=pd.Timestamp.now()
        )
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Получение важности признаков"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 20):
        """Визуализация важности признаков"""
        importance_df = self.get_feature_importance(top_n)
        
        if importance_df.empty:
            logger.warning("Важность признаков недоступна")
            return
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Топ-{top_n} наиболее важных признаков для предсказания RSI')
        plt.xlabel('Важность признака')
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str, export_onnx: bool = True):
        """Сохранение модели в PKL и ONNX форматах"""
        # Сохранение в PKL
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"PKL модель сохранена: {filepath}")
        
        # Экспорт в ONNX
        if export_onnx and ONNX_AVAILABLE and self.is_trained:
            try:
                onnx_path = filepath.replace('.pkl', '.onnx')
                self._export_onnx(onnx_path)
                logger.info(f"ONNX модель сохранена: {onnx_path}")
            except Exception as e:
                logger.error(f"Ошибка экспорта в ONNX: {e}")
        elif export_onnx and not ONNX_AVAILABLE:
            logger.warning("ONNX экспорт недоступен. Установите необходимые библиотеки.")
    
    def _export_onnx(self, onnx_path: str):
        """Экспорт модели в ONNX формат"""
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        # Определяем размерность входных данных
        n_features = len(self.feature_names)
        initial_type = [('float_input', OnnxFloatTensorType([None, n_features]))]
        
        try:
            if self.config.model_type == 'catboost':
                # Экспорт CatBoost
                onnx_model = convert_catboost(self.model, initial_types=initial_type)
            elif self.config.model_type == 'xgboost':
                # Экспорт XGBoost
                onnx_model = convert_xgboost(self.model, initial_types=initial_type)
            else:
                raise ValueError(f"ONNX экспорт не поддерживается для {self.config.model_type}")
            
            # Сохранение ONNX модели
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
                
            # Сохранение метаданных для ONNX (scaler и feature names)
            metadata = {
                'feature_names': self.feature_names,
                'scaler_mean': self.scaler.center_.tolist() if hasattr(self.scaler, 'center_') else None,
                'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                'scaler_type': type(self.scaler).__name__
            }
            
            metadata_path = onnx_path.replace('.onnx', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"ONNX метаданные сохранены: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте ONNX: {e}")
            raise
    
    def load(self, filepath: str):
        """Загрузка модели"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Модель загружена: {filepath}")

# Интеграция с существующим проектом
def integrate_with_existing_data(csv_path: str, **csv_kwargs) -> RSIPredictor:
    """
    Интеграция с существующими данными проекта
    
    Args:
        csv_path: Путь к CSV файлу с данными
        **csv_kwargs: Дополнительные параметры для pd.read_csv
        
    Returns:
        Обученная модель RSIPredictor
    """
    logger.info(f"Загрузка данных из: {csv_path}")
    
    # Загрузка данных
    df = DataAdapter.load_csv(csv_path, **csv_kwargs)
    logger.info(f"Загружено строк: {len(df)}, колонок: {len(df.columns)}")
    logger.info(f"Колонки: {list(df.columns)}")
    
    # Адаптация к OHLCV формату
    df_ohlcv = DataAdapter.adapt_to_ohlcv(df)
    
    # Создание и обучение модели
    config = ModelConfig(
        model_type='catboost',
        test_size=0.2,
        cv_folds=5
    )
    
    predictor = RSIPredictor(config)
    
    # Создание папки для моделей
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Обучение с сохранением
    model_path = model_dir / "rsi_predictor.pkl"
    metrics = predictor.train(df_ohlcv, save_path=str(model_path))
    
    logger.info("Модель успешно обучена и сохранена")
    return predictor

def quick_predict_from_csv(csv_path: str, model_path: str = "models/rsi_predictor.pkl") -> PredictionResult:
    """
    Быстрое предсказание RSI из CSV файла
    
    Args:
        csv_path: Путь к CSV с данными
        model_path: Путь к сохраненной модели
        
    Returns:
        Результат предсказания
    """
    # Загрузка модели
    predictor = RSIPredictor()
    predictor.load(model_path)
    
    # Загрузка данных
    df = DataAdapter.load_csv(csv_path)
    df_ohlcv = DataAdapter.adapt_to_ohlcv(df)
    
    # Предсказание
    result = predictor.predict(df_ohlcv, return_confidence=True)
    return result

# Утилиты для работы с вашими данными
def analyze_your_csv(csv_path: str):
    """Анализ структуры CSV файла"""
    df = DataAdapter.load_csv(csv_path)
    
    print(f"\n📊 АНАЛИЗ CSV ФАЙЛА: {csv_path}")
    print("="*60)
    print(f"Размер данных: {df.shape}")
    print(f"Колонки ({len(df.columns)}): {list(df.columns)}")
    
    # Определение формата
    format_type = DataAdapter.detect_format(df)
    print(f"Определенный формат: {format_type}")
    
    # Статистика по колонкам
    print(f"\nСтатистика:")
    try:
        for col in df.columns[:10]:  # Первые 10 колонок
            non_null = df[col].count()
            null_count = len(df) - non_null
            dtype = str(df[col].dtype)  # Конвертируем в строку
            print(f"  {col:<25} | {dtype:<10} | Non-null: {non_null:<6} | Null: {null_count}")
    except Exception as e:
        print(f"Ошибка при выводе статистики: {e}")
    
    # Проверка на совместимость с RSI предиктором
    if format_type in ['ohlcv', 'price_only']:
        print(f"✅ Данные совместимы с RSI предиктором")
    elif format_type == 'indicators_only':
        if 'close' in df.columns:
            print(f"✅ Данные совместимы (есть цена закрытия)")
        else:
            print(f"⚠️  Нужна цена закрытия для работы RSI предиктора")
    else:
        print(f"❌ Данные требуют доработки")
    
    return df

# Пример использования с вашими данными
if __name__ == "__main__":
    # 1. АНАЛИЗ ВАШИХ ДАННЫХ
    print("🔍 Анализ ваших CSV файлов:")
    
    # Ваши файлы с данными
    your_csv_files = [
        "accumulatedData_2024.csv",
        "accumulatedData_2025.csv",
        "data.csv"  # Оставляем для сравнения
    ]
    
    for csv_file in your_csv_files:
        try:
            print(f"\n--- Анализ {csv_file} ---")
            analyze_your_csv(csv_file)
        except FileNotFoundError:
            print(f"Файл {csv_file} не найден")
        except Exception as e:
            print(f"Ошибка анализа {csv_file}: {e}")
    
    # 2. ОБУЧЕНИЕ НА ВАШИХ РЕАЛЬНЫХ ДАННЫХ
    print(f"\n🚀 Обучение модели RSI предиктора на ваших данных...")
    
    # Приоритет: сначала 2024, потом 2025
    priority_files = [
        "accumulatedData_2024.csv",
        "accumulatedData_2025.csv"
    ]
    
    trained_successfully = False
    
    for csv_file in priority_files:
        try:
            print(f"\n📊 Попытка обучения на {csv_file}...")
            
            # Специальная обработка для ваших данных
            config = ModelConfig(
                model_type='catboost',
                test_size=0.2,
                cv_folds=3,  # Меньше фолдов для быстрого тестирования
                catboost_params={
                    'iterations': 500,  # Меньше итераций для быстрого тестирования
                    'learning_rate': 0.1,
                    'depth': 6,
                    'random_seed': 42,
                    'verbose': False,
                    'early_stopping_rounds': 50
                }
            )
            
            predictor = RSIPredictor(config)
            
            # Загрузка и проверка данных
            df = DataAdapter.load_csv(csv_file)
            print(f"Загружено: {df.shape[0]} строк, {df.shape[1]} колонок")
            
            # Специальная проверка для accumulatedData
            if 'open_time' in df.columns:
                print(f"📅 Временной диапазон: {df['open_time'].iloc[0]} - {df['open_time'].iloc[-1]}")
            
            # Обучение
            model_path = f"models/rsi_predictor_{csv_file.replace('.csv', '')}.pkl"
            metrics = predictor.train(df, save_path=model_path)
            
            print(f"✅ Модель успешно обучена на {csv_file}")
            print(f"📁 Сохранено в: {model_path}")
            
            # Тестовое предсказание
            result = predictor.predict(df, return_confidence=True)
            print(f"🔮 Тестовое предсказание: {result}")
            
            # Показать важность признаков
            print(f"\n📊 Топ-10 важных признаков:")
            importance_df = predictor.get_feature_importance(10)
            for idx, row in importance_df.iterrows():
                print(f"  {row['feature']:<25} - {row['importance']:.4f}")
            
            # Визуализация (если нужно)
            try:
                predictor.plot_feature_importance(top_n=15)
            except:
                print("Визуализация недоступна")
            
            trained_successfully = True
            break
            
        except Exception as e:
            print(f"❌ Ошибка обучения на {csv_file}: {e}")
            import traceback
            print(f"Детали ошибки:\n{traceback.format_exc()}")
            continue
    
    if not trained_successfully:
        print(f"\n⚠️  Не удалось обучить модель на ваших данных")
        print(f"🔧 Создаем тестовую модель...")
        
        # Создание тестовых данных как fallback
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        price_base = 100
        prices = [price_base]
        
        for i in range(499):
            trend = 0.0001 * i
            volatility = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + trend + volatility)
            prices.append(max(new_price, 1))
        
        df = pd.DataFrame({
            'open_time': dates,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 500)
        })
        
        config = ModelConfig(model_type='catboost', test_size=0.2, cv_folds=3)
        predictor = RSIPredictor(config)
        
        metrics = predictor.train(df, save_path="models/rsi_predictor_fallback.pkl")
        result = predictor.predict(df, return_confidence=True)
        
        print(f"✅ Тестовая модель создана")
        print(f"🔮 Результат: {result}")

# Специальная функция для ваших данных
def train_on_accumulated_data(file_2024: str = "accumulatedData_2024.csv", 
                             file_2025: str = "accumulatedData_2025.csv") -> RSIPredictor:
    """
    Обучение модели на ваших accumulated данных
    
    Args:
        file_2024: Путь к данным 2024 года
        file_2025: Путь к данным 2025 года
        
    Returns:
        Обученная модель
    """
    # Загрузка данных
    df_2024 = DataAdapter.load_csv(file_2024)
    df_2025 = DataAdapter.load_csv(file_2025)
    
    # Объединение данных
    df_combined = pd.concat([df_2024, df_2025], ignore_index=True)
    
    # Сортировка по времени
    if 'open_time' in df_combined.columns:
        df_combined['open_time'] = pd.to_datetime(df_combined['open_time'])
        df_combined = df_combined.sort_values('open_time').reset_index(drop=True)
    
    print(f"Объединенные данные: {df_combined.shape}")
    print(f"Временной диапазон: {df_combined['open_time'].iloc[0]} - {df_combined['open_time'].iloc[-1]}")
    
    # Настройка модели
    config = ModelConfig(
        model_type='catboost',
        test_size=0.15,  # Меньше тестовая выборка для большего объема обучения
        cv_folds=5,
        catboost_params={
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 8,  # Больше глубина для сложных данных
            'random_seed': 42,
            'verbose': 100,
            'early_stopping_rounds': 100
        }
    )
    
    # Обучение
    predictor = RSIPredictor(config)
    metrics = predictor.train(df_combined, save_path="models/rsi_predictor_combined.pkl")
    
    return predictor

# Дополнительные утилиты для интеграции
def batch_predict_rsi(csv_directory: str, model_path: str = "models/rsi_predictor.pkl"):
    """Пакетное предсказание RSI для всех CSV файлов в директории"""
    results = {}
    
    for csv_file in Path(csv_directory).glob("*.csv"):
        try:
            result = quick_predict_from_csv(str(csv_file), model_path)
            results[csv_file.name] = result
            print(f"✅ {csv_file.name}: {result}")
        except Exception as e:
            print(f"❌ {csv_file.name}: {e}")
    
    return results

def export_prediction_to_csv(prediction_result: PredictionResult, output_path: str):
    """Экспорт результата предсказания в CSV"""
    df = pd.DataFrame([{
        'prediction_date': prediction_result.prediction_date,
        'current_rsi': prediction_result.current_rsi,
        'predicted_rsi': prediction_result.predicted_rsi,
        'change': prediction_result.change,
        'confidence': prediction_result.confidence
    }])
    
    df.to_csv(output_path, index=False)
    print(f"Результат сохранен в: {output_path}")