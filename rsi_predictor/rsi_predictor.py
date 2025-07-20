"""
Основной класс RSI предиктора
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Импорты наших модулей
from config import ModelConfig
from data_types import PredictionResult
from feature_engineer import FeatureEngineer
from model_evaluator import ModelEvaluator

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

logger = logging.getLogger(__name__)

class RSIPredictor:
    """Улучшенный предиктор RSI"""
    
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