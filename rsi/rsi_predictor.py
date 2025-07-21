"""
Основной класс RSI предиктора (ИСПРАВЛЕННАЯ ВЕРСИЯ)
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
from data_adapter import DataAdapter

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
        """Подготовка признаков (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
        # Исключаем только целевую переменную и временные колонки
        exclude_columns = {
            'rsi_next', 'open_time', 'close_time'
        }
        
        # ИСПРАВЛЕНИЕ: Включаем все существующие индикаторы из данных
        all_columns = set(df.columns) - exclude_columns
        
        # Удаляем колонки с Unnamed (если есть)
        feature_columns = [col for col in all_columns if not col.startswith('Unnamed')]
        
        if not feature_columns:
            raise ValueError("Не найдено ни одного признака для обучения")
        
        logger.info(f"Доступные колонки для признаков: {len(feature_columns)}")
        logger.debug(f"Колонки: {feature_columns[:10]}...")  # Показываем первые 10
        
        # Проверяем наличие целевой переменной
        if 'rsi_next' not in df.columns:
            raise ValueError("Отсутствует целевая переменная 'rsi_next'")
        
        # Получаем данные
        X = df[feature_columns].copy()
        y = df['rsi_next'].copy()
        
        # Конвертируем все в числовой формат
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    # Заменяем запятые на точки и конвертируем
                    X[col] = X[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Не удалось конвертировать колонку {col}: {e}")
        
        # Определяем числовые колонки
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Числовые колонки: {len(numeric_columns)}")
        
        if not numeric_columns:
            raise ValueError("Не найдено числовых признаков")
        
        # Используем только числовые колонки
        X = X[numeric_columns]
        
        # Удаляем строки с NaN в важных колонках
        # Сначала проверяем целевую переменную
        valid_target_mask = ~y.isna()
        
        # Затем проверяем основные OHLC колонки (если есть)
        ohlc_columns = ['open', 'high', 'low', 'close']
        available_ohlc = [col for col in ohlc_columns if col in X.columns]
        
        if available_ohlc:
            valid_ohlc_mask = ~X[available_ohlc].isna().any(axis=1)
            valid_mask = valid_target_mask & valid_ohlc_mask
        else:
            # Если нет OHLC, проверяем первые 5 числовых колонок
            important_cols = numeric_columns[:5]
            valid_features_mask = ~X[important_cols].isna().any(axis=1)
            valid_mask = valid_target_mask & valid_features_mask
        
        # Применяем маску
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        logger.info(f"После очистки: строк {len(X_clean)} из {len(X)}")
        
        if len(X_clean) == 0:
            raise ValueError("После удаления NaN не осталось данных для обучения")
        
        # Удаляем признаки с нулевой дисперсией
        variance = X_clean.var()
        non_zero_variance_features = variance[variance > 1e-8].index.tolist()
        
        if len(non_zero_variance_features) == 0:
            raise ValueError("Все признаки имеют нулевую дисперсию")
        
        # Ограничиваем количество признаков для стабильности
        max_features = min(len(non_zero_variance_features), 50)
        
        # Сортируем по дисперсии и берем топ признаков
        top_features = variance[non_zero_variance_features].sort_values(ascending=False).head(max_features).index.tolist()
        
        X_final = X_clean[top_features]
        self.feature_names = top_features
        
        logger.info(f"Отобрано {len(self.feature_names)} признаков из {len(numeric_columns)}")
        
        return X_final, y_clean
    
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
        
        try:
            # Создание признаков
            df_features = FeatureEngineer.create_all_features(df)
            
            # ИСПРАВЛЕНИЕ: Отладочная информация о типах данных
            logger.info(f"Типы колонок после создания признаков:")
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            datetime_cols = df_features.select_dtypes(include=['datetime64']).columns
            object_cols = df_features.select_dtypes(include=['object']).columns
            
            logger.info(f"Числовые колонки: {len(numeric_cols)}")
            logger.info(f"Временные колонки: {len(datetime_cols)}")
            logger.info(f"Object колонки: {len(object_cols)}")
            
            if len(object_cols) > 0:
                logger.warning(f"Найдены object колонки: {list(object_cols)[:5]}")
            
            total_features = len([col for col in df_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']])

            logger.info(f"Создано признаков: {total_features}")

            # Статистика пропусков
            missing = (df_features.isna().sum() / len(df_features) * 100).sort_values(ascending=False)
            logger.info("Пропуски в данных (топ 5):")
            for col, pct in missing.head(5).items():
                if pct > 0:
                    logger.info(f"  {col}: {pct:.1f}%")

            # Базовые метрики качества
            quality = DataAdapter.validate_data_quality(df_features)
            logger.info(
                f"Качество данных: строк={quality['total_rows']}, колонки={quality['total_columns']}, score={quality['quality_score']:.2f}"
            )
            
            # Подготовка данных
            X, y = self._prepare_features(df_features)
            logger.info(f"Размер данных: {X.shape}, целевая переменная: {y.shape}")
            
            # Проверяем минимальное количество данных
            if len(X) < 50:
                raise ValueError(f"Недостаточно данных для обучения: {len(X)} < 50")
            
            # Временное разделение
            split_idx = int(len(X) * (1 - self.config.test_size))
            if split_idx < 20:  # Минимум 20 точек для обучения
                raise ValueError("Слишком мало данных для создания train/test разделения")
                
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
            try:
                cv_scores = self._cross_validate(X_train_scaled, y_train)
                logger.info(f"CV R² score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            except Exception as cv_error:
                logger.warning(f"Кросс-валидация не удалась: {cv_error}")
            
            self.is_trained = True
            
            # Сохранение модели
            if save_path:
                self.save(save_path)
            
            return test_metrics
            
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            raise
    
    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Кросс-валидация для временных рядов"""
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            try:
                X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
                y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = self._get_model()
                model.fit(X_cv_train, y_cv_train)
                
                y_cv_pred = model.predict(X_cv_val)
                score = r2_score(y_cv_val, y_cv_pred)
                cv_scores.append(score)
            except Exception as e:
                logger.warning(f"Ошибка в одном из фолдов кросс-валидации: {e}")
                continue
        
        if not cv_scores:
            raise ValueError("Все фолды кросс-валидации завершились с ошибкой")
        
        return np.array(cv_scores)
    
    def predict(self, df: pd.DataFrame, return_confidence: bool = False) -> Union[float, PredictionResult]:
        """
        Предсказание RSI для следующего периода (ИСПРАВЛЕННАЯ ВЕРСИЯ)
        
        Args:
            df: DataFrame с данными
            return_confidence: Возвращать ли детальный результат
            
        Returns:
            float или PredictionResult
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена! Используйте метод train()")
        
        try:
            # Создание признаков
            df_features = FeatureEngineer.create_all_features(df)
            
            # ИСПРАВЛЕНИЕ: Используем тот же процесс подготовки, что и при обучении
            # Исключаем только целевую переменную и временные колонки
            exclude_columns = {
                'rsi_next', 'open_time', 'close_time'
            }
            
            # Получаем все доступные колонки
            all_columns = set(df_features.columns) - exclude_columns
            feature_columns = [col for col in all_columns if not col.startswith('Unnamed')]
            
            if not feature_columns:
                raise ValueError("Недостаточно данных для предсказания")
            
            # Получаем данные
            X = df_features[feature_columns].copy()
            
            # Конвертируем все в числовой формат
            for col in X.columns:
                if X[col].dtype == 'object':
                    try:
                        X[col] = X[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"Не удалось конвертировать колонку {col}: {e}")
            
            # Используем только числовые колонки
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = X[numeric_columns]
            
            # Берем последнюю доступную строку без NaN
            # Сначала проверяем основные OHLC колонки (если есть)
            ohlc_columns = ['open', 'high', 'low', 'close']
            available_ohlc = [col for col in ohlc_columns if col in X_numeric.columns]
            
            if available_ohlc:
                valid_mask = ~X_numeric[available_ohlc].isna().any(axis=1)
            else:
                # Если нет OHLC, проверяем первые 5 числовых колонок
                important_cols = numeric_columns[:5]
                valid_mask = ~X_numeric[important_cols].isna().any(axis=1)
            
            valid_rows = X_numeric[valid_mask]
            
            if len(valid_rows) == 0:
                raise ValueError("Нет валидных строк для предсказания")
            
            # Берем последнюю валидную строку
            X_last = valid_rows.iloc[[-1]]
            
            # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Подгоняем признаки под обученную модель
            # Создаем DataFrame с нужными признаками в правильном порядке
            X_aligned = pd.DataFrame(index=X_last.index, columns=self.feature_names)
            
            # Заполняем доступными признаками
            for feature in self.feature_names:
                if feature in X_last.columns:
                    X_aligned[feature] = X_last[feature]
                else:
                    # Если признака нет, заполняем медианным значением или 0
                    X_aligned[feature] = 0
                    logger.debug(f"Признак {feature} отсутствует, заполнен нулем")
            
            # Конвертируем в числовой формат
            X_aligned = X_aligned.astype(float)
            
            # Проверяем на NaN и заполняем
            X_aligned = X_aligned.fillna(0)
            
            # Масштабирование
            X_scaled = self.scaler.transform(X_aligned)
            
            # Предсказание
            prediction = self.model.predict(X_scaled)[0]
            prediction = np.clip(prediction, 0, 100)  # Ограничиваем RSI
            
            if not return_confidence:
                return prediction
            
            # Создаем детальный результат
            current_rsi = df_features['rsi'].iloc[-1] if 'rsi' in df_features.columns else 50.0
            change = prediction - current_rsi
            
            # Простая оценка уверенности
            confidence = min(95.0, max(50.0, 100 - abs(change) * 5))
            
            return PredictionResult(
                predicted_rsi=prediction,
                current_rsi=current_rsi,
                confidence=confidence,
                change=change,
                prediction_date=pd.Timestamp.now()
            )
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            raise
    
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Получение важности признаков"""
        if not self.is_trained:
            logger.warning("Модель не обучена")
            return pd.DataFrame()
            
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Модель не поддерживает важность признаков")
            return pd.DataFrame()
        
        try:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
        except Exception as e:
            logger.error(f"Ошибка при получении важности признаков: {e}")
            return pd.DataFrame()
    
    def plot_feature_importance(self, top_n: int = 20):
        """Визуализация важности признаков"""
        importance_df = self.get_feature_importance(top_n)
        
        if importance_df.empty:
            logger.warning("Важность признаков недоступна")
            return
        
        try:
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
            plt.title(f'Топ-{top_n} наиболее важных признаков для предсказания RSI')
            plt.xlabel('Важность признака')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Ошибка при создании графика: {e}")
    
    def save(self, filepath: str, export_onnx: bool = True):
        """Сохранение модели в PKL и ONNX форматах"""
        try:
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
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {e}")
            raise
    
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
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.config = model_data['config']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Модель загружена: {filepath}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
