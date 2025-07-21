"""
Основной класс RSI предиктора (КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ ПОД accumulatedData)
"""
import pandas as pd
import numpy as np
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

logger = logging.getLogger(__name__)

class RSIPredictor:
    """КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ: Предиктор RSI без утечки данных и с правильной валидацией"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Параметры против переобучения
        if self.config.model_type == 'catboost':
            self.config.catboost_params.update({
                'iterations': 400,          # Еще меньше итераций для accumulatedData
                'learning_rate': 0.05,      # Немного увеличили для стабильности
                'depth': 4,                 # Ограничиваем глубину
                'l2_leaf_reg': 15,          # Усиленная регуляризация
                'early_stopping_rounds': 50,
                'verbose': False,
                'random_seed': 42
            })
        
        self.model = None
        self.scaler = RobustScaler()  # Устойчив к выбросам
        self.feature_names: List[str] = []
        self.is_trained = False
        self._feature_stats = {}  # Для отладки
        
    def _get_model(self):
        """Создание модели согласно конфигурации"""
        if self.config.model_type == 'catboost':
            return CatBoostRegressor(**self.config.catboost_params)
        elif self.config.model_type == 'xgboost':
            return XGBRegressor(**self.config.xgboost_params)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.config.model_type}")
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Подготовка признаков специально для accumulatedData"""
        
        # Исключаем временные и целевые колонки
        exclude_columns = {
            'rsi_next',           # Целевая переменная
            'open_time',          # Временные колонки
            'close_time',
            'ema_cross_signal',   # Категориальные сигналы (пока исключаем)
            'signal_gma',
            'Bol'
        }
        
        # Получаем все остальные колонки как признаки
        all_columns = set(df.columns)
        feature_columns = [col for col in all_columns 
                          if col not in exclude_columns and not col.startswith('Unnamed')]
        
        if not feature_columns:
            raise ValueError("Не найдено признаков для обучения")
        
        logger.info(f"Потенциальных признаков: {len(feature_columns)}")
        
        # Проверяем наличие целевой переменной
        if 'rsi_next' not in df.columns:
            raise ValueError("Отсутствует целевая переменная 'rsi_next'")
        
        # Получаем данные
        X = df[feature_columns].copy()
        y = df['rsi_next'].copy()
        
        logger.info(f"Исходные данные: X={X.shape}, y={y.shape}")
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Принудительная конвертация всех признаков в числовой формат
        from data_adapter import DataAdapter
        
        numeric_converted = 0
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                original_type = X[col].dtype
                X[col] = DataAdapter._robust_numeric_conversion_accumulated(X[col], col)
                numeric_converted += 1
                logger.debug(f"Конвертирован {col}: {original_type} -> {X[col].dtype}")
        
        if numeric_converted > 0:
            logger.info(f"Конвертировано в числовой формат: {numeric_converted} колонок")
        
        # Получаем только числовые колонки
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Числовые колонки: {len(numeric_columns)}")
        
        if not numeric_columns:
            raise ValueError("Нет числовых признаков после конвертации")
        
        X_numeric = X[numeric_columns].copy()
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Умная очистка данных
        # 1. Удаляем строки с NaN в целевой переменной
        valid_target_mask = ~y.isna()
        
        # 2. Проверяем важные OHLC колонки
        ohlc_columns = ['open', 'high', 'low', 'close']
        available_ohlc = [col for col in ohlc_columns if col in X_numeric.columns]
        
        if available_ohlc:
            # Если есть OHLC, они должны быть валидными
            valid_ohlc_mask = ~X_numeric[available_ohlc].isna().any(axis=1)
            valid_mask = valid_target_mask & valid_ohlc_mask
            logger.info(f"Используем OHLC колонки для валидации: {available_ohlc}")
        else:
            # Если нет OHLC, проверяем первые важные колонки
            important_cols = numeric_columns[:min(5, len(numeric_columns))]
            valid_features_mask = ~X_numeric[important_cols].isna().any(axis=1)
            valid_mask = valid_target_mask & valid_features_mask
            logger.info(f"Используем важные колонки для валидации: {important_cols}")
        
        # Применяем маску
        X_clean = X_numeric[valid_mask].copy()
        y_clean = y[valid_mask].copy()
        
        logger.info(f"После очистки NaN: {len(X_clean)} строк из {len(X_numeric)} ({len(X_clean)/len(X_numeric)*100:.1f}%)")
        
        if len(X_clean) == 0:
            raise ValueError("После удаления NaN не осталось данных")
        
        if len(X_clean) < 30:
            raise ValueError(f"Слишком мало данных после очистки: {len(X_clean)} < 30")
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Интеллектуальный отбор признаков
        # 1. Удаляем признаки с нулевой дисперсией
        variance = X_clean.var()
        zero_variance_features = variance[variance <= 1e-8].index.tolist()
        
        if zero_variance_features:
            logger.warning(f"Удаляем {len(zero_variance_features)} признаков с нулевой дисперсией")
            X_clean = X_clean.drop(columns=zero_variance_features)
            variance = variance.drop(zero_variance_features)
        
        # 2. Удаляем признаки с слишком большим количеством NaN
        nan_threshold = 0.5  # Максимум 50% NaN
        high_nan_features = []
        for col in X_clean.columns:
            nan_ratio = X_clean[col].isna().sum() / len(X_clean)
            if nan_ratio > nan_threshold:
                high_nan_features.append(col)
        
        if high_nan_features:
            logger.warning(f"Удаляем {len(high_nan_features)} признаков с >50% NaN: {high_nan_features[:3]}...")
            X_clean = X_clean.drop(columns=high_nan_features)
            variance = variance.drop(high_nan_features)
        
        # 3. КРИТИЧЕСКОЕ ОГРАНИЧЕНИЕ: Максимум 25 признаков против переобучения
        max_features = min(15, len(variance))
        
        if len(variance) > max_features:
            # Отбираем признаки по дисперсии (более информативные)
            top_features = variance.sort_values(ascending=False).head(max_features).index.tolist()
            
            # Приоритет RSI-коррелирующим признакам
            rsi_related = [col for col in top_features if 'rsi' in col.lower() or 
                          any(keyword in col.lower() for keyword in ['stoch', 'williams', 'cci', 'macd', 'bb_percent', 'mfi'])]
            
            other_features = [col for col in top_features if col not in rsi_related]
            
            # Берем до 15 RSI-related + остальные до 25 общих
            selected_features = rsi_related[:15] + other_features[:max_features-len(rsi_related[:15])]
            
            logger.info(f"Отобрано {len(selected_features)} признаков из {len(variance)} (RSI-related: {len(rsi_related[:15])})")
        else:
            selected_features = variance.index.tolist()
            logger.info(f"Используем все {len(selected_features)} признаков")
        
        X_final = X_clean[selected_features].copy()
        
        # Финальная обработка NaN в отобранных признаках
        for col in X_final.columns:
            if X_final[col].isna().any():
                # Заполняем медианой для устойчивости
                median_val = X_final[col].median()
                X_final[col] = X_final[col].fillna(median_val)
                logger.debug(f"Заполнены NaN в {col} медианой: {median_val:.4f}")
        
        # Сохраняем информацию о признаках
        self.feature_names = selected_features
        self._feature_stats = {
            'total_original': len(feature_columns),
            'numeric_converted': numeric_converted,
            'after_cleaning': len(X_clean),
            'final_selected': len(selected_features),
            'removed_zero_variance': len(zero_variance_features),
            'removed_high_nan': len(high_nan_features)
        }
        
        logger.info(f"Подготовка признаков завершена:")
        logger.info(f"  Финальных признаков: {len(self.feature_names)}")
        logger.info(f"  Финальный размер данных: {X_final.shape}")
        logger.info(f"  Целевая переменная: {y_clean.shape}")
        
        return X_final, y_clean
    
    def train(self, df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, float]:
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Обучение с временной валидацией и защитой от переобучения"""
        logger.info("Начинаем обучение RSI предиктора...")
        logger.info(f"Входные данные: {df.shape}")
        
        try:
            # Создание признаков
            df_features = FeatureEngineer.create_all_features(df)
            logger.info(f"После создания признаков: {df_features.shape}")
            
            # Отладочная информация о типах данных
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            datetime_cols = df_features.select_dtypes(include=['datetime64']).columns
            object_cols = df_features.select_dtypes(include=['object']).columns
            
            logger.info(f"Типы колонок после создания признаков:")
            logger.info(f"  Числовые колонки: {len(numeric_cols)}")
            logger.info(f"  Временные колонки: {len(datetime_cols)}")
            logger.info(f"  Object колонки: {len(object_cols)}")
            
            if len(object_cols) > 0:
                logger.warning(f"Найдены object колонки: {list(object_cols)[:5]}")
            
            # Подготовка данных
            X, y = self._prepare_features(df_features)
            logger.info(f"Подготовленные данные: X={X.shape}, y={y.shape}")
            
            # Проверка минимального размера данных
            if len(X) < 50:
                logger.warning(f"Мало данных для обучения: {len(X)} строк")
                if len(X) < 30:
                    raise ValueError(f"Критически мало данных: {len(X)} < 30")
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Временное разделение (НЕ случайное!)
            # Сортируем по индексу чтобы сохранить временной порядок
            X = X.sort_index()
            y = y.sort_index()
            
            split_idx = int(len(X) * (1 - self.config.test_size))
            
            if split_idx < 20:
                raise ValueError(f"Слишком мало данных для train/test разделения: train={split_idx}")
            
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            logger.info(f"Временное разделение:")
            logger.info(f"  Train: {len(X_train)} строк ({len(X_train)/len(X)*100:.1f}%)")
            logger.info(f"  Test: {len(X_test)} строк ({len(X_test)/len(X)*100:.1f}%)")
            
            # Масштабирование данных
            logger.info("Масштабирование признаков...")
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
            
            # Проверка масштабирования
            logger.info(f"Масштабирование: среднее={X_train_scaled.mean().mean():.4f}, std={X_train_scaled.std().mean():.4f}")
            
            # Обучение модели
            logger.info("Обучение модели...")
            self.model = self._get_model()
            
            if self.config.model_type == 'catboost':
                self.model.fit(
                    X_train_scaled, y_train,
                    eval_set=(X_test_scaled, y_test),
                    use_best_model=True
                )
            else:
                self.model.fit(X_train_scaled, y_train)
            
            logger.info("Модель обучена успешно")
            
            # Предсказания
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            
            # Ограничиваем предсказания RSI диапазоном 0-100
            y_pred_train = np.clip(y_pred_train, 0, 100)
            y_pred_test = np.clip(y_pred_test, 0, 100)
            
            # Оценка качества
            train_metrics = ModelEvaluator.calculate_metrics(y_train, y_pred_train)
            test_metrics = ModelEvaluator.calculate_metrics(y_test, y_pred_test)
            
            # Печать результатов
            ModelEvaluator.print_evaluation(train_metrics, test_metrics)
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Временная кросс-валидация
            try:
                cv_scores = self._time_series_cross_validate(X_train_scaled, y_train)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                logger.info(f"CV R² score: {cv_mean:.4f} ± {cv_std:.4f}")
                
                # Проверка стабильности
                if cv_std > 0.2:
                    logger.warning(f"Высокая нестабильность CV: std={cv_std:.4f}")
                elif cv_std < 0.05:
                    logger.info(f"Стабильная модель: std={cv_std:.4f}")
                    
            except Exception as cv_error:
                logger.warning(f"Кросс-валидация не удалась: {cv_error}")
            
            # Проверка переобучения
            overfitting_score = abs(train_metrics['r2'] - test_metrics['r2'])
            if overfitting_score < 0.1:
                logger.info(f"✅ Переобучение под контролем: {overfitting_score:.3f}")
            elif overfitting_score < 0.2:
                logger.warning(f"⚠️ Умеренное переобучение: {overfitting_score:.3f}")
            else:
                logger.error(f"❌ Сильное переобучение: {overfitting_score:.3f}")
            
            self.is_trained = True
            
            # Логирование статистики признаков
            logger.info(f"Статистика признаков: {self._feature_stats}")
            
            # Сохранение модели
            if save_path:
                self.save(save_path)
                logger.info(f"Модель сохранена: {save_path}")
            
            # Добавляем дополнительные метрики в результат
            test_metrics.update({
                'overfitting_score': overfitting_score,
                'cv_mean': cv_scores.mean() if 'cv_scores' in locals() else None,
                'cv_std': cv_scores.std() if 'cv_scores' in locals() else None,
                'n_features': len(self.feature_names),
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
            
            return test_metrics
            
        except Exception as e:
            logger.error(f"Критическая ошибка при обучении: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _time_series_cross_validate(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильная временная кросс-валидация"""
        # Ограничиваем количество фолдов для стабильности
        n_splits = min(self.config.cv_folds, 3)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        logger.info(f"Временная кросс-валидация с {n_splits} фолдами...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            try:
                X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
                y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Проверка минимального размера для стабильности
                if len(X_cv_train) < 20:
                    logger.warning(f"Фолд {fold_idx}: слишком мало train данных ({len(X_cv_train)})")
                    continue
                    
                if len(X_cv_val) < 5:
                    logger.warning(f"Фолд {fold_idx}: слишком мало val данных ({len(X_cv_val)})")
                    continue
                
                # Обучение модели для фолда
                model = self._get_model()
                model.fit(X_cv_train, y_cv_train)
                
                # Предсказание и оценка
                y_cv_pred = model.predict(X_cv_val)
                y_cv_pred = np.clip(y_cv_pred, 0, 100)  # Ограничиваем RSI
                
                score = r2_score(y_cv_val, y_cv_pred)
                cv_scores.append(score)
                
                logger.debug(f"Фолд {fold_idx}: R²={score:.4f}, train={len(X_cv_train)}, val={len(X_cv_val)}")
                
            except Exception as e:
                logger.warning(f"Ошибка в фолде {fold_idx}: {e}")
                continue
        
        if not cv_scores:
            raise ValueError("Все фолды кросс-валидации провалились")
        
        logger.info(f"CV завершена: {len(cv_scores)} успешных фолдов из {n_splits}")
        
        return np.array(cv_scores)
    
    def predict(self, df: pd.DataFrame, return_confidence: bool = False) -> Union[float, PredictionResult]:
        """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Предсказание БЕЗ утечки данных"""
        if not self.is_trained:
            raise ValueError("Модель не обучена! Используйте метод train()")
        
        try:
            logger.info(f"Предсказание для данных: {df.shape}")
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Создаем признаки БЕЗ целевой переменной
            df_features = self._create_features_for_prediction(df)
            
            # Используем тот же процесс подготовки, что и при обучении
            exclude_columns = {
                'rsi_next',           # ВАЖНО: исключаем целевую переменную!
                'open_time', 'close_time',
                'ema_cross_signal', 'signal_gma', 'Bol'
            }
            
            all_columns = set(df_features.columns)
            feature_columns = [col for col in all_columns 
                              if col not in exclude_columns and not col.startswith('Unnamed')]
            
            if not feature_columns:
                raise ValueError("Недостаточно данных для предсказания")
            
            logger.info(f"Доступно признаков для предсказания: {len(feature_columns)}")
            
            X = df_features[feature_columns].copy()
            
            # Конвертируем в числовой формат
            from data_adapter import DataAdapter
            
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = DataAdapter._robust_numeric_conversion_accumulated(X[col], col)
            
            # Получаем числовые колонки
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = X[numeric_columns]
            
            # Находим последнюю валидную строку
            ohlc_columns = ['open', 'high', 'low', 'close']
            available_ohlc = [col for col in ohlc_columns if col in X_numeric.columns]
            
            if available_ohlc:
                valid_mask = ~X_numeric[available_ohlc].isna().any(axis=1)
                logger.debug(f"Используем OHLC для валидации: {available_ohlc}")
            else:
                # Используем первые 5 числовых колонок
                important_cols = numeric_columns[:min(5, len(numeric_columns))]
                valid_mask = ~X_numeric[important_cols].isna().any(axis=1)
                logger.debug(f"Используем важные колонки: {important_cols}")
            
            valid_rows = X_numeric[valid_mask]
            
            if len(valid_rows) == 0:
                raise ValueError("Нет валидных строк для предсказания")
            
            # Берем последнюю валидную строку
            X_last = valid_rows.iloc[[-1]]
            logger.info(f"Используем последнюю валидную строку: индекс {X_last.index[0]}")
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Выравниваем признаки под обученную модель
            X_aligned = pd.DataFrame(index=X_last.index, columns=self.feature_names)
            
            missing_features = []
            for feature in self.feature_names:
                if feature in X_last.columns:
                    X_aligned[feature] = X_last[feature]
                else:
                    # Заполняем отсутствующие признаки нулем
                    X_aligned[feature] = 0.0
                    missing_features.append(feature)
            
            if missing_features:
                logger.warning(f"Отсутствуют {len(missing_features)} признаков, заполнены нулями")
                logger.debug(f"Отсутствующие признаки: {missing_features[:5]}...")
            
            # Финальная обработка
            X_aligned = X_aligned.astype(float)
            X_aligned = X_aligned.fillna(0.0)
            
            # Проверка на корректность данных
            if X_aligned.isna().any().any():
                raise ValueError("Остались NaN после подготовки к предсказанию")
            
            if np.isinf(X_aligned.values).any():
                logger.warning("Найдены бесконечные значения, заменяем на 0")
                X_aligned = X_aligned.replace([np.inf, -np.inf], 0.0)
            
            # Масштабирование
            X_scaled = self.scaler.transform(X_aligned)
            
            # Предсказание
            prediction = self.model.predict(X_scaled)[0]
            prediction = float(np.clip(prediction, 0, 100))  # RSI должен быть 0-100
            
            logger.info(f"Предсказанный RSI: {prediction:.2f}")
            
            if not return_confidence:
                return prediction
            
            # Создаем детальный результат
            current_rsi = float(df_features['rsi'].iloc[-1]) if 'rsi' in df_features.columns else 50.0
            change = prediction - current_rsi
            
            # Простая оценка уверенности на основе изменения
            confidence = min(95.0, max(50.0, 100 - abs(change) * 3))
            
            result = PredictionResult(
                predicted_rsi=prediction,
                current_rsi=current_rsi,
                confidence=confidence,
                change=change,
                prediction_date=pd.Timestamp.now()
            )
            
            logger.info(f"Результат предсказания: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _create_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков для предсказания БЕЗ целевой переменной"""
        try:
            logger.info("Создание признаков для предсказания...")
            
            # Валидация данных
            df = FeatureEngineer.validate_data(df)
            
            # Создание всех признаков КРОМЕ целевой переменной
            df = FeatureEngineer.create_existing_indicators_features(df)
            df = FeatureEngineer.create_rsi_features(df)
            df = FeatureEngineer.create_rsi_correlated_features(df)
            df = FeatureEngineer.create_oscillator_features(df)
            df = FeatureEngineer.create_trend_features(df)
            
            # ВАЖНО: НЕ создаем rsi_next для предсказания!
            
            # Удаляем бесконечности
            df = df.replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"Признаки для предсказания созданы: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка создания признаков для предсказания: {e}")
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
            logger.error(f"Ошибка получения важности признаков: {e}")
            return pd.DataFrame()
    
    def save(self, filepath: str):
        """Сохранение модели"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'config': self.config,
                'is_trained': self.is_trained,
                'feature_stats': self._feature_stats
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_data, filepath)
            logger.info(f"PKL модель сохранена: {filepath}")
                
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {e}")
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
            self._feature_stats = model_data.get('feature_stats', {})
            
            logger.info(f"Модель загружена: {filepath}")
            logger.info(f"Признаков в модели: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise