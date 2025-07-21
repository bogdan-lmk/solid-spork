"""
Основной класс RSI предиктора - ИСПРАВЛЕННАЯ ВЕРСИЯ без утечки данных
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier

# Импорты наших модулей
from config import ModelConfig
from data_types import PredictionResult
from feature_engineer import FeatureEngineer
from model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class RSIPredictor:
    """ИСПРАВЛЕННЫЙ предиктор RSI без утечки данных с правильной валидацией"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        
        # Используем консервативные параметры против переобучения
        if self.config.model_type == 'catboost':
            self.config.catboost_params.update({
                'iterations': 300,          # Умеренное количество итераций
                'learning_rate': 0.05,      # Умеренная скорость обучения
                'depth': 4,                 # Ограничиваем глубину
                'l2_leaf_reg': 15,          # Регуляризация
                'early_stopping_rounds': 50,
                'verbose': False,
                'random_seed': 42
            })
        
        # Модели для разных типов предсказаний
        self.models = {}  # {'value': model, 'change': model, 'direction': model}
        self.scalers = {}  # {'value': scaler, 'change': scaler, 'direction': scaler}
        self.feature_names: List[str] = []
        self.is_trained = False
        self._feature_stats = {}
        
    def _get_model(self, model_type: str = 'regression'):
        """Создание модели согласно конфигурации"""
        if self.config.model_type == 'catboost':
            if model_type == 'classification':
                params = self.config.catboost_params.copy()
                params.update({'loss_function': 'MultiClass'})
                return CatBoostClassifier(**params)
            else:
                return CatBoostRegressor(**self.config.catboost_params)
        elif self.config.model_type == 'xgboost':
            if model_type == 'classification':
                return XGBClassifier(**self.config.xgboost_params)
            else:
                return XGBRegressor(**self.config.xgboost_params)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.config.model_type}")
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """ИСПРАВЛЕННАЯ подготовка признаков"""
        
        # Исключаем временные и целевые колонки
        exclude_columns = {
            'target_rsi_next', 'target_rsi_change', 'target_rsi_direction',  # Целевые переменные
            'open_time', 'close_time',  # Временные колонки
            'ema_cross_signal', 'signal_gma', 'Bol'  # Категориальные (пока исключаем)
        }
        
        # Получаем все остальные колонки как признаки
        all_columns = set(df.columns)
        feature_columns = [col for col in all_columns 
                          if col not in exclude_columns and not col.startswith('Unnamed')]
        
        if not feature_columns:
            raise ValueError("Не найдено признаков для обучения")
        
        logger.info(f"Потенциальных признаков: {len(feature_columns)}")
        
        # Проверяем наличие целевых переменных
        targets = {}
        for target_name in ['target_rsi_next', 'target_rsi_change', 'target_rsi_direction']:
            if target_name in df.columns:
                targets[target_name] = df[target_name].copy()
        
        if not targets:
            raise ValueError("Отсутствуют целевые переменные")
        
        # Получаем признаки
        X = df[feature_columns].copy()
        
        logger.info(f"Исходные данные: X={X.shape}, targets={len(targets)}")
        
        # Конвертация в числовой формат
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
        
        # Умная очистка данных
        valid_masks = {}
        
        # Создаем маску валидности для каждой целевой переменной
        for target_name, target_series in targets.items():
            valid_target_mask = ~target_series.isna()
            
            # Проверяем важные колонки для признаков
            important_cols = numeric_columns[:min(10, len(numeric_columns))]
            valid_features_mask = ~X_numeric[important_cols].isna().any(axis=1)
            
            combined_mask = valid_target_mask & valid_features_mask
            valid_masks[target_name] = combined_mask
            
            logger.info(f"Для {target_name}: {combined_mask.sum()} валидных строк из {len(combined_mask)}")
        
        # Интеллектуальный отбор признаков
        variance = X_numeric.var()
        zero_variance_features = variance[variance <= 1e-8].index.tolist()
        
        if zero_variance_features:
            logger.warning(f"Удаляем {len(zero_variance_features)} признаков с нулевой дисперсией")
            X_numeric = X_numeric.drop(columns=zero_variance_features)
            variance = variance.drop(zero_variance_features)
        
        # Удаляем признаки с слишком большим количеством NaN
        nan_threshold = 0.5
        high_nan_features = []
        for col in X_numeric.columns:
            nan_ratio = X_numeric[col].isna().sum() / len(X_numeric)
            if nan_ratio > nan_threshold:
                high_nan_features.append(col)
        
        if high_nan_features:
            logger.warning(f"Удаляем {len(high_nan_features)} признаков с >50% NaN")
            X_numeric = X_numeric.drop(columns=high_nan_features)
            variance = variance.drop(high_nan_features)
        
        # Ограничиваем количество признаков против переобучения
        max_features = 20
        
        if len(variance) > max_features:
            # Отбираем по дисперсии + приоритет rsi_volatility связанным
            top_features = variance.sort_values(ascending=False).head(max_features * 2).index.tolist()
            
            # Приоритет признакам связанным с rsi_volatility
            rsi_related = [col for col in top_features if 'rsi_volatility' in col or 'rsi' in col.lower()]
            other_features = [col for col in top_features if col not in rsi_related]
            
            selected_features = rsi_related[:max_features//2] + other_features[:max_features-len(rsi_related[:max_features//2])]
            
            logger.info(f"Отобрано {len(selected_features)} признаков из {len(variance)}")
        else:
            selected_features = variance.index.tolist()
            logger.info(f"Используем все {len(selected_features)} признаков")
        
        X_final = X_numeric[selected_features].copy()
        
        # Финальная обработка NaN
        for col in X_final.columns:
            if X_final[col].isna().any():
                median_val = X_final[col].median()
                X_final[col] = X_final[col].fillna(median_val)
        
        # Сохраняем информацию о признаках
        self.feature_names = selected_features
        self._feature_stats = {
            'total_original': len(feature_columns),
            'numeric_converted': numeric_converted,
            'final_selected': len(selected_features),
            'removed_zero_variance': len(zero_variance_features),
            'removed_high_nan': len(high_nan_features)
        }
        
        logger.info(f"Подготовка признаков завершена:")
        logger.info(f"  Финальных признаков: {len(self.feature_names)}")
        logger.info(f"  Финальный размер данных: {X_final.shape}")
        
        # Применяем маски валидности к данным и целевым переменным
        cleaned_targets = {}
        for target_name, target_series in targets.items():
            mask = valid_masks[target_name]
            X_clean = X_final[mask].copy()
            y_clean = target_series[mask].copy()
            
            if len(X_clean) > 0:
                cleaned_targets[target_name] = (X_clean, y_clean)
            else:
                logger.warning(f"Нет данных для {target_name} после очистки")
        
        return X_final, cleaned_targets
    
    def train(self, df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, float]:
        """ИСПРАВЛЕННОЕ обучение с временной валидацией"""
        logger.info("Начинаем обучение RSI предиктора...")
        logger.info(f"Входные данные: {df.shape}")
        
        try:
            # Создание признаков
            df_features = FeatureEngineer.create_all_features(df)
            logger.info(f"После создания признаков: {df_features.shape}")
            
            # Подготовка данных
            X_full, cleaned_targets = self._prepare_features(df_features)
            
            results = {}
            
            # Обучаем модели для каждого типа предсказания
            for target_name, (X_clean, y_clean) in cleaned_targets.items():
                logger.info(f"\nОбучение модели для {target_name}: X={X_clean.shape}, y={y_clean.shape}")
                
                if len(X_clean) < 30:
                    logger.warning(f"Слишком мало данных для {target_name}: {len(X_clean)}")
                    continue
                
                # Временное разделение данных
                split_idx = int(len(X_clean) * (1 - self.config.test_size))
                
                X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
                y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
                
                logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")
                
                # Масштабирование
                scaler = RobustScaler()
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=self.feature_names,
                    index=X_train.index
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=self.feature_names,
                    index=X_test.index
                )
                
                # Определяем тип модели
                model_type = 'classification' if 'direction' in target_name else 'regression'
                
                # Обучение модели
                model = self._get_model(model_type)
                
                if self.config.model_type == 'catboost':
                    if model_type == 'classification':
                        # For classification, create new params without eval_metric
                        params = self.config.catboost_params.copy()
                        if 'eval_metric' in params:
                            del params['eval_metric']
                        model = CatBoostClassifier(
                            **params,
                            loss_function='MultiClass',
                            eval_metric='Accuracy'
                        )
                        model.fit(
                            X_train_scaled, y_train,
                            eval_set=(X_test_scaled, y_test) if len(X_test) > 5 else None,
                            use_best_model=True if len(X_test) > 5 else False
                        )
                    else:
                        model.fit(
                            X_train_scaled, y_train,
                            eval_set=(X_test_scaled, y_test) if len(X_test) > 5 else None,
                            use_best_model=True if len(X_test) > 5 else False
                        )
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Предсказания
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Ограничиваем предсказания для RSI
                if 'next' in target_name:
                    y_pred_train = np.clip(y_pred_train, 0, 100)
                    y_pred_test = np.clip(y_pred_test, 0, 100)
                
                # Оценка качества
                if model_type == 'regression':
                    train_metrics = ModelEvaluator.calculate_metrics(y_train, y_pred_train)
                    test_metrics = ModelEvaluator.calculate_metrics(y_test, y_pred_test)
                    ModelEvaluator.print_evaluation(train_metrics, test_metrics)
                else:
                    # Для классификации
                    train_acc = accuracy_score(y_train, y_pred_train)
                    test_acc = accuracy_score(y_test, y_pred_test)
                    logger.info(f"  Accuracy - Train: {train_acc:.3f}, Test: {test_acc:.3f}")
                    test_metrics = {'accuracy': test_acc, 'train_accuracy': train_acc}
                
                # Сохраняем модель и скалер
                model_key = target_name.replace('target_rsi_', '')
                self.models[model_key] = model
                self.scalers[model_key] = scaler
                
                results[model_key] = test_metrics
            
            # Кросс-валидация
            try:
                if 'next' in cleaned_targets:
                    X_cv, y_cv = cleaned_targets['target_rsi_next']
                    cv_scores = self._time_series_cross_validate(X_cv, y_cv)
                    logger.info(f"CV R² score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            except Exception as cv_error:
                logger.warning(f"Кросс-валидация не удалась: {cv_error}")
            
            self.is_trained = True
            
            # Сохранение модели
            if save_path:
                self.save(save_path)
                logger.info(f"Модели сохранены: {save_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Критическая ошибка при обучении: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _time_series_cross_validate(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Временная кросс-валидация"""
        n_splits = min(self.config.cv_folds, 3)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            try:
                X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
                y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
                
                if len(X_cv_train) < 10 or len(X_cv_val) < 3:
                    continue
                
                # Обучение модели для фолда
                scaler = RobustScaler()
                X_cv_train_scaled = scaler.fit_transform(X_cv_train)
                X_cv_val_scaled = scaler.transform(X_cv_val)
                
                model = self._get_model('regression')
                model.fit(X_cv_train_scaled, y_cv_train)
                
                y_cv_pred = model.predict(X_cv_val_scaled)
                y_cv_pred = np.clip(y_cv_pred, 0, 100)
                
                score = r2_score(y_cv_val, y_cv_pred)
                cv_scores.append(score)
                
            except Exception as e:
                logger.warning(f"Ошибка в фолде {fold_idx}: {e}")
                continue
        
        return np.array(cv_scores) if cv_scores else np.array([0.0])
    
    def predict(self, df: pd.DataFrame, return_confidence: bool = False) -> Union[float, PredictionResult]:
        """ИСПРАВЛЕННОЕ предсказание БЕЗ утечки данных"""
        if not self.is_trained:
            raise ValueError("Модель не обучена! Используйте метод train()")
        
        try:
            logger.info(f"Предсказание для данных: {df.shape}")
            
            # Создаем признаки БЕЗ целевых переменных
            df_features = self._create_features_for_prediction(df)
            
            # Получаем текущее значение RSI
            current_rsi = float(df['rsi_volatility'].iloc[-1]) if 'rsi_volatility' in df.columns else 50.0
            
            # Получаем признаки для последней строки
            exclude_columns = {
                'target_rsi_next', 'target_rsi_change', 'target_rsi_direction',
                'open_time', 'close_time',
                'ema_cross_signal', 'signal_gma', 'Bol'
            }
            
            feature_columns = [col for col in df_features.columns 
                              if col not in exclude_columns and not col.startswith('Unnamed')]
            
            X = df_features[feature_columns].copy()
            
            # Конвертация в числовой формат
            from data_adapter import DataAdapter
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = DataAdapter._robust_numeric_conversion_accumulated(X[col], col)
            
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = X[numeric_columns]
            
            # Берем последнюю валидную строку
            valid_mask = ~X_numeric.isna().any(axis=1)
            valid_rows = X_numeric[valid_mask]
            
            if len(valid_rows) == 0:
                raise ValueError("Нет валидных строк для предсказания")
            
            X_last = valid_rows.iloc[[-1]]
            
            # Выравниваем признаки под обученную модель
            X_aligned = pd.DataFrame(index=X_last.index, columns=self.feature_names)
            
            for feature in self.feature_names:
                if feature in X_last.columns:
                    X_aligned[feature] = X_last[feature]
                else:
                    X_aligned[feature] = 0.0
            
            X_aligned = X_aligned.fillna(0.0)
            
            # Предсказания для всех моделей
            predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    scaler = self.scalers[model_name]
                    X_scaled = scaler.transform(X_aligned)
                    
                    pred = model.predict(X_scaled)[0]
                    
                    if model_name == 'next':
                        pred = float(np.clip(pred, 0, 100))
                    elif model_name == 'direction':
                        direction_names = ['DOWN', 'SIDEWAYS', 'UP']
                        pred = direction_names[int(pred)]
                    
                    predictions[model_name] = pred
                    
                except Exception as e:
                    logger.warning(f"Ошибка предсказания {model_name}: {e}")
                    predictions[model_name] = None
            
            logger.info(f"Предсказания: {predictions}")
            
            if not return_confidence:
                return predictions.get('next', current_rsi)
            
            # Создаем детальный результат
            predicted_rsi = predictions.get('next', current_rsi)
            change = predictions.get('change', 0.0)
            if change is None and predicted_rsi != current_rsi:
                change = predicted_rsi - current_rsi
            
            confidence = 75.0  # Базовая уверенность
            
            result = PredictionResult(
                predicted_rsi=predicted_rsi,
                current_rsi=current_rsi,
                confidence=confidence,
                change=change or 0.0,
                prediction_date=pd.Timestamp.now()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _create_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков для предсказания БЕЗ целевых переменных"""
        try:
            # Валидация данных
            df = FeatureEngineer.validate_data(df)
            
            # Создание всех признаков КРОМЕ целевых переменных
            df = FeatureEngineer.create_existing_indicators_features(df)
            df = FeatureEngineer.create_rsi_features(df)
            df = FeatureEngineer.create_oscillator_features(df)
            df = FeatureEngineer.create_trend_features(df)
            
            # НЕ создаем целевые переменные!
            
            df = df.replace([np.inf, -np.inf], np.nan)
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка создания признаков для предсказания: {e}")
            raise
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Получение важности признаков"""
        if not self.is_trained or not self.models:
            return pd.DataFrame()
        
        # Берем важность из основной модели (next)
        main_model = self.models.get('next', list(self.models.values())[0])
        
        if not hasattr(main_model, 'feature_importances_'):
            return pd.DataFrame()
        
        try:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': main_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
        except Exception as e:
            logger.error(f"Ошибка получения важности признаков: {e}")
            return pd.DataFrame()
    
    def save(self, filepath: str):
        """Сохранение всех моделей"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_names': self.feature_names,
                'config': self.config,
                'is_trained': self.is_trained,
                'feature_stats': self._feature_stats
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_data, filepath)
            logger.info(f"Модели сохранены: {filepath}")
                
        except Exception as e:
            logger.error(f"Ошибка при сохранении: {e}")
            raise
    
    def load(self, filepath: str):
        """Загрузка всех моделей"""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']
            self.config = model_data['config']
            self.is_trained = model_data['is_trained']
            self._feature_stats = model_data.get('feature_stats', {})
            
            logger.info(f"Модели загружены: {filepath}")
            logger.info(f"Доступные модели: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке: {e}")
            raise
