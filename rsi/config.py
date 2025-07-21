"""
Конфигурация для RSI предиктора (ФИНАЛЬНАЯ НАСТРОЙКА против переобучения)
"""
from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelConfig:
    """ФИНАЛЬНАЯ НАСТРОЙКА: Максимальная защита от переобучения"""
    model_type: str = 'catboost'
    test_size: float = 0.2
    cv_folds: int = 3
    random_state: int = 42
    
    catboost_params: Dict = None
    xgboost_params: Dict = None
    
    def __post_init__(self):
        if self.catboost_params is None:
            # ФИНАЛЬНАЯ НАСТРОЙКА: Максимальная защита от переобучения
            self.catboost_params = {
                # КРИТИЧЕСКОЕ УМЕНЬШЕНИЕ итераций
                'iterations': 200,              # Еще меньше итераций (было 400)
                'learning_rate': 0.03,          # Еще меньше learning rate (было 0.05)
                'depth': 3,                     # Еще меньше глубина (было 4)
                
                # МАКСИМАЛЬНАЯ регуляризация
                'l2_leaf_reg': 25,              # Максимальная L2 регуляризация (было 15)
                'reg_lambda': 3.0,              # Усиленная регуляризация (было 1.0)
                'random_strength': 1.0,         # Добавляем случайность
                
                # АГРЕССИВНАЯ ранняя остановка
                'early_stopping_rounds': 30,   # Раньше остановка (было 50)
                'od_type': 'Iter',
                'od_wait': 30,                  # Меньше ожидание (было 50)
                
                # МАКСИМАЛЬНЫЙ сэмплинг
                'subsample': 0.6,               # Только 60% данных (было 0.8)
                'colsample_bylevel': 0.6,       # Только 60% признаков (было 0.8)
                'rsm': 0.6,                     # Random subspace method
                
                # Дополнительная защита
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 0.5,     # Меньше температура
                'leaf_estimation_method': 'Newton',  # Более консервативный метод
                'leaf_estimation_iterations': 1,     # Меньше итераций для листьев
                
                # Ограничения модели
                'max_leaves': 8,                # Очень мало листьев (было без ограничений)
                'min_data_in_leaf': 10,         # Минимум данных в листе
                'feature_border_type': 'Median', # Медианное разделение
                
                # Стандартные настройки
                'random_seed': self.random_state,
                'verbose': False,
                'thread_count': -1,
                'eval_metric': 'RMSE',
                'use_best_model': True,
                'task_type': 'CPU'
            }
            
        if self.xgboost_params is None:
            # Аналогично для XGBoost
            self.xgboost_params = {
                'n_estimators': 150,            # Очень мало деревьев (было 300)
                'learning_rate': 0.03,          # Медленное обучение
                'max_depth': 3,                 # Очень мелкие деревья
                
                # Максимальная регуляризация
                'reg_alpha': 3.0,               # Максимальная L1
                'reg_lambda': 25.0,             # Максимальная L2
                'gamma': 3.0,                   # Высокий порог для разделения
                'min_child_weight': 10,         # Минимум данных для листа
                
                # Агрессивный сэмплинг
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'colsample_bylevel': 0.6,
                'colsample_bynode': 0.6,
                
                'early_stopping_rounds': 30,
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': 0
            }

# Специальные конфигурации для разных случаев

def get_ultra_conservative_config() -> ModelConfig:
    """Ультра-консервативная конфигурация для сильного переобучения"""
    config = ModelConfig()
    config.catboost_params.update({
        'iterations': 100,              # Минимум итераций
        'learning_rate': 0.01,          # Очень медленное обучение
        'depth': 2,                     # Минимальная глубина
        'l2_leaf_reg': 50,              # Максимальная регуляризация
        'subsample': 0.5,               # Половина данных
        'colsample_bylevel': 0.5,       # Половина признаков
        'early_stopping_rounds': 20,   # Очень ранняя остановка
        'max_leaves': 4                 # Минимум листьев
    })
    return config

def get_config_for_overfitting_score(overfitting_score: float) -> ModelConfig:
    """Автоматический выбор конфигурации на основе уровня переобучения"""
    
    if overfitting_score > 0.3:
        # Критическое переобучение
        return get_ultra_conservative_config()
    elif overfitting_score > 0.2:
        # Сильное переобучение - финальная настройка
        return ModelConfig()
    elif overfitting_score > 0.1:
        # Умеренное переобучение - стандартная настройка
        config = ModelConfig()
        config.catboost_params.update({
            'iterations': 300,
            'learning_rate': 0.05,
            'depth': 4,
            'l2_leaf_reg': 15
        })
        return config
    else:
        # Переобучение под контролем - можно быть менее консервативным
        config = ModelConfig()
        config.catboost_params.update({
            'iterations': 500,
            'learning_rate': 0.07,
            'depth': 5,
            'l2_leaf_reg': 10
        })
        return config

# Дополнительные конфигурации данных

@dataclass
class DataConfig:
    """Конфигурация для обработки данных"""
    
    # ФИНАЛЬНАЯ НАСТРОЙКА: Еще более строгие ограничения
    max_features: int = 12              # Еще меньше признаков 
    min_feature_importance: float = 0.5  # Только очень важные признаки
    max_correlation: float = 0.9        # Удаляем сильно коррелирующие
    
    # Очистка данных
    max_price_change_threshold: float = 0.15  # Строже против выбросов
    min_data_completeness: float = 0.9        # Выше требования к полноте
    max_nan_ratio_per_feature: float = 0.3    # Строже к NaN
    
    # Валидация
    min_train_size: int = 50
    min_test_size: int = 15
    
    # Новые параметры против переобучения
    feature_selection_method: str = 'mutual_info'  # Лучший отбор признаков
    remove_highly_correlated: bool = True          # Удаляем коррелирующие
    use_feature_importance_threshold: bool = True   # Используем порог важности

@dataclass 
class QualityConfig:
    """Строгий контроль качества"""
    
    # ФИНАЛЬНЫЕ пороги
    max_overfitting_score: float = 0.1    # Очень строго (было 0.15)
    min_r2_score: float = 0.2             # Минимальное качество
    max_mape: float = 25.0                # Реалистичный MAPE
    max_cv_std: float = 0.15              # Стабильность CV
    
    # Действия при превышении порогов
    auto_adjust_params: bool = True       # Автоматическая настройка
    max_adjustment_attempts: int = 3      # Максимум попыток настройки