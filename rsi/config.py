"""
Конфигурация для RSI предиктора
"""
from dataclasses import dataclass
from typing import Dict

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
