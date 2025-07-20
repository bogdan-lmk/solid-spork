"""
Оценка качества модели
"""
import numpy as np
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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