"""
Типы данных для RSI предиктора
"""
from dataclasses import dataclass
import pandas as pd

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