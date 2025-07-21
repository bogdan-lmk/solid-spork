"""
Утилиты и вспомогательные функции для RSI предиктора
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict

# Импорты наших модулей
from config import ModelConfig
from data_adapter import DataAdapter
from rsi_predictor import RSIPredictor
from data_types import PredictionResult

logger = logging.getLogger(__name__)

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

def create_test_data(n_rows: int = 500) -> pd.DataFrame:
    """Создание тестовых данных для демонстрации"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_rows, freq='D')
    
    price_base = 100
    prices = [price_base]
    
    for i in range(n_rows - 1):
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
        'volume': np.random.randint(1000000, 10000000, n_rows)
    })
    
    return df
