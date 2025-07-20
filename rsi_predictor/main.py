"""
Главный исполняемый файл для RSI предиктора
"""
import logging
import warnings
from pathlib import Path

# Импорты наших модулей
from config import ModelConfig
from rsi_predictor import RSIPredictor
from utilities import (
    analyze_your_csv, 
    train_on_accumulated_data, 
    integrate_with_existing_data,
    create_test_data
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def main():
    """Основная функция"""
    print("🚀 RSI Predictor - Система предсказания RSI")
    print("=" * 60)
    
    # 1. АНАЛИЗ ВАШИХ ДАННЫХ
    print("\n🔍 Анализ ваших CSV файлов:")
    
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
            
            # Обучение через интеграционную функцию
            predictor = integrate_with_existing_data(csv_file)
            
            print(f"✅ Модель успешно обучена на {csv_file}")
            
            # Тестовое предсказание
            from data_adapter import DataAdapter
            df = DataAdapter.load_csv(csv_file)
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
    
    # 3. FALLBACK - создание тестовой модели
    if not trained_successfully:
        print(f"\n⚠️  Не удалось обучить модель на ваших данных")
        print(f"🔧 Создаем тестовую модель...")
        
        # Создание тестовых данных как fallback
        df = create_test_data(500)
        
        config = ModelConfig(model_type='catboost', test_size=0.2, cv_folds=3)
        predictor = RSIPredictor(config)
        
        # Создание папки для моделей
        Path("models").mkdir(exist_ok=True)
        
        metrics = predictor.train(df, save_path="models/rsi_predictor_fallback.pkl")
        result = predictor.predict(df, return_confidence=True)
        
        print(f"✅ Тестовая модель создана")
        print(f"🔮 Результат: {result}")
    
    # 4. ПОПЫТКА ОБЪЕДИНЕНИЯ ДАННЫХ
    try:
        print(f"\n🔄 Попытка объединения данных 2024 и 2025...")
        combined_predictor = train_on_accumulated_data()
        print(f"✅ Модель на объединенных данных создана")
    except Exception as e:
        print(f"❌ Ошибка создания объединенной модели: {e}")
    
    print(f"\n🎉 Процесс завершен!")
    print(f"📁 Проверьте папку 'models/' для сохраненных моделей")

if __name__ == "__main__":
    main()