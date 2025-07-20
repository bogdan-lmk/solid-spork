"""
Главный исполняемый файл для RSI предиктора
"""
import sys
import os
from pathlib import Path

# Добавляем текущую директорию в Python path (absolute path)
sys.path.insert(0, os.path.abspath(str(Path(__file__).parent)))

import logging
import warnings

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def main():
    """Основная функция"""
    current_dir = Path(__file__).parent
    
    print("🚀 RSI Predictor - Система предсказания RSI")
    print("=" * 60)
    
    try:
        # Импорты внутри функции для избежания циркулярных импортов
        from config import ModelConfig
        from rsi_predictor import RSIPredictor
        from utilities import (
            analyze_your_csv, 
            train_on_accumulated_data, 
            integrate_with_existing_data,
            create_test_data
        )
        from data_adapter import DataAdapter
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print(f"💡 Убедитесь, что все файлы находятся в папке: {current_dir}")
        print(f"📁 Текущая директория: {os.getcwd()}")
        return
    
    # 1. АНАЛИЗ ВАШИХ ДАННЫХ
    print("\n🔍 Анализ ваших CSV файлов:")
    
    # Поиск CSV файлов в текущей директории и родительской
    csv_search_paths = [
        current_dir,  # Текущая папка
        current_dir.parent,  # Родительская папка
        Path.cwd()  # Рабочая директория
    ]
    
    found_files = []
    target_files = [
        "accumulatedData_2024.csv",
        "accumulatedData_2025.csv",
        "data.csv"
    ]
    
    for search_path in csv_search_paths:
        for target_file in target_files:
            file_path = search_path / target_file
            if file_path.exists():
                found_files.append(str(file_path))
                print(f"📁 Найден файл: {file_path}")
    
    if not found_files:
        print("⚠️ CSV файлы не найдены в стандартных местах")
        print("🔍 Ищем любые CSV файлы...")
        
        # Ищем любые CSV файлы
        for search_path in csv_search_paths:
            csv_files = list(search_path.glob("*.csv"))
            if csv_files:
                print(f"📁 Найдены CSV файлы в {search_path}:")
                for csv_file in csv_files[:5]:  # Показываем первые 5
                    print(f"  - {csv_file.name}")
                    found_files.append(str(csv_file))
                break
    
    # Анализ найденных файлов
    for csv_file in found_files[:3]:  # Анализируем первые 3
        try:
            print(f"\n--- Анализ {Path(csv_file).name} ---")
            analyze_your_csv(csv_file)
        except FileNotFoundError:
            print(f"Файл {csv_file} не найден")
        except Exception as e:
            print(f"Ошибка анализа {csv_file}: {e}")
    
    # 2. ОБУЧЕНИЕ НА ДАННЫХ
    print(f"\n🚀 Обучение модели RSI предиктора...")
    
    trained_successfully = False
    
    # Пробуем обучиться на найденных файлах
    for csv_file in found_files[:2]:  # Пробуем первые 2 файла
        try:
            print(f"\n📊 Попытка обучения на {Path(csv_file).name}...")
            
            # Обучение через интеграционную функцию
            predictor = integrate_with_existing_data(csv_file)
            
            print(f"✅ Модель успешно обучена на {Path(csv_file).name}")
            
            # Тестовое предсказание
            df = DataAdapter.load_csv(csv_file)
            result = predictor.predict(df, return_confidence=True)
            print(f"🔮 Тестовое предсказание: {result}")
            
            # Показать важность признаков
            print(f"\n📊 Топ-10 важных признаков:")
            importance_df = predictor.get_feature_importance(10)
            if not importance_df.empty:
                for idx, row in importance_df.iterrows():
                    print(f"  {row['feature']:<25} - {row['importance']:.4f}")
            
            trained_successfully = True
            break
            
        except Exception as e:
            print(f"❌ Ошибка обучения на {Path(csv_file).name}: {e}")
            continue
    
    # 3. FALLBACK - создание тестовой модели
    if not trained_successfully:
        print(f"\n⚠️ Не удалось обучить модель на реальных данных")
        print(f"🔧 Создаем тестовую модель...")
        
        try:
            # Создание тестовых данных как fallback
            df = create_test_data(500)
            
            from config import ModelConfig
            config = ModelConfig(model_type='catboost', test_size=0.2, cv_folds=3)
            predictor = RSIPredictor(config)
            
            # Создание папки для моделей
            models_dir = current_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            metrics = predictor.train(df, save_path=str(models_dir / "rsi_predictor_fallback.pkl"))
            result = predictor.predict(df, return_confidence=True)
            
            print(f"✅ Тестовая модель создана")
            print(f"🔮 Результат: {result}")
            trained_successfully = True
        except Exception as e:
            print(f"❌ Ошибка создания тестовой модели: {e}")
            import traceback
            print(traceback.format_exc())
    
    # 4. ИНФОРМАЦИЯ О РЕЗУЛЬТАТАХ
    if trained_successfully:
        models_dir = current_dir / "models"
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl"))
            print(f"\n📁 Сохраненные модели в {models_dir}:")
            for model_file in model_files:
                print(f"  - {model_file.name}")
    
    print(f"\n🎉 Процесс завершен!")
    
    # Показываем полезную информацию
    print(f"\n📋 Полезная информация:")
    print(f"📁 Директория проекта: {current_dir}")
    print(f"🐍 Python path: {sys.path[0]}")
    print(f"💾 Рабочая директория: {os.getcwd()}")

if __name__ == "__main__":
    main()
