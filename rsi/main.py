"""
Главный исполняемый файл для RSI предиктора (УЛУЧШЕННАЯ ВЕРСИЯ)
"""
import sys
import os
from pathlib import Path
import pandas as pd

# Добавляем текущую директорию в Python path (absolute path)
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

import logging
import warnings

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def combine_all_csv_files(data_folder: Path):
    """
    НОВАЯ ФУНКЦИЯ: Объединение всех CSV файлов из папки data
    
    Args:
        data_folder: Путь к папке с CSV файлами
        
    Returns:
        Объединенный DataFrame
    """
    from data_adapter import DataAdapter
    
    all_dataframes = []
    processed_files = []
    
    print(f"\n🔗 Объединение всех файлов из папки: {data_folder}")
    
    # Найти все CSV файлы
    csv_files = list(data_folder.glob("*.csv"))
    
    if not csv_files:
        return None, []
    
    print(f"📁 Найдено CSV файлов: {len(csv_files)}")
    
    for csv_file in csv_files:
        try:
            print(f"  📄 Обработка {csv_file.name}...")
            
            # Загрузка файла
            df = DataAdapter.load_csv(str(csv_file))
            
            # Анализ формата
            format_type = DataAdapter.detect_format(df)
            
            # Проверка совместимости
            if format_type in ['ohlcv', 'price_only']:
                # Адаптация к OHLCV формату
                df_adapted = DataAdapter.adapt_to_ohlcv(df)
                
                # Добавляем источник данных
                df_adapted['data_source'] = csv_file.stem
                
                all_dataframes.append(df_adapted)
                processed_files.append(csv_file.name)
                print(f"  ✅ Обработано: {len(df_adapted)} строк")
                
            elif format_type == 'indicators_only':
                # Проверяем наличие базовых колонок
                if 'close' in df.columns or any(col in df.columns for col in ['open', 'high', 'low']):
                    df_adapted = DataAdapter.adapt_to_ohlcv(df)
                    df_adapted['data_source'] = csv_file.stem
                    all_dataframes.append(df_adapted)
                    processed_files.append(csv_file.name)
                    print(f"  ✅ Обработано как индикаторы: {len(df_adapted)} строк")
                else:
                    print(f"  ⚠️ Пропущен (нет цен): {csv_file.name}")
            else:
                print(f"  ❌ Неподдерживаемый формат: {csv_file.name}")
                
        except Exception as e:
            print(f"  ❌ Ошибка обработки {csv_file.name}: {e}")
            continue
    
    if not all_dataframes:
        return None, []
    
    # Объединение всех DataFrame
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    # Сортировка по времени (если есть временная колонка)
    if 'open_time' in combined_df.columns:
        try:
            combined_df['open_time'] = pd.to_datetime(combined_df['open_time'], errors='coerce')
            combined_df = combined_df.sort_values('open_time').reset_index(drop=True)
            print(f"  📅 Данные отсортированы по времени")
        except Exception as e:
            print(f"  ⚠️ Не удалось отсортировать по времени: {e}")
    
    print(f"✅ Объединение завершено:")
    print(f"   📊 Итого строк: {len(combined_df)}")
    print(f"   📁 Файлов обработано: {len(processed_files)}")
    print(f"   📋 Источники: {', '.join(processed_files)}")
    
    return combined_df, processed_files

def train_on_combined_data(combined_df, data_source_info):
    """
    НОВАЯ ФУНКЦИЯ: Обучение на объединенных данных
    """
    from config import ModelConfig
    from rsi_predictor import RSIPredictor
    
    # Настройка модели для большого датасета
    config = ModelConfig(
        model_type='catboost',
        test_size=0.15,  # Меньше тестовая выборка для большего объема обучения
        cv_folds=5,
        catboost_params={
            'iterations': 1500,  # Больше итераций для большого датасета
            'learning_rate': 0.03,  # Меньше learning rate для стабильности
            'depth': 6,
            'random_seed': 42,
            'verbose': 100,
            'early_stopping_rounds': 150,
            'l2_leaf_reg': 3,  # Больше регуляризации против переобучения
        }
    )
    
    # Создание модели
    predictor = RSIPredictor(config)
    
    # Создание папки для моделей
    models_dir = current_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"\n🚀 Обучение модели на объединенных данных...")
    print(f"📊 Размер датасета: {len(combined_df)} строк")
    print(f"📁 Источники: {', '.join(data_source_info)}")
    
    # Обучение
    model_path = models_dir / "rsi_predictor_combined_data.pkl"
    metrics = predictor.train(combined_df, save_path=str(model_path))
    
    return predictor

def main():
    """Основная функция (УЛУЧШЕННАЯ ВЕРСИЯ)"""
    
    print("🚀 RSI Predictor - Система предсказания RSI")
    print("=" * 60)
    
    try:
        # Импорты внутри функции для избежания циркулярных импортов
        from config import ModelConfig
        from rsi_predictor import RSIPredictor
        from data_adapter import DataAdapter
        
        # Попробуем импортировать utilities с разными именами
        try:
            from utilities import (
                analyze_your_csv, 
                train_on_accumulated_data, 
                integrate_with_existing_data,
                create_test_data
            )
        except ImportError:
            # Если utilities с пробелом в конце
            import importlib.util
            utilities_path = current_dir / "utilities.py "
            if utilities_path.exists():
                spec = importlib.util.spec_from_file_location("utilities", utilities_path)
                utilities_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(utilities_module)
                
                analyze_your_csv = utilities_module.analyze_your_csv
                train_on_accumulated_data = utilities_module.train_on_accumulated_data
                integrate_with_existing_data = utilities_module.integrate_with_existing_data
                create_test_data = utilities_module.create_test_data
            else:
                raise ImportError("Не удалось найти модуль utilities")
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print(f"💡 Убедитесь, что все файлы находятся в папке: {current_dir}")
        print(f"📁 Текущая директория: {os.getcwd()}")
        
        # Проверим наличие всех файлов
        required_files = [
            "config.py",
            "rsi_predictor.py", 
            "data_adapter.py",
            "utilities.py",
            "utilities.py ",  # с пробелом
            "feature_engineer.py",
            "model_evaluator.py",
            "data_types.py"
        ]
        
        print("\n📋 Проверка файлов:")
        for file in required_files:
            file_path = current_dir / file
            exists = "✅" if file_path.exists() else "❌"
            print(f"{exists} {file}")
        
        return
    
    # НОВОЕ: Проверяем наличие папки data
    data_folder = current_dir / "data"
    use_data_folder = False
    
    if data_folder.exists() and list(data_folder.glob("*.csv")):
        print(f"\n📁 Обнаружена папка data с CSV файлами!")
        csv_in_data = list(data_folder.glob("*.csv"))
        print(f"📊 Найдено файлов в data/: {len(csv_in_data)}")
        
        for csv_file in csv_in_data:
            print(f"  📄 {csv_file.name}")
        
        use_data_folder = True
        
        # Анализ файлов в папке data
        print(f"\n🔍 Анализ файлов в папке data:")
        for csv_file in csv_in_data:
            try:
                print(f"\n--- Анализ {csv_file.name} ---")
                analyze_your_csv(str(csv_file))
            except Exception as e:
                print(f"Ошибка анализа {csv_file.name}: {e}")
    else:
        print(f"\n📁 Папка data не найдена или пуста")
        print(f"💡 Создайте папку data и поместите туда CSV файлы для лучших результатов")
    
    # 1. АНАЛИЗ ВАШИХ ДАННЫХ (существующая логика как fallback)
    if not use_data_folder:
        print("\n🔍 Анализ CSV файлов в стандартных местах:")
        
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
    predictor = None
    
    # НОВОЕ: Приоритет папке data
    if use_data_folder:
        try:
            print(f"\n🎯 ПРИОРИТЕТ: Обучение на всех файлах из папки data...")
            
            # Объединяем все файлы из папки data
            combined_df, processed_files = combine_all_csv_files(data_folder)
            
            if combined_df is not None and len(combined_df) > 50:
                # Обучение на объединенных данных
                predictor = train_on_combined_data(combined_df, processed_files)
                
                print(f"✅ Модель успешно обучена на объединенных данных!")
                
                # Тестовое предсказание
                result = predictor.predict(combined_df, return_confidence=True)
                print(f"🔮 Тестовое предсказание: {result}")
                
                # Показать важность признаков
                print(f"\n📊 Топ-10 важных признаков:")
                importance_df = predictor.get_feature_importance(10)
                if not importance_df.empty:
                    for idx, row in importance_df.iterrows():
                        print(f"  {row['feature']:<25} - {row['importance']:.4f}")
                
                trained_successfully = True
            else:
                print(f"⚠️ Недостаточно данных в папке data для обучения")
                
        except Exception as e:
            print(f"❌ Ошибка обучения на данных из папки data: {e}")
            import traceback
            print(traceback.format_exc())
    
    # СУЩЕСТВУЮЩАЯ ЛОГИКА как fallback
    if not trained_successfully:
        print(f"\n📊 Fallback: Обучение на отдельных файлах...")
        
        # Определяем файлы для обучения
        files_to_try = []
        
        if use_data_folder:
            # Если есть папка data, но что-то пошло не так
            files_to_try = [str(f) for f in data_folder.glob("*.csv")]
        else:
            # Используем найденные файлы из существующей логики
            files_to_try = found_files[:2] if 'found_files' in locals() else []
        
        # Пробуем обучиться на отдельных файлах
        for csv_file in files_to_try:
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
    
    # 3. FALLBACK - создание тестовой модели (существующая логика)
    if not trained_successfully:
        print(f"\n⚠️ Не удалось обучить модель на реальных данных")
        print(f"🔧 Создаем тестовую модель...")
        
        try:
            # Создание тестовых данных как fallback
            df = create_test_data(500)
            
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
                print(f"  📄 {model_file.name}")
    
    print(f"\n🎉 Процесс завершен!")
    
    # НОВОЕ: Улучшенная информация
    print(f"\n📋 Полезная информация:")
    print(f"📁 Директория проекта: {current_dir}")
    if data_folder.exists():
        csv_count = len(list(data_folder.glob("*.csv")))
        print(f"📊 Папка data: {csv_count} CSV файлов")
    else:
        print(f"💡 Создайте папку data и поместите туда CSV файлы")
        print(f"   mkdir data")
        print(f"   mv *.csv data/")
    
    print(f"🐍 Python path: {sys.path[0]}")
    print(f"💾 Рабочая директория: {os.getcwd()}")

if __name__ == "__main__":
    main()
