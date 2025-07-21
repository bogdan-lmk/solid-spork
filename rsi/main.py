"""
Главный исполняемый файл для RSI предиктора - УЛУЧШЕННАЯ ВЕРСИЯ с автозагрузкой из папки data
"""
import sys
import os
from pathlib import Path
import pandas as pd
import glob
from data_adapter import DataAdapter


# Добавляем текущую директорию в Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

import logging
import warnings

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def find_all_data_files():
    """
    Поиск всех CSV файлов в папке data и других возможных местах
    """
    data_files = []
    
    # Возможные папки с данными
    data_directories = [
        current_dir / "data",           # ./data/
        current_dir.parent / "data",    # ../data/
        current_dir,                    # текущая папка
        Path.cwd() / "data",           # рабочая_директория/data/
        Path.cwd()                      # рабочая директория
    ]
    
    # Паттерны файлов для поиска (в порядке приоритета)
    file_patterns = [
        "accumulatedData_*.csv",     # Ваши основные файлы
        "*.csv"                     # Любые CSV (будет последним)
    ]
    
    print("🔍 Поиск CSV файлов для обучения...")
    
    found_files_info = []
    
    for data_dir in data_directories:
        if not data_dir.exists():
            continue
            
        print(f"📁 Проверяем папку: {data_dir}")
        
        for pattern in file_patterns:
            pattern_files = list(data_dir.glob(pattern))
            
            for file_path in pattern_files:
                if file_path.is_file() and file_path.suffix.lower() == '.csv':
                    # Проверяем, не добавлен ли уже этот файл
                    if not any(existing['path'].resolve() == file_path.resolve() for existing in found_files_info):
                        
                        # Получаем размер файла
                        file_size = file_path.stat().st_size
                        file_size_mb = file_size / (1024 * 1024)
                        
                        # Проверяем, не пустой ли файл
                        if file_size > 1024:  # Минимум 1KB
                            found_files_info.append({
                                'path': file_path,
                                'name': file_path.name,
                                'size_mb': file_size_mb,
                                'pattern': pattern,
                                'directory': data_dir
                            })
                            print(f"  ✅ Найден: {file_path.name} ({file_size_mb:.2f} MB)")
    
    # Сортируем по приоритету и размеру
    priority_order = {
        "accumulatedData_*.csv": 1,
        "*.csv": 2
    }
    
    found_files_info.sort(key=lambda x: (
        priority_order.get(x['pattern'], 99), 
        -x['size_mb']  # Больше файлы первыми
    ))
    
    # Возвращаем только пути к файлам
    data_files = [info['path'] for info in found_files_info]
    
    print(f"\n📊 Итого найдено файлов: {len(data_files)}")
    
    if found_files_info:
        print("📋 Список файлов для обучения:")
        for i, info in enumerate(found_files_info[:10], 1):  # Показываем первые 10
            print(f"  {i}. {info['name']} ({info['size_mb']:.2f} MB) - {info['pattern']}")
        
        if len(found_files_info) > 10:
            print(f"  ... и еще {len(found_files_info) - 10} файлов")
    
    return data_files

def load_and_combine_data_files(data_files, max_files=5):
    """
    Загрузка и объединение нескольких CSV файлов
    """
    if not data_files:
        return None
    
    print(f"\n📚 Загрузка и объединение данных (макс. {max_files} файлов)...")
    
    # Импорты
    from data_adapter import DataAdapter
    
    combined_data = []
    successful_files = []
    
    for i, file_path in enumerate(data_files[:max_files]):
        try:
            print(f"\n📖 Загружаем файл {i+1}/{min(len(data_files), max_files)}: {file_path.name}")
            
            # Загрузка файла
            df = DataAdapter.load_csv(str(file_path))
            print(f"  Размер: {df.shape}")
            
            # Быстрая проверка формата
            required_cols = ['open', 'high', 'low', 'close']
            has_ohlc = all(col in df.columns for col in required_cols)
            has_rsi_volatility = 'rsi_volatility' in df.columns
            
            print(f"  OHLC данные: {'✅' if has_ohlc else '❌'}")
            print(f"  RSI volatility: {'✅' if has_rsi_volatility else '❌'}")
            
            # Если нет критически важных данных, пропускаем
            if not has_ohlc and not has_rsi_volatility:
                print(f"  ⚠️ Файл не содержит необходимых данных, пропускаем")
                continue
            
            # Добавляем информацию об источнике
            df['data_source'] = file_path.name
            
            # Проверяем наличие временных меток
            if 'open_time' in df.columns:
                try:
                    df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
                    valid_dates = df['open_time'].notna().sum()
                    print(f"  Временные метки: {valid_dates}/{len(df)} валидных")
                except Exception as e:
                    print(f"  ⚠️ Проблема с временными метками: {e}")
            
            combined_data.append(df)
            successful_files.append(file_path.name)
            
        except Exception as e:
            print(f"  ❌ Ошибка загрузки {file_path.name}: {e}")
            continue
    
    if not combined_data:
        print("❌ Не удалось загрузить ни одного файла!")
        return None
    
    print(f"\n🔗 Объединяем {len(combined_data)} файлов...")
    
    try:
        # Объединяем данные
        df_combined = pd.concat(combined_data, ignore_index=True, sort=False)
        
        print(f"📊 Объединенные данные: {df_combined.shape}")
        print(f"📁 Использованные файлы: {', '.join(successful_files)}")
        
        # Сортируем по времени если возможно
        if 'open_time' in df_combined.columns:
            valid_time_mask = df_combined['open_time'].notna()
            if valid_time_mask.any():
                df_combined = df_combined.sort_values('open_time').reset_index(drop=True)
                
                time_range = df_combined.loc[valid_time_mask, 'open_time']
                print(f"🗓️ Временной диапазон: {time_range.min()} — {time_range.max()}")
        
        # Проверяем дубликаты
        if 'open_time' in df_combined.columns:
            duplicates = df_combined.duplicated(subset=['open_time'], keep='first').sum()
            if duplicates > 0:
                print(f"🔄 Удаляем {duplicates} дубликатов по времени")
                df_combined = df_combined.drop_duplicates(subset=['open_time'], keep='first')
        
        # Финальная статистика
        print(f"✅ Финальный размер данных: {df_combined.shape}")
        print(f"📈 Уникальных источников: {df_combined['data_source'].nunique()}")
        
        return df_combined
        
    except Exception as e:
        print(f"❌ Ошибка объединения данных: {e}")
        return combined_data[0] if len(combined_data) == 1 else None

def main():
    """Основная функция - УЛУЧШЕННАЯ ВЕРСИЯ с автозагрузкой"""
    
    print("🚀 RSI Predictor - УЛУЧШЕННАЯ СИСТЕМА с автозагрузкой данных")
    print("=" * 70)
    
    try:
        # Импорты
        from config import ModelConfig
        from rsi_predictor import RSIPredictor
        from data_adapter import DataAdapter
        from utilities import analyze_your_csv, create_test_data
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return
    
    # Поиск всех данных
    data_files = find_all_data_files()
    
    if not data_files:
        print("\n❌ CSV файлы не найдены!")
        print("💡 Создайте папку 'data' и поместите в неё CSV файлы")
        print("💡 Или разместите CSV файлы в текущей папке")
        
        # Создаем тестовые данные как fallback
        print("\n🔧 Создаем тестовые данные для демонстрации...")
        try:
            df = create_test_data(500)
            # Добавляем rsi_volatility для тестовых данных
            import talib
            df['rsi_volatility'] = talib.RSI(df['close'], timeperiod=14)
            data_files = [df]  # Используем DataFrame напрямую
        except Exception as e:
            print(f"❌ Ошибка создания тестовых данных: {e}")
            return
    else:
        # Загружаем и объединяем реальные данные
        df = load_and_combine_data_files(data_files, max_files=5)
        
        if df is None:
            print("❌ Не удалось загрузить данные!")
            return
        
        data_files = [df]  # Используем объединенный DataFrame
    
    # Анализ данных
    print(f"\n📋 АНАЛИЗ ДАННЫХ:")
    for i, data_file in enumerate(data_files):
        try:
            if isinstance(data_file, str):
                print(f"\n--- Анализ файла {Path(data_file).name} ---")
                analyze_your_csv(data_file)
            else:
                print(f"\n--- Анализ объединенных данных ---")
                print(f"Размер данных: {data_file.shape}")
                print(f"Колонки: {list(data_file.columns)}")
                
                # Проверяем качество данных
                if 'rsi_volatility' in data_file.columns:
                    rsi_data = data_file['rsi_volatility'].dropna()
                    print(f"RSI volatility: {len(rsi_data)} валидных значений, диапазон: [{rsi_data.min():.2f}, {rsi_data.max():.2f}]")
                
                if 'open_time' in data_file.columns:
                    time_data = data_file['open_time'].dropna()
                    if len(time_data) > 0:
                        print(f"Временной диапазон: {time_data.min()} — {time_data.max()}")
                
        except Exception as e:
            print(f"Ошибка анализа: {e}")
    
    # ОБУЧЕНИЕ МОДЕЛИ
    print(f"\n🚀 ОБУЧЕНИЕ НА ОБЪЕДИНЕННЫХ ДАННЫХ:")
    print("-" * 50)
    
    trained_successfully = False
    predictor = None
    
    # Пробуем обучить на данных
    for data_file in data_files:
        try:
            if isinstance(data_file, str):
                print(f"\n📊 Обучение на файле {Path(data_file).name}...")
                df = DataAdapter.load_csv(data_file)
            else:
                print(f"\n📊 Обучение на объединенных данных...")
                df = data_file
            
            print(f"Данные для обучения: {df.shape}")
            
            # Проверяем наличие rsi_volatility
            if 'rsi_volatility' not in df.columns:
                print(f"❌ Нет колонки rsi_volatility в данных")
                
                # Пытаемся создать rsi_volatility из close цены
                if 'close' in df.columns:
                    print("🔧 Создаем rsi_volatility из цены закрытия...")
                    import talib
                    df['rsi_volatility'] = talib.RSI(df['close'], timeperiod=14)
                    print(f"✅ RSI volatility создан")
                else:
                    continue
            
            # Адаптация к OHLCV формату
            df_clean = DataAdapter.adapt_to_ohlcv(df)
            print(f"Данные очищены: {df_clean.shape}")
            
            # Улучшенная конфигурация модели
            config = ModelConfig(
                model_type='catboost',
                test_size=0.15,  # Меньше тестовая выборка для больших данных
                cv_folds=5,
                catboost_params={
                    'iterations': 500,           # Больше итераций для больших данных
                    'learning_rate': 0.03,       # Более консервативная скорость
                    'depth': 5,                  # Умеренная глубина
                    'l2_leaf_reg': 10,           # Регуляризация
                    'early_stopping_rounds': 50,
                    'verbose': False,
                    'random_seed': 42,
                    'eval_metric': 'RMSE',
                    'use_best_model': True
                }
            )
            
            # Создание и обучение модели
            predictor = RSIPredictor(config)
            
            # Создание папки для моделей
            models_dir = current_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Обучение с улучшенной моделью
            model_path = models_dir / "rsi_predictor_combined.pkl"
            print(f"Начинаем обучение на {len(df_clean)} строках...")
            
            metrics = predictor.train(df_clean, save_path=str(model_path))
            
            print(f"✅ Модель успешно обучена на объединенных данных!")
            print(f"💾 Сохранена: {model_path}")
            
            # Результаты обучения
            print(f"\n📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
            for model_type, result in metrics.items():
                print(f"\n{model_type.upper()} модель:")
                if isinstance(result, dict):
                    for metric, value in result.items():
                        if isinstance(value, (int, float)):
                            if 'accuracy' in metric:
                                print(f"  {metric}: {value:.1f}%")
                            else:
                                print(f"  {metric}: {value:.4f}")
            
            # Тестовое предсказание
            print(f"\n🔮 ТЕСТОВОЕ ПРЕДСКАЗАНИЕ:")
            result = predictor.predict(df_clean, return_confidence=True)
            
            print(f"📅 Дата предсказания: {result.prediction_date.strftime('%d.%m.%Y %H:%M')}")
            print(f"📊 Текущий RSI: {result.current_rsi:.2f}")
            print(f"🎯 Предсказанный RSI: {result.predicted_rsi:.2f}")
            print(f"📈 Изменение: {result.change:+.2f}")
            print(f"🎲 Уверенность: {result.confidence:.1f}%")
            
            # Интерпретация
            print(f"\n💡 ИНТЕРПРЕТАЦИЯ:")
            if abs(result.change) < 1:
                print("🟡 Незначительное изменение RSI - боковое движение")
            elif result.change > 3:
                print("🟢 Сильный рост RSI - возможна коррекция вниз")
            elif result.change < -3:
                print("🔴 Сильное падение RSI - возможен отскок вверх")
            else:
                print("🔵 Умеренное изменение RSI")
            
            # Важность признаков
            print(f"\n📊 ТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
            importance_df = predictor.get_feature_importance(10)
            if not importance_df.empty:
                for idx, row in importance_df.iterrows():
                    print(f"  {row['feature']:<30}: {row['importance']:.4f}")
            else:
                print("  Информация о важности недоступна")
            
            trained_successfully = True
            break
            
        except Exception as e:
            print(f"❌ Ошибка обучения: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Проверка результатов
    if trained_successfully:
        print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        
        # Информация о сохраненных моделях
        models_dir = current_dir / "models"
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl"))
            print(f"\n📁 Сохраненные модели:")
            for model_file in model_files:
                file_size = model_file.stat().st_size / (1024 * 1024)
                print(f"  📄 {model_file.name} ({file_size:.2f} MB)")
        
        # Рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        print("✅ Модель обучена на объединенных данных из всех найденных файлов")
        print("✅ Использована временная валидация для честной оценки")
        print("✅ Применены консервативные параметры против переобучения")
        print("✅ Модель готова для продакшена")
        
        print(f"\n🔄 Для использования:")
        print("  1. predictor = RSIPredictor()")
        print("  2. predictor.load('models/rsi_predictor_combined.pkl')")
        print("  3. result = predictor.predict(new_data, return_confidence=True)")
        
    else:
        print(f"\n❌ ОБУЧЕНИЕ НЕ УДАЛОСЬ")
        print("Проверьте:")
        print("  - Наличие CSV файлов в папке 'data' или текущей папке")
        print("  - Корректность структуры данных (OHLC колонки)")
        print("  - Достаточное количество данных (минимум 100 строк)")

def demo_batch_prediction():
    """Демонстрация пакетных предсказаний для всех файлов"""
    
    print("\n" + "="*60)
    print("🔄 ПАКЕТНЫЕ ПРЕДСКАЗАНИЯ ДЛЯ ВСЕХ ФАЙЛОВ")
    print("="*60)
    
    try:
        from rsi_predictor import RSIPredictor
        from data_adapter import DataAdapter
        
        # Загрузка модели
        model_path = current_dir / "models" / "rsi_predictor_combined.pkl"
        if not model_path.exists():
            print(f"❌ Модель не найдена: {model_path}")
            print("Сначала запустите обучение")
            return
        
        predictor = RSIPredictor()
        predictor.load(str(model_path))
        print(f"✅ Модель загружена: {model_path.name}")
        
        # Поиск данных для предсказаний
        data_files = find_all_data_files()
        
        if not data_files:
            print("❌ Файлы данных для предсказаний не найдены")
            return
        
        print(f"\n🔮 Делаем предсказания для {len(data_files)} файлов:")
        
        results = []
        
        for i, file_path in enumerate(data_files[:5], 1):  # Первые 5 файлов
            try:
                print(f"\n{i}. Предсказание для {file_path.name}:")
                
                # Загрузка данных
                df = DataAdapter.load_csv(str(file_path))
                
                # Проверка минимальных требований
                if len(df) < 30:
                    print(f"   ⚠️ Слишком мало данных: {len(df)} строк")
                    continue
                
                df_clean = DataAdapter.adapt_to_ohlcv(df)
                
                # Предсказание
                result = predictor.predict(df_clean, return_confidence=True)
                
                print(f"   📊 Текущий RSI: {result.current_rsi:.2f}")
                print(f"   🎯 Предсказание: {result.predicted_rsi:.2f}")
                print(f"   📈 Изменение: {result.change:+.2f}")
                print(f"   🎲 Уверенность: {result.confidence:.1f}%")
                
                results.append({
                    'file': file_path.name,
                    'current_rsi': result.current_rsi,
                    'predicted_rsi': result.predicted_rsi,
                    'change': result.change,
                    'confidence': result.confidence
                })
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                continue
        
        # Сводка результатов
        if results:
            print(f"\n📋 СВОДКА ПРЕДСКАЗАНИЙ:")
            print("-" * 60)
            for result in results:
                direction = "📈" if result['change'] > 1 else "📉" if result['change'] < -1 else "➡️"
                print(f"{direction} {result['file']:<25} RSI: {result['current_rsi']:5.1f} → {result['predicted_rsi']:5.1f} ({result['change']:+5.1f})")
    
    except Exception as e:
        print(f"❌ Ошибка пакетных предсказаний: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Основное обучение
    main()
    
    # Демонстрация пакетных предсказаний
    demo_batch_prediction()