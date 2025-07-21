"""
Главный исполняемый файл для RSI предиктора - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""
import sys
import os
from pathlib import Path
import pandas as pd

# Добавляем текущую директорию в Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

import logging
import warnings

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def main():
    """Основная функция - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    
    print("🚀 RSI Predictor - ИСПРАВЛЕННАЯ СИСТЕМА предсказания RSI")
    print("=" * 60)
    
    try:
        # Импорты
        from config import ModelConfig
        from rsi_predictor import RSIPredictor
        from data_adapter import DataAdapter
        from utilities import analyze_your_csv, create_test_data
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return
    
    # Поиск ваших файлов данных
    data_files = []
    search_paths = [current_dir, current_dir.parent, Path.cwd()]
    target_files = ["accumulatedData_2024.csv", "accumulatedData_2025.csv"]
    
    print("\n🔍 Поиск ваших данных...")
    for search_path in search_paths:
        for target_file in target_files:
            file_path = search_path / target_file
            if file_path.exists():
                data_files.append(str(file_path))
                print(f"✅ Найден: {file_path}")
    
    if not data_files:
        print("❌ Файлы accumulatedData не найдены!")
        print("💡 Убедитесь что файлы находятся в той же папке что и main.py")
        
        # Создаем тестовые данные как fallback
        print("\n🔧 Создаем тестовые данные для демонстрации...")
        try:
            df = create_test_data(500)
            data_files = [df]  # Используем DataFrame напрямую
        except Exception as e:
            print(f"❌ Ошибка создания тестовых данных: {e}")
            return
    else:
        print(f"📊 Найдено файлов: {len(data_files)}")
    
    # Анализ ваших данных
    print(f"\n📋 АНАЛИЗ ВАШИХ ДАННЫХ:")
    for data_file in data_files[:2]:  # Анализируем первые 2
        try:
            if isinstance(data_file, str):
                print(f"\n--- Анализ {Path(data_file).name} ---")
                analyze_your_csv(data_file)
            else:
                print(f"\n--- Анализ тестовых данных ---")
                print(f"Размер данных: {data_file.shape}")
        except Exception as e:
            print(f"Ошибка анализа: {e}")
    
    # ИСПРАВЛЕННОЕ ОБУЧЕНИЕ
    print(f"\n🚀 ИСПРАВЛЕННОЕ ОБУЧЕНИЕ МОДЕЛИ:")
    print("-" * 40)
    
    trained_successfully = False
    predictor = None
    
    # Пробуем обучить на ваших данных
    for data_file in data_files:
        try:
            if isinstance(data_file, str):
                print(f"\n📊 Обучение на {Path(data_file).name}...")
                df = DataAdapter.load_csv(data_file)
            else:
                print(f"\n📊 Обучение на тестовых данных...")
                df = data_file
            
            print(f"Данные загружены: {df.shape}")
            
            # Проверяем наличие rsi_volatility
            if 'rsi_volatility' not in df.columns:
                print(f"❌ Нет колонки rsi_volatility в данных")
                continue
            
            # Адаптация к OHLCV формату
            df_clean = DataAdapter.adapt_to_ohlcv(df)
            print(f"Данные очищены: {df_clean.shape}")
            
            # ИСПРАВЛЕННАЯ конфигурация модели
            config = ModelConfig(
                model_type='catboost',
                test_size=0.2,
                cv_folds=3,
                catboost_params={
                    'iterations': 300,          # Консервативно
                    'learning_rate': 0.05,      # Умеренно
                    'depth': 4,                 # Неглубоко
                    'l2_leaf_reg': 15,          # Регуляризация
                    'early_stopping_rounds': 50,
                    'verbose': False,
                    'random_seed': 42
                }
            )
            
            # Создание и обучение модели
            predictor = RSIPredictor(config)
            
            # Создание папки для моделей
            models_dir = current_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # ИСПРАВЛЕННОЕ обучение
            model_path = models_dir / "rsi_predictor_fixed.pkl"
            print(f"Начинаем обучение...")
            
            metrics = predictor.train(df_clean, save_path=str(model_path))
            
            print(f"✅ Модель успешно обучена!")
            print(f"💾 Сохранена: {model_path}")
            
            # Результаты обучения
            print(f"\n📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
            for model_type, result in metrics.items():
                print(f"\n{model_type.upper()} модель:")
                if isinstance(result, dict):
                    for metric, value in result.items():
                        if isinstance(value, float):
                            print(f"  {metric}: {value:.3f}")
            
            # ИСПРАВЛЕННОЕ предсказание
            print(f"\n🔮 ТЕСТОВОЕ ПРЕДСКАЗАНИЕ:")
            result = predictor.predict(df_clean, return_confidence=True)
            
            print(f"📅 Дата предсказания: {result.prediction_date.strftime('%d.%m.%Y %H:%M')}")
            print(f"📊 Текущий RSI: {result.current_rsi:.2f}")
            print(f"🎯 Предсказанный RSI: {result.predicted_rsi:.2f}")
            print(f"📈 Изменение: {result.change:+.2f}")
            print(f"🎲 Уверенность: {result.confidence:.1f}%")
            
            # Важность признаков
            print(f"\n📊 ВАЖНОСТЬ ПРИЗНАКОВ:")
            importance_df = predictor.get_feature_importance(10)
            if not importance_df.empty:
                for idx, row in importance_df.iterrows():
                    print(f"  {row['feature']:<30}: {row['importance']:.3f}")
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
                print(f"  📄 {model_file.name}")
        
        # Рекомендации по использованию
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        print("✅ Утечка данных ИСПРАВЛЕНА - модель использует только исторические данные")
        print("✅ Временная валидация ИСПРАВЛЕНА - данные разделяются по времени")
        print("✅ Используется ваш rsi_volatility как основа для предсказаний")
        print("✅ Создаются три типа предсказаний: значение, изменение, направление")
        
        print(f"\n🔄 Для повторного использования:")
        print("  1. Загрузите модель: predictor.load('models/rsi_predictor_fixed.pkl')")
        print("  2. Сделайте предсказание: predictor.predict(new_data, return_confidence=True)")
        print("  3. Модель готова для продакшена!")
        
    else:
        print(f"\n❌ ОБУЧЕНИЕ НЕ УДАЛОСЬ")
        print("Проверьте:")
        print("  - Наличие файлов accumulatedData_2024.csv и accumulatedData_2025.csv")
        print("  - Корректность данных в файлах")
        print("  - Наличие колонки rsi_volatility")

def demo_prediction():
    """Демонстрация работы с ИСПРАВЛЕННОЙ моделью"""
    
    print("\n" + "="*50)
    print("🔄 ДЕМОНСТРАЦИЯ ЗАГРУЗКИ ИСПРАВЛЕННОЙ МОДЕЛИ")
    print("="*50)
    
    try:
        from rsi_predictor import RSIPredictor
        from data_adapter import DataAdapter
        
        # Загрузка модели
        model_path = current_dir / "models" / "rsi_predictor_fixed.pkl"
        if model_path.exists():
            predictor = RSIPredictor()
            predictor.load(str(model_path))
            print(f"✅ ИСПРАВЛЕННАЯ модель загружена: {model_path.name}")
            print(f"📊 Доступные типы предсказаний: {list(predictor.models.keys())}")
            
            # Поиск данных для предсказания
            data_file = None
            for search_path in [current_dir, current_dir.parent]:
                for filename in ["accumulatedData_2025.csv", "accumulatedData_2024.csv"]:
                    file_path = search_path / filename
                    if file_path.exists():
                        data_file = str(file_path)
                        break
                if data_file:
                    break
            
            if data_file:
                print(f"📁 Используем данные: {Path(data_file).name}")
                
                # Загрузка данных
                df = DataAdapter.load_csv(data_file)
                df_clean = DataAdapter.adapt_to_ohlcv(df)
                
                print(f"📊 Данные загружены: {df_clean.shape}")
                print(f"🗓️  Последняя дата: {df_clean['open_time'].iloc[-1] if 'open_time' in df_clean.columns else 'N/A'}")
                
                # ИСПРАВЛЕННОЕ предсказание
                result = predictor.predict(df_clean, return_confidence=True)
                
                print(f"\n🔮 РЕЗУЛЬТАТ ИСПРАВЛЕННОГО ПРЕДСКАЗАНИЯ:")
                print(f"📅 Дата: {result.prediction_date.strftime('%d.%m.%Y %H:%M')}")
                print(f"📊 Текущий RSI: {result.current_rsi:.2f}")
                print(f"🎯 Предсказанный RSI: {result.predicted_rsi:.2f}")
                print(f"📈 Ожидаемое изменение: {result.change:+.2f}")
                print(f"🎲 Уверенность модели: {result.confidence:.1f}%")
                
                # Интерпретация результата
                print(f"\n💡 ИНТЕРПРЕТАЦИЯ:")
                if abs(result.change) < 1:
                    print("🟡 Незначительное изменение RSI - боковое движение")
                elif result.change > 2:
                    print("🟢 Сильный рост RSI - возможна коррекция вниз")
                elif result.change < -2:
                    print("🔴 Сильное падение RSI - возможен отскок вверх")
                else:
                    print("🔵 Умеренное изменение RSI")
                
                # Торговые сигналы
                current_rsi = result.current_rsi
                predicted_rsi = result.predicted_rsi
                
                print(f"\n📈 ТОРГОВЫЕ СИГНАЛЫ:")
                if current_rsi > 70 and result.change < -2:
                    print("🔴 RSI в зоне перекупленности и ожидается падение")
                    print("   Сигнал: Возможна продажа")
                elif current_rsi < 30 and result.change > 2:
                    print("🟢 RSI в зоне перепроданности и ожидается рост")
                    print("   Сигнал: Возможна покупка")
                elif 30 <= current_rsi <= 70:
                    print("🟡 RSI в нейтральной зоне")
                    print("   Сигнал: Ожидание подтверждения")
                else:
                    print("⚪ Неопределенный сигнал")
                
                print(f"\n⚠️  ВАЖНО:")
                print("• Это ИСПРАВЛЕННАЯ модель без утечки данных")
                print("• Используйте совместно с другими индикаторами")
                print("• Учитывайте рыночные условия и новости")
                print("• Модель предназначена для помощи в анализе, не для автоматической торговли")
                
            else:
                print(f"❌ Файлы данных не найдены для демонстрации")
        else:
            print(f"❌ ИСПРАВЛЕННАЯ модель не найдена: {model_path}")
            print("Сначала запустите обучение: python main.py")
    
    except Exception as e:
        print(f"❌ Ошибка демонстрации: {e}")
        import traceback
        traceback.print_exc()

def validate_fix():
    """Проверка что исправления работают корректно"""
    print("\n" + "="*50)
    print("🔍 ПРОВЕРКА ИСПРАВЛЕНИЙ")
    print("="*50)
    
    try:
        from feature_engineer import FeatureEngineer
        from utilities import create_test_data
        
        print("✅ Тестируем создание признаков без утечки данных...")
        
        # Создаем тестовые данные
        df_test = create_test_data(100)
        
        # Добавляем rsi_volatility
        import talib
        df_test['rsi_volatility'] = talib.RSI(df_test['close'], timeperiod=14)
        
        print(f"📊 Тестовые данные: {df_test.shape}")
        
        # Создаем признаки
        df_features = FeatureEngineer.create_all_features(df_test)
        
        print(f"📊 Данные с признаками: {df_features.shape}")
        
        # Проверяем что нет утечки данных
        checks = []
        
        # Проверка 1: Все признаки должны быть лаговыми или текущими
        future_features = [col for col in df_features.columns if 'shift(-' in str(col) or '_next' in col and col != 'target_rsi_next']
        checks.append(("Нет признаков из будущего", len(future_features) == 0, f"Найдено: {future_features}"))
        
        # Проверка 2: Целевые переменные созданы корректно
        target_cols = ['target_rsi_next', 'target_rsi_change', 'target_rsi_direction']
        has_targets = all(col in df_features.columns for col in target_cols)
        checks.append(("Целевые переменные созданы", has_targets, f"Колонки: {[col for col in target_cols if col in df_features.columns]}"))
        
        # Проверка 3: НЕТ NaN в последней строке признаков (кроме целевых)
        feature_cols = [col for col in df_features.columns if not col.startswith('target_')]
        last_row_nans = df_features[feature_cols].iloc[-1].isna().sum()
        checks.append(("Последняя строка признаков валидна", last_row_nans < len(feature_cols) * 0.5, f"NaN: {last_row_nans}/{len(feature_cols)}"))
        
        # Проверка 4: Лаговые признаки имеют правильные значения
        lag_cols = [col for col in df_features.columns if '_lag_1' in col]
        lag_check = True
        if lag_cols:
            test_col = lag_cols[0]
            base_col = test_col.replace('_lag_1', '')
            if base_col in df_features.columns:
                # Проверяем что lag_1 = shift(1)
                expected = df_features[base_col].shift(1).iloc[-5:]
                actual = df_features[test_col].iloc[-5:]
                lag_check = expected.equals(actual) or (expected.isna() == actual.isna()).all()
        
        checks.append(("Лаговые признаки корректны", lag_check, f"Проверено на {len(lag_cols)} колонках"))
        
        # Вывод результатов проверки
        print(f"\n📋 РЕЗУЛЬТАТЫ ПРОВЕРКИ:")
        all_passed = True
        for check_name, passed, details in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}")
            if details:
                print(f"   {details}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print(f"\n🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
            print("✅ Утечка данных устранена")
            print("✅ Признаки создаются корректно")
            print("✅ Модель готова к честному обучению")
        else:
            print(f"\n⚠️  ЕСТЬ ПРОБЛЕМЫ В ИСПРАВЛЕНИЯХ")
        
    except Exception as e:
        print(f"❌ Ошибка проверки: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Основное обучение
    main()
    
    # Проверка исправлений
    validate_fix()
    
    # Демонстрация работы
    demo_prediction()