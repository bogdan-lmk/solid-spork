# RSI Aggregator - Extended Cryptocurrency Market Indicator

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Агрегированный индикатор RSI для анализа состояния криптовалютного рынка. Система рассчитывает взвешенный RSI для топ-20 альткоинов с использованием различных стратегий взвешивания и обходом ограничений API.

## 🎯 Описание

RSI Aggregator - это **расширенный технический индикатор**, который предоставляет макроэкономический взгляд на динамику криптовалютного рынка путем агрегации значений RSI от нескольких активов с правильным взвешиванием.

### Ключевые особенности

- **Взвешенный RSI по рыночной капитализации** для топ-20 криптовалют
- **Включение ETH** - важно для анализа крипторынка
- **Исключение Bitcoin** - фокус на движения альткоинов  
- **Автоматическая фильтрация стейблкоинов** - исключение USDT, USDC и др.
- **Множественные стратегии взвешивания** - по капитализации, объему, равновесная, гибридная
- **Реальные данные** из Binance API с надежным fetcher
- **Мониторинг качества данных** с оценками достоверности
- **Обход ограничений API** - многоуровневая система кеширования и альтернативные источники
- **Специальная обработка XMR** - корректное получение данных Monero

## 📁 Структура проекта

```
rsi_aggregator/
├── src/                           # Основной исходный код
│   ├── core/                      # Ядро системы
│   │   ├── aggregator.py          # Главный класс агрегатора RSI
│   │   ├── data_models.py         # Модели данных и структуры
│   │   └── weighting_strategies.py # Стратегии взвешивания
│   ├── fetchers/                  # Получение рыночных данных
│   │   └── improved_klines_fetcher.py # Надежный Binance API fetcher
│   ├── indicators/                # Технические индикаторы
│   │   └── indicator.py           # RSI и другие индикаторы
│   └── utils/                     # Утилиты
│       ├── alerts.py              # Система уведомлений
│       └── data_persistence.py    # Хранение данных
├── scripts/                       # Вспомогательные скрипты
│   ├── quick_rsi.py              # Быстрое получение RSI
│   ├── show_top_coins.py         # Показать топ-20 монет
│   ├── test_xmr.py               # Тест получения данных XMR
│   └── test_xmr_simple.py        # Простой тест XMR
├── tests/                         # Тестирование
│   ├── test_rsi_fix.py           # Тест исправлений RSI
│   └── test_aggregated_rsi.py    # Тест агрегированного RSI
├── cache/                         # Кеш файлы (создается автоматически)
├── logs/                          # Лог файлы (создается автоматически)
├── main.py                        # Главная точка входа
├── requirements.txt               # Зависимости Python
├── .gitignore                     # Исключения Git
└── README.md                      # Эта документация
```

## 🚀 Быстрый старт

### Установка

1. **Клонирование репозитория:**
```bash
git clone <repository_url>
cd rsi_aggregator
```

2. **Создание виртуального окружения:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate     # Windows
```

3. **Установка зависимостей:**
```bash
pip install -r requirements.txt
```

### Использование

#### 🌐 Веб-дашборд (рекомендуется):
```bash
# Запуск интерактивного дашборда с amCharts
python scripts/run_dashboard.py

# Откройте браузер: http://localhost:5000
```

#### 📊 Демо-дашборд:
```bash
# Статичный HTML демо с примерами данных
open demo_dashboard.html
```

#### 💻 Командная строка:
```bash
# Полный анализ с несколькими стратегиями
python main.py

# Быстрое получение текущего RSI
python scripts/quick_rsi.py

# Показать текущий топ-20 список монет
python scripts/show_top_coins.py
```

#### Программно:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.core.aggregator import MarketCapRSIAggregator

# Создание агрегатора с настройками по умолчанию
aggregator = MarketCapRSIAggregator()

# Запуск анализа
result = aggregator.run_full_analysis()

if result.success:
    snapshot = result.snapshot
    print(f"Агрегированный RSI: {snapshot.aggregated_rsi:.2f}")
    print(f"Рыночные настроения: {snapshot.market_sentiment}")
    print(f"Уверенность: {snapshot.confidence_score:.1f}%")
    print(f"Количество активов: {snapshot.num_assets}")
```

## 📊 Текущий топ-20 криптовалют

Система автоматически выбирает топ-20 криптовалют по рыночной капитализации (исключая BTC и стейблкоины):

1. **ETH** (~42% веса) - Всегда включен
2. **XRP** (~18% веса) - Второй по величине
3. **BNB, SOL, DOGE, TRX, ADA, XLM, SUI, LINK...**

**Специальные исключения:**
- **HYPE** - не торгуется на Binance
- **MATIC** - сетевой токен инфраструктуры
- **Стейблкоины** - USDT, USDC, BUSD и др.
- **Wrapped токены** - WBTC, WETH, STETH и др.

*Общая совокупная капитализация: ~$1.04 триллиона*

## ⚙️ Конфигурация

### Пользовательская конфигурация

```python
from src.core.data_models import IndicatorConfig
from src.core.weighting_strategies import HybridWeightingStrategy

# Кастомная конфигурация
config = IndicatorConfig(
    top_n_assets=20,              # Количество активов
    rsi_period=14,                # Период расчета RSI
    min_market_cap=1e9,           # Минимум $1B капитализации
    min_volume_24h=10e6,          # Минимум $10M объема в день
    cache_duration_minutes=30,     # Длительность кеша данных
    min_confidence_score=50.0      # Минимальный порог качества данных
)

# Кастомная стратегия взвешивания
strategy = HybridWeightingStrategy(
    market_cap_weight=0.6,        # 60% веса по капитализации
    volume_weight=0.4,            # 40% веса по объему
    volatility_adjustment=False   # Без корректировки на волатильность
)

aggregator = MarketCapRSIAggregator(config=config, weighting_strategy=strategy)
```

## 📈 Стратегии взвешивания

### 1. Market Cap Weighting (по умолчанию)
- Больше капитализация = больше веса в итоговом RSI
- ETH доминирует с ~42% веса
- Отражает важность на рынке

### 2. Volume Weighting
- Больше 24ч объема = больше веса
- Акцентирует активность рынка и ликвидность
- Подходит для анализа моментума

### 3. Equal Weighting
- Все активы взвешены равно
- Демократический подход
- Дает равный голос малым капитализациям

### 4. Hybrid Weighting
- Комбинирует капитализацию + объем
- Настраиваемые соотношения
- Опциональная корректировка на волатильность

## 🎯 Интерпретация результатов

### Значения RSI
- **70-100**: Перекупленность - возможная коррекция
- **55-70**: Бычий моментум
- **45-55**: Нейтрально/боковое движение
- **30-45**: Медвежье давление
- **0-30**: Перепроданность - возможный отскок

### Рыночные настроения
- **Overbought**: RSI ≥ 70 - рассмотреть фиксацию прибыли
- **Oversold**: RSI ≤ 30 - потенциальная возможность покупки
- **Neutral**: 45 ≤ RSI ≤ 55 - движение в диапазоне
- **Trending**: Направленное смещение выше/ниже нейтрального

### Оценка уверенности
- **80-100%**: Высокое качество данных, надежный сигнал
- **60-80%**: Хорошее качество, незначительные проблемы с данными
- **40-60%**: Удовлетворительное качество, некоторые данные отсутствуют
- **0-40%**: Плохое качество, использовать с осторожностью

### Консенсус стратегий
- **Strategy consensus RSI** - среднее арифметическое RSI всех стратегий взвешивания
- Показывает общий тренд рынка независимо от выбранной стратегии
- Обеспечивает робастность сигнала

## 🔄 Решение проблем с API лимитами

Система использует **5-уровневую систему** для обеспечения стабильной работы:

1. **Tier 1**: Persistent cache (2 часа) - избегает API вызовов
2. **Tier 2**: CoinGecko API - основной источник данных
3. **Tier 3**: CoinMarketCap API - альтернативный источник
4. **Tier 4**: Последние успешные данные - если все API недоступны
5. **Tier 5**: Hardcoded fallback - только в крайнем случае

### Особенности кеширования:
- **Автоматическое сохранение** успешных запросов в `cache/top_coins_cache.json`
- **Проверка актуальности** кеша по времени
- **Детекция 429 ошибок** (rate limiting) с автоматическим переключением

## 🧪 Тестирование

### Запуск тестов:
```bash
# Тест основной функциональности
python tests/test_rsi_fix.py

# Тест получения данных XMR
python scripts/test_xmr.py

# Простой тест XMR через Binance API
python scripts/test_xmr_simple.py
```

### Проверка работы системы:
```bash
# Показать какие монеты будут выбраны
python scripts/show_top_coins.py

# Быстрый тест текущего RSI
python scripts/quick_rsi.py
```

## 📊 Архитектура системы

### Компоненты ядра:

#### MarketCapRSIAggregator
- Главный класс для расчета агрегированного RSI
- Управляет получением данных, кешированием и расчетами
- Поддерживает различные стратегии взвешивания

#### Data Models
- `CoinMarketData` - данные о монете (символ, капитализация, RSI)
- `AggregatedRSISnapshot` - снимок результата расчета
- `IndicatorConfig` - конфигурация параметров индикатора

#### Weighting Strategies
- Абстрактная база `WeightingStrategy`
- Реализации: MarketCap, Volume, Equal, Hybrid
- Расширяемая архитектура для новых стратегий

### Получение данных:

#### improved_klines_fetcher.py
- Надежное получение данных с Binance API
- Обработка таймаутов и ошибок сети
- Специальная логика для XMR (Monero)
- Множественные временные интервалы и агрегация

## 💡 Примеры использования

### 1. Мониторинг рынка
```python
# Получение обзора рынка
aggregator = MarketCapRSIAggregator()
result = aggregator.run_full_analysis()

if result.success:
    rsi = result.snapshot.aggregated_rsi
    if rsi >= 70:
        print("⚠️ Рынок перекуплен - возможна коррекция")
    elif rsi <= 30:
        print("🟢 Рынок перепродан - возможен отскок")
    else:
        print(f"📊 Рынок в состоянии: {result.snapshot.market_sentiment}")
```

### 2. Сравнение стратегий
```python
# Использование разных стратегий взвешивания
strategies = [
    MarketCapWeightingStrategy(),
    VolumeWeightingStrategy(), 
    EqualWeightingStrategy(),
    HybridWeightingStrategy(0.7, 0.3)
]

for strategy in strategies:
    aggregator = MarketCapRSIAggregator(weighting_strategy=strategy)
    result = aggregator.run_full_analysis()
    if result.success:
        print(f"{strategy.name}: RSI {result.snapshot.aggregated_rsi:.2f}")
```

### 3. Долгосрочный мониторинг
```python
# Ежедневное отслеживание с сохранением истории
import time
from datetime import datetime

while True:
    aggregator = MarketCapRSIAggregator()
    result = aggregator.run_full_analysis()
    
    if result.success:
        snapshot = result.snapshot
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"[{timestamp}] RSI: {snapshot.aggregated_rsi:.2f} "
              f"| Confidence: {snapshot.confidence_score:.1f}% "
              f"| Sentiment: {snapshot.market_sentiment}")
        
        # Сохранение в базу данных или файл
        # save_to_database(snapshot)
    
    time.sleep(3600)  # Проверка каждый час
```

## ⚠️ Важные замечания

- Это **запаздывающий индикатор** - используйте с другими видами анализа
- **Не является финансовой консультацией** - только для образовательных/исследовательских целей
- Требует стабильного интернета для получения данных в реальном времени
- Веса по капитализации изменяются вместе с рыночными условиями
- Большой вес ETH (~42%) значительно влияет на результаты
- Система оптимизирована для **защитного анализа торговли**

## 🔗 Зависимости

```
Core:
- pandas, numpy - манипуляция данными
- requests - API вызовы  
- talib - технические индикаторы

Visualization (optional):
- matplotlib, seaborn, plotly - графики

Development:
- pytest - тестирование
- jupyter - интерактивная разработка
```

## 📞 Поддержка

При возникновении проблем:

1. **Проверьте логи** в `logs/aggregated_rsi.log`
2. **Убедитесь в наличии интернет-соединения** для API вызовов
3. **Проверьте установку зависимостей**: `pip install -r requirements.txt`
4. **Просмотрите настройки конфигурации**
5. **Проверьте кеш**: удалите `cache/` для принудительного обновления

### Распространенные проблемы:

**API Limits (429 errors):**
- Система автоматически переключается на альтернативные источники
- Кеш предотвращает частые вызовы API

**Empty XMR data:**
- Используется специальная логика обработки для Monero
- Прямые API вызовы обходят сложную логику временных рамок

**Import errors:**
- Убедитесь, что вы запускаете скрипты из корневой директории проекта
- Проверьте структуру путей в `sys.path`

---

**RSI Aggregator v2.0**  
*Extended Cryptocurrency Market Indicator*  
*Разработано для анализа агрегированных трендов криптовалютного рынка*