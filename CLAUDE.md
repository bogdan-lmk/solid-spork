# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RSI Aggregator is a defensive cryptocurrency market analysis tool that calculates aggregated RSI (Relative Strength Index) for the top 20 altcoins. The system combines market cap and volume weighting strategies to provide a single RSI value representing overall altcoin market sentiment.

**Key Focus**: Defensive trading analysis - excludes Bitcoin, includes ETH, filters out stablecoins and wrapped tokens.

## Common Commands

### Development & Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Run main RSI calculation (simplified single value output)
python main.py

# Quick RSI check
python scripts/quick_rsi.py

# Show current top-20 coin selection
python scripts/show_top_coins.py

# Start web dashboard (Chart.js visualization)
python scripts/run_dashboard.py
# Opens at: http://localhost:5000

# Run tests
python tests/test_rsi_fix.py
python tests/test_aggregated_rsi.py
python scripts/test_xmr.py
```

### API Endpoints
- `/api/rsi` - Current aggregated RSI data  
- `/api/rsi/historical` - **Real historical RSI data** (90 days of calculated values)
- `/simple` - Simple dashboard view

**Note**: `/api/rsi/historical` now returns actual calculated RSI from real price data, not simulated values.

## Architecture Overview

### Core RSI Calculation Strategy
The system calculates RSI using a **dual-weighting approach**:
1. Market cap weighted RSI for top 20 altcoins
2. Volume weighted RSI for the same assets  
3. Final result: `(market_cap_rsi + volume_rsi) / 2`

This is implemented in `main.py:get_single_rsi()` function.

### Key Components

**MarketCapRSIAggregator** (`src/core/aggregator.py`)
- Core calculation engine
- Manages Binance API integration with 5-tier fallback system
- Handles special cases (XMR data, API limits, caching)

**Real Historical Data System** (`web/app.py:get_real_historical_rsi_data()`)
- Fetches 90 days of real historical price data for each top-20 coin
- Calculates genuine RSI using `calculate_wilder_rsi` (14-period)
- Aggregates daily RSI with proper market cap + volume weighting
- NO synthetic/random data - only real market data

**Weighting Strategies** (`src/core/weighting_strategies.py`)
- Abstract base: `WeightingStrategy`
- Active implementations: `MarketCapWeightingStrategy`, `VolumeWeightingStrategy`
- Removed: Equal, Hybrid, Volatility strategies (simplified architecture)

**Data Models** (`src/core/data_models.py`)
- `CoinMarketData`: Individual coin data structure
- `AggregatedRSISnapshot`: Final calculation result
- `IndicatorConfig`: System configuration parameters

### Coin Selection Logic

**Included**: Top 20 by market cap (ETH always included despite being #2)
**Excluded**: 
- Bitcoin (focus on altcoin movements)
- Stablecoins (USDT, USDC, DAI, etc.)
- Wrapped tokens (WBTC, WETH, STETH)
- Exchange tokens (LEO, CRO, OKB) 
- Infrastructure tokens (MATIC)
- Non-Binance assets (HYPE, BGB)

### API Resilience System

5-tier fallback for market data:
1. Persistent cache (2 hours)
2. CoinGecko API (primary)
3. CoinMarketCap API (secondary)  
4. Last successful data
5. Hardcoded fallback

Cache stored in: `cache/top_coins_cache.json`

### Web Interface

**Frontend**: Chart.js (switched from amCharts for reliability)
**Backend**: Flask with real-time RSI API
**Visualization**: Single green line showing aggregated RSI with overbought/oversold zones

Chart displays:
- Green RSI line (#4CAF50) with **real historical data**
- Red reference lines at 70 (overbought) and 30 (oversold)
- Gray middle line at 50
- **Consistent data** - same RSI values for same dates on every reload

### Historical Data Pipeline

**Real Data Flow** (no synthetic data):
1. Get current top-20 coins from aggregator
2. Fetch 90 days of 1d OHLC data for each coin via Binance API
3. Calculate 14-period Wilder's RSI for each coin daily
4. Aggregate daily RSI using market cap + volume weights
5. Return time series of real aggregated RSI values

**Data Consistency**: RSI values are deterministic - same input dates produce identical RSI outputs

## Project Structure Notes

- **`main.py`**: Simplified entry point, outputs single RSI value
- **`src/core/`**: All calculation logic
- **`src/fetchers/`**: Binance API integration with reliability features
- **`src/indicators/`**: RSI calculation utilities  
- **`web/app.py`**: Flask application with historical data generation
- **`templates/rsi_chart.html`**: Chart.js visualization
- **`scripts/`**: Utility scripts for testing and quick checks
- **`logs/`**: Auto-created, contains `aggregated_rsi.log`

## Important Implementation Details

### Real Data Requirements
**CRITICAL**: System only works with real market data - no synthetic/random generation allowed.

- Historical RSI must be calculated from actual OHLC price data
- Every coin's RSI calculated using `calculate_wilder_rsi()` with 14-period
- Aggregation uses real market cap and volume weights from current market state
- Data consistency: identical input dates = identical RSI outputs

### Path Management
All scripts must run from project root. The codebase uses:
```python
sys.path.append(str(Path(__file__).parent))
```

### RSI Interpretation
- 70-100: Overbought (potential correction)
- 30-70: Normal range  
- 0-30: Oversold (potential buying opportunity)

### Special Handling
- **XMR (Monero)**: Custom fetching logic due to API limitations
- **ETH Weight**: ~42% of total weight (largest component)
- **API Rate Limits**: Automatic detection and fallback switching
- **Data Validation**: Must have 'close' column and minimum 20 data points for RSI calculation

## Logging
System logs to both console and `logs/aggregated_rsi.log`. Check logs for:
- API failures and fallback activations
- Data quality issues
- Coin selection process
- RSI calculation warnings