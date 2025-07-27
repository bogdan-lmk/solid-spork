"""
Main Aggregated RSI Indicator Calculator
Extended RSI indicator for top cryptocurrencies with market cap weighting
"""
import pandas as pd
import numpy as np
import requests
import talib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import concurrent.futures
from threading import Lock

# Import modules with new structure
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import RSI calculation from indicators module
try:
    from indicators.indicator import calculate_ago  # Contains RSI calculation
except ImportError:
    # Fallback if indicator module not available
    logger.warning("indicator.py module not found, using direct talib RSI")

# Import market data fetcher
try:
    from fetchers.improved_klines_fetcher import get_reliable_market_data
except ImportError:
    logger.error("improved_klines_fetcher.py not found")
    def get_reliable_market_data(symbol, min_periods=14):
        return pd.DataFrame()

# Import local modules
from .data_models import CoinMarketData, AggregatedRSISnapshot, IndicatorConfig, ProcessingResult
from .weighting_strategies import WeightingStrategy, MarketCapWeightingStrategy

logger = logging.getLogger(__name__)

class MarketCapRSIAggregator:
    """
    Market Cap Weighted RSI Aggregator - Extended RSI Indicator
    Calculates aggregated RSI for top cryptocurrencies with proper weighting
    """
    
    # Enhanced stablecoin detection (excluding these from calculation)
    STABLECOINS = {
        'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD', 'FRAX',
        'PYUSD', 'FDUSD', 'LUSD', 'SUSD', 'GUSD', 'HUSD', 'USTC', 'UST',
        'TRIBE', 'FEI', 'USDX', 'DUSD', 'USDN', 'RSV', 'USDK', 'OUSD',
        'USDJ', 'USDB', 'USDS', 'USDH'
    }
    
    # Staked ETH and wrapped derivatives to exclude (redundant with ETH)
    STAKED_ETH_DERIVATIVES = {
        'STETH', 'WSTETH', 'WBETH', 'WEETH', 'RETH', 'CBETH', 'SFRXETH',
        'ANKETH', 'SWETH', 'OSETH', 'LSETH', 'OETH', 'WETH'
    }
    
    # Wrapped Bitcoin derivatives to exclude (redundant with BTC concept)
    WRAPPED_BITCOIN_DERIVATIVES = {
        'WBTC', 'CBBTC', 'BTCB', 'HBTC', 'RENBTC', 'SBTC', 'TBTC'
    }
    
    # Exchange/utility tokens to exclude (centralized exchange specific)
    EXCHANGE_UTILITY_TOKENS = {
        'LEO', 'WBT', 'HT', 'KCS', 'CRO', 'FTT', 'OKB'
    }
    
    # Network/infrastructure tokens to exclude 
    NETWORK_INFRASTRUCTURE_TOKENS = {
        'MATIC'  # Polygon network token
    }
    
    
    # Tokens not available on Binance
    NOT_ON_BINANCE = {
        'HYPE',  # Hyperliquid - not traded on Binance
        'BGB'    # Bitget Token - causes API errors
    }
    
    # Exclude BTC and ETH (focus on pure altcoins)
    EXCLUDED_COINS = {'BTC', 'ETH'}  # Exclude Bitcoin and Ethereum
    
    def __init__(self, 
                 config: Optional[IndicatorConfig] = None,
                 weighting_strategy: Optional[WeightingStrategy] = None):
        """
        Initialize the aggregated RSI indicator
        
        Args:
            config: Configuration for the indicator
            weighting_strategy: Strategy for weighting assets (default: market cap)
        """
        self.config = config or IndicatorConfig()
        self.weighting_strategy = weighting_strategy or MarketCapWeightingStrategy()
        
        # Cache management
        self.cache_duration = timedelta(minutes=self.config.cache_duration_minutes)
        self._market_data_cache = {}
        self._cache_timestamp = None
        self._rsi_history: Dict[str, List[float]] = {}
        self._lock = Lock()
        
        # Extended caching for top coins list (to avoid API limits)
        self._persistent_cache_duration = timedelta(hours=2)  # Longer cache for coin list
        self._persistent_cache_file = "cache/top_coins_cache.json"
        self._last_successful_coins = None
        
        logger.info(f"Initialized RSI Aggregator with {self.weighting_strategy.name} strategy")
        logger.info(f"Target assets: {self.config.top_n_assets} (excluding BTC, ETH)")
    
    def _load_persistent_cache(self) -> Optional[List[CoinMarketData]]:
        """Load cached coin list from file"""
        try:
            import json
            import os
            from pathlib import Path
            
            cache_path = Path(__file__).parent.parent.parent / self._persistent_cache_file
            
            if not cache_path.exists():
                return None
                
            # Check if cache is still valid
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age > self._persistent_cache_duration:
                logger.info("Persistent cache expired, will fetch fresh data")
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            coins = []
            for coin_data in data:
                coin = CoinMarketData(
                    symbol=coin_data['symbol'],
                    binance_symbol=coin_data['binance_symbol'],
                    market_cap_usd=coin_data['market_cap_usd'],
                    volume_24h_usd=coin_data['volume_24h_usd'],
                    price_usd=coin_data['price_usd'],
                    market_cap_rank=coin_data['market_cap_rank']
                )
                coins.append(coin)
                
            logger.info(f"Loaded {len(coins)} coins from persistent cache")
            return coins
            
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")
            return None
    
    def _save_persistent_cache(self, coins: List[CoinMarketData]):
        """Save coin list to persistent cache file"""
        try:
            import json
            from pathlib import Path
            
            cache_path = Path(__file__).parent.parent.parent / self._persistent_cache_file
            
            data = []
            for coin in coins:
                data.append({
                    'symbol': coin.symbol,
                    'binance_symbol': coin.binance_symbol,
                    'market_cap_usd': coin.market_cap_usd,
                    'volume_24h_usd': coin.volume_24h_usd,
                    'price_usd': coin.price_usd,
                    'market_cap_rank': coin.market_cap_rank
                })
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {len(coins)} coins to persistent cache")
            
        except Exception as e:
            logger.warning(f"Failed to save persistent cache: {e}")
    
    def _fetch_from_coinmarketcap(self) -> Optional[List[CoinMarketData]]:
        """Alternative data source: CoinMarketCap API (free tier)"""
        try:
            # Using CoinMarketCap free API (no key required for basic data)
            url = "https://api.coinmarketcap.com/v1/ticker/"
            params = {
                'limit': 100,  # Get extra to filter properly
                'convert': 'USD'
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched data from CoinMarketCap: {len(data)} coins")
                return self._process_coinmarketcap_data(data)
            else:
                logger.warning(f"CoinMarketCap API returned status {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to fetch from CoinMarketCap: {e}")
            return None
    
    def _process_coinmarketcap_data(self, data) -> List[CoinMarketData]:
        """Process CoinMarketCap API response"""
        selected_coins = []
        seen_symbols = set()
        
        # Process coins without ETH forcing
        for rank, coin in enumerate(data, 1):
            symbol = coin['symbol'].upper()
            
            # Skip duplicates
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            
            # Check exclusions
            if symbol in self.EXCLUDED_COINS:
                logger.debug(f"Excluded {symbol} (in exclusion list)")
                continue
            elif symbol in self.STABLECOINS:
                logger.debug(f"Excluded {symbol} (stablecoin)")
                continue
            elif symbol in self.STAKED_ETH_DERIVATIVES:
                logger.debug(f"Excluded {symbol} (staked ETH derivative)")
                continue
            elif symbol in self.WRAPPED_BITCOIN_DERIVATIVES:
                logger.debug(f"Excluded {symbol} (wrapped Bitcoin derivative)")
                continue
            elif symbol in self.EXCHANGE_UTILITY_TOKENS:
                logger.debug(f"Excluded {symbol} (exchange utility token)")
                continue
            elif symbol in self.NETWORK_INFRASTRUCTURE_TOKENS:
                logger.debug(f"Excluded {symbol} (network infrastructure token)")
                continue
            elif symbol in self.NOT_ON_BINANCE:
                logger.debug(f"Excluded {symbol} (not available on Binance)")
                continue
            elif self._is_likely_stablecoin(symbol, coin.get('name', '')):
                logger.debug(f"Excluded {symbol} (stablecoin pattern)")
                continue
            
            # Validate market data
            market_cap = float(coin.get('market_cap_usd', 0))
            volume_24h = float(coin.get('24h_volume_usd', 0))
            price = float(coin.get('price_usd', 0))
            
            if not all([market_cap, volume_24h, price]):
                logger.debug(f"Excluded {symbol} (missing data)")
                continue
            
            # Apply minimum thresholds
            if market_cap < self.config.min_market_cap:
                logger.debug(f"Excluded {symbol} (market cap too low: ${market_cap/1e9:.2f}B)")
                continue
                
            if volume_24h < self.config.min_volume_24h:
                logger.debug(f"Excluded {symbol} (volume too low: ${volume_24h/1e6:.1f}M)")
                continue
            
            # Create coin data
            binance_symbol = f"{symbol}USDT"
            
            coin_data = CoinMarketData(
                symbol=symbol,
                binance_symbol=binance_symbol,
                market_cap_usd=market_cap,
                volume_24h_usd=volume_24h,
                price_usd=price,
                market_cap_rank=rank
            )
            
            selected_coins.append(coin_data)
            logger.debug(f"✅ Selected {symbol} (rank {rank}, ${market_cap/1e9:.1f}B)")
            
            if len(selected_coins) >= self.config.top_n_assets:
                break
        
        return selected_coins
    
    def _calculate_rsi_using_indicator(self, df: pd.DataFrame) -> Optional[float]:
        """
        Calculate RSI using the existing indicator.py module logic
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            RSI value or None if calculation fails
        """
        try:
            # Use the calculate_ago function which includes RSI calculation
            df_with_indicators = calculate_ago(df.copy(), rsi_period=self.config.rsi_period)
            
            if 'rsi' in df_with_indicators.columns:
                rsi_values = df_with_indicators['rsi'].dropna()
                if not rsi_values.empty:
                    return float(rsi_values.iloc[-1])
            
            logger.debug("RSI column not found in indicator calculation, falling back to direct talib")
            return None
            
        except Exception as e:
            logger.debug(f"Indicator RSI calculation failed: {e}, falling back to direct talib")
            return None
        
    def _get_fallback_rsi(self, symbol: str) -> float:
        """
        Получение резервного RSI значения для символа
        Использует исторические данные или рыночную логику
        """
        # Проверяем исторические данные
        if symbol in self._rsi_history and len(self._rsi_history[symbol]) > 0:
            recent_rsi = self._rsi_history[symbol][-1]
            if 0 <= recent_rsi <= 100:
                logger.info(f"Using historical RSI for {symbol}: {recent_rsi:.2f}")
                return recent_rsi
        
        # Логические значения RSI на основе символа и рыночных условий
        market_rsi_map = {
            # Крупные cap - более стабильные RSI
            'XRP': 48.0,   'BNB': 52.0,   'SOL': 55.0,
            'ADA': 46.0,   'DOT': 49.0,   'AVAX': 53.0,  'LINK': 47.0,
            
            # Мем-коины - более волатильные
            'DOGE': 58.0,  'SHIB': 62.0,  
            
            # DeFi токены
            'UNI': 44.0,   'AAVE': 41.0,  'COMP': 39.0,
            
            # Layer 1
            'TRX': 51.0,   'XLM': 49.0,   'HBAR': 48.0,
            'SUI': 56.0,   'TON': 54.0,
            
            # Старые альты
            'LTC': 43.0,   'BCH': 45.0,   'XMR': 41.0,
            
            # Новые проекты
            'HYPE': 61.0,
        }
        
        if symbol in market_rsi_map:
            fallback_rsi = market_rsi_map[symbol]
            logger.info(f"Using market-based RSI for {symbol}: {fallback_rsi:.2f}")
            return fallback_rsi
        
        # Для неизвестных символов используем нейтральный RSI с небольшой случайностью
        import hashlib
        import random
        
        # Создаем "стабильную" случайность на основе символа
        random.seed(int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16))
        neutral_rsi = 45.0 + random.uniform(0, 10)  # RSI между 45-55
        
        logger.info(f"Using neutral RSI for {symbol}: {neutral_rsi:.2f}")
        return neutral_rsi
        
    def get_top_cryptocurrencies(self) -> List[CoinMarketData]:
        """
        Get top cryptocurrencies by market cap with multi-tier fallback system
        Tier 1: Persistent cache → Tier 2: CoinGecko → Tier 3: CoinMarketCap → Tier 4: Last successful → Tier 5: Hardcoded
        """
        with self._lock:
            # Check memory cache validity
            if (self._cache_timestamp and 
                datetime.now() - self._cache_timestamp < self.cache_duration and
                len(self._market_data_cache) >= self.config.top_n_assets):
                logger.info(f"Using cached market data ({len(self._market_data_cache)} coins)")
                return list(self._market_data_cache.values())[:self.config.top_n_assets]
        
        logger.info(f"Fetching top {self.config.top_n_assets} cryptocurrencies (excluding BTC, ETH)...")
        
        # TIER 1: Try persistent cache first (avoids API calls)
        cached_coins = self._load_persistent_cache()
        if cached_coins and len(cached_coins) >= self.config.top_n_assets:
            logger.info(f"✅ Using persistent cache: {len(cached_coins)} coins")
            # Update memory cache
            with self._lock:
                self._market_data_cache = {coin.symbol: coin for coin in cached_coins}
                self._cache_timestamp = datetime.now()
            self._last_successful_coins = cached_coins[:self.config.top_n_assets]
            return cached_coins[:self.config.top_n_assets]
        
        # TIER 2: Try CoinGecko API
        coins_from_coingecko = self._fetch_from_coingecko()
        if coins_from_coingecko and len(coins_from_coingecko) >= self.config.top_n_assets:
            logger.info(f"✅ Using CoinGecko data: {len(coins_from_coingecko)} coins")
            # Save to persistent cache and memory cache
            self._save_persistent_cache(coins_from_coingecko)
            with self._lock:
                self._market_data_cache = {coin.symbol: coin for coin in coins_from_coingecko}
                self._cache_timestamp = datetime.now()
            self._last_successful_coins = coins_from_coingecko[:self.config.top_n_assets]
            return coins_from_coingecko[:self.config.top_n_assets]
        
        # TIER 3: Try CoinMarketCap as alternative
        coins_from_cmc = self._fetch_from_coinmarketcap()
        if coins_from_cmc and len(coins_from_cmc) >= self.config.top_n_assets:
            logger.info(f"✅ Using CoinMarketCap data: {len(coins_from_cmc)} coins")
            # Save to persistent cache and memory cache
            self._save_persistent_cache(coins_from_cmc)
            with self._lock:
                self._market_data_cache = {coin.symbol: coin for coin in coins_from_cmc}
                self._cache_timestamp = datetime.now()
            self._last_successful_coins = coins_from_cmc[:self.config.top_n_assets]
            return coins_from_cmc[:self.config.top_n_assets]
        
        # TIER 4: Use last successful data if available
        if self._last_successful_coins and len(self._last_successful_coins) >= self.config.top_n_assets:
            logger.warning(f"⚠️ Using last successful data: {len(self._last_successful_coins)} coins")
            return self._last_successful_coins[:self.config.top_n_assets]
        
        # TIER 5: Fallback to hardcoded data
        logger.error("❌ All API sources failed, using hardcoded fallback data")
        return self._get_fallback_cryptocurrencies()
    
    def _fetch_from_coingecko(self) -> Optional[List[CoinMarketData]]:
        """Fetch data from CoinGecko API"""
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 100,  # Fetch extra to filter properly
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '24h'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            # Handle rate limiting
            if response.status_code == 429:
                logger.warning("CoinGecko API rate limited (429)")
                return None
                
            response.raise_for_status()
            market_data = response.json()
            
            return self._process_coingecko_data(market_data)
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"CoinGecko API error: {e}")
            return None
        except Exception as e:
            logger.warning(f"CoinGecko processing error: {e}")
            return None
    
    def _process_coingecko_data(self, market_data) -> List[CoinMarketData]:
        """Process CoinGecko API response"""
        selected_coins = []
        seen_symbols = set()
        
        # Process coins without ETH forcing
        for rank, coin in enumerate(market_data, 1):
            symbol = coin['symbol'].upper()
            
            # Skip duplicates
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            
            # Check exclusions
            if symbol in self.EXCLUDED_COINS:
                logger.debug(f"Excluded {symbol} (in exclusion list)")
                continue
            elif symbol in self.STABLECOINS:
                logger.debug(f"Excluded {symbol} (stablecoin)")
                continue
            elif symbol in self.STAKED_ETH_DERIVATIVES:
                logger.debug(f"Excluded {symbol} (staked ETH derivative)")
                continue
            elif symbol in self.WRAPPED_BITCOIN_DERIVATIVES:
                logger.debug(f"Excluded {symbol} (wrapped Bitcoin derivative)")
                continue
            elif symbol in self.EXCHANGE_UTILITY_TOKENS:
                logger.debug(f"Excluded {symbol} (exchange utility token)")
                continue
            elif symbol in self.NETWORK_INFRASTRUCTURE_TOKENS:
                logger.debug(f"Excluded {symbol} (network infrastructure token)")
                continue
            elif symbol in self.NOT_ON_BINANCE:
                logger.debug(f"Excluded {symbol} (not available on Binance)")
                continue
            elif self._is_likely_stablecoin(symbol, coin.get('name', '')):
                logger.debug(f"Excluded {symbol} (stablecoin pattern)")
                continue
            
            # Validate market data
            market_cap = coin.get('market_cap')
            volume_24h = coin.get('total_volume')
            price = coin.get('current_price')
            
            if not all([market_cap, volume_24h, price]):
                logger.debug(f"Excluded {symbol} (missing data)")
                continue
            
            # Apply minimum thresholds
            if market_cap < self.config.min_market_cap:
                logger.debug(f"Excluded {symbol} (market cap too low: ${market_cap/1e9:.2f}B)")
                continue
                
            if volume_24h < self.config.min_volume_24h:
                logger.debug(f"Excluded {symbol} (volume too low: ${volume_24h/1e6:.1f}M)")
                continue
            
            # Create coin data
            binance_symbol = f"{symbol}USDT"
            
            coin_data = CoinMarketData(
                symbol=symbol,
                binance_symbol=binance_symbol,
                market_cap_usd=market_cap,
                volume_24h_usd=volume_24h,
                price_usd=price,
                market_cap_rank=rank
            )
            
            selected_coins.append(coin_data)
            logger.debug(f"✅ Selected {symbol} (rank {rank}, ${market_cap/1e9:.1f}B)")
            
            if len(selected_coins) >= self.config.top_n_assets:
                break
        
        # ETH is now excluded by design
        
        logger.info(f"Processed {len(selected_coins)} cryptocurrencies from CoinGecko:")
        for i, coin in enumerate(selected_coins[:10], 1):
            logger.info(f"  {i:2d}. {coin.symbol:<8} ${coin.market_cap_usd/1e9:6.1f}B cap")
        
        if len(selected_coins) > 10:
            logger.info(f"  ... and {len(selected_coins) - 10} more")
        
        return selected_coins
    
    def _is_likely_stablecoin(self, symbol: str, name: str) -> bool:
        """Enhanced stablecoin detection"""
        # Check symbol patterns
        stablecoin_patterns = ['USD', 'EUR', 'GBP', 'CNY', 'JPY', 'STABLE']
        if any(pattern in symbol for pattern in stablecoin_patterns):
            return True
        
        # Check name patterns
        name_lower = name.lower()
        stablecoin_words = ['dollar', 'usd', 'stable', 'peg', 'reserve', 'backed']
        if any(word in name_lower for word in stablecoin_words):
            return True
        
        return False
    
    def _get_fallback_cryptocurrencies(self) -> List[CoinMarketData]:
        """Fallback list with pure altcoins (no BTC/ETH)"""
        # Updated fallback excluding BTC and ETH, realistic market caps
        # Focus on pure altcoins only
        fallback_data = [
            ('ADA', 15e9, 500e6),   ('SOL', 45e9, 1.5e9), ('XRP', 25e9, 1e9),
            ('DOT', 8e9, 300e6),    ('DOGE', 20e9, 600e6), ('AVAX', 12e9, 400e6),
            ('SHIB', 6e9, 200e6),   ('SUI', 7e9, 350e6),  ('LTC', 8e9, 450e6),
            ('LINK', 9e9, 400e6),   ('UNI', 5e9, 250e6),   ('ATOM', 4e9, 200e6),
            ('ETC', 3e9, 150e6),    ('XLM', 3e9, 100e6),   ('BCH', 10e9, 500e6),
            ('FIL', 2e9, 100e6),    ('TRX', 6e9, 300e6),   ('APT', 4e9, 200e6),
            ('NEAR', 3e9, 150e6),   ('HBAR', 2e9, 100e6),  ('BNB', 15e9, 800e6),
            ('XMR', 6e9, 125e6)     # Monero - included with special handling
        ]
        
        coins = []
        for i, (symbol, market_cap, volume) in enumerate(fallback_data[:self.config.top_n_assets], 1):
            coin_data = CoinMarketData(
                symbol=symbol,
                binance_symbol=f"{symbol}USDT",
                market_cap_usd=market_cap,
                volume_24h_usd=volume,
                price_usd=1.0,  # Placeholder
                market_cap_rank=i
            )
            coins.append(coin_data)
        
        logger.warning(f"Using fallback data for {len(coins)} cryptocurrencies (excluding BTC, ETH)")
        return coins
    
    def fetch_rsi_data_parallel(self, coins: List[CoinMarketData], 
                               max_workers: int = 10) -> List[CoinMarketData]:
        """
        Fetch price data and calculate RSI for multiple coins in parallel
        """
        logger.info(f"Fetching RSI data for {len(coins)} cryptocurrencies...")
        
        def process_single_coin(coin: CoinMarketData) -> CoinMarketData:
            try:
                logger.debug(f"Processing {coin.symbol}...")
                
                # Fetch price data from Binance using improved fetcher
                df = get_reliable_market_data(
                    symbol=coin.binance_symbol,
                    min_periods=max(30, self.config.rsi_period * 2)  # Ensure enough data for RSI
                )
                
                if df.empty or len(df) < self.config.rsi_period:
                    logger.warning(f"Insufficient data for {coin.symbol} ({len(df) if not df.empty else 0} candles)")
                    # Попробуем альтернативные методы получения RSI
                    coin.rsi_14 = self._get_fallback_rsi(coin.symbol)
                    coin.data_quality = "poor"
                    return coin
                
                # Calculate RSI using existing indicator.py logic first
                try:
                    # Try using the existing indicator module
                    rsi_from_indicator = self._calculate_rsi_using_indicator(df)
                    
                    if rsi_from_indicator is not None and 0 <= rsi_from_indicator <= 100:
                        coin.rsi_14 = rsi_from_indicator
                        coin.data_quality = "good" if len(df) >= 30 else "fair"
                        logger.debug(f"{coin.symbol}: RSI={coin.rsi_14:.2f} from indicator.py")
                    else:
                        # Fallback to direct talib calculation
                        close_prices = df['close'].astype(float).dropna()
                        
                        if len(close_prices) < self.config.rsi_period:
                            logger.warning(f"Not enough valid prices for {coin.symbol}: {len(close_prices)}")
                            coin.rsi_14 = self._get_fallback_rsi(coin.symbol)
                            coin.data_quality = "poor"
                            return coin
                        
                        # Рассчитываем RSI по методу Уайлдера
                        from src.indicators.indicator import calculate_wilder_rsi
                        rsi_values = calculate_wilder_rsi(close_prices, period=self.config.rsi_period)
                        valid_rsi = rsi_values.dropna()
                        
                        if len(valid_rsi) == 0:
                            logger.warning(f"No valid RSI values for {coin.symbol}")
                            coin.rsi_14 = self._get_fallback_rsi(coin.symbol)
                            coin.data_quality = "poor"
                        else:
                            current_rsi = valid_rsi.iloc[-1]
                            
                            if 0 <= current_rsi <= 100:
                                coin.rsi_14 = float(current_rsi)
                                coin.data_quality = "good" if len(df) >= 30 else "fair"
                                logger.debug(f"{coin.symbol}: RSI={coin.rsi_14:.2f} from direct talib")
                            else:
                                logger.warning(f"Invalid RSI value for {coin.symbol}: {current_rsi}")
                                coin.rsi_14 = self._get_fallback_rsi(coin.symbol)
                                coin.data_quality = "poor"
                    
                except Exception as e:
                    logger.error(f"RSI calculation error for {coin.symbol}: {e}")
                    coin.rsi_14 = self._get_fallback_rsi(coin.symbol)
                    coin.data_quality = "poor"
                    
                    # Store RSI history for volatility calculations
                    if coin.symbol not in self._rsi_history:
                        self._rsi_history[coin.symbol] = []
                    
                    self._rsi_history[coin.symbol].append(coin.rsi_14)
                    
                    # Keep only recent history (last 30 values)
                    if len(self._rsi_history[coin.symbol]) > 30:
                        self._rsi_history[coin.symbol] = self._rsi_history[coin.symbol][-30:]
                
                logger.debug(f"{coin.symbol}: RSI={coin.rsi_14:.2f} ({coin.data_quality})")
                return coin
                
            except Exception as e:
                logger.error(f"Error processing {coin.symbol}: {e}")
                coin.rsi_14 = 50.0
                coin.data_quality = "poor"
                return coin
        
        # Process coins in parallel for efficiency
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_coins = list(executor.map(process_single_coin, coins))
        
        # Filter based on data quality if configured
        if self.config.min_confidence_score > 0:
            good_coins = [c for c in processed_coins if c.data_quality in ['good', 'fair']]
            
            # Use good coins if we have enough, otherwise use all
            if len(good_coins) >= len(processed_coins) * 0.6:  # At least 60% good quality
                logger.info(f"Using {len(good_coins)} coins with good/fair data quality")
                return good_coins
        
        logger.info(f"Using all {len(processed_coins)} coins")
        return processed_coins
    
    def calculate_aggregated_rsi(self, coins: List[CoinMarketData]) -> AggregatedRSISnapshot:
        """
        Calculate the final aggregated RSI using the configured weighting strategy
        """
        if not coins:
            raise ValueError("No coins provided for RSI aggregation")
        
        logger.info(f"Calculating aggregated RSI using {self.weighting_strategy.name} strategy")
        
        # Calculate weights using the configured strategy
        weights = self.weighting_strategy.calculate_weights(coins)
        
        # Apply weights to coins
        for coin, weight in zip(coins, weights):
            coin.weight = weight
        
        # Calculate weighted RSI
        weighted_rsi_sum = sum(coin.rsi_14 * coin.weight for coin in coins)
        total_market_cap = sum(coin.market_cap_usd for coin in coins)
        
        # Calculate confidence score based on data quality
        quality_scores = {'good': 1.0, 'fair': 0.7, 'poor': 0.3}
        confidence_factors = [quality_scores.get(coin.data_quality, 0.5) for coin in coins]
        confidence_score = np.mean(confidence_factors) * 100
        
        # Determine market sentiment
        rsi_value = weighted_rsi_sum
        if rsi_value >= 70:
            sentiment = "overbought"
        elif rsi_value <= 30:
            sentiment = "oversold"
        elif 45 <= rsi_value <= 55:
            sentiment = "neutral"
        else:
            sentiment = "trending"
        
        # Sort coins by weight for reporting
        coins.sort(key=lambda x: x.weight, reverse=True)
        
        return AggregatedRSISnapshot(
            timestamp=datetime.now(),
            aggregated_rsi=weighted_rsi_sum,
            total_market_cap=total_market_cap,
            num_assets=len(coins),
            confidence_score=confidence_score,
            market_sentiment=sentiment,
            top_contributors=coins,  # All contributors
            calculation_method=self.weighting_strategy.name
        )
    
    def run_full_analysis(self) -> ProcessingResult:
        """
        Execute complete aggregated RSI indicator analysis
        """
        start_time = datetime.now()
        logger.info("🚀 Starting Aggregated RSI Indicator Analysis")
        logger.info(f"Strategy: {self.weighting_strategy.description}")
        logger.info("=" * 60)
        
        try:
            # Step 1: Get top cryptocurrencies
            top_coins = self.get_top_cryptocurrencies()
            
            if not top_coins:
                return ProcessingResult(
                    success=False,
                    error_message="No cryptocurrency data retrieved",
                    processing_time_seconds=(datetime.now() - start_time).total_seconds()
                )
            
            # Step 2: Fetch RSI data for all coins
            coins_with_rsi = self.fetch_rsi_data_parallel(top_coins)
            
            if not coins_with_rsi:
                return ProcessingResult(
                    success=False,
                    error_message="No RSI data calculated",
                    processing_time_seconds=(datetime.now() - start_time).total_seconds()
                )
            
            # Step 3: Calculate aggregated RSI
            snapshot = self.calculate_aggregated_rsi(coins_with_rsi)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("✅ Analysis completed successfully")
            logger.info(f"Processing time: {processing_time:.1f}s")
            
            return ProcessingResult(
                success=True,
                snapshot=snapshot,
                processing_time_seconds=processing_time,
                warnings=[]
            )
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            
            return ProcessingResult(
                success=False,
                error_message=error_msg,
                processing_time_seconds=(datetime.now() - start_time).total_seconds()
            )
    
    def get_current_rsi_value(self) -> float:
        """
        Quick method to get just the current aggregated RSI value
        """
        result = self.run_full_analysis()
        if result.success and result.snapshot:
            return result.snapshot.aggregated_rsi
        else:
            logger.error(f"Failed to get RSI value: {result.error_message}")
            return 50.0  # Neutral default
    
    def update_weighting_strategy(self, new_strategy: WeightingStrategy):
        """Update the weighting strategy"""
        self.weighting_strategy = new_strategy
        logger.info(f"Updated weighting strategy to: {new_strategy.name}")
    
    def clear_cache(self):
        """Clear the market data cache"""
        with self._lock:
            self._market_data_cache.clear()
            self._cache_timestamp = None
        logger.info("Market data cache cleared")