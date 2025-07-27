"""
Data models for Aggregated RSI Indicator
Clean data structures for the extended RSI indicator system
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class CoinMarketData:
    """Market data structure for individual cryptocurrency in the indicator"""
    symbol: str                    # Base symbol (e.g., 'BTC')
    binance_symbol: str           # Trading pair format (e.g., 'BTCUSDT')
    market_cap_usd: float         # Market capitalization in USD
    volume_24h_usd: float         # 24h trading volume in USD
    price_usd: float              # Current price in USD
    market_cap_rank: int          # Market cap ranking
    rsi_14: float = 0.0          # 14-period RSI value
    weight: float = 0.0          # Weight in aggregated calculation
    data_quality: str = "unknown" # Data quality: good, fair, poor
    price_change_24h: float = 0.0 # 24h price change percentage
    # OHLC data for candlestick charts
    price_open_24h: float = 0.0   # Opening price 24h ago
    price_high_24h: float = 0.0   # Highest price in 24h
    price_low_24h: float = 0.0    # Lowest price in 24h

    def __post_init__(self):
        """Validate data after initialization"""
        if self.market_cap_usd < 0:
            raise ValueError(f"Market cap cannot be negative: {self.market_cap_usd}")
        if not (0 <= self.rsi_14 <= 100):
            if self.rsi_14 != 0.0:  # Allow 0.0 as default unset value
                raise ValueError(f"RSI must be between 0-100: {self.rsi_14}")
        if not (0 <= self.weight <= 1.0):
            if self.weight != 0.0:  # Allow 0.0 as default unset value
                raise ValueError(f"Weight must be between 0-1: {self.weight}")

@dataclass
class AggregatedRSISnapshot:
    """Complete snapshot of aggregated RSI indicator calculation"""
    timestamp: datetime           # When the calculation was performed
    aggregated_rsi: float        # Final aggregated RSI value (0-100)
    total_market_cap: float      # Combined market cap of all assets
    num_assets: int              # Number of assets included
    confidence_score: float      # Data quality confidence (0-100)
    market_sentiment: str        # Interpreted sentiment: overbought, oversold, neutral, trending
    top_contributors: List[CoinMarketData]  # Assets with highest weights
    
    # Extended indicator metadata
    market_cap_weighted_rsi: float = None  # Same as aggregated_rsi for market cap weighting
    calculation_method: str = "market_cap_weighted"  # Method used for aggregation
    
    def __post_init__(self):
        """Validate snapshot data"""
        if not (0 <= self.aggregated_rsi <= 100):
            raise ValueError(f"Aggregated RSI must be between 0-100: {self.aggregated_rsi}")
        if not (0 <= self.confidence_score <= 100):
            raise ValueError(f"Confidence score must be between 0-100: {self.confidence_score}")
        if self.num_assets != len(self.top_contributors):
            # Note: top_contributors might be subset, so this is just a warning
            pass
        
        # Set default market_cap_weighted_rsi if not provided
        if self.market_cap_weighted_rsi is None:
            self.market_cap_weighted_rsi = self.aggregated_rsi
    
    @property
    def is_overbought(self) -> bool:
        """Check if indicator shows overbought condition (RSI >= 70)"""
        return self.aggregated_rsi >= 70.0
    
    @property 
    def is_oversold(self) -> bool:
        """Check if indicator shows oversold condition (RSI <= 30)"""
        return self.aggregated_rsi <= 30.0
    
    @property
    def is_neutral(self) -> bool:
        """Check if indicator is in neutral zone (45 <= RSI <= 55)"""
        return 45.0 <= self.aggregated_rsi <= 55.0
    
    def get_market_interpretation(self) -> str:
        """Get detailed market interpretation based on RSI value"""
        rsi = self.aggregated_rsi
        
        if rsi >= 80:
            return "Extremely Overbought - Strong reversal signal"
        elif rsi >= 70:
            return "Overbought - Consider taking profits"
        elif rsi >= 60:
            return "Bullish momentum - Monitor for continuation"
        elif rsi >= 55:
            return "Slight bullish bias"
        elif rsi >= 45:
            return "Neutral - Range-bound movement expected"
        elif rsi >= 40:
            return "Slight bearish bias"
        elif rsi >= 30:
            return "Bearish pressure - Monitor for support"
        elif rsi >= 20:
            return "Oversold - Potential bounce expected"
        else:
            return "Extremely Oversold - Strong reversal opportunity"
    
    def to_dict(self) -> dict:
        """Convert snapshot to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'aggregated_rsi': self.aggregated_rsi,
            'total_market_cap': self.total_market_cap,
            'num_assets': self.num_assets,
            'confidence_score': self.confidence_score,
            'market_sentiment': self.market_sentiment,
            'calculation_method': self.calculation_method,
            'market_interpretation': self.get_market_interpretation(),
            'is_overbought': self.is_overbought,
            'is_oversold': self.is_oversold,
            'is_neutral': self.is_neutral,
            'top_contributors': [
                {
                    'symbol': asset.symbol,
                    'rsi_14': asset.rsi_14,
                    'weight': asset.weight,
                    'market_cap_usd': asset.market_cap_usd,
                    'data_quality': asset.data_quality,
                    'price_usd': asset.price_usd,
                    'price_change_24h': asset.price_change_24h,
                    'price_open_24h': asset.price_open_24h,
                    'price_high_24h': asset.price_high_24h,
                    'price_low_24h': asset.price_low_24h
                }
                for asset in self.top_contributors[:10]  # Limit to top 10
            ]
        }

@dataclass
class IndicatorConfig:
    """Configuration for aggregated RSI indicator calculations"""
    top_n_assets: int = 20                    # Number of top assets to include
    rsi_period: int = 14                      # RSI calculation period
    min_market_cap: float = 1e9               # Minimum market cap (1B USD)
    min_volume_24h: float = 10e6              # Minimum 24h volume (10M USD)
    cache_duration_minutes: int = 30          # Data cache duration
    include_stablecoins: bool = False         # Whether to include stablecoins
    include_major_coins: bool = False         # Whether to include BTC/ETH
    
    # Data quality thresholds
    min_confidence_score: float = 50.0        # Minimum acceptable confidence
    max_data_age_hours: int = 25              # Maximum age of price data
    
    def __post_init__(self):
        """Validate configuration"""
        if self.top_n_assets <= 0:
            raise ValueError("top_n_assets must be positive")
        if not (1 <= self.rsi_period <= 50):
            raise ValueError("rsi_period must be between 1-50")
        if self.min_market_cap < 0:
            raise ValueError("min_market_cap cannot be negative")

@dataclass 
class ProcessingResult:
    """Result of indicator processing operation"""
    success: bool
    snapshot: Optional[AggregatedRSISnapshot] = None
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []