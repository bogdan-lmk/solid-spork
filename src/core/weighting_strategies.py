"""
Weighting strategies for aggregated RSI indicator
Different approaches to weight individual asset RSI values in the final calculation
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict
try:
    from .data_models import CoinMarketData
except ImportError:
    from aggregated_rsi_indicator.core.data_models import CoinMarketData

class WeightingStrategy(ABC):
    """Abstract base class for different weighting strategies"""
    
    @abstractmethod
    def calculate_weights(self, assets: List[CoinMarketData]) -> List[float]:
        """
        Calculate weights for each asset
        
        Args:
            assets: List of coin market data
            
        Returns:
            List of weights (should sum to 1.0)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name for identification"""
        pass
    
    @property
    def description(self) -> str:
        """Return strategy description"""
        return f"{self.name} weighting strategy"
    
    def validate_weights(self, weights: List[float]) -> List[float]:
        """Validate and normalize weights to ensure they sum to 1.0"""
        if not weights:
            raise ValueError("No weights calculated")
        
        weights = np.array(weights, dtype=float)
        
        # Check for negative weights
        if np.any(weights < 0):
            raise ValueError("Weights cannot be negative")
        
        # Normalize to sum to 1.0
        total_weight = np.sum(weights)
        if total_weight == 0:
            # Equal weights fallback
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights = weights / total_weight
        
        return weights.tolist()

class MarketCapWeightingStrategy(WeightingStrategy):
    """
    Market capitalization based weighting
    Assets with larger market caps get proportionally higher weights
    """
    
    def calculate_weights(self, assets: List[CoinMarketData]) -> List[float]:
        """Calculate weights based on market capitalization"""
        if not assets:
            return []
        
        market_caps = [asset.market_cap_usd for asset in assets]
        total_market_cap = sum(market_caps)
        
        if total_market_cap <= 0:
            # Fallback to equal weights
            return [1.0 / len(assets)] * len(assets)
        
        weights = [cap / total_market_cap for cap in market_caps]
        return self.validate_weights(weights)
    
    @property
    def name(self) -> str:
        return "market_cap_weighted"
    
    @property
    def description(self) -> str:
        return "Weights based on market capitalization - larger caps have more influence"

class VolumeWeightingStrategy(WeightingStrategy):
    """
    24h trading volume based weighting
    Assets with higher trading volumes get proportionally higher weights
    """
    
    def calculate_weights(self, assets: List[CoinMarketData]) -> List[float]:
        """Calculate weights based on 24h trading volume"""
        if not assets:
            return []
        
        volumes = [asset.volume_24h_usd for asset in assets]
        total_volume = sum(volumes)
        
        if total_volume <= 0:
            return [1.0 / len(assets)] * len(assets)
        
        weights = [vol / total_volume for vol in volumes]
        return self.validate_weights(weights)
    
    @property
    def name(self) -> str:
        return "volume_weighted"
    
    @property
    def description(self) -> str:
        return "Weights based on 24h trading volume - more liquid assets have more influence"




# Strategy registry for easy access
STRATEGY_REGISTRY = {
    'market_cap': MarketCapWeightingStrategy,
    'volume': VolumeWeightingStrategy
}

def get_strategy(strategy_name: str, **kwargs) -> WeightingStrategy:
    """
    Get a weighting strategy by name
    
    Args:
        strategy_name: Name of the strategy ('market_cap', 'volume')
        **kwargs: Additional arguments for strategy initialization
        
    Returns:
        WeightingStrategy instance
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = ', '.join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class()

def list_available_strategies() -> Dict[str, str]:
    """Get list of available strategies with descriptions"""
    strategies = {}
    for name, strategy_class in STRATEGY_REGISTRY.items():
        instance = strategy_class()
        strategies[name] = instance.description
    
    return strategies