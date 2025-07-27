"""
Main entry point for RSI Aggregator
Simple RSI indicator combining market cap and volume weighting
"""
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from src.core.aggregator import MarketCapRSIAggregator
from src.core.weighting_strategies import (
    MarketCapWeightingStrategy,
    VolumeWeightingStrategy
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/aggregated_rsi.log')
    ]
)
logger = logging.getLogger(__name__)

def get_single_rsi():
    """Get single RSI value combining market cap and volume weighting"""
    try:
        # Run market cap weighted analysis
        mc_aggregator = MarketCapRSIAggregator(weighting_strategy=MarketCapWeightingStrategy())
        mc_result = mc_aggregator.run_full_analysis()
        
        # Run volume weighted analysis  
        vol_aggregator = MarketCapRSIAggregator(weighting_strategy=VolumeWeightingStrategy())
        vol_result = vol_aggregator.run_full_analysis()
        
        if not mc_result.success or not vol_result.success:
            return None
        
        mc_rsi = mc_result.snapshot.aggregated_rsi
        vol_rsi = vol_result.snapshot.aggregated_rsi
        
        # Calculate single RSI value (average of market cap and volume weighted)
        return (mc_rsi + vol_rsi) / 2
        
    except Exception as e:
        logger.error(f"Error calculating single RSI: {e}")
        return None

def run_analysis():
    """Run simplified RSI analysis"""
    print("📊 RSI Aggregator")
    print("=" * 30)
    
    try:
        rsi = get_single_rsi()
        
        if rsi is None:
            print("❌ Analysis failed")
            return None
        
        print(f"RSI: {rsi:.2f}")
        
        # Simple sentiment
        if rsi >= 70:
            print("Status: Overbought")
        elif rsi <= 30:
            print("Status: Oversold") 
        else:
            print("Status: Normal")
        
        return rsi
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        print(f"❌ Error: {e}")
        return None

def main():
    """Main execution function"""
    try:
        rsi = run_analysis()
        if rsi is None:
            print("Failed to calculate RSI")
    except KeyboardInterrupt:
        print("\n👋 Analysis interrupted")
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()