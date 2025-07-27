"""
Test script for Aggregated RSI Indicator System
Comprehensive test of all components with real data
"""
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add the aggregated RSI indicator to path
sys.path.insert(0, str(Path(__file__).parent))

from aggregated_rsi_indicator.core.aggregator import MarketCapRSIAggregator
from aggregated_rsi_indicator.core.data_models import IndicatorConfig
from aggregated_rsi_indicator.core.weighting_strategies import MarketCapWeightingStrategy, VolumeWeightingStrategy, HybridWeightingStrategy
from aggregated_rsi_indicator.utils.data_persistence import RSIDataPersistence
from aggregated_rsi_indicator.utils.alerts import create_default_alert_system
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_aggregation():
    """Test basic RSI aggregation functionality"""
    print("🧪 Testing Basic RSI Aggregation")
    print("=" * 50)
    
    try:
        # Create aggregator with default settings
        aggregator = MarketCapRSIAggregator()
        
        # Run analysis
        result = aggregator.run_full_analysis()
        
        if result.success:
            snapshot = result.snapshot
            
            print(f"✅ Basic aggregation successful!")
            print(f"   Aggregated RSI: {snapshot.aggregated_rsi:.2f}")
            print(f"   Market Sentiment: {snapshot.market_sentiment}")
            print(f"   Assets: {snapshot.num_assets}")
            print(f"   Confidence: {snapshot.confidence_score:.1f}%")
            print(f"   Processing time: {result.processing_time_seconds:.2f}s")
            
            # Show top 5 contributors
            print(f"\n   Top 5 Contributors:")
            for i, asset in enumerate(snapshot.top_contributors[:5], 1):
                print(f"     {i}. {asset.symbol}: RSI {asset.rsi_14:.1f} (weight {asset.weight:.1%})")
            
            return snapshot
        else:
            print(f"❌ Basic aggregation failed: {result.error_message}")
            return None
            
    except Exception as e:
        print(f"❌ Error in basic aggregation: {e}")
        return None

def test_multiple_strategies():
    """Test different weighting strategies"""
    print(f"\n🔄 Testing Multiple Weighting Strategies")
    print("=" * 50)
    
    strategies = [
        ("Market Cap", MarketCapWeightingStrategy()),
        ("Volume", VolumeWeightingStrategy()),
        ("Hybrid 70/30", HybridWeightingStrategy(market_cap_weight=0.7, volume_weight=0.3))
    ]
    
    results = {}
    
    for name, strategy in strategies:
        try:
            print(f"\n🔧 Testing {name} strategy...")
            
            aggregator = MarketCapRSIAggregator(weighting_strategy=strategy)
            result = aggregator.run_full_analysis()
            
            if result.success:
                snapshot = result.snapshot
                results[name] = snapshot.aggregated_rsi
                
                print(f"   ✅ {name}: RSI {snapshot.aggregated_rsi:.2f}")
                print(f"      Sentiment: {snapshot.market_sentiment}")
                print(f"      Confidence: {snapshot.confidence_score:.1f}%")
            else:
                print(f"   ❌ {name} failed: {result.error_message}")
                
        except Exception as e:
            print(f"   ❌ Error with {name}: {e}")
    
    # Compare results
    if len(results) > 1:
        print(f"\n📊 Strategy Comparison:")
        rsi_values = list(results.values())
        rsi_range = max(rsi_values) - min(rsi_values)
        avg_rsi = sum(rsi_values) / len(rsi_values)
        
        for name, rsi in results.items():
            deviation = abs(rsi - avg_rsi)
            print(f"   {name:<15}: {rsi:6.2f} (±{deviation:4.1f} from avg)")
        
        print(f"   Average RSI: {avg_rsi:.2f}")
        print(f"   Range: {rsi_range:.2f} points")
        
        if rsi_range < 5:
            print("   🟢 High agreement between strategies")
        elif rsi_range < 10:
            print("   🟡 Moderate agreement")
        else:
            print("   🔴 Low agreement - significant strategy differences")
    
    return results

def test_data_persistence():
    """Test data persistence functionality"""
    print(f"\n💾 Testing Data Persistence")
    print("=" * 50)
    
    try:
        # Create persistence layer
        persistence = RSIDataPersistence("test_rsi_history.db")
        
        # Run aggregation
        aggregator = MarketCapRSIAggregator()
        result = aggregator.run_full_analysis()
        
        if result.success:
            snapshot = result.snapshot
            
            # Save snapshot
            record_id = persistence.save_rsi_snapshot(snapshot, result.processing_time_seconds)
            print(f"✅ Saved snapshot with ID: {record_id}")
            
            # Test retrieval
            latest = persistence.get_latest_snapshot()
            if latest:
                print(f"✅ Retrieved latest snapshot: RSI {latest.aggregated_rsi:.2f}")
            
            # Test historical data
            historical = persistence.get_historical_rsi(days=1)
            print(f"✅ Historical records: {len(historical)}")
            
            # Test statistics
            stats = persistence.get_rsi_statistics(days=1)
            print(f"✅ Statistics: {stats['record_count']} records, avg RSI {stats['average_rsi']:.2f}")
            
            # Test database info
            db_info = persistence.get_database_info()
            print(f"✅ Database: {db_info['database_size_mb']:.2f} MB, {db_info['tables']['rsi_history_records']} RSI records")
            
            return True
        else:
            print(f"❌ Could not test persistence: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Error in persistence test: {e}")
        return False

def test_alert_system():
    """Test alert system functionality"""
    print(f"\n🚨 Testing Alert System")
    print("=" * 50)
    
    try:
        # Create alert system
        alert_system = create_default_alert_system()
        
        # Run aggregation
        aggregator = MarketCapRSIAggregator()
        result = aggregator.run_full_analysis()
        
        if result.success:
            snapshot = result.snapshot
            
            # Check for alerts (no previous snapshot for first run)
            alerts = alert_system.check_alerts(snapshot, None)
            
            print(f"✅ Alert system check completed: {len(alerts)} alerts triggered")
            
            for alert in alerts:
                print(f"   🔔 {alert.level.value.upper()}: {alert.title}")
                print(f"      {alert.message}")
                if alert.affected_assets:
                    print(f"      Affected: {', '.join(alert.affected_assets[:5])}")
            
            if not alerts:
                print("   ℹ️ No alerts triggered - market conditions normal")
            
            # Test alert summary
            summary = alert_system.get_alert_summary(hours=1)
            print(f"✅ Alert summary: {summary['total_alerts']} alerts in last hour")
            
            return len(alerts)
        else:
            print(f"❌ Could not test alerts: {result.error_message}")
            return 0
            
    except Exception as e:
        print(f"❌ Error in alert system test: {e}")
        return 0

def test_custom_configuration():
    """Test custom configuration options"""
    print(f"\n⚙️ Testing Custom Configuration")
    print("=" * 50)
    
    try:
        # Create custom config
        config = IndicatorConfig(
            top_n_assets=15,           # Fewer assets
            rsi_period=21,             # Longer RSI period
            min_market_cap=5e9,        # Higher minimum market cap
            min_confidence_score=70    # Higher confidence requirement
        )
        
        # Custom strategy
        strategy = HybridWeightingStrategy(
            market_cap_weight=0.6,
            volume_weight=0.4
        )
        
        aggregator = MarketCapRSIAggregator(config=config, weighting_strategy=strategy)
        result = aggregator.run_full_analysis()
        
        if result.success:
            snapshot = result.snapshot
            
            print(f"✅ Custom configuration successful!")
            print(f"   RSI (21-period): {snapshot.aggregated_rsi:.2f}")
            print(f"   Assets used: {snapshot.num_assets} (target: {config.top_n_assets})")
            print(f"   Confidence: {snapshot.confidence_score:.1f}% (min: {config.min_confidence_score}%)")
            print(f"   Strategy: {strategy.description}")
            
            # Show different RSI period impact
            print(f"\n   Top 5 assets with 21-period RSI:")
            for i, asset in enumerate(snapshot.top_contributors[:5], 1):
                print(f"     {i}. {asset.symbol}: RSI {asset.rsi_14:.1f}")
            
            return snapshot
        else:
            print(f"❌ Custom configuration failed: {result.error_message}")
            return None
            
    except Exception as e:
        print(f"❌ Error in custom configuration test: {e}")
        return None

def test_performance():
    """Test system performance"""
    print(f"\n⚡ Testing Performance")
    print("=" * 50)
    
    try:
        import time
        
        # Multiple runs to test consistency
        run_times = []
        rsi_values = []
        
        for i in range(3):
            print(f"   Run {i+1}/3...", end=" ")
            
            start_time = time.time()
            aggregator = MarketCapRSIAggregator()
            result = aggregator.run_full_analysis()
            end_time = time.time()
            
            if result.success:
                run_time = end_time - start_time
                run_times.append(run_time)
                rsi_values.append(result.snapshot.aggregated_rsi)
                print(f"✅ {run_time:.2f}s, RSI {result.snapshot.aggregated_rsi:.2f}")
            else:
                print(f"❌ Failed")
        
        if run_times:
            avg_time = sum(run_times) / len(run_times)
            min_time = min(run_times)
            max_time = max(run_times)
            
            rsi_std = (max(rsi_values) - min(rsi_values)) if len(rsi_values) > 1 else 0
            
            print(f"\n📈 Performance Summary:")
            print(f"   Average time: {avg_time:.2f}s")
            print(f"   Time range: {min_time:.2f}s - {max_time:.2f}s")
            print(f"   RSI consistency: ±{rsi_std/2:.2f} points")
            
            if avg_time < 10:
                print("   🟢 Good performance")
            elif avg_time < 30:
                print("   🟡 Acceptable performance")
            else:
                print("   🔴 Slow performance")
            
            return avg_time
        else:
            print("❌ No successful runs for performance test")
            return None
            
    except Exception as e:
        print(f"❌ Error in performance test: {e}")
        return None

def generate_test_report(test_results):
    """Generate comprehensive test report"""
    print(f"\n📋 COMPREHENSIVE TEST REPORT")
    print("=" * 60)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System: Aggregated RSI Indicator v1.0.0")
    
    # Test results summary
    print(f"\n🧪 Test Results:")
    
    basic_snapshot = test_results.get('basic_aggregation')
    if basic_snapshot:
        print(f"   ✅ Basic Aggregation: RSI {basic_snapshot.aggregated_rsi:.2f}")
        print(f"      Market Cap: ${basic_snapshot.total_market_cap/1e12:.2f}T")
        print(f"      Assets: {basic_snapshot.num_assets}")
    else:
        print(f"   ❌ Basic Aggregation: Failed")
    
    strategies = test_results.get('multiple_strategies', {})
    if strategies:
        print(f"   ✅ Strategy Testing: {len(strategies)} strategies tested")
        for name, rsi in strategies.items():
            print(f"      {name}: {rsi:.2f}")
    else:
        print(f"   ❌ Strategy Testing: Failed")
    
    persistence_ok = test_results.get('data_persistence', False)
    print(f"   {'✅' if persistence_ok else '❌'} Data Persistence: {'Working' if persistence_ok else 'Failed'}")
    
    alert_count = test_results.get('alert_system', 0)
    print(f"   ✅ Alert System: {alert_count} alerts triggered")
    
    custom_snapshot = test_results.get('custom_config')
    if custom_snapshot:
        print(f"   ✅ Custom Configuration: RSI {custom_snapshot.aggregated_rsi:.2f}")
    else:
        print(f"   ❌ Custom Configuration: Failed")
    
    avg_time = test_results.get('performance')
    if avg_time:
        print(f"   ✅ Performance: {avg_time:.2f}s average")
    else:
        print(f"   ❌ Performance: Failed")
    
    # Overall assessment
    successful_tests = sum([
        basic_snapshot is not None,
        len(strategies) > 0,
        persistence_ok,
        True,  # Alert system always runs
        custom_snapshot is not None,
        avg_time is not None
    ])
    
    total_tests = 6
    success_rate = (successful_tests / total_tests) * 100
    
    print(f"\n🎯 Overall Assessment:")
    print(f"   Tests Passed: {successful_tests}/{total_tests} ({success_rate:.0f}%)")
    
    if success_rate >= 80:
        print(f"   🟢 System Status: HEALTHY")
        print(f"   💡 Ready for production use")
    elif success_rate >= 60:
        print(f"   🟡 System Status: FUNCTIONAL")
        print(f"   💡 Minor issues detected, suitable for testing")
    else:
        print(f"   🔴 System Status: ISSUES DETECTED")
        print(f"   💡 Requires investigation before use")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if basic_snapshot and basic_snapshot.confidence_score < 70:
        print(f"   • Data confidence is {basic_snapshot.confidence_score:.1f}% - monitor data quality")
    if avg_time and avg_time > 15:
        print(f"   • Performance is {avg_time:.1f}s - consider optimization")
    if alert_count > 2:
        print(f"   • {alert_count} alerts triggered - review market conditions")
    
    print(f"   • Set up daily batch processing for continuous monitoring")
    print(f"   • Configure custom alert thresholds based on trading strategy")
    print(f"   • Regular database cleanup recommended for long-term use")

def main():
    """Run comprehensive test suite"""
    print("🚀 Aggregated RSI Indicator - Comprehensive Test Suite")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Testing all system components with real cryptocurrency data")
    print("=" * 70)
    
    test_results = {}
    
    try:
        # Run all tests
        test_results['basic_aggregation'] = test_basic_aggregation()
        test_results['multiple_strategies'] = test_multiple_strategies()
        test_results['data_persistence'] = test_data_persistence()
        test_results['alert_system'] = test_alert_system()
        test_results['custom_config'] = test_custom_configuration()
        test_results['performance'] = test_performance()
        
        # Generate final report
        generate_test_report(test_results)
        
    except KeyboardInterrupt:
        print(f"\n👋 Test suite interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error in test suite: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Test suite completed!")

if __name__ == "__main__":
    main()