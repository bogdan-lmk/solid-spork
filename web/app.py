#!/usr/bin/env python3
"""
Simplified Flask Web Application for RSI Dashboard
"""
import sys
import os
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.aggregator import MarketCapRSIAggregator
from src.core.weighting_strategies import (
    MarketCapWeightingStrategy,
    VolumeWeightingStrategy
)
from main import get_single_rsi
import numpy as np

app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')

def get_simple_rsi_data():
    """Get simplified RSI data for dashboard"""
    try:
        print("🔄 Getting RSI data...")
        
        # Get single RSI value
        rsi = get_single_rsi()
        
        if rsi is None:
            return None, "Failed to calculate RSI"
        
        # Determine sentiment
        if rsi >= 70:
            sentiment = "OVERBOUGHT"
        elif rsi <= 30:
            sentiment = "OVERSOLD"
        else:
            sentiment = "NORMAL"
        
        # Get basic market data for context
        aggregator = MarketCapRSIAggregator()
        result = aggregator.run_full_analysis()
        
        contributors = []
        num_assets = 0
        market_cap = "0T"
        
        if result.success:
            snapshot = result.snapshot
            num_assets = snapshot.num_assets
            market_cap = f"{snapshot.total_market_cap/1e12:.2f}T"
            
            # Get all 20 contributors for display
            for coin in snapshot.top_contributors:
                contributors.append({
                    "symbol": coin.symbol,
                    "rsi": round(coin.rsi_14, 1),
                    "weight": round(coin.weight * 100, 1),
                    "market_cap": coin.market_cap_usd,
                    "price": round(coin.price_usd, 4),
                    "change": round(coin.price_change_24h, 2) if hasattr(coin, 'price_change_24h') else 0.0
                })
        
        dashboard_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "rsi": round(rsi, 2),
            "sentiment": sentiment,
            "confidence": 95.0,  # Static for simplified version
            "num_assets": num_assets,
            "market_cap": market_cap,
            "processing_time": 1.0,  # Static for simplified version
            "contributors": contributors
        }
        
        print(f"✅ RSI calculated: {rsi:.2f} ({sentiment})")
        return dashboard_data, None
        
    except Exception as e:
        error_msg = f"Error getting RSI data: {str(e)}"
        print(f"❌ {error_msg}")
        return None, error_msg

@app.route('/')
def dashboard():
    """RSI Chart dashboard route"""
    return render_template('rsi_chart.html')

@app.route('/simple')
def simple_dashboard():
    """Simple dashboard route"""
    data, error = get_simple_rsi_data()
    
    if error:
        return render_template('simple_dashboard.html', 
                             error=error,
                             rsi=0,
                             sentiment="ERROR",
                             contributors=[])
    
    return render_template('simple_dashboard.html', **data)

@app.route('/api/rsi')
def api_rsi():
    """API endpoint for RSI data"""
    data, error = get_simple_rsi_data()
    
    if error:
        return jsonify({"error": error}), 500
    
    return jsonify(data)

@app.route('/api/rsi/historical')
def api_rsi_historical():
    """API endpoint for historical RSI data"""
    try:
        # Get current RSI
        current_rsi = get_single_rsi()
        
        if current_rsi is None:
            return jsonify({"error": "Failed to get current RSI"}), 500
        
        # Get real historical RSI data - ONLY real data
        historical_data = get_real_historical_rsi_data()
        
        if historical_data is None:
            return jsonify({
                "error": "Failed to get real historical RSI data"
            }), 500
        
        print(f"✅ Using real historical data with {len(historical_data)} points")
        
        return jsonify({
            "current_rsi": current_rsi,
            "historical_data": historical_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_real_historical_rsi_data():
    """Get real historical RSI data for top 20 altcoins"""
    from datetime import datetime, timedelta
    import sys
    from pathlib import Path
    
    # Add paths for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from src.fetchers.improved_klines_fetcher import get_historical_klines_robust
        from src.indicators.indicator import calculate_wilder_rsi
        from src.core.aggregator import MarketCapRSIAggregator
        from src.core.weighting_strategies import MarketCapWeightingStrategy, VolumeWeightingStrategy
        
        print("🔄 Getting real historical RSI data...")
        
        # Get aggregator to get current top coins
        aggregator = MarketCapRSIAggregator()
        result = aggregator.run_full_analysis()
        
        if not result.success:
            print(f"❌ Failed to get coin list: {result.error_message}")
            return None
        
        top_coins = result.snapshot.top_contributors
        print(f"📊 Got {len(top_coins)} coins for historical analysis")
        
        # Get historical data for 90 days
        days = 90
        historical_dates = []
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i-1)
            historical_dates.append({
                "date": int(date.timestamp() * 1000),
                "datetime": date
            })
        
        # Get historical RSI for each coin
        coin_historical_rsi = {}
        
        for coin in top_coins:
            try:
                symbol = coin.symbol + "USDT"
                print(f"📈 Getting data for {symbol}...")
                
                # Get historical price data
                df = get_historical_klines_robust(symbol, interval='1d', limit=days+14)  # +14 for RSI calculation
                
                if df.empty or len(df) < 20 or 'close' not in df.columns:
                    print(f"⚠️ Insufficient data for {symbol}")
                    continue
                
                # Calculate RSI
                rsi_values = calculate_wilder_rsi(df['close'], period=14)
                
                # Take last 'days' values
                recent_rsi = rsi_values.tail(days).values
                
                if len(recent_rsi) == days:
                    coin_historical_rsi[coin.symbol] = recent_rsi
                    print(f"✅ Got RSI data for {symbol}")
                else:
                    print(f"⚠️ Incomplete RSI data for {symbol}")
                    
            except Exception as e:
                print(f"❌ Error getting data for {coin.symbol}: {e}")
                continue
        
        print(f"📊 Successfully got historical data for {len(coin_historical_rsi)} coins")
        
        # Calculate aggregated RSI for each day
        aggregated_historical_rsi = []
        
        for day_idx in range(days):
            # Get RSI values for this day
            day_rsi_data = []
            day_weights_mc = []
            day_weights_vol = []
            
            for coin in top_coins:
                if coin.symbol in coin_historical_rsi:
                    rsi_val = coin_historical_rsi[coin.symbol][day_idx]
                    if not np.isnan(rsi_val):
                        day_rsi_data.append(rsi_val)
                        day_weights_mc.append(coin.market_cap_usd)
                        day_weights_vol.append(coin.volume_24h_usd)
            
            if day_rsi_data:
                # Calculate weighted RSI (market cap + volume average)
                total_mc = sum(day_weights_mc)
                total_vol = sum(day_weights_vol)
                
                if total_mc > 0 and total_vol > 0:
                    # Market cap weighted RSI
                    mc_weights = [w/total_mc for w in day_weights_mc]
                    mc_weighted_rsi = sum(rsi * weight for rsi, weight in zip(day_rsi_data, mc_weights))
                    
                    # Volume weighted RSI  
                    vol_weights = [w/total_vol for w in day_weights_vol]
                    vol_weighted_rsi = sum(rsi * weight for rsi, weight in zip(day_rsi_data, vol_weights))
                    
                    # Final aggregated RSI (average of both)
                    final_rsi = (mc_weighted_rsi + vol_weighted_rsi) / 2
                    
                    aggregated_historical_rsi.append({
                        "date": historical_dates[day_idx]["date"],
                        "rsi": round(final_rsi, 2)
                    })
                else:
                    # Fallback to simple average
                    avg_rsi = sum(day_rsi_data) / len(day_rsi_data)
                    aggregated_historical_rsi.append({
                        "date": historical_dates[day_idx]["date"],
                        "rsi": round(avg_rsi, 2)
                    })
        
        print(f"✅ Generated {len(aggregated_historical_rsi)} days of real aggregated RSI")
        return aggregated_historical_rsi
        
    except Exception as e:
        print(f"❌ Error in real historical RSI calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print("🚀 Starting RSI Dashboard...")
    print("🌐 Dashboard: http://localhost:5000")
    print("📊 API: http://localhost:5000/api/rsi")
    
    app.run(host='0.0.0.0', port=5000, debug=True)