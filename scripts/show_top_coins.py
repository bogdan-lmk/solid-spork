"""
Quick script to show top 20 cryptocurrencies for aggregated RSI indicator
Including ETH, excluding BTC and stablecoins
"""
import requests
import pandas as pd
from datetime import datetime

def get_top_cryptos_preview(top_n=20):
    """Get preview of top cryptocurrencies that would be selected"""
    
    # Stablecoins to exclude
    STABLECOINS = {
        'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD', 'FRAX',
        'PYUSD', 'FDUSD', 'LUSD', 'SUSD', 'GUSD', 'HUSD', 'USTC', 'UST',
        'TRIBE', 'FEI', 'USDX', 'DUSD', 'USDN', 'RSV', 'USDK', 'OUSD'
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
    
    # Only exclude Bitcoin
    EXCLUDED_COINS = {'BTC'}
    
    def is_likely_stablecoin(symbol, name):
        """Enhanced stablecoin detection"""
        stablecoin_patterns = ['USD', 'EUR', 'GBP', 'CNY', 'JPY', 'STABLE']
        if any(pattern in symbol for pattern in stablecoin_patterns):
            return True
            
        name_lower = name.lower()
        stablecoin_words = ['dollar', 'usd', 'stable', 'peg', 'reserve', 'backed']
        if any(word in name_lower for word in stablecoin_words):
            return True
            
        return False
    
    print("🔍 Fetching cryptocurrency market data...")
    
    # Use CoinGecko API
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 100,
        'page': 1,
        'sparkline': False,
        'price_change_percentage': '24h'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        market_data = response.json()
        
        selected_coins = []
        seen_symbols = set()
        
        print(f"\n📊 TOP {top_n} CRYPTOCURRENCIES FOR AGGREGATED RSI")
        print("=" * 80)
        print(f"{'Rank':<4} {'Symbol':<8} {'Name':<25} {'Market Cap':<12} {'24h Volume':<12} {'Price':<10}")
        print("-" * 80)
        
        for rank, coin in enumerate(market_data, 1):
            symbol = coin['symbol'].upper()
            name = coin['name']
            
            # Skip duplicates
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            
            # Apply exclusion rules
            if symbol in EXCLUDED_COINS:
                continue
            if symbol in STABLECOINS:
                continue
            if symbol in STAKED_ETH_DERIVATIVES:
                continue
            if symbol in WRAPPED_BITCOIN_DERIVATIVES:
                continue
            if symbol in EXCHANGE_UTILITY_TOKENS:
                continue
            if is_likely_stablecoin(symbol, name):
                continue
            
            # Get market data
            market_cap = coin.get('market_cap', 0)
            volume_24h = coin.get('total_volume', 0)
            price = coin.get('current_price', 0)
            
            if not all([market_cap, volume_24h, price]) or market_cap <= 0:
                continue
            
            # Format values
            market_cap_str = f"${market_cap/1e9:.1f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.0f}M"
            volume_str = f"${volume_24h/1e9:.1f}B" if volume_24h >= 1e9 else f"${volume_24h/1e6:.0f}M"
            price_str = f"${price:.2f}" if price >= 0.01 else f"${price:.6f}"
            
            # Highlight ETH
            highlight = "🟢" if symbol == "ETH" else "  "
            
            print(f"{highlight}{len(selected_coins)+1:<2} {symbol:<8} {name[:23]:<25} {market_cap_str:<12} {volume_str:<12} {price_str:<10}")
            
            selected_coins.append({
                'rank': len(selected_coins) + 1,
                'symbol': symbol,
                'name': name,
                'binance_symbol': f"{symbol}USDT",
                'market_cap_usd': market_cap,
                'volume_24h_usd': volume_24h,
                'price_usd': price,
                'market_cap_formatted': market_cap_str,
                'volume_formatted': volume_str
            })
            
            if len(selected_coins) >= top_n:
                break
        
        print("-" * 80)
        print(f"✅ Selected {len(selected_coins)} cryptocurrencies")
        
        # Summary statistics
        total_market_cap = sum(coin['market_cap_usd'] for coin in selected_coins)
        total_volume = sum(coin['volume_24h_usd'] for coin in selected_coins)
        
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   Total Market Cap: ${total_market_cap/1e12:.2f}T")
        print(f"   Total 24h Volume: ${total_volume/1e9:.1f}B")
        print(f"   Average Market Cap: ${total_market_cap/len(selected_coins)/1e9:.1f}B")
        
        # Show top contributors by market cap
        print(f"\n🏆 TOP 5 BY MARKET CAP WEIGHT:")
        for i, coin in enumerate(selected_coins[:5], 1):
            weight = coin['market_cap_usd'] / total_market_cap
            print(f"   {i}. {coin['symbol']:<8} {weight:6.1%} weight ({coin['market_cap_formatted']})")
        
        # Check if ETH is included
        eth_included = any(coin['symbol'] == 'ETH' for coin in selected_coins)
        if eth_included:
            eth_coin = next(coin for coin in selected_coins if coin['symbol'] == 'ETH')
            eth_weight = eth_coin['market_cap_usd'] / total_market_cap
            print(f"\n🟢 ETH INCLUSION: Rank #{eth_coin['rank']} with {eth_weight:.1%} weight")
        else:
            print(f"\n⚠️  ETH NOT FOUND in top {top_n}")
        
        # Show excluded major coins
        print(f"\n❌ EXCLUDED COINS:")
        excluded_found = []
        for coin in market_data[:20]:  # Check top 20 overall
            symbol = coin['symbol'].upper()
            if symbol in EXCLUDED_COINS:
                excluded_found.append(f"{symbol} (rank #{market_data.index(coin)+1})")
        
        if excluded_found:
            print(f"   {', '.join(excluded_found)}")
        else:
            print("   None in top 20")
        
        print(f"\n💡 NOTES:")
        print(f"   • Including ETH as it's essential for altcoin market analysis")
        print(f"   • Excluding Bitcoin to focus on altcoin movements")
        print(f"   • All stablecoins automatically excluded")
        print(f"   • Using Binance trading pairs (USDT)")
        
        return selected_coins
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return []

if __name__ == "__main__":
    print("🧠 Aggregated RSI Indicator - Top Cryptocurrencies Preview")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    coins = get_top_cryptos_preview(20)
    
    if coins:
        print(f"\n✅ These {len(coins)} cryptocurrencies will be used for the aggregated RSI calculation")
    else:
        print(f"\n❌ Failed to fetch cryptocurrency data")