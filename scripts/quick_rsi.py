#!/usr/bin/env python3
"""
Quick RSI check - simplified version
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import get_single_rsi

def get_current_rsi():
    """Get current simplified RSI"""
    rsi = get_single_rsi()
    
    if rsi is not None:
        print(f"RSI: {rsi:.2f}")
        
        if rsi >= 70:
            print("Status: Overbought")
        elif rsi <= 30:
            print("Status: Oversold")
        else:
            print("Status: Normal")
        
        return rsi
    else:
        print("❌ Failed to get RSI")
        return None

if __name__ == "__main__":
    get_current_rsi()