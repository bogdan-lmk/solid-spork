#!/usr/bin/env python3
"""
Script to run simplified RSI Dashboard
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to project directory for relative paths
os.chdir(project_root)

# Import and run the Flask app
from web.app import app

if __name__ == '__main__':
    print("🚀 Starting RSI Dashboard...")
    print("🌐 Dashboard available at: http://localhost:5000")
    print("📊 API endpoint: http://localhost:5000/api/rsi")
    print("=" * 50)
    
    # Start the Flask development server
    app.run(host='0.0.0.0', port=5000, debug=True)