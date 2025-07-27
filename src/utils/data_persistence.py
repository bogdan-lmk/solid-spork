"""
Data persistence utilities for Aggregated RSI Indicator
SQLite-based storage for historical RSI data and system logs
"""
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

try:
    from ..core.data_models import AggregatedRSISnapshot, CoinMarketData
except ImportError:
    from aggregated_rsi_indicator.core.data_models import AggregatedRSISnapshot, CoinMarketData

logger = logging.getLogger(__name__)

class RSIDataPersistence:
    """
    Handles SQLite-based persistence for aggregated RSI historical data
    """
    
    def __init__(self, db_path: str = "aggregated_rsi_history.db"):
        self.db_path = Path(db_path)
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main aggregated RSI history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS aggregated_rsi_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    aggregated_rsi REAL NOT NULL,
                    total_market_cap REAL NOT NULL,
                    num_assets INTEGER NOT NULL,
                    confidence_score REAL NOT NULL,
                    market_sentiment TEXT NOT NULL,
                    calculation_method TEXT NOT NULL,
                    processing_duration_seconds REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date) ON CONFLICT REPLACE
                )
            ''')
            
            # Individual asset contributions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS asset_contributions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    aggregated_rsi_id INTEGER,
                    symbol TEXT NOT NULL,
                    binance_symbol TEXT NOT NULL,
                    rsi_14 REAL NOT NULL,
                    market_cap_usd REAL NOT NULL,
                    volume_24h_usd REAL NOT NULL,
                    weight REAL NOT NULL,
                    market_cap_rank INTEGER,
                    data_quality TEXT,
                    price_usd REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (aggregated_rsi_id) REFERENCES aggregated_rsi_history (id)
                )
            ''')
            
            # System processing logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    error_details TEXT,
                    processing_time_seconds REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # RSI analysis summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rsi_analysis_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    avg_rsi REAL NOT NULL,
                    max_rsi REAL NOT NULL,
                    min_rsi REAL NOT NULL,
                    overbought_count INTEGER NOT NULL,
                    oversold_count INTEGER NOT NULL,
                    neutral_count INTEGER NOT NULL,
                    market_cap_dominance TEXT,
                    top_performer TEXT,
                    worst_performer TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rsi_date ON aggregated_rsi_history (date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rsi_timestamp ON aggregated_rsi_history (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_contributions_symbol ON asset_contributions (symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_contributions_date ON asset_contributions (aggregated_rsi_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON processing_logs (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_status ON processing_logs (status)')
            
            conn.commit()
            logger.info(f"Database initialized: {self.db_path}")
    
    def save_rsi_snapshot(self, snapshot: AggregatedRSISnapshot, 
                         processing_duration: float = 0.0) -> int:
        """Save complete RSI snapshot to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                # Insert main RSI record
                cursor.execute('''
                    INSERT OR REPLACE INTO aggregated_rsi_history 
                    (timestamp, date, aggregated_rsi, total_market_cap, num_assets, 
                     confidence_score, market_sentiment, calculation_method, processing_duration_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.timestamp.isoformat(),
                    snapshot.timestamp.date().isoformat(),
                    snapshot.aggregated_rsi,
                    snapshot.total_market_cap,
                    snapshot.num_assets,
                    snapshot.confidence_score,
                    snapshot.market_sentiment,
                    snapshot.calculation_method,
                    processing_duration
                ))
                
                main_record_id = cursor.lastrowid
                
                # Delete existing asset contributions for this date (in case of replacement)
                cursor.execute('''
                    DELETE FROM asset_contributions 
                    WHERE aggregated_rsi_id = ?
                ''', (main_record_id,))
                
                # Insert asset contributions
                for asset in snapshot.top_contributors:
                    cursor.execute('''
                        INSERT INTO asset_contributions 
                        (aggregated_rsi_id, symbol, binance_symbol, rsi_14, market_cap_usd, 
                         volume_24h_usd, weight, market_cap_rank, data_quality, price_usd)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        main_record_id,
                        asset.symbol,
                        asset.binance_symbol,
                        asset.rsi_14,
                        asset.market_cap_usd,
                        asset.volume_24h_usd,
                        asset.weight,
                        asset.market_cap_rank,
                        asset.data_quality,
                        asset.price_usd
                    ))
                
                # Generate analysis summary
                self._update_analysis_summary(cursor, snapshot)
                
                conn.commit()
                logger.info(f"Saved RSI snapshot with ID {main_record_id}")
                return main_record_id
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error saving snapshot: {e}")
                raise
    
    def _update_analysis_summary(self, cursor, snapshot: AggregatedRSISnapshot):
        """Update daily analysis summary"""
        date_str = snapshot.timestamp.date().isoformat()
        
        # Calculate summary statistics
        rsi_values = [asset.rsi_14 for asset in snapshot.top_contributors]
        overbought_count = sum(1 for rsi in rsi_values if rsi >= 70)
        oversold_count = sum(1 for rsi in rsi_values if rsi <= 30)
        neutral_count = sum(1 for rsi in rsi_values if 45 <= rsi <= 55)
        
        # Find top and worst performers
        sorted_assets = sorted(snapshot.top_contributors, key=lambda x: x.rsi_14, reverse=True)
        top_performer = f"{sorted_assets[0].symbol}({sorted_assets[0].rsi_14:.1f})"
        worst_performer = f"{sorted_assets[-1].symbol}({sorted_assets[-1].rsi_14:.1f})"
        
        # Market cap dominance (top 3 combined weight)
        top_3_weight = sum(asset.weight for asset in snapshot.top_contributors[:3])
        dominance_desc = f"Top3: {top_3_weight:.1%}"
        
        cursor.execute('''
            INSERT OR REPLACE INTO rsi_analysis_summary
            (date, avg_rsi, max_rsi, min_rsi, overbought_count, oversold_count, 
             neutral_count, market_cap_dominance, top_performer, worst_performer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            date_str,
            snapshot.aggregated_rsi,
            max(rsi_values),
            min(rsi_values),
            overbought_count,
            oversold_count,
            neutral_count,
            dominance_desc,
            top_performer,
            worst_performer
        ))
    
    def get_historical_rsi(self, days: int = 30) -> pd.DataFrame:
        """Get historical aggregated RSI data"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT timestamp, date, aggregated_rsi, total_market_cap, 
                       num_assets, confidence_score, market_sentiment, calculation_method
                FROM aggregated_rsi_history 
                WHERE date >= date('now', '-{} day')
                ORDER BY timestamp DESC
            '''.format(days)
            
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = pd.to_datetime(df['date'])
            return df
    
    def get_asset_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data for specific asset"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT h.date, h.timestamp, a.symbol, a.rsi_14, a.weight, 
                       a.market_cap_usd, a.data_quality
                FROM aggregated_rsi_history h
                JOIN asset_contributions a ON h.id = a.aggregated_rsi_id
                WHERE a.symbol = ? AND h.date >= date('now', '-{} day')
                ORDER BY h.timestamp DESC
            '''.format(days)
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = pd.to_datetime(df['date'])
            return df
    
    def get_latest_snapshot(self) -> Optional[AggregatedRSISnapshot]:
        """Get the most recent RSI snapshot"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get latest main record
            cursor.execute('''
                SELECT * FROM aggregated_rsi_history 
                ORDER BY timestamp DESC LIMIT 1
            ''')
            
            main_row = cursor.fetchone()
            if not main_row:
                return None
            
            # Get asset contributions for this snapshot
            cursor.execute('''
                SELECT * FROM asset_contributions 
                WHERE aggregated_rsi_id = ?
                ORDER BY weight DESC
            ''', (main_row[0],))
            
            asset_rows = cursor.fetchall()
            
            # Reconstruct snapshot
            assets = []
            for row in asset_rows:
                asset = CoinMarketData(
                    symbol=row[2],
                    binance_symbol=row[3],
                    market_cap_usd=row[5],
                    volume_24h_usd=row[6],
                    price_usd=row[10] or 0.0,
                    market_cap_rank=row[8] or 0,
                    rsi_14=row[4],
                    weight=row[7],
                    data_quality=row[9] or 'unknown'
                )
                assets.append(asset)
            
            return AggregatedRSISnapshot(
                timestamp=datetime.fromisoformat(main_row[1]),
                aggregated_rsi=main_row[3],
                total_market_cap=main_row[4],
                num_assets=main_row[5],
                confidence_score=main_row[6],
                market_sentiment=main_row[7],
                top_contributors=assets,
                calculation_method=main_row[8]
            )
    
    def log_processing_event(self, level: str, status: str, message: str = None, 
                           error_details: str = None, processing_time: float = 0.0):
        """Log processing events and errors"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO processing_logs 
                (timestamp, level, status, message, error_details, processing_time_seconds)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                level,
                status,
                message,
                error_details,
                processing_time
            ))
            conn.commit()
    
    def get_processing_logs(self, days: int = 7, status_filter: str = None) -> pd.DataFrame:
        """Get recent processing logs"""
        with sqlite3.connect(self.db_path) as conn:
            base_query = '''
                SELECT timestamp, level, status, message, error_details, processing_time_seconds
                FROM processing_logs 
                WHERE timestamp >= datetime('now', '-{} day')
            '''.format(days)
            
            params = []
            if status_filter:
                base_query += " AND status = ?"
                params.append(status_filter)
            
            base_query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql_query(base_query, conn, params=params)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    
    def get_rsi_statistics(self, days: int = 30) -> Dict:
        """Get statistical summary of RSI data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Basic RSI statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as record_count,
                    AVG(aggregated_rsi) as avg_rsi,
                    MIN(aggregated_rsi) as min_rsi,
                    MAX(aggregated_rsi) as max_rsi,
                    AVG(confidence_score) as avg_confidence,
                    AVG(num_assets) as avg_assets
                FROM aggregated_rsi_history 
                WHERE date >= date('now', '-{} day')
            '''.format(days))
            
            stats = cursor.fetchone()
            
            # Sentiment distribution
            cursor.execute('''
                SELECT market_sentiment, COUNT(*) as count
                FROM aggregated_rsi_history 
                WHERE date >= date('now', '-{} day')
                GROUP BY market_sentiment
            '''.format(days))
            
            sentiment_dist = dict(cursor.fetchall())
            
            # Recent trend (last 7 days)
            cursor.execute('''
                SELECT aggregated_rsi, date
                FROM aggregated_rsi_history 
                WHERE date >= date('now', '-7 day')
                ORDER BY date ASC
            ''')
            
            recent_data = cursor.fetchall()
            trend = "stable"
            if len(recent_data) >= 2:
                start_rsi = recent_data[0][0]
                end_rsi = recent_data[-1][0]
                change = end_rsi - start_rsi
                if change > 5:
                    trend = "rising"
                elif change < -5:
                    trend = "falling"
            
            return {
                'period_days': days,
                'record_count': stats[0] or 0,
                'average_rsi': stats[1] or 0,
                'min_rsi': stats[2] or 0,
                'max_rsi': stats[3] or 0,
                'average_confidence': stats[4] or 0,
                'average_assets': stats[5] or 0,
                'sentiment_distribution': sentiment_dist,
                'recent_trend': trend,
                'rsi_range': (stats[3] or 0) - (stats[2] or 0)
            }
    
    def export_data(self, output_dir: str, days: int = 30) -> List[str]:
        """Export historical data to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        exported_files = []
        
        # Export main RSI history
        rsi_df = self.get_historical_rsi(days)
        if not rsi_df.empty:
            rsi_file = output_path / f"aggregated_rsi_history_{days}days.csv"
            rsi_df.to_csv(rsi_file, index=False)
            exported_files.append(str(rsi_file))
        
        # Export processing logs
        logs_df = self.get_processing_logs(days)
        if not logs_df.empty:
            logs_file = output_path / f"processing_logs_{days}days.csv"
            logs_df.to_csv(logs_file, index=False)
            exported_files.append(str(logs_file))
        
        # Export asset contributions (latest snapshot)
        with sqlite3.connect(self.db_path) as conn:
            assets_query = '''
                SELECT h.date, a.symbol, a.rsi_14, a.weight, a.market_cap_usd, a.data_quality
                FROM aggregated_rsi_history h
                JOIN asset_contributions a ON h.id = a.aggregated_rsi_id
                WHERE h.date >= date('now', '-{} day')
                ORDER BY h.date DESC, a.weight DESC
            '''.format(days)
            
            assets_df = pd.read_sql_query(assets_query, conn)
            
            if not assets_df.empty:
                assets_file = output_path / f"asset_contributions_{days}days.csv"
                assets_df.to_csv(assets_file, index=False)
                exported_files.append(str(assets_file))
        
        logger.info(f"Exported {len(exported_files)} files to {output_path}")
        return exported_files
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data beyond specified days"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get IDs of records to delete
            cursor.execute('''
                SELECT id FROM aggregated_rsi_history 
                WHERE date < date('now', '-{} day')
            '''.format(days_to_keep))
            
            old_ids = [row[0] for row in cursor.fetchall()]
            
            if old_ids:
                # Delete asset contributions
                placeholders = ','.join('?' * len(old_ids))
                cursor.execute(f'''
                    DELETE FROM asset_contributions 
                    WHERE aggregated_rsi_id IN ({placeholders})
                ''', old_ids)
                
                # Delete main records
                cursor.execute(f'''
                    DELETE FROM aggregated_rsi_history 
                    WHERE id IN ({placeholders})
                ''', old_ids)
                
                # Delete old logs
                cursor.execute('''
                    DELETE FROM processing_logs 
                    WHERE timestamp < datetime('now', '-{} day')
                '''.format(days_to_keep))
                
                conn.commit()
                logger.info(f"Cleaned up {len(old_ids)} old records beyond {days_to_keep} days")
            else:
                logger.info("No old data to clean up")
    
    def get_database_info(self) -> Dict:
        """Get database information and statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Table sizes
            cursor.execute("SELECT COUNT(*) FROM aggregated_rsi_history")
            rsi_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM asset_contributions")
            assets_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM processing_logs")
            logs_count = cursor.fetchone()[0]
            
            # Database file size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            # Date range
            cursor.execute('''
                SELECT MIN(date) as earliest, MAX(date) as latest 
                FROM aggregated_rsi_history
            ''')
            date_range = cursor.fetchone()
            
            return {
                'database_path': str(self.db_path),
                'database_size_mb': db_size / (1024 * 1024),
                'tables': {
                    'rsi_history_records': rsi_count,
                    'asset_contributions': assets_count,
                    'processing_logs': logs_count
                },
                'date_range': {
                    'earliest': date_range[0],
                    'latest': date_range[1]
                } if date_range[0] else None
            }