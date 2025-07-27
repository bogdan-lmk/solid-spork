"""
Alert system for Aggregated RSI Indicator
Monitors RSI conditions and triggers notifications for significant events
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

try:
    from ..core.data_models import AggregatedRSISnapshot, CoinMarketData
except ImportError:
    from aggregated_rsi_indicator.core.data_models import AggregatedRSISnapshot, CoinMarketData

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Individual alert data structure"""
    timestamp: datetime
    level: AlertLevel
    title: str
    message: str
    rsi_value: float
    confidence_score: float
    affected_assets: List[str]
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'rsi_value': self.rsi_value,
            'confidence_score': self.confidence_score,
            'affected_assets': self.affected_assets,
            'metadata': self.metadata
        }

class AlertCondition:
    """Base class for alert conditions"""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
    
    def check(self, current_snapshot: AggregatedRSISnapshot, 
              previous_snapshot: Optional[AggregatedRSISnapshot] = None) -> List[Alert]:
        """Check condition and return alerts if triggered"""
        if not self.enabled:
            return []
        return self._evaluate(current_snapshot, previous_snapshot)
    
    def _evaluate(self, current: AggregatedRSISnapshot, 
                  previous: Optional[AggregatedRSISnapshot]) -> List[Alert]:
        """Override this method in subclasses"""
        raise NotImplementedError

class OverboughtCondition(AlertCondition):
    """Alert when aggregated RSI enters overbought territory"""
    
    def __init__(self, threshold: float = 75.0, enabled: bool = True):
        super().__init__("Overbought Alert", enabled)
        self.threshold = threshold
    
    def _evaluate(self, current: AggregatedRSISnapshot, 
                  previous: Optional[AggregatedRSISnapshot]) -> List[Alert]:
        alerts = []
        
        if current.aggregated_rsi >= self.threshold:
            # Check if this is a new condition (not triggered in previous snapshot)
            is_new = True
            if previous and previous.aggregated_rsi >= self.threshold:
                is_new = False
            
            level = AlertLevel.CRITICAL if current.aggregated_rsi >= 80 else AlertLevel.WARNING
            
            # Find overbought individual assets
            overbought_assets = [
                asset.symbol for asset in current.top_contributors 
                if asset.rsi_14 >= 70
            ]
            
            alert = Alert(
                timestamp=current.timestamp,
                level=level,
                title=f"Overbought Condition - RSI {current.aggregated_rsi:.1f}",
                message=f"Market appears overbought with aggregated RSI at {current.aggregated_rsi:.1f}. "
                       f"Consider profit-taking opportunities. {len(overbought_assets)} individual assets also overbought.",
                rsi_value=current.aggregated_rsi,
                confidence_score=current.confidence_score,
                affected_assets=overbought_assets,
                metadata={
                    'threshold': self.threshold,
                    'is_new_condition': is_new,
                    'overbought_asset_count': len(overbought_assets)
                }
            )
            alerts.append(alert)
        
        return alerts

class OversoldCondition(AlertCondition):
    """Alert when aggregated RSI enters oversold territory"""
    
    def __init__(self, threshold: float = 25.0, enabled: bool = True):
        super().__init__("Oversold Alert", enabled)
        self.threshold = threshold
    
    def _evaluate(self, current: AggregatedRSISnapshot, 
                  previous: Optional[AggregatedRSISnapshot]) -> List[Alert]:
        alerts = []
        
        if current.aggregated_rsi <= self.threshold:
            is_new = True
            if previous and previous.aggregated_rsi <= self.threshold:
                is_new = False
            
            level = AlertLevel.CRITICAL if current.aggregated_rsi <= 20 else AlertLevel.WARNING
            
            # Find oversold individual assets
            oversold_assets = [
                asset.symbol for asset in current.top_contributors 
                if asset.rsi_14 <= 30
            ]
            
            alert = Alert(
                timestamp=current.timestamp,
                level=level,
                title=f"Oversold Condition - RSI {current.aggregated_rsi:.1f}",
                message=f"Market appears oversold with aggregated RSI at {current.aggregated_rsi:.1f}. "
                       f"Potential buying opportunity. {len(oversold_assets)} individual assets also oversold.",
                rsi_value=current.aggregated_rsi,
                confidence_score=current.confidence_score,
                affected_assets=oversold_assets,
                metadata={
                    'threshold': self.threshold,
                    'is_new_condition': is_new,
                    'oversold_asset_count': len(oversold_assets)
                }
            )
            alerts.append(alert)
        
        return alerts

class SignificantChangeCondition(AlertCondition):
    """Alert when RSI changes significantly from previous reading"""
    
    def __init__(self, min_change: float = 10.0, enabled: bool = True):
        super().__init__("Significant Change Alert", enabled)
        self.min_change = min_change
    
    def _evaluate(self, current: AggregatedRSISnapshot, 
                  previous: Optional[AggregatedRSISnapshot]) -> List[Alert]:
        alerts = []
        
        if not previous:
            return alerts
        
        change = current.aggregated_rsi - previous.aggregated_rsi
        abs_change = abs(change)
        
        if abs_change >= self.min_change:
            direction = "increased" if change > 0 else "decreased"
            level = AlertLevel.CRITICAL if abs_change >= 15 else AlertLevel.WARNING
            
            # Find assets with significant changes
            significant_assets = []
            for current_asset in current.top_contributors:
                prev_asset = next((a for a in previous.top_contributors if a.symbol == current_asset.symbol), None)
                if prev_asset:
                    asset_change = abs(current_asset.rsi_14 - prev_asset.rsi_14)
                    if asset_change >= 15:  # Individual asset significant change threshold
                        significant_assets.append(f"{current_asset.symbol}({asset_change:+.1f})")
            
            alert = Alert(
                timestamp=current.timestamp,
                level=level,
                title=f"Significant RSI Change - {direction.title()} {abs_change:.1f} points",
                message=f"Aggregated RSI {direction} by {abs_change:.1f} points from {previous.aggregated_rsi:.1f} "
                       f"to {current.aggregated_rsi:.1f}. This indicates strong momentum shift.",
                rsi_value=current.aggregated_rsi,
                confidence_score=current.confidence_score,
                affected_assets=[asset.split('(')[0] for asset in significant_assets],
                metadata={
                    'change_amount': change,
                    'previous_rsi': previous.aggregated_rsi,
                    'min_change_threshold': self.min_change,
                    'significant_asset_changes': significant_assets,
                    'time_between_readings': (current.timestamp - previous.timestamp).total_seconds() / 3600
                }
            )
            alerts.append(alert)
        
        return alerts

class LowConfidenceCondition(AlertCondition):
    """Alert when data confidence is low"""
    
    def __init__(self, min_confidence: float = 60.0, enabled: bool = True):
        super().__init__("Low Confidence Alert", enabled)
        self.min_confidence = min_confidence
    
    def _evaluate(self, current: AggregatedRSISnapshot, 
                  previous: Optional[AggregatedRSISnapshot]) -> List[Alert]:
        alerts = []
        
        if current.confidence_score < self.min_confidence:
            # Find assets with poor data quality
            poor_quality_assets = [
                asset.symbol for asset in current.top_contributors 
                if asset.data_quality == 'poor'
            ]
            
            level = AlertLevel.CRITICAL if current.confidence_score < 40 else AlertLevel.WARNING
            
            alert = Alert(
                timestamp=current.timestamp,
                level=level,
                title=f"Low Data Confidence - {current.confidence_score:.1f}%",
                message=f"RSI calculation confidence is only {current.confidence_score:.1f}%. "
                       f"Results may be unreliable due to data quality issues. "
                       f"{len(poor_quality_assets)} assets have poor data quality.",
                rsi_value=current.aggregated_rsi,
                confidence_score=current.confidence_score,
                affected_assets=poor_quality_assets,
                metadata={
                    'min_confidence_threshold': self.min_confidence,
                    'poor_quality_asset_count': len(poor_quality_assets),
                    'total_assets': current.num_assets
                }
            )
            alerts.append(alert)
        
        return alerts

class MarketDivergenceCondition(AlertCondition):
    """Alert when individual assets show strong divergence from aggregated RSI"""
    
    def __init__(self, divergence_threshold: float = 30.0, min_assets: int = 3, enabled: bool = True):
        super().__init__("Market Divergence Alert", enabled)
        self.divergence_threshold = divergence_threshold
        self.min_assets = min_assets
    
    def _evaluate(self, current: AggregatedRSISnapshot, 
                  previous: Optional[AggregatedRSISnapshot]) -> List[Alert]:
        alerts = []
        
        # Find assets with strong divergence from aggregated RSI
        divergent_assets = []
        for asset in current.top_contributors:
            divergence = abs(asset.rsi_14 - current.aggregated_rsi)
            if divergence >= self.divergence_threshold:
                direction = "higher" if asset.rsi_14 > current.aggregated_rsi else "lower"
                divergent_assets.append(f"{asset.symbol}({asset.rsi_14:.1f},{direction})")
        
        if len(divergent_assets) >= self.min_assets:
            level = AlertLevel.WARNING
            
            alert = Alert(
                timestamp=current.timestamp,
                level=level,
                title=f"Market Divergence - {len(divergent_assets)} assets diverging",
                message=f"{len(divergent_assets)} assets show significant divergence from aggregated RSI "
                       f"({current.aggregated_rsi:.1f}). This may indicate sector rotation or "
                       f"individual asset-specific movements.",
                rsi_value=current.aggregated_rsi,
                confidence_score=current.confidence_score,
                affected_assets=[asset.split('(')[0] for asset in divergent_assets],
                metadata={
                    'divergence_threshold': self.divergence_threshold,
                    'divergent_asset_details': divergent_assets,
                    'divergent_count': len(divergent_assets),
                    'total_assets': current.num_assets
                }
            )
            alerts.append(alert)
        
        return alerts

class AlertSystem:
    """Main alert system coordinator"""
    
    def __init__(self, alert_handlers: List[Callable[[Alert], None]] = None):
        """
        Initialize alert system
        
        Args:
            alert_handlers: List of functions to handle triggered alerts
        """
        self.conditions: List[AlertCondition] = []
        self.alert_handlers = alert_handlers or []
        self.alert_history: List[Alert] = []
        
        # Add default conditions
        self._setup_default_conditions()
    
    def _setup_default_conditions(self):
        """Setup default alert conditions"""
        self.conditions = [
            OverboughtCondition(threshold=75.0),
            OversoldCondition(threshold=25.0),
            SignificantChangeCondition(min_change=10.0),
            LowConfidenceCondition(min_confidence=60.0),
            MarketDivergenceCondition(divergence_threshold=25.0, min_assets=3)
        ]
    
    def add_condition(self, condition: AlertCondition):
        """Add custom alert condition"""
        self.conditions.append(condition)
        logger.info(f"Added alert condition: {condition.name}")
    
    def remove_condition(self, condition_name: str):
        """Remove alert condition by name"""
        self.conditions = [c for c in self.conditions if c.name != condition_name]
        logger.info(f"Removed alert condition: {condition_name}")
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, current_snapshot: AggregatedRSISnapshot,
                    previous_snapshot: Optional[AggregatedRSISnapshot] = None) -> List[Alert]:
        """
        Check all conditions and return triggered alerts
        """
        all_alerts = []
        
        for condition in self.conditions:
            try:
                alerts = condition.check(current_snapshot, previous_snapshot)
                all_alerts.extend(alerts)
                
                if alerts:
                    logger.info(f"Condition '{condition.name}' triggered {len(alerts)} alerts")
                    
            except Exception as e:
                logger.error(f"Error checking condition '{condition.name}': {e}")
                continue
        
        # Store in history
        self.alert_history.extend(all_alerts)
        
        # Keep only recent history (last 100 alerts)
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        # Handle alerts
        for alert in all_alerts:
            self._handle_alert(alert)
        
        return all_alerts
    
    def _handle_alert(self, alert: Alert):
        """Process individual alert through all handlers"""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from recent hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """Get summary of recent alerts"""
        recent_alerts = self.get_recent_alerts(hours)
        
        if not recent_alerts:
            return {
                'period_hours': hours,
                'total_alerts': 0,
                'by_level': {},
                'by_condition': {},
                'latest_alert': None
            }
        
        # Count by level
        level_counts = {}
        for alert in recent_alerts:
            level = alert.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Count by condition (extracted from title)
        condition_counts = {}
        for alert in recent_alerts:
            condition = alert.title.split(' - ')[0]  # Get condition name from title
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        return {
            'period_hours': hours,
            'total_alerts': len(recent_alerts),
            'by_level': level_counts,
            'by_condition': condition_counts,
            'latest_alert': recent_alerts[-1].to_dict() if recent_alerts else None
        }
    
    def configure_condition(self, condition_name: str, **kwargs):
        """Configure existing condition parameters"""
        for condition in self.conditions:
            if condition.name == condition_name:
                for key, value in kwargs.items():
                    if hasattr(condition, key):
                        setattr(condition, key, value)
                        logger.info(f"Updated {condition_name}.{key} = {value}")
                    else:
                        logger.warning(f"Condition {condition_name} has no parameter {key}")
                break
        else:
            logger.error(f"Condition {condition_name} not found")

# Default alert handlers
def console_alert_handler(alert: Alert):
    """Simple console alert handler"""
    level_emoji = {
        AlertLevel.INFO: "ℹ️",
        AlertLevel.WARNING: "⚠️", 
        AlertLevel.CRITICAL: "🚨"
    }
    
    emoji = level_emoji.get(alert.level, "📢")
    print(f"\n{emoji} {alert.level.value.upper()} ALERT")
    print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Title: {alert.title}")
    print(f"Message: {alert.message}")
    if alert.affected_assets:
        print(f"Affected Assets: {', '.join(alert.affected_assets)}")
    print("-" * 50)

def file_alert_handler(alert: Alert, log_file: str = "rsi_alerts.log"):
    """File-based alert handler"""
    with open(log_file, 'a') as f:
        f.write(f"{alert.timestamp.isoformat()} | {alert.level.value.upper()} | {alert.title}\n")
        f.write(f"Message: {alert.message}\n")
        if alert.affected_assets:
            f.write(f"Assets: {', '.join(alert.affected_assets)}\n")
        f.write("-" * 80 + "\n")

def create_default_alert_system() -> AlertSystem:
    """Create alert system with default handlers"""
    return AlertSystem(alert_handlers=[console_alert_handler])