import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    NEUTRAL = 0
    LONG = 1
    SHORT = 2

class DayOfWeek(Enum):
    SUNDAY = 0
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6

@dataclass
class CandleData:
    """Data structure for candle information"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_volume: float  # Number of price changes

@dataclass
class TradingParameters:
    """All trading parameters in one place"""
    # Window sizes
    window_size: int = 20
    tick_analysis_window: int = 100
    volume_analysis_window: int = 100
    
    # Threshold multipliers
    tick_multiplier: float = 2.5
    volume_multiplier: float = 2.5
    
    # Threshold bounds
    tick_min_multiplier: float = 0.3
    tick_max_multiplier: float = 3.0
    volume_min_multiplier: float = 0.3
    volume_max_multiplier: float = 3.0
    
    # Absolute minimums
    tick_min_absolute: float = 100
    volume_min_absolute: float = 80
    
    # Base thresholds
    base_tick_buy: float = 1000
    base_tick_sell: float = -1000
    base_volume_buy: float = 800
    base_volume_sell: float = -800
    
    # Risk management
    risk_percent: float = 2.0
    fixed_lot_size: float = 0.01
    stop_loss_points: int = 300
    take_profit_points: int = 450
    max_spread: float = 3.0
    max_slippage: float = 3.0
    trailing_start: int = 200
    trailing_step: int = 100
    
    # Trading filters
    enable_trading: bool = True
    enable_tick_dynamic: bool = True
    enable_volume_dynamic: bool = True
    enable_time_filter: bool = True
    
    # Day filters (0=Sunday, 6=Saturday)
    allowed_days: List[int] = None
    
    # Hour filters (0-23)
    allowed_hours: List[int] = None
    
    def __post_init__(self):
        if self.allowed_days is None:
            self.allowed_days = [1, 2, 3, 4, 5]  # Monday-Friday
        if self.allowed_hours is None:
            self.allowed_hours = list(range(7, 18))  # 7am to 6pm

class DynamicThresholdCalculator:
    """Calculates dynamic thresholds using Median + MAD"""
    
    @staticmethod
    def calculate_median(data: np.ndarray) -> float:
        """Calculate median of array"""
        if len(data) == 0:
            return 0.0
        return float(np.median(data))
    
    @staticmethod
    def calculate_mad(data: np.ndarray, median: float = None) -> float:
        """Calculate Median Absolute Deviation"""
        if len(data) == 0:
            return 0.0
        
        if median is None:
            median = DynamicThresholdCalculator.calculate_median(data)
        
        # Calculate absolute deviations
        deviations = np.abs(data - median)
        
        # Calculate median of deviations
        mad_raw = DynamicThresholdCalculator.calculate_median(deviations)
        
        # Scale for normal distribution
        mad_scaled = mad_raw * 1.4826
        
        return float(mad_scaled)
    
    @staticmethod
    def calculate_thresholds(
        data: np.ndarray,
        multiplier: float,
        base_threshold: float,
        min_multiplier: float,
        max_multiplier: float,
        min_absolute: float,
        is_positive: bool
    ) -> float:
        """
        Calculate dynamic threshold with bounds
        """
        if len(data) < 10:
            return base_threshold
        
        median = DynamicThresholdCalculator.calculate_median(data)
        mad = DynamicThresholdCalculator.calculate_mad(data, median)
        
        # Apply minimum MAD
        mad = max(mad, 10.0)
        
        # Calculate raw threshold
        if is_positive:
            raw_threshold = median + (multiplier * mad)
        else:
            raw_threshold = median - (multiplier * mad)
        
        # Apply absolute minimum
        if abs(raw_threshold) < min_absolute:
            raw_threshold = min_absolute if raw_threshold >= 0 else -min_absolute
        
        # Apply multiplier bounds relative to base threshold
        min_bound = base_threshold * min_multiplier
        max_bound = base_threshold * max_multiplier
        
        if is_positive:
            # For positive thresholds, ensure positive and bound
            if raw_threshold < 0:
                raw_threshold = abs(raw_threshold)
            return min(max(raw_threshold, min_bound), max_bound)
        else:
            # For negative thresholds, ensure negative and bound
            if raw_threshold > 0:
                raw_threshold = -raw_threshold
            return max(min(raw_threshold, min_bound), max_bound)

class DeltaCalculator:
    """Calculates various delta values from candle data"""
    
    @staticmethod
    def calculate_volume_delta(candle: CandleData) -> float:
        """Calculate volume delta based on candle direction"""
        if candle.close > candle.open:
            return candle.tick_volume  # Positive delta for bullish
        elif candle.close < candle.open:
            return -candle.tick_volume  # Negative delta for bearish
        else:
            return 0.0  # Neutral
    
    @staticmethod
    def calculate_tick_delta(candle: CandleData) -> float:
        """Calculate tick delta based on price movement ratio"""
        price_change = candle.close - candle.open
        price_range = candle.high - candle.low
        
        if price_range > 0:
            bullish_ratio = price_change / price_range
            return candle.tick_volume * bullish_ratio
        return 0.0
    
    @staticmethod
    def calculate_ticks_per_second(candle: CandleData, period_seconds: int) -> float:
        """Calculate ticks per second for a candle"""
        if period_seconds > 0:
            return candle.tick_volume / period_seconds
        return 0.0
    
    @staticmethod
    def calculate_volume_weighted_price(candle: CandleData) -> float:
        """Calculate volume-weighted price for footprint line"""
        typical_price = (candle.high + candle.low + candle.close) / 3.0
        
        if candle.tick_volume > 0:
            # Weighted average
            close_weight = 0.4
            typical_weight = 0.4
            open_weight = 0.2
            
            return (
                candle.close * close_weight +
                typical_price * typical_weight +
                candle.open * open_weight
            )
        return typical_price

class PositionManager:
    """Manages trading positions and risk"""
    
    def __init__(self, params: TradingParameters, symbol_info: Dict):
        self.params = params
        self.symbol_info = symbol_info
        self.positions = []
        self.closed_positions = []
        
        # Track trading statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        
    def calculate_position_size(
        self, 
        current_price: float, 
        stop_loss_distance: float,
        account_balance: float = 10000.0
    ) -> float:
        """Calculate position size based on risk management rules"""
        
        if self.params.fixed_lot_size > 0:
            return self.params.fixed_lot_size
        
        # Dynamic position sizing based on risk percentage
        risk_amount = account_balance * (self.params.risk_percent / 100)
        
        # Calculate position size based on stop loss
        # This is simplified - adjust based on your broker's calculation
        point_value = 0.0001  # Adjust for your symbol
        tick_value = 10.0  # Adjust for your symbol
        
        if stop_loss_distance > 0:
            risk_per_point = tick_value * point_value
            points_at_risk = stop_loss_distance / point_value
            dollar_risk = points_at_risk * risk_per_point
            
            if dollar_risk > 0:
                lots = risk_amount / dollar_risk
                return round(lots, 2)
        
        return 0.01  # Default minimum lot size
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        is_long: bool
    ) -> float:
        """Calculate stop loss price"""
        if is_long:
            return entry_price - (self.params.stop_loss_points * 0.0001)
        else:
            return entry_price + (self.params.stop_loss_points * 0.0001)
    
    def calculate_take_profit(
        self, 
        entry_price: float, 
        is_long: bool
    ) -> float:
        """Calculate take profit price"""
        if is_long:
            return entry_price + (self.params.take_profit_points * 0.0001)
        else:
            return entry_price - (self.params.take_profit_points * 0.0001)
    
    def check_trailing_stop(
        self, 
        position: Dict, 
        current_price: float
    ) -> Optional[float]:
        """Check and update trailing stop if needed"""
        if position['is_long']:
            new_sl = current_price - (self.params.trailing_start * 0.0001)
            if (new_sl > position['stop_loss'] and 
                new_sl > position['entry_price']):
                return new_sl
        else:
            new_sl = current_price + (self.params.trailing_start * 0.0001)
            if (new_sl < position['stop_loss'] and 
                new_sl < position['entry_price']):
                return new_sl
        return None
    
    def add_position(
        self, 
        timestamp: datetime,
        entry_price: float,
        is_long: bool,
        position_size: float,
        signal_strength: float = 1.0
    ):
        """Add a new position"""
        position = {
            'id': len(self.positions) + 1,
            'timestamp': timestamp,
            'entry_price': entry_price,
            'stop_loss': self.calculate_stop_loss(entry_price, is_long),
            'take_profit': self.calculate_take_profit(entry_price, is_long),
            'size': position_size,
            'is_long': is_long,
            'signal_strength': signal_strength,
            'exit_price': None,
            'exit_time': None,
            'pnl': 0.0,
            'pnl_percent': 0.0,
            'status': 'OPEN'
        }
        
        self.positions.append(position)
        self.total_trades += 1
        
        logger.info(f"Opened {'LONG' if is_long else 'SHORT'} position "
                   f"at {entry_price:.5f}, SL: {position['stop_loss']:.5f}, "
                   f"TP: {position['take_profit']:.5f}")
        
        return position
    
    def update_positions(self, current_price: float, timestamp: datetime):
        """Update all open positions"""
        positions_to_close = []
        
        for position in self.positions:
            if position['status'] != 'OPEN':
                continue
            
            # Check for stop loss or take profit
            if position['is_long']:
                if current_price <= position['stop_loss']:
                    position['exit_reason'] = 'SL'
                    positions_to_close.append(position)
                elif current_price >= position['take_profit']:
                    position['exit_reason'] = 'TP'
                    positions_to_close.append(position)
                else:
                    # Check trailing stop
                    new_sl = self.check_trailing_stop(position, current_price)
                    if new_sl:
                        position['stop_loss'] = new_sl
            else:
                if current_price >= position['stop_loss']:
                    position['exit_reason'] = 'SL'
                    positions_to_close.append(position)
                elif current_price <= position['take_profit']:
                    position['exit_reason'] = 'TP'
                    positions_to_close.append(position)
                else:
                    # Check trailing stop
                    new_sl = self.check_trailing_stop(position, current_price)
                    if new_sl:
                        position['stop_loss'] = new_sl
            
            # Update unrealized P&L
            if position['is_long']:
                position['unrealized_pnl'] = (
                    (current_price - position['entry_price']) * 
                    position['size'] * 100000  # Simplified P&L calculation
                )
            else:
                position['unrealized_pnl'] = (
                    (position['entry_price'] - current_price) * 
                    position['size'] * 100000
                )
        
        # Close positions that hit SL/TP
        for position in positions_to_close:
            self.close_position(position, current_price, timestamp)

    def close_position(
        self, 
        position: Dict, 
        exit_price: float, 
        timestamp: datetime
    ):
        """Close a position and calculate P&L"""
        position['exit_price'] = exit_price
        position['exit_time'] = timestamp
        position['status'] = 'CLOSED'
        
        # Calculate P&L
        if position['is_long']:
            pnl = (exit_price - position['entry_price']) * position['size'] * 100000
        else:
            pnl = (position['entry_price'] - exit_price) * position['size'] * 100000
        
        position['pnl'] = pnl
        position['pnl_percent'] = (pnl / (position['entry_price'] * position['size'] * 100000)) * 100
        
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update drawdown
        self.peak_equity = max(self.peak_equity, self.total_pnl)
        self.current_drawdown = self.peak_equity - self.total_pnl
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Move to closed positions
        self.closed_positions.append(position)
        self.positions.remove(position)
        
        logger.info(f"Closed position #{position['id']} at {exit_price:.5f}, "
                   f"P&L: ${pnl:.2f} ({position['pnl_percent']:.2f}%)")

class TimeFilter:
    """Manages time-based trading filters"""
    
    def __init__(self, params: TradingParameters):
        self.params = params
    
    def is_trading_allowed(self, timestamp: datetime) -> bool:
        """Check if trading is allowed at the given time"""
        if not self.params.enable_time_filter:
            return True
        
        # Check day of week
        day_of_week = timestamp.weekday()  # Monday=0, Sunday=6
        if day_of_week not in self.params.allowed_days:
            return False
        
        # Check hour
        current_hour = timestamp.hour
        if current_hour not in self.params.allowed_hours:
            return False
        
        return True
    
    def get_trading_status(self, timestamp: datetime) -> Dict:
        """Get detailed trading status"""
        return {
            'allowed': self.is_trading_allowed(timestamp),
            'day': timestamp.strftime('%A'),
            'hour': timestamp.hour,
            'minute': timestamp.minute
        }

class AegisScalper:
    """
    Main AegisScalper trading system
    Implements the dynamic delta threshold scalping strategy
    """
    
    def __init__(self, params: TradingParameters, symbol: str = "EURUSD"):
        self.params = params
        self.symbol = symbol
        
        # Initialize calculators
        self.threshold_calc = DynamicThresholdCalculator()
        self.delta_calc = DeltaCalculator()
        self.time_filter = TimeFilter(params)
        
        # Initialize data structures
        self.window_data = []  # List of CandleData for the trading window
        self.tick_analysis_data = []  # Tick deltas for threshold calculation
        self.volume_analysis_data = []  # Volume deltas for threshold calculation
        
        # Current deltas and thresholds
        self.cumulative_tick_delta = 0.0
        self.cumulative_volume_delta = 0.0
        self.dynamic_tick_buy_threshold = params.base_tick_buy
        self.dynamic_tick_sell_threshold = params.base_tick_sell
        self.dynamic_volume_buy_threshold = params.base_volume_buy
        self.dynamic_volume_sell_threshold = params.base_volume_sell
        
        # Volume footprint data
        self.volume_weighted_prices = []
        self.volume_line_slope = 0  # -1: downtrend, 0: neutral, 1: uptrend
        
        # Performance tracking
        self.signals_generated = 0
        self.signals_executed = 0
        self.last_trade_time = None
        
        # Position manager
        self.position_manager = PositionManager(params, {})
        
        logger.info(f"AegisScalper initialized for {symbol}")
        logger.info(f"Window size: {params.window_size}")
        logger.info(f"Tick analysis window: {params.tick_analysis_window}")
        logger.info(f"Volume analysis window: {params.volume_analysis_window}")
    
    def update_with_candle(self, candle: CandleData):
        """
        Update the system with a new candle
        """
        # Add candle to window
        self.window_data.append(candle)
        
        # Maintain window size
        if len(self.window_data) > self.params.window_size:
            removed_candle = self.window_data.pop(0)
            
            # Update analysis windows with removed candle's deltas
            tick_delta = self.delta_calc.calculate_tick_delta(removed_candle)
            volume_delta = self.delta_calc.calculate_volume_delta(removed_candle)
            
            self.update_analysis_windows(tick_delta, volume_delta)
        
        # Recalculate all metrics
        self.recalculate_deltas()
        self.recalculate_volume_footprint()
        self.calculate_dynamic_thresholds()
        
        # Check for trading signals
        if self.params.enable_trading:
            signal = self.check_trading_signals(candle.timestamp)
            return signal
        
        return SignalType.NEUTRAL
    
    def update_analysis_windows(
        self, 
        tick_delta: float, 
        volume_delta: float
    ):
        """Update the analysis windows with new delta values"""
        # Update tick analysis window
        self.tick_analysis_data.append(tick_delta)
        if len(self.tick_analysis_data) > self.params.tick_analysis_window:
            self.tick_analysis_data.pop(0)
        
        # Update volume analysis window
        self.volume_analysis_data.append(volume_delta)
        if len(self.volume_analysis_data) > self.params.volume_analysis_window:
            self.volume_analysis_data.pop(0)
    
    def recalculate_deltas(self):
        """Recalculate all delta values for the current window"""
        self.cumulative_tick_delta = 0.0
        self.cumulative_volume_delta = 0.0
        
        for candle in self.window_data:
            tick_delta = self.delta_calc.calculate_tick_delta(candle)
            volume_delta = self.delta_calc.calculate_volume_delta(candle)
            
            self.cumulative_tick_delta += tick_delta
            self.cumulative_volume_delta += volume_delta
    
    def recalculate_volume_footprint(self):
        """Recalculate volume footprint line and slope"""
        self.volume_weighted_prices = []
        
        for candle in self.window_data:
            vwp = self.delta_calc.calculate_volume_weighted_price(candle)
            self.volume_weighted_prices.append(vwp)
        
        # Calculate slope
        if len(self.volume_weighted_prices) >= 2:
            start_price = self.volume_weighted_prices[0]
            end_price = self.volume_weighted_prices[-1]
            
            if end_price > start_price:
                self.volume_line_slope = 1  # Uptrend
            elif end_price < start_price:
                self.volume_line_slope = -1  # Downtrend
            else:
                self.volume_line_slope = 0  # Neutral
    
    def calculate_dynamic_thresholds(self):
        """Calculate dynamic thresholds using MAD"""
        # Convert to numpy arrays for calculations
        tick_data = np.array(self.tick_analysis_data)
        volume_data = np.array(self.volume_analysis_data)
        
        # Calculate tick thresholds
        if (self.params.enable_tick_dynamic and 
            len(tick_data) >= 10):
            
            self.dynamic_tick_buy_threshold = (
                self.threshold_calc.calculate_thresholds(
                    data=tick_data,
                    multiplier=self.params.tick_multiplier,
                    base_threshold=self.params.base_tick_buy,
                    min_multiplier=self.params.tick_min_multiplier,
                    max_multiplier=self.params.tick_max_multiplier,
                    min_absolute=self.params.tick_min_absolute,
                    is_positive=True
                )
            )
            
            self.dynamic_tick_sell_threshold = (
                self.threshold_calc.calculate_thresholds(
                    data=tick_data,
                    multiplier=self.params.tick_multiplier,
                    base_threshold=self.params.base_tick_sell,
                    min_multiplier=self.params.tick_min_multiplier,
                    max_multiplier=self.params.tick_max_multiplier,
                    min_absolute=self.params.tick_min_absolute,
                    is_positive=False
                )
            )
        else:
            # Use base thresholds
            self.dynamic_tick_buy_threshold = self.params.base_tick_buy
            self.dynamic_tick_sell_threshold = self.params.base_tick_sell
        
        # Calculate volume thresholds
        if (self.params.enable_volume_dynamic and 
            len(volume_data) >= 10):
            
            self.dynamic_volume_buy_threshold = (
                self.threshold_calc.calculate_thresholds(
                    data=volume_data,
                    multiplier=self.params.volume_multiplier,
                    base_threshold=self.params.base_volume_buy,
                    min_multiplier=self.params.volume_min_multiplier,
                    max_multiplier=self.params.volume_max_multiplier,
                    min_absolute=self.params.volume_min_absolute,
                    is_positive=True
                )
            )
            
            self.dynamic_volume_sell_threshold = (
                self.threshold_calc.calculate_thresholds(
                    data=volume_data,
                    multiplier=self.params.volume_multiplier,
                    base_threshold=self.params.base_volume_sell,
                    min_multiplier=self.params.volume_min_multiplier,
                    max_multiplier=self.params.volume_max_multiplier,
                    min_absolute=self.params.volume_min_absolute,
                    is_positive=False
                )
            )
        else:
            # Use base thresholds
            self.dynamic_volume_buy_threshold = self.params.base_volume_buy
            self.dynamic_volume_sell_threshold = self.params.base_volume_sell
    
    def check_trading_signals(self, timestamp: datetime) -> SignalType:
        """
        Check for trading signals based on current conditions
        Returns: SignalType (LONG, SHORT, or NEUTRAL)
        """
        # Check time filter
        if not self.time_filter.is_trading_allowed(timestamp):
            return SignalType.NEUTRAL
        
        # Check minimum time between trades
        if (self.last_trade_time and 
            (timestamp - self.last_trade_time).seconds < 60):  # 1 minute minimum
            return SignalType.NEUTRAL
        
        # Check if we have enough data
        if len(self.window_data) < self.params.window_size:
            return SignalType.NEUTRAL
        
        # Check signal conditions
        tick_buy_signal = self.cumulative_tick_delta < self.dynamic_tick_sell_threshold
        tick_sell_signal = self.cumulative_tick_delta > self.dynamic_tick_buy_threshold
        
        volume_buy_signal = self.cumulative_volume_delta < self.dynamic_volume_sell_threshold
        volume_sell_signal = self.cumulative_volume_delta > self.dynamic_volume_buy_threshold
        
        # Both deltas must agree
        long_signal = tick_buy_signal and volume_buy_signal
        short_signal = tick_sell_signal and volume_sell_signal
        
        # Volume footprint confirmation
        if long_signal and self.volume_line_slope == -1:  # Downtrend
            self.signals_generated += 1
            logger.info(f"LONG signal generated at {timestamp}")
            return SignalType.LONG
        elif short_signal and self.volume_line_slope == 1:  # Uptrend
            self.signals_generated += 1
            logger.info(f"SHORT signal generated at {timestamp}")
            return SignalType.SHORT
        
        return SignalType.NEUTRAL
    
    def execute_trade(
        self, 
        signal: SignalType, 
        current_price: float,
        timestamp: datetime,
        account_balance: float = 10000.0
    ):
        """Execute a trade based on signal"""
        if signal == SignalType.NEUTRAL:
            return None
        
        # Calculate position size
        position_size = self.position_manager.calculate_position_size(
            current_price=current_price,
            stop_loss_distance=self.params.stop_loss_points * 0.0001,
            account_balance=account_balance
        )
        
        if signal == SignalType.LONG:
            position = self.position_manager.add_position(
                timestamp=timestamp,
                entry_price=current_price,
                is_long=True,
                position_size=position_size
            )
        elif signal == SignalType.SHORT:
            position = self.position_manager.add_position(
                timestamp=timestamp,
                entry_price=current_price,
                is_long=False,
                position_size=position_size
            )
        
        self.last_trade_time = timestamp
        self.signals_executed += 1
        
        return position
    
    def get_current_stats(self) -> Dict:
        """Get current system statistics"""
        return {
            'window_candles': len(self.window_data),
            'tick_analysis_points': len(self.tick_analysis_data),
            'volume_analysis_points': len(self.volume_analysis_data),
            'cumulative_tick_delta': self.cumulative_tick_delta,
            'cumulative_volume_delta': self.cumulative_volume_delta,
            'tick_buy_threshold': self.dynamic_tick_buy_threshold,
            'tick_sell_threshold': self.dynamic_tick_sell_threshold,
            'volume_buy_threshold': self.dynamic_volume_buy_threshold,
            'volume_sell_threshold': self.dynamic_volume_sell_threshold,
            'volume_line_slope': self.volume_line_slope,
            'signals_generated': self.signals_generated,
            'signals_executed': self.signals_executed,
            'open_positions': len(self.position_manager.positions),
            'total_pnl': self.position_manager.total_pnl,
            'win_rate': (
                self.position_manager.winning_trades / 
                max(1, self.position_manager.total_trades)
            ) * 100
        }
    
    def get_trading_recommendation(self) -> Dict:
        """Get trading recommendation based on current state"""
        stats = self.get_current_stats()
        
        recommendation = {
            'action': 'HOLD',
            'confidence': 0.0,
            'reason': '',
            'stats': stats
        }
        
        # Check for long signal
        if (stats['cumulative_tick_delta'] < stats['tick_sell_threshold'] and
            stats['cumulative_volume_delta'] < stats['volume_sell_threshold'] and
            stats['volume_line_slope'] == -1):
            
            signal_strength = min(
                abs(stats['cumulative_tick_delta'] / stats['tick_sell_threshold']),
                abs(stats['cumulative_volume_delta'] / stats['volume_sell_threshold'])
            )
            
            recommendation['action'] = 'BUY'
            recommendation['confidence'] = min(signal_strength, 1.0) * 100
            recommendation['reason'] = 'Both deltas below sell thresholds in downtrend'
        
        # Check for short signal
        elif (stats['cumulative_tick_delta'] > stats['tick_buy_threshold'] and
              stats['cumulative_volume_delta'] > stats['volume_buy_threshold'] and
              stats['volume_line_slope'] == 1):
            
            signal_strength = min(
                stats['cumulative_tick_delta'] / stats['tick_buy_threshold'],
                stats['cumulative_volume_delta'] / stats['volume_buy_threshold']
            )
            
            recommendation['action'] = 'SELL'
            recommendation['confidence'] = min(signal_strength, 1.0) * 100
            recommendation['reason'] = 'Both deltas above buy thresholds in uptrend'
        
        return recommendation

# Example usage and backtesting
def run_backtest(
    historical_data: List[CandleData],
    params: TradingParameters = None
) -> Dict:
    """
    Run a backtest on historical data
    """
    if params is None:
        params = TradingParameters()
    
    # Initialize scalper
    scalper = AegisScalper(params)
    
    # Track results
    trades = []
    equity_curve = []
    
    for i, candle in enumerate(historical_data):
        # Update scalper with new candle
        signal = scalper.update_with_candle(candle)
        
        # Get current price (use close for backtesting)
        current_price = candle.close
        
        # Update existing positions
        scalper.position_manager.update_positions(current_price, candle.timestamp)
        
        # Execute new trade if signal exists
        if (signal != SignalType.NEUTRAL and 
            scalper.params.enable_trading):
            
            # Check if we already have a position
            has_position = len(scalper.position_manager.positions) > 0
            
            if not has_position:
                position = scalper.execute_trade(
                    signal=signal,
                    current_price=current_price,
                    timestamp=candle.timestamp,
                    account_balance=10000.0
                )
                
                if position:
                    trades.append(position)
        
        # Record equity
        if i % 10 == 0:  # Record every 10 candles
            equity_curve.append({
                'timestamp': candle.timestamp,
                'equity': 10000 + scalper.position_manager.total_pnl,
                'drawdown': scalper.position_manager.current_drawdown
            })
    
    # Close any remaining positions at final price
    final_price = historical_data[-1].close
    for position in scalper.position_manager.positions:
        scalper.position_manager.close_position(
            position, final_price, historical_data[-1].timestamp
        )
    
    # Generate performance report
    stats = scalper.get_current_stats()
    
    performance_report = {
        'total_trades': scalper.position_manager.total_trades,
        'winning_trades': scalper.position_manager.winning_trades,
        'losing_trades': scalper.position_manager.losing_trades,
        'win_rate': (
            scalper.position_manager.winning_trades / 
            max(1, scalper.position_manager.total_trades)
        ) * 100,
        'total_pnl': scalper.position_manager.total_pnl,
        'max_drawdown': scalper.position_manager.max_drawdown,
        'profit_factor': (
            sum(p['pnl'] for p in scalper.position_manager.closed_positions if p['pnl'] > 0) /
            max(1, abs(sum(p['pnl'] for p in scalper.position_manager.closed_positions if p['pnl'] < 0)))
        ),
        'sharpe_ratio': calculate_sharpe_ratio(equity_curve),
        'signals_generated': scalper.signals_generated,
        'signals_executed': scalper.signals_executed
    }
    
    return {
        'performance': performance_report,
        'trades': scalper.position_manager.closed_positions,
        'equity_curve': equity_curve,
        'final_stats': stats
    }

def calculate_sharpe_ratio(equity_curve: List[Dict], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from equity curve"""
    if len(equity_curve) < 2:
        return 0.0
    
    returns = []
    for i in range(1, len(equity_curve)):
        ret = (equity_curve[i]['equity'] - equity_curve[i-1]['equity']) / equity_curve[i-1]['equity']
        returns.append(ret)
    
    if len(returns) == 0:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return float(sharpe)

# Main execution
if __name__ == "__main__":
    # Example usage
    print("AegisScalper - Dynamic Delta Threshold Scalping System")
    print("=" * 60)
    
    # Create sample historical data
    sample_data = []
    start_time = datetime.now() - timedelta(days=30)
    
    for i in range(1000):
        candle_time = start_time + timedelta(minutes=i*5)
        sample_data.append(CandleData(
            timestamp=candle_time,
            open=1.1000 + np.random.randn() * 0.001,
            high=1.1010 + np.random.randn() * 0.001,
            low=1.0990 + np.random.randn() * 0.001,
            close=1.1005 + np.random.randn() * 0.001,
            volume=1000 + np.random.randn() * 100,
            tick_volume=500 + np.random.randn() * 50
        ))
    
    # Configure parameters
    params = TradingParameters(
        window_size=20,
        tick_analysis_window=100,
        volume_analysis_window=100,
        tick_multiplier=2.5,
        volume_multiplier=2.5,
        enable_trading=True
    )
    
    # Run backtest
    print("Running backtest...")
    results = run_backtest(sample_data, params)
    
    # Display results
    print("\nBacktest Results:")
    print("-" * 40)
    perf = results['performance']
    
    print(f"Total Trades: {perf['total_trades']}")
    print(f"Win Rate: {perf['win_rate']:.2f}%")
    print(f"Total P&L: ${perf['total_pnl']:.2f}")
    print(f"Max Drawdown: ${perf['max_drawdown']:.2f}")
    print(f"Profit Factor: {perf['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"Signals Generated: {perf['signals_generated']}")
    print(f"Signals Executed: {perf['signals_executed']}")
    
    # Get current recommendation
    scalper = AegisScalper(params)
    for candle in sample_data[-50:]:  # Warm up with recent data
        scalper.update_with_candle(candle)
    
    recommendation = scalper.get_trading_recommendation()
    print(f"\nCurrent Recommendation: {recommendation['action']}")
    print(f"Confidence: {recommendation['confidence']:.1f}%")
    print(f"Reason: {recommendation['reason']}")