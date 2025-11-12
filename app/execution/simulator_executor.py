import random
from typing import Dict

def simulate_fill(price: float, side: str, size: int, slippage_std: float = 0.25):
    slip = random.normalvariate(0, slippage_std)
    if side == "LONG":
        fill = price + abs(slip)
    elif side == "SHORT":
        fill = price - abs(slip)
    else:
        fill = price
    return float(fill), float(slip)

def _apply_sl_tp(side: str, entry: float, next_high: float, next_low: float, tick_value: float, stop_ticks: int, tp_ticks: int):
    """
    Simplified: evaluate next bar high/low for SL/TP triggers.
    """
    if side == "LONG":
        tp_price = entry + tp_ticks
        sl_price = entry - stop_ticks
        # if low breaches SL first -> SL; else if high hits TP -> TP; else close at next close (handled by caller)
        if next_low <= sl_price:
            exit_price = sl_price
            reason = "SL"
        elif next_high >= tp_price:
            exit_price = tp_price
            reason = "TP"
        else:
            exit_price = None
            reason = "HOLD"
    else:
        tp_price = entry - tp_ticks
        sl_price = entry + stop_ticks
        if next_high >= sl_price:
            exit_price = sl_price
            reason = "SL"
        elif next_low <= tp_price:
            exit_price = tp_price
            reason = "TP"
        else:
            exit_price = None
            reason = "HOLD"
    return exit_price, reason

def execute_trade(bar, signal: Dict, config: Dict):
    if signal["side"] == "HOLD":
        return None
    size = min(config["risk"]["max_contracts"], max(1, int(config["risk"]["max_contracts"] * 1)))
    price = float(bar["close"])
    fill_price, slip = simulate_fill(price, signal["side"], size, slippage_std=0.1)
    commission = config["risk"]["contract_cost"] * size
    return {
        "timestamp": bar["timestamp"],
        "side": signal["side"],
        "size": size,
        "price": fill_price,
        "slippage": float(slip),
        "commission": float(commission),
        "status": "FILLED"
    }

def exit_with_sl_tp(entry_trade: Dict, next_bar, cfg: Dict):
    """
    Decide exit using next bar high/low for SL/TP; else exit at next close.
    """
    entry = entry_trade["price"]
    side = entry_trade["side"]
    tick_value = cfg["risk"]["tick_value"]
    stop_ticks = cfg["risk"]["stop_loss_ticks"]
    tp_ticks = cfg["risk"]["take_profit_ticks"]
    ex_price, reason = _apply_sl_tp(side, entry, float(next_bar["high"]), float(next_bar["low"]), tick_value, stop_ticks, tp_ticks)
    if ex_price is None:
        ex_price = float(next_bar["close"])
        reason = "EOD"
    if side == "LONG":
        pnl = (ex_price - entry) * entry_trade["size"] * tick_value
    else:
        pnl = (entry - ex_price) * entry_trade["size"] * tick_value
    return ex_price, pnl, reason
