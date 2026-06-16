"""
SMC ICT Signal Scanner Pro - Analysis Engine
----------------------------------------------
Pure-python / pandas / numpy analysis core. Contains NO trade-execution code,
NO broker connections, and NO API-key requirements anywhere in this file.
This module only ever READS price data and PRODUCES read-only signal objects.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Basic indicators
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], min(200, max(20, len(df) - 1)))
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)
    return df


# ---------------------------------------------------------------------------
# Swing point / market structure detection
# ---------------------------------------------------------------------------

def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Mark fractal swing highs/lows: a bar whose high/low is the most
    extreme within `left` bars before and `right` bars after it."""
    df = df.copy()
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    highs = df["high"].values
    lows = df["low"].values
    for i in range(left, n - right):
        window_h = highs[i - left : i + right + 1]
        window_l = lows[i - left : i + right + 1]
        if highs[i] == window_h.max() and np.argmax(window_h) == left:
            swing_high[i] = True
        if lows[i] == window_l.min() and np.argmin(window_l) == left:
            swing_low[i] = True
    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    return df


@dataclass
class StructureState:
    trend: str = "Ranging"          # Bullish / Bearish / Ranging
    last_event: str = "None"        # BOS / CHOCH / None
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    prior_swing_high: Optional[float] = None
    prior_swing_low: Optional[float] = None


def analyze_structure(df: pd.DataFrame) -> StructureState:
    """Walk the swing points in chronological order and classify the
    current trend plus the most recent structural event (BOS/CHOCH)."""
    swings = df[(df["swing_high"]) | (df["swing_low"])].copy()
    if len(swings) < 4:
        return StructureState()

    highs = swings[swings["swing_high"]]["high"]
    lows = swings[swings["swing_low"]]["low"]

    state = StructureState()
    if len(highs) >= 2:
        state.prior_swing_high = float(highs.iloc[-2])
        state.last_swing_high = float(highs.iloc[-1])
    if len(lows) >= 2:
        state.prior_swing_low = float(lows.iloc[-2])
        state.last_swing_low = float(lows.iloc[-1])

    making_hh = state.last_swing_high is not None and state.prior_swing_high is not None \
        and state.last_swing_high > state.prior_swing_high
    making_hl = state.last_swing_low is not None and state.prior_swing_low is not None \
        and state.last_swing_low > state.prior_swing_low
    making_ll = state.last_swing_low is not None and state.prior_swing_low is not None \
        and state.last_swing_low < state.prior_swing_low
    making_lh = state.last_swing_high is not None and state.prior_swing_high is not None \
        and state.last_swing_high < state.prior_swing_high

    last_close = df["close"].iloc[-1]

    if making_hh and making_hl:
        state.trend = "Bullish"
    elif making_ll and making_lh:
        state.trend = "Bearish"
    else:
        state.trend = "Ranging"

    # BOS: price closes beyond the last swing point in the direction of
    # the existing trend. CHOCH: price closes beyond the last swing point
    # AGAINST the existing trend (first sign of reversal).
    if state.trend == "Bullish" and state.last_swing_high and last_close > state.last_swing_high:
        state.last_event = "BOS"
    elif state.trend == "Bearish" and state.last_swing_low and last_close < state.last_swing_low:
        state.last_event = "BOS"
    elif state.trend == "Bullish" and state.last_swing_low and last_close < state.last_swing_low:
        state.last_event = "CHOCH"
        state.trend = "Bearish"
    elif state.trend == "Bearish" and state.last_swing_high and last_close > state.last_swing_high:
        state.last_event = "CHOCH"
        state.trend = "Bullish"
    else:
        state.last_event = "None"

    return state


# ---------------------------------------------------------------------------
# Order blocks
# ---------------------------------------------------------------------------

def find_order_blocks(df: pd.DataFrame, lookback: int = 40):
    """Very last bearish candle before an impulsive bullish leg = bullish OB.
    Very last bullish candle before an impulsive bearish leg = bearish OB."""
    recent = df.tail(lookback).reset_index(drop=True)
    bullish_ob = None
    bearish_ob = None
    avg_range = (recent["high"] - recent["low"]).mean()
    for i in range(1, len(recent) - 1):
        candle = recent.iloc[i]
        nxt = recent.iloc[i + 1]
        body = abs(nxt["close"] - nxt["open"])
        is_down_candle = candle["close"] < candle["open"]
        is_up_candle = candle["close"] > candle["open"]
        impulsive_up = nxt["close"] > nxt["open"] and body > avg_range * 0.9
        impulsive_down = nxt["close"] < nxt["open"] and body > avg_range * 0.9
        if is_down_candle and impulsive_up:
            bullish_ob = {"high": float(candle["high"]), "low": float(candle["low"])}
        if is_up_candle and impulsive_down:
            bearish_ob = {"high": float(candle["high"]), "low": float(candle["low"])}
    return bullish_ob, bearish_ob


# ---------------------------------------------------------------------------
# Fair Value Gaps
# ---------------------------------------------------------------------------

def find_fvgs(df: pd.DataFrame, lookback: int = 40):
    recent = df.tail(lookback).reset_index(drop=True)
    bullish_fvg = None
    bearish_fvg = None
    for i in range(2, len(recent)):
        c1, c3 = recent.iloc[i - 2], recent.iloc[i]
        if c1["high"] < c3["low"]:
            bullish_fvg = {"top": float(c3["low"]), "bottom": float(c1["high"])}
        if c1["low"] > c3["high"]:
            bearish_fvg = {"top": float(c1["low"]), "bottom": float(c3["high"])}
    return bullish_fvg, bearish_fvg


# ---------------------------------------------------------------------------
# Liquidity sweeps
# ---------------------------------------------------------------------------

def find_liquidity_sweep(df: pd.DataFrame, structure: StructureState, recent_n: int = 6):
    """Looks for a wick that pierces the prior swing high/low and then
    closes back inside range within the most recent `recent_n` candles."""
    recent = df.tail(recent_n)
    swept_high = False
    swept_low = False
    if structure.prior_swing_high:
        swept_high = bool(((recent["high"] > structure.prior_swing_high) &
                            (recent["close"] < structure.prior_swing_high)).any())
    if structure.prior_swing_low:
        swept_low = bool(((recent["low"] < structure.prior_swing_low) &
                           (recent["close"] > structure.prior_swing_low)).any())
    return swept_high, swept_low


# ---------------------------------------------------------------------------
# Premium / discount zone
# ---------------------------------------------------------------------------

def premium_discount_zone(structure: StructureState, last_close: float):
    if structure.last_swing_high is None or structure.last_swing_low is None:
        return "Unknown", None
    hi, lo = structure.last_swing_high, structure.last_swing_low
    if hi <= lo:
        return "Unknown", None
    eq = (hi + lo) / 2
    zone = "Premium" if last_close > eq else "Discount"
    return zone, eq


# ---------------------------------------------------------------------------
# Support / resistance clustering
# ---------------------------------------------------------------------------

def support_resistance_levels(df: pd.DataFrame, n_levels: int = 2, lookback: int = 80):
    recent = find_swings(df.tail(lookback).reset_index(drop=True))
    res = sorted(recent[recent["swing_high"]]["high"].tail(6).tolist(), reverse=True)
    sup = sorted(recent[recent["swing_low"]]["low"].tail(6).tolist(), reverse=True)
    return sup[:n_levels] if sup else [], res[:n_levels] if res else []


# ---------------------------------------------------------------------------
# Price action patterns (evaluated on the most recently CLOSED candle)
# ---------------------------------------------------------------------------

def price_action_signal(df: pd.DataFrame):
    if len(df) < 3:
        return None
    c0, c1 = df.iloc[-1], df.iloc[-2]
    rng = c0["high"] - c0["low"]
    if rng <= 0:
        return None
    body = abs(c0["close"] - c0["open"])
    upper_wick = c0["high"] - max(c0["close"], c0["open"])
    lower_wick = min(c0["close"], c0["open"]) - c0["low"]

    # Bullish pin bar / rejection
    if lower_wick > body * 2 and lower_wick > rng * 0.5 and c0["close"] > c0["open"]:
        return "bullish"
    # Bearish pin bar / rejection
    if upper_wick > body * 2 and upper_wick > rng * 0.5 and c0["close"] < c0["open"]:
        return "bearish"
    # Bullish engulfing
    if c1["close"] < c1["open"] and c0["close"] > c0["open"] \
            and c0["close"] >= c1["open"] and c0["open"] <= c1["close"]:
        return "bullish"
    # Bearish engulfing
    if c1["close"] > c1["open"] and c0["close"] < c0["open"] \
            and c0["open"] >= c1["close"] and c0["close"] <= c1["open"]:
        return "bearish"
    return None


# ---------------------------------------------------------------------------
# Scoring engine  (weights sum to 100, matches spec exactly)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "market_structure": 20,
    "trend_alignment": 15,
    "order_block": 15,
    "liquidity_sweep": 10,
    "fvg": 10,
    "rsi": 10,
    "support_resistance": 10,
    "price_action": 10,
}


def classify_setup(score: float) -> str:
    if score >= 95:
        return "Elite Setup"
    if score >= 85:
        return "Strong Setup"
    if score >= 75:
        return "Moderate Setup"
    return "NO TRADE"


# ---------------------------------------------------------------------------
# Full multi-timeframe signal generation
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    symbol: str
    timeframe: str
    direction: str               # BUY / SELL / NO TRADE
    confidence: float
    setup_class: str
    trend_h4: str
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    rr: str = "-"
    support: list = field(default_factory=list)
    resistance: list = field(default_factory=list)
    reason: str = ""
    warning: str = ""
    last_price: Optional[float] = None
    components: dict = field(default_factory=dict)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    df = find_swings(df)
    return df


def generate_signal(symbol: str, h4: pd.DataFrame, h1: pd.DataFrame, m15: pd.DataFrame) -> Signal:
    h4, h1, m15 = _prep(h4), _prep(h1), _prep(m15)

    struct_h4 = analyze_structure(h4)
    struct_h1 = analyze_structure(h1)
    struct_m15 = analyze_structure(m15)

    last = m15.iloc[-1]
    last_close = float(last["close"])

    bull_ob, bear_ob = find_order_blocks(m15)
    bull_fvg, bear_fvg = find_fvgs(m15)
    swept_high, swept_low = find_liquidity_sweep(m15, struct_h1)
    zone, eq = premium_discount_zone(struct_h1, last_close)
    support, resistance = support_resistance_levels(h1)
    pa = price_action_signal(m15)

    rsi_val = float(last["rsi14"])
    ema50_val = float(last["ema50"])

    bias = struct_h4.trend  # Directional bias from H4

    bullish_case = (
        struct_h1.trend == "Bullish"
        and bull_ob is not None
        and swept_low
        and rsi_val > 50
        and last_close > ema50_val
    )
    bearish_case = (
        struct_h1.trend == "Bearish"
        and bear_ob is not None
        and swept_high
        and rsi_val < 50
        and last_close < ema50_val
    )

    direction = "NO TRADE"
    if bullish_case and not bearish_case:
        direction = "BUY"
    elif bearish_case and not bullish_case:
        direction = "SELL"

    # ---- scoring ----
    comp = {k: 0 for k in WEIGHTS}
    if direction == "BUY":
        comp["market_structure"] = WEIGHTS["market_structure"] if struct_h1.trend == "Bullish" else WEIGHTS["market_structure"] * 0.4
        comp["trend_alignment"] = WEIGHTS["trend_alignment"] if bias == "Bullish" else WEIGHTS["trend_alignment"] * 0.3
        comp["order_block"] = WEIGHTS["order_block"] if bull_ob else 0
        comp["liquidity_sweep"] = WEIGHTS["liquidity_sweep"] if swept_low else 0
        comp["fvg"] = WEIGHTS["fvg"] if bull_fvg else 0
        comp["rsi"] = WEIGHTS["rsi"] if rsi_val > 50 else WEIGHTS["rsi"] * 0.3
        comp["support_resistance"] = WEIGHTS["support_resistance"] if zone == "Discount" else WEIGHTS["support_resistance"] * 0.4
        comp["price_action"] = WEIGHTS["price_action"] if pa == "bullish" else 0
    elif direction == "SELL":
        comp["market_structure"] = WEIGHTS["market_structure"] if struct_h1.trend == "Bearish" else WEIGHTS["market_structure"] * 0.4
        comp["trend_alignment"] = WEIGHTS["trend_alignment"] if bias == "Bearish" else WEIGHTS["trend_alignment"] * 0.3
        comp["order_block"] = WEIGHTS["order_block"] if bear_ob else 0
        comp["liquidity_sweep"] = WEIGHTS["liquidity_sweep"] if swept_high else 0
        comp["fvg"] = WEIGHTS["fvg"] if bear_fvg else 0
        comp["rsi"] = WEIGHTS["rsi"] if rsi_val < 50 else WEIGHTS["rsi"] * 0.3
        comp["support_resistance"] = WEIGHTS["support_resistance"] if zone == "Premium" else WEIGHTS["support_resistance"] * 0.4
        comp["price_action"] = WEIGHTS["price_action"] if pa == "bearish" else 0

    score = sum(comp.values())

    warning = ""
    against_htf = (direction == "BUY" and bias == "Bearish") or (direction == "SELL" and bias == "Bullish")
    if direction != "NO TRADE" and against_htf:
        warning = "Trade is against higher timeframe trend. Exercise caution."
        score *= 0.85

    score = round(min(score, 100), 1)

    if score < 75:
        direction = "NO TRADE"

    setup_class = classify_setup(score) if direction != "NO TRADE" else "NO TRADE"

    sig = Signal(
        symbol=symbol,
        timeframe="M15 entry / H1 confirm / H4 bias",
        direction=direction,
        confidence=score,
        setup_class=setup_class,
        trend_h4=bias,
        support=support,
        resistance=resistance,
        last_price=last_close,
        components=comp,
        warning=warning,
    )

    if direction in ("BUY", "SELL"):
        atr_val = float(last["atr14"])
        if direction == "BUY":
            structure_low = struct_m15.last_swing_low or (last_close - atr_val * 1.5)
            entry = last_close
            stop = min(structure_low, entry - atr_val * 0.5)
            risk = entry - stop
            sig.entry, sig.stop_loss = entry, stop
            sig.tp1, sig.tp2, sig.tp3 = entry + risk * 2, entry + risk * 3, entry + risk * 5
        else:
            structure_high = struct_m15.last_swing_high or (last_close + atr_val * 1.5)
            entry = last_close
            stop = max(structure_high, entry + atr_val * 0.5)
            risk = stop - entry
            sig.entry, sig.stop_loss = entry, stop
            sig.tp1, sig.tp2, sig.tp3 = entry - risk * 2, entry - risk * 3, entry - risk * 5
        sig.rr = "1:2 / 1:3 / 1:5"

        reasons = []
        reasons.append(f"H4 directional bias is {bias}.")
        reasons.append(f"H1 structure shows a {struct_h1.trend.lower()} {struct_h1.last_event if struct_h1.last_event != 'None' else 'structure'}.")
        if (direction == "BUY" and swept_low) or (direction == "SELL" and swept_high):
            reasons.append("Liquidity was swept just before the move, suggesting stops were cleared.")
        if (direction == "BUY" and bull_ob) or (direction == "SELL" and bear_ob):
            reasons.append("Price is reacting from a fresh order block on the M15 chart.")
        if (direction == "BUY" and bull_fvg) or (direction == "SELL" and bear_fvg):
            reasons.append("An unfilled fair value gap supports continuation in this direction.")
        reasons.append(f"RSI(14) is at {rsi_val:.1f}, price is {zone.lower()} relative to the recent range.")
        sig.reason = " ".join(reasons)
        sig.warning = sig.warning or "Standard market risk applies. Past structure does not guarantee future price behavior."
    else:
        sig.reason = "Conditions for a high-probability BUY or SELL setup were not met (score below 75 or conflicting confirmations)."
        sig.warning = "No trade is the correct decision here. Capital preservation takes priority over trade frequency."

    return sig
  
