"""
SMC ICT Signal Scanner Pro
--------------------------
Read-only market analysis tool. This application:
  - NEVER places trades
  - NEVER connects to a broker account
  - NEVER requires an API key from the user
It only reads public market price data to compute ICT / SMC style signals.
"""

import io
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from data_engine import (
    add_indicators, find_swings, analyze_structure, generate_signal,
    find_order_blocks, find_fvgs, classify_setup,
)

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SMC ICT Signal Scanner Pro",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_ORANGE = "#FF8C00"
ROYAL_BLUE = "#4169E1"
BLACK = "#000000"
WHITE = "#FFFFFF"
CARD = "#161616"
GREEN = "#22C55E"
RED = "#EF4444"
GRAY = "#9CA3AF"

st.markdown(f"""
<style>
.stApp {{ background-color: {BLACK}; color: {WHITE}; }}
section[data-testid="stSidebar"] {{ background-color: #0a0a0a; border-right: 1px solid {DARK_ORANGE}; }}
h1, h2, h3 {{ color: {WHITE} !important; }}
.smc-card {{
    background-color: {CARD}; border: 1px solid #2a2a2a; border-left: 4px solid {DARK_ORANGE};
    border-radius: 10px; padding: 16px 18px; margin-bottom: 14px;
}}
.smc-pill {{
    display: inline-block; padding: 3px 12px; border-radius: 20px; font-weight: 700;
    font-size: 0.8rem; letter-spacing: 0.5px;
}}
.pill-buy {{ background-color: rgba(34,197,94,0.15); color: {GREEN}; border: 1px solid {GREEN}; }}
.pill-sell {{ background-color: rgba(239,68,68,0.15); color: {RED}; border: 1px solid {RED}; }}
.pill-no {{ background-color: rgba(156,163,175,0.15); color: {GRAY}; border: 1px solid {GRAY}; }}
.smc-badge {{ color: {DARK_ORANGE}; font-weight: 700; }}
.smc-sub {{ color: {ROYAL_BLUE}; font-weight: 600; }}
hr {{ border-color: #2a2a2a; }}
div[data-testid="stMetricValue"] {{ color: {DARK_ORANGE}; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Supported markets
# ---------------------------------------------------------------------------

FOREX_PAIRS = ["NZDUSD", "AUDCHF", "AUDUSD", "USDCHF", "AUDCAD",
               "USDJPY", "GBPUSD", "NZDJPY", "CADCHF", "EURUSD"]

CRYPTO_PAIRS = {"LTCUSD": "LTC-USD", "XRPUSD": "XRP-USD", "BCHUSD": "BCH-USD"}

SYNTHETIC_INDICES = [
    "Boom 300", "Boom 500", "Boom 1000",
    "Crash 300", "Crash 500", "Crash 1000",
    "Volatility 10", "Volatility 25", "Volatility 50", "Volatility 75", "Volatility 100",
]

ALL_SYMBOLS = FOREX_PAIRS + list(CRYPTO_PAIRS.keys()) + SYNTHETIC_INDICES

# ---------------------------------------------------------------------------
# Data fetching  (read-only public market data — no keys, no broker accounts)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=20, show_spinner=False)
def fetch_crypto_live_price(yf_symbol: str) -> float:
    """Last traded price via yfinance (1-minute bar, most recent close)."""
    import yfinance as yf
    ticker = yf.Ticker(yf_symbol)
    hist = ticker.history(period="1d", interval="1m")
    if hist is None or hist.empty:
        raise ValueError(f"No live price data for {yf_symbol}")
    return float(hist["Close"].iloc[-1])


# yfinance interval mapping: our internal key -> yfinance interval string
_YF_INTERVAL = {"4h": "1h", "1h": "1h", "15m": "15m"}
# period to request for enough bars
_YF_PERIOD   = {"4h": "60d", "1h": "60d", "15m": "30d"}


@st.cache_data(ttl=50, show_spinner=False)
def fetch_crypto_ohlc(yf_symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """Fetch OHLCV data for a crypto pair using yfinance.
    `interval` accepts '4h', '1h', or '15m'."""
    import yfinance as yf
    yf_interval = _YF_INTERVAL.get(interval, interval)
    period = _YF_PERIOD.get(interval, "60d")
    data = yf.Ticker(yf_symbol).history(period=period, interval=yf_interval)
    if data is None or data.empty:
        raise ValueError(f"No OHLC data returned for {yf_symbol} @ {yf_interval}")
    data = data.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                 "Close": "close", "Volume": "volume"})
    df = data[["open", "high", "low", "close", "volume"]].copy()
    # For 4h interval, yfinance only goes to 1h — resample up
    if interval == "4h":
        df = resample_ohlc(df, "4h")
    return df.tail(limit)


@st.cache_data(ttl=50, show_spinner=False)
def fetch_forex_ohlc(pair: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    ticker = f"{pair}=X"
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data is None or data.empty:
        raise ValueError(f"No data returned for {ticker}")
    data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    return data[["open", "high", "low", "close", "volume"]]


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["open", "high", "low", "close", "volume"]
    return out.dropna()


@st.cache_data(ttl=50, show_spinner=False)
def fetch_synthetic_ohlc(index_name: str, bars: int, seed_bucket: int) -> pd.DataFrame:
    """
    Boom/Crash/Volatility indices are proprietary to a single broker (Deriv)
    and have no independent public data feed. To support scanning these
    symbols WITHOUT connecting to any broker account or requiring an API
    key, this generates a clearly-labeled simulated price series whose
    volatility/spike behaviour mirrors the public description of each
    index family. It is for structure/pattern demonstration only and is
    NOT a live quote.
    """
    rng = np.random.default_rng(seed_bucket + abs(hash(index_name)) % 10000)
    is_boom = index_name.startswith("Boom")
    is_crash = index_name.startswith("Crash")
    vol_level = 0.15
    if "Volatility" in index_name:
        vol_level = float(index_name.split()[-1]) / 100.0

    price = 1000.0
    rows = []
    for _ in range(bars):
        shock = rng.normal(0, vol_level)
        spike = 0.0
        if is_boom and rng.random() < 0.012:
            spike = abs(rng.normal(4, 1.5)) * vol_level * 10
        if is_crash and rng.random() < 0.012:
            spike = -abs(rng.normal(4, 1.5)) * vol_level * 10
        o = price
        price = max(price + shock + spike, 1.0)
        c = price
        h = max(o, c) + abs(rng.normal(0, vol_level * 0.3))
        l = min(o, c) - abs(rng.normal(0, vol_level * 0.3))
        rows.append((o, h, l, c, 0))
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=bars, freq="15min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def get_multi_timeframe(symbol: str):
    """Returns (h4, h1, m15, live_price, is_simulated)."""
    if symbol in CRYPTO_PAIRS:
        bsym = CRYPTO_PAIRS[symbol]
        h4 = fetch_crypto_ohlc(bsym, "4h", 200)
        h1 = fetch_crypto_ohlc(bsym, "1h", 300)
        m15 = fetch_crypto_ohlc(bsym, "15m", 300)
        live = fetch_crypto_live_price(bsym)
        return h4, h1, m15, live, False

    if symbol in FOREX_PAIRS:
        h1_raw = fetch_forex_ohlc(symbol, "60m", "60d")
        m15 = fetch_forex_ohlc(symbol, "15m", "30d")
        h4 = resample_ohlc(h1_raw, "4h")
        live = float(m15["close"].iloc[-1])
        return h4, h1_raw, m15, live, False

    # synthetic indices
    seed_bucket = int(time.time() // 60)  # changes once per minute
    m15 = fetch_synthetic_ohlc(symbol, 300, seed_bucket)
    h1 = resample_ohlc(m15, "1h")
    h4 = resample_ohlc(m15, "4h")
    if len(h1) < 10:
        h1 = fetch_synthetic_ohlc(symbol, 300, seed_bucket).iloc[::4]
    if len(h4) < 10:
        h4 = fetch_synthetic_ohlc(symbol, 300, seed_bucket).iloc[::16]
    live = float(m15["close"].iloc[-1])
    return h4, h1, m15, live, True


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []
if "last_scan" not in st.session_state:
    st.session_state.last_scan = None
if "scan_results" not in st.session_state:
    st.session_state.scan_results = {}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"<h2 style='color:{DARK_ORANGE};'>📡 SMC ICT Scanner</h2>", unsafe_allow_html=True)
    st.caption("Signals only. No trade execution. No broker connection. No API key required.")
    st.markdown("---")

    st.markdown("**Market Selection**")
    sel_forex = st.multiselect("Forex Pairs", FOREX_PAIRS, default=["EURUSD", "AUDUSD"])
    sel_crypto = st.multiselect("Crypto", list(CRYPTO_PAIRS.keys()), default=["LTCUSD"])
    sel_synth = st.multiselect("Synthetic Indices", SYNTHETIC_INDICES, default=[])

    st.markdown("**Timeframe**")
    st.caption("H4 = bias · H1 = confirmation · M15 = entry timing (always analyzed together)")

    st.markdown("---")
    auto_scan = st.toggle("Auto-scan every 1 minute", value=True)
    if auto_scan and not AUTOREFRESH_AVAILABLE:
        st.warning("Install `streamlit-autorefresh` (see requirements.txt) to enable automatic 1-minute scanning. Falling back to manual refresh.")

    scan_clicked = st.button("🔍 Scan Now", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**Chart Screenshot Analyzer**")
    uploaded_img = st.file_uploader("Upload a chart screenshot", type=["png", "jpg", "jpeg"])

    st.markdown("---")
    st.caption("⚠️ Educational signal tool only. Not financial advice. Synthetic indices use simulated data (see notes below).")

selected_symbols = sel_forex + sel_crypto + sel_synth

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(f"<h1>SMC <span style='color:{DARK_ORANGE};'>ICT</span> Signal Scanner <span style='color:{ROYAL_BLUE};'>Pro</span></h1>", unsafe_allow_html=True)
st.caption("Institutional-style ICT / Smart Money Concepts market scanner — analysis only, never executes trades.")

if AUTOREFRESH_AVAILABLE and auto_scan:
    st_autorefresh(interval=60_000, key="auto_refresh_tick")

run_scan = scan_clicked or (auto_scan and True)  # autorefresh reruns the script every 60s

# ---------------------------------------------------------------------------
# Run the scan
# ---------------------------------------------------------------------------

def run_full_scan(symbols):
    results = {}
    for sym in symbols:
        try:
            h4, h1, m15, live_price, simulated = get_multi_timeframe(sym)
            sig = generate_signal(sym, h4, h1, m15)
            sig.last_price = live_price
            results[sym] = {"signal": sig, "h4": h4, "h1": h1, "m15": m15, "simulated": simulated, "error": None}
        except Exception as e:
            results[sym] = {"signal": None, "error": str(e)}
    return results


if selected_symbols and (run_scan or not st.session_state.scan_results):
    with st.spinner("Scanning markets for ICT/SMC structure..."):
        st.session_state.scan_results = run_full_scan(selected_symbols)
        st.session_state.last_scan = datetime.now(timezone.utc)
        for sym, data in st.session_state.scan_results.items():
            sig = data.get("signal")
            if sig and sig.direction != "NO TRADE":
                st.session_state.history.insert(0, {
                    "time": st.session_state.last_scan.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "symbol": sym, "direction": sig.direction,
                    "confidence": sig.confidence, "class": sig.setup_class,
                })
        st.session_state.history = st.session_state.history[:60]

results = st.session_state.scan_results

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Markets Scanned", len(results) if results else 0)
with col_b:
    active = sum(1 for d in results.values() if d.get("signal") and d["signal"].direction != "NO TRADE") if results else 0
    st.metric("Active Signals", active)
with col_c:
    elite = sum(1 for d in results.values() if d.get("signal") and d["signal"].setup_class == "Elite Setup") if results else 0
    st.metric("Elite Setups", elite)
with col_d:
    last_scan_str = st.session_state.last_scan.strftime("%H:%M:%S UTC") if st.session_state.last_scan else "—"
    st.metric("Last Scan", last_scan_str)

st.markdown("---")

if not selected_symbols:
    st.info("Select at least one market in the sidebar, then click **Scan Now**.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_scanner, tab_signals, tab_structure, tab_history, tab_shot = st.tabs(
    ["📊 Market Scanner", "🎯 Active Signals", "🧭 Structure Visualization", "🕓 Signal History", "🖼️ Screenshot Analyzer"]
)

# ---- Market Scanner table ----
with tab_scanner:
    rows = []
    for sym, data in results.items():
        if data.get("error"):
            rows.append({"Symbol": sym, "Price": "—", "Trend (H4)": "—", "Signal": "ERROR", "Confidence": "—", "Setup": data["error"][:60]})
            continue
        sig = data["signal"]
        price = sig.last_price
        decimals = 5 if (sym in FOREX_PAIRS and "JPY" not in sym) else (3 if "JPY" in sym else 2)
        rows.append({
            "Symbol": sym + (" 🧪" if data.get("simulated") else ""),
            "Price": f"{price:,.{decimals}f}" if price is not None else "—",
            "Trend (H4)": sig.trend_h4,
            "Signal": sig.direction,
            "Confidence": f"{sig.confidence:.0f}%",
            "Setup": sig.setup_class,
        })
    if rows:
        df_scan = pd.DataFrame(rows)
        st.dataframe(df_scan, use_container_width=True, hide_index=True)
        st.caption("🧪 = Synthetic index using simulated price data (no independent public feed exists without connecting to a broker).")
    else:
        st.info("No results yet — click Scan Now.")

# ---- Active signals (full output format) ----
with tab_signals:
    any_active = False
    for sym, data in results.items():
        if data.get("error") or not data.get("signal"):
            continue
        sig = data["signal"]
        if sig.direction == "NO TRADE":
            continue
        any_active = True
        pill_class = "pill-buy" if sig.direction == "BUY" else "pill-sell"
        decimals = 5 if (sym in FOREX_PAIRS and "JPY" not in sym) else (3 if "JPY" in sym else 2)

        def fmt(v):
            return f"{v:,.{decimals}f}" if v is not None else "—"

        meter_color = GREEN if sig.confidence >= 95 else ROYAL_BLUE if sig.confidence >= 85 else DARK_ORANGE if sig.confidence >= 75 else RED

        st.markdown(f"""
        <div class="smc-card">
          <h3>{sym} <span class="smc-pill {pill_class}">{sig.direction}</span>
              <span style="color:{GRAY}; font-size:0.85rem;">{sig.setup_class}</span></h3>
          <p><b>TIMEFRAME:</b> {sig.timeframe}</p>
          <p><b>ENTRY:</b> {fmt(sig.entry)} &nbsp;|&nbsp; <b>STOP LOSS:</b> {fmt(sig.stop_loss)}</p>
          <p><b>TAKE PROFIT 1:</b> {fmt(sig.tp1)} &nbsp;|&nbsp; <b>TP2:</b> {fmt(sig.tp2)} &nbsp;|&nbsp; <b>TP3:</b> {fmt(sig.tp3)}</p>
          <p><b>RISK:REWARD:</b> {sig.rr}</p>
          <p><b>CONFIDENCE:</b> <span style="color:{meter_color}; font-weight:700;">{sig.confidence:.0f}%</span></p>
          <p><b>TREND (H4 bias):</b> {sig.trend_h4}</p>
          <p><b>SUPPORT LEVELS:</b> {", ".join(fmt(x) for x in sig.support) if sig.support else "—"}</p>
          <p><b>RESISTANCE LEVELS:</b> {", ".join(fmt(x) for x in sig.resistance) if sig.resistance else "—"}</p>
          <p><b>TRADE REASON:</b><br>{sig.reason}</p>
          <p style="color:{DARK_ORANGE};"><b>RISK WARNING:</b> {sig.warning}</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(int(sig.confidence), 100))

    if not any_active:
        st.info("No setups currently meet the 75% confidence threshold. NO TRADE is the conservative, correct call right now.")

# ---- Structure visualization ----
with tab_structure:
    valid_syms = [s for s, d in results.items() if not d.get("error")]
    if valid_syms:
        chosen = st.selectbox("Select symbol", valid_syms)
        d = results[chosen]
        m15 = add_indicators(d["m15"])
        m15 = find_swings(m15)
        bull_ob, bear_ob = find_order_blocks(m15)
        bull_fvg, bear_fvg = find_fvgs(m15)

        fig = go.Figure(data=[go.Candlestick(
            x=m15.index, open=m15["open"], high=m15["high"], low=m15["low"], close=m15["close"],
            increasing_line_color=GREEN, decreasing_line_color=RED, name=chosen,
        )])
        fig.add_trace(go.Scatter(x=m15.index, y=m15["ema20"], line=dict(color=DARK_ORANGE, width=1.3), name="EMA 20"))
        fig.add_trace(go.Scatter(x=m15.index, y=m15["ema50"], line=dict(color=ROYAL_BLUE, width=1.3), name="EMA 50"))

        if bull_ob:
            fig.add_hrect(y0=bull_ob["low"], y1=bull_ob["high"], fillcolor=GREEN, opacity=0.15, line_width=0, annotation_text="Bullish OB")
        if bear_ob:
            fig.add_hrect(y0=bear_ob["low"], y1=bear_ob["high"], fillcolor=RED, opacity=0.15, line_width=0, annotation_text="Bearish OB")
        if bull_fvg:
            fig.add_hrect(y0=bull_fvg["bottom"], y1=bull_fvg["top"], fillcolor=ROYAL_BLUE, opacity=0.12, line_width=0, annotation_text="Bullish FVG")
        if bear_fvg:
            fig.add_hrect(y0=bear_fvg["bottom"], y1=bear_fvg["top"], fillcolor=DARK_ORANGE, opacity=0.12, line_width=0, annotation_text="Bearish FVG")

        swing_highs = m15[m15["swing_high"]]
        swing_lows = m15[m15["swing_low"]]
        fig.add_trace(go.Scatter(x=swing_highs.index, y=swing_highs["high"], mode="markers",
                                  marker=dict(color=RED, size=7, symbol="triangle-down"), name="Swing High"))
        fig.add_trace(go.Scatter(x=swing_lows.index, y=swing_lows["low"], mode="markers",
                                  marker=dict(color=GREEN, size=7, symbol="triangle-up"), name="Swing Low"))

        fig.update_layout(
            template="plotly_dark", paper_bgcolor=BLACK, plot_bgcolor=BLACK,
            height=560, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)
        if d.get("simulated"):
            st.caption("🧪 Simulated price series for this synthetic index — see sidebar note.")
    else:
        st.info("No structure data available yet — run a scan first.")

# ---- Signal history ----
with tab_history:
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No signals have crossed the 75% confidence threshold yet this session.")

# ---- Screenshot analyzer ----
with tab_shot:
    st.markdown("Upload a chart screenshot for a heuristic structure read. This is a lightweight pixel-based "
                 "heuristic (no external AI vision API is called, since this app never requires an API key) — "
                 "treat it as a rough supplement to the live scanner above, not a primary signal source.")
    if uploaded_img is not None:
        from PIL import Image
        img = Image.open(uploaded_img).convert("RGB")
        w, h = img.size
        st.image(img, caption=f"Uploaded chart ({w}x{h})", use_container_width=True)

        if w < 400 or h < 250:
            st.error("NO TRADE - Additional timeframe or clearer screenshot required.")
        else:
            arr = np.asarray(img).astype(float)
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            green_mask = (g > r + 15) & (g > b + 15)
            red_mask = (r > g + 15) & (r > b + 15)
            green_px, red_px = green_mask.sum(), red_mask.sum()
            total_candle_px = green_px + red_px
            contrast = arr.std()

            if total_candle_px < (w * h * 0.01) or contrast < 18:
                st.error("NO TRADE - Additional timeframe or clearer screenshot required.")
            else:
                left_green = green_mask[:, : w // 2].sum()
                right_green = green_mask[:, w // 2 :].sum()
                left_red = red_mask[:, : w // 2].sum()
                right_red = red_mask[:, w // 2 :].sum()
                left_bias = left_green - left_red
                right_bias = right_green - right_red
                trend_guess = "Bullish" if right_bias > left_bias else "Bearish" if right_bias < left_bias else "Ranging"
                bull_ratio = green_px / max(total_candle_px, 1)

                st.markdown(f"""
                <div class="smc-card">
                  <p><b>PAIR:</b> Unknown (auto-detection from a screenshot needs a visible price/symbol label)</p>
                  <p><b>TIMEFRAME:</b> Unknown (not labeled in image — confirm manually)</p>
                  <p><b>DETECTED TREND BIAS:</b> {trend_guess} &nbsp;(bullish candle ratio: {bull_ratio*100:.0f}%)</p>
                  <p><b>SIGNAL:</b> NO TRADE</p>
                  <p style="color:{DARK_ORANGE};"><b>RISK WARNING:</b> Screenshot analysis cannot reliably confirm order blocks,
                  FVGs, or liquidity sweeps from pixels alone. Use the live Market Scanner tab for confidence-scored signals,
                  and treat this tab as a directional sanity-check only.</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.caption("No screenshot uploaded yet.")

st.markdown("---")
st.caption(
    "⚠️ RISK DISCLOSURE: This tool provides educational market structure analysis only. It does not place trades, "
    "manage funds, or connect to any broker. Trading forex, crypto, and synthetic indices carries a high risk of "
    "loss. Past structure does not guarantee future price behavior. Always do your own due diligence."
)
