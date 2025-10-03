import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time

# === Helper: RSI ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# === Signal Generator ===
def generate_signal(ticker, bench, cfg):
    time.sleep(1)
    end = datetime.today().strftime("%Y-%m-%d")
    start = "2023-01-01"

    prices = yf.download([ticker, bench], start=start, end=end,
                         progress=False, auto_adjust=True)["Close"].dropna(how="all")
    if ticker not in prices.columns or bench not in prices.columns:
        return {"Symbol": ticker, "Error": "Data not available"}

    price = prices[ticker].to_frame(name="Price")
    bench_series = prices[bench]

    price["SMA200"] = price["Price"].rolling(cfg["sma_window"]).mean()
    price["RSI"] = compute_rsi(price["Price"], period=cfg["rsi_period"])

    if cfg["use_atr_stop"]:
        ohlc = yf.download(ticker, start=start, end=end,
                           progress=False, auto_adjust=True)[["High","Low","Close"]]
        hl = ohlc["High"] - ohlc["Low"]
        hp = (ohlc["High"] - ohlc["Close"].shift()).abs()
        lp = (ohlc["Low"] - ohlc["Close"].shift()).abs()
        tr = pd.concat([hl, hp, lp], axis=1).max(axis=1)
        price["ATR"] = tr.rolling(cfg["atr_period"]).mean()

    bench_quarterly = bench_series.resample("QE").last()
    quarterly_returns = bench_quarterly.pct_change(fill_method=None)
    boom_quarters_idx = quarterly_returns[quarterly_returns > cfg["boom_threshold"]].index
    boom_quarters_set = set((q.year, q.quarter) for q in boom_quarters_idx)
    price["IsBoom"] = [(d.year, d.quarter) in boom_quarters_set for d in price.index]

    latest = price.iloc[-1]
    live_price = yf.Ticker(ticker).info.get("currentPrice")
    if live_price:
        p = live_price
    else:
        p = latest["Price"]
    rsi, sma, boom = latest["RSI"], latest["SMA200"], latest["IsBoom"]

    signal = "HOLD"
    reason = ""


    if boom:
        signal = "BUY"
        reason = "Boom quarter"
    
    elif pd.notna(rsi) and pd.notna(sma):
        # Entry logic
        if (rsi < rsi_oversold) and (p > sma):
            signal = "BUY"
            reason = f"RSI {rsi:.1f} < {rsi_oversold} and Price > SMA200 ({p:.2f} > {sma:.2f})"
    
        # Exit logic
        elif in_position:
            # Trend-aware RSI threshold
            if p > sma * (1 + strongtrend_sma_gap):
                overbought_level = rsi_overbought_strongtrend
            else:
                overbought_level = rsi_overbought
    
            if (rsi > overbought_level) or (p < sma):
                signal = "SELL"
                if rsi > overbought_level and p < sma:
                    reason = f"RSI {rsi:.1f} > {overbought_level} and Price < SMA200 ({p:.2f} < {sma:.2f})"
                elif rsi > overbought_level:
                    reason = f"RSI {rsi:.1f} > {overbought_level}"
                elif p < sma:
                    reason = f"Price < SMA200 ({p:.2f} < {sma:.2f})"
    
            # ATR trailing stop
            if use_atr_stop and not pd.isna(row.get("ATR")):
                if trailing_stop is None:
                    trailing_stop = p - atr_mult * row["ATR"]
                else:
                    trailing_stop = max(trailing_stop, p - atr_mult * row["ATR"])
                if p < trailing_stop:
                    signal = "SELL"
                    reason = f"Price hit ATR trailing stop ({p:.2f} < {trailing_stop:.2f})"

    return {
        "Symbol": ticker,
        "Date": latest.name.strftime("%Y-%m-%d"),
        "Price": round(p,2),
        "RSI": round(rsi,2) if pd.notna(rsi) else None,
        "SMA200": round(sma,2) if pd.notna(sma) else None,
        "ATR": round(latest.get("ATR", float("nan")),2) if cfg["use_atr_stop"] else None,
        "IsBoom": bool(boom),
        "Signal": signal,
        "Reason": reason
    }

# === Streamlit UI ===
st.set_page_config(page_title="Signal Dashboard", layout="wide")
st.title("📈 Signal Dashboard")

# Default config
default_cfg = {
    "sma_window": st.sidebar.number_input("SMA Window", value=200),
    "rsi_period": st.sidebar.number_input("RSI Period", value=14),
    "boom_threshold": st.sidebar.number_input("Boom Threshold", value=0.05),
    "rsi_overbought": st.sidebar.number_input("RSI Overbought", value=70),
    "rsi_oversold": st.sidebar.number_input("RSI Oversold", value=30),
    "use_atr_stop": st.sidebar.checkbox("Use ATR Stop", value=True),
    "atr_period": st.sidebar.number_input("ATR Period", value=14),
    "atr_mult": st.sidebar.number_input("ATR Multiplier", value=2.0)
}

benchmark = st.sidebar.text_input("Benchmark Symbol", value="NIFTYBEES.NS")
symbols_input = st.text_area("Enter stock symbols (comma-separated)", value="HDFCBANK.NS,HERITGFOOD.NS,ADANIGREEN.NS,NIFTYBEES.NS,HDFCSML250.NS")
symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

if st.button("Generate Signals"):
    results = []
    for sym in symbols:
        res = generate_signal(sym, benchmark, default_cfg)
        results.append(res)

    df = pd.DataFrame(results)
    if not df.empty:
        def highlight(row):
            color = {"BUY": "#d4f4dd", "SELL": "#f4d4d4", "HOLD": "#f4f4d4"}
            return [f"background-color: {color.get(row['Signal'], '')}" for _ in row]

        st.dataframe(df.style.apply(highlight, axis=1), use_container_width=True)
    else:
        st.warning("No data returned. Check symbols or benchmark.")


