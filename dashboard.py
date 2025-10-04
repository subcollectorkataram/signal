# import streamlit as st
# import yfinance as yf
# import pandas as pd
# from datetime import datetime
# import time

# # === Helper: RSI ===
# def compute_rsi(series, period=14):
#     delta = series.diff()
#     gain = delta.where(delta > 0, 0.0)
#     loss = -delta.where(delta < 0, 0.0)
#     avg_gain = gain.rolling(period).mean()
#     avg_loss = loss.rolling(period).mean()
#     rs = avg_gain / avg_loss
#     return 100 - (100 / (1 + rs))

# # === Signal Generator ===
# def generate_signal(ticker, cfg, simulate_tax=False):
#     time.sleep(1)
#     end = datetime.today().strftime("%Y-%m-%d")
#     start = "2023-01-01"

#     prices = yf.download([ticker], start=start, end=end,
#                          progress=False, auto_adjust=True)["Close"].dropna()
#     if ticker not in prices.columns:
#         return {"Symbol": ticker, "Error": "Data not available"}

#     price = prices[ticker].to_frame(name="Price")

#     # === Config variables from sidebar ===
#     sma_window = cfg["sma_window"]
#     rsi_period = cfg["rsi_period"]
#     rsi_overbought = cfg["rsi_overbought"]
#     rsi_oversold = cfg["rsi_oversold"]
#     use_atr_stop = cfg["use_atr_stop"]
#     atr_period = cfg["atr_period"]
#     atr_mult = cfg["atr_mult"]
#     boom_threshold = cfg["boom_threshold"]

#     # For demo, assume in_position and strongtrend_sma_gap
#     in_position = True
#     strongtrend_sma_gap = 0.05
#     rsi_overbought_strongtrend = 80
#     trailing_stop = None

#     # Calculate indicators
#     price["SMA200"] = price["Price"].rolling(sma_window).mean()
#     price["RSI"] = compute_rsi(price["Price"], period=rsi_period)

#     if use_atr_stop:
#         ohlc = yf.download(ticker, start=start, end=end,
#                            progress=False, auto_adjust=True)[["High","Low","Close"]]
#         hl = ohlc["High"] - ohlc["Low"]
#         hp = (ohlc["High"] - ohlc["Close"].shift()).abs()
#         lp = (ohlc["Low"] - ohlc["Close"].shift()).abs()
#         tr = pd.concat([hl, hp, lp], axis=1).max(axis=1)
#         price["ATR"] = tr.rolling(atr_period).mean()

#     # Boom quarter (per stock)
#     stock_quarterly = price["Price"].resample("QE").last()
#     quarterly_returns = stock_quarterly.pct_change(fill_method=None)
#     boom_quarters_idx = quarterly_returns[quarterly_returns > boom_threshold].index
#     boom_quarters_set = set((q.year, q.quarter) for q in boom_quarters_idx)
#     price["IsBoom"] = [(d.year, d.quarter) in boom_quarters_set for d in price.index]

#     latest = price.iloc[-1]
#     live_price = yf.Ticker(ticker).info.get("currentPrice")
#     p = live_price if live_price else latest["Price"]
#     rsi, sma, boom = latest["RSI"], latest["SMA200"], latest["IsBoom"]

#     signal = "HOLD"
#     reason = ""

#     # Adjust thresholds if tax simulation is on
#     if simulate_tax:
#         rsi_overbought += 5
#         rsi_overbought_strongtrend += 5
#         sma = sma * 0.85 if pd.notna(sma) else sma
#         reason += "[Tax Sim] "

#     # Signal logic
#     if boom:
#         if rsi > 90:
#             signal = "SELL"
#         else:
#             signal = "BUY"
#         reason += "Boom quarter"
        
#     elif pd.notna(rsi) and pd.notna(sma):
#         if (rsi < rsi_oversold) and (p > sma):
#             signal = "BUY"
#             reason += f"RSI {rsi:.1f} < {rsi_oversold} and Price > SMA200 ({p:.2f} > {sma:.2f})"        
#         elif in_position:
#             if p > sma * (1 + strongtrend_sma_gap):
#                 overbought_level = rsi_overbought_strongtrend
#             else:
#                 overbought_level = rsi_overbought
#             if (rsi > overbought_level) or (p < sma):
#                 signal = "SELL"
#                 if rsi > overbought_level and p < sma:
#                     reason += f"RSI {rsi:.1f} > {overbought_level} and Price < SMA200 ({p:.2f} < {sma:.2f})"
#                 elif rsi > overbought_level:
#                     reason += f"RSI {rsi:.1f} > {overbought_level}"
#                 elif p < sma:
#                     reason += f"Price < SMA200 ({p:.2f} < {sma:.2f})"
#             if use_atr_stop and "ATR" in latest and not pd.isna(latest["ATR"]):
#                 if trailing_stop is None:
#                     trailing_stop = p - atr_mult * latest["ATR"]
#                 else:
#                     trailing_stop = max(trailing_stop, p - atr_mult * latest["ATR"])
#                 if p < trailing_stop:
#                     signal = "SELL"
#                     reason += f" Price hit ATR trailing stop ({p:.2f} < {trailing_stop:.2f})"

#     return {
#         "Symbol": ticker,
#         "Date": latest.name.strftime("%Y-%m-%d"),
#         "Price": round(p,2),
#         "RSI": round(rsi,2) if pd.notna(rsi) else None,
#         "SMA200": round(sma,2) if pd.notna(sma) else None,
#         "ATR": round(latest.get("ATR", float("nan")),2) if use_atr_stop else None,
#         "IsBoom": bool(boom),
#         "Signal": signal,
#         "Reason": reason
#     }

# # === Streamlit UI ===
# st.set_page_config(page_title="Signal Dashboard", layout="wide")
# st.title("📈 Signal Dashboard")

# # Default config from sidebar
# default_cfg = {
#     "sma_window": st.sidebar.number_input("SMA Window", value=200),
#     "rsi_period": st.sidebar.number_input("RSI Period", value=14),
#     "boom_threshold": st.sidebar.number_input("Boom Threshold", value=0.05),
#     "rsi_overbought": st.sidebar.number_input("RSI Overbought", value=70),
#     "rsi_oversold": st.sidebar.number_input("RSI Oversold", value=30),
#     "use_atr_stop": st.sidebar.checkbox("Use ATR Stop", value=True),
#     "atr_period": st.sidebar.number_input("ATR Period", value=14),
#     "atr_mult": st.sidebar.number_input("ATR Multiplier", value=2.0)
# }

# symbols_input = st.text_area("Enter stock symbols (comma-separated)", 
#                              value="NIFTYBEES.NS")
# symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

# if st.button("Generate Signals"):
#     normal_results, tax_results = [], []
#     for sym in symbols:
#         normal_results.append(generate_signal(sym, default_cfg, simulate_tax=False))
#         tax_results.append(generate_signal(sym, default_cfg, simulate_tax=True))

#     df_normal = pd.DataFrame(normal_results)
#     df_tax = pd.DataFrame(tax_results)

#     def highlight(row):
#         color = {"BUY": "#d4f4dd", "SELL": "#f4d4d4", "HOLD": "#f4f4d4"}
#         return [f"background-color: {color.get(row['Signal'], '')}" for _ in row]

#     if not df_normal.empty:
#         st.subheader("Normal Signals")
#         st.dataframe(df_normal.style.apply(highlight, axis=1), use_container_width=True)

#     if not df_tax.empty:
#         st.subheader("Signals with 15% Tax Implications")
#         st.dataframe(df_tax.style.apply(highlight, axis=1), use_container_width=True)


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
def generate_signal(ticker, cfg, tax_mode=None):
    """
    tax_mode: None (normal), 'sma_shift', 'rsi_sma', 'atr_scale'
    """
    time.sleep(1)
    end = datetime.today().strftime("%Y-%m-%d")
    start = "2023-01-01"

    prices = yf.download([ticker], start=start, end=end,
                         progress=False, auto_adjust=True)["Close"].dropna()
    if ticker not in prices.columns:
        return {"Symbol": ticker, "Error": "Data not available"}

    price = prices[ticker].to_frame(name="Price")

    # === Config variables from sidebar ===
    sma_window = cfg["sma_window"]
    rsi_period = cfg["rsi_period"]
    rsi_overbought = cfg["rsi_overbought"]
    rsi_oversold = cfg["rsi_oversold"]
    use_atr_stop = cfg["use_atr_stop"]
    atr_period = cfg["atr_period"]
    atr_mult = cfg["atr_mult"]
    boom_threshold = cfg["boom_threshold"]

    # For demo, assume in_position and strongtrend_sma_gap
    in_position = True
    strongtrend_sma_gap = 0.05
    rsi_overbought_strongtrend = 80
    trailing_stop = None

    # Calculate indicators
    price["SMA200"] = price["Price"].rolling(sma_window).mean()
    price["RSI"] = compute_rsi(price["Price"], period=rsi_period)

    if use_atr_stop:
        ohlc = yf.download(ticker, start=start, end=end,
                           progress=False, auto_adjust=True)[["High","Low","Close"]]
        hl = ohlc["High"] - ohlc["Low"]
        hp = (ohlc["High"] - ohlc["Close"].shift()).abs()
        lp = (ohlc["Low"] - ohlc["Close"].shift()).abs()
        tr = pd.concat([hl, hp, lp], axis=1).max(axis=1)
        price["ATR"] = tr.rolling(atr_period).mean()

    # Boom quarter (per stock)
    stock_quarterly = price["Price"].resample("QE").last()
    quarterly_returns = stock_quarterly.pct_change(fill_method=None)
    boom_quarters_idx = quarterly_returns[quarterly_returns > boom_threshold].index
    boom_quarters_set = set((q.year, q.quarter) for q in boom_quarters_idx)
    price["IsBoom"] = [(d.year, d.quarter) in boom_quarters_set for d in price.index]

    latest = price.iloc[-1]
    live_price = yf.Ticker(ticker).info.get("currentPrice")
    p = live_price if live_price else latest["Price"]
    rsi, sma, boom = latest["RSI"], latest["SMA200"], latest["IsBoom"]

    signal = "HOLD"
    reason = ""

    # === Tax simulation adjustments ===
    if tax_mode == "sma_shift":
        sma = sma * 0.85 if pd.notna(sma) else sma
        reason += "[Tax SMA Shift] "
    elif tax_mode == "rsi_sma":
        reason += "[Tax RSI+SMA] "
    elif tax_mode == "atr_scale":
        reason += "[Tax ATR Scale] "

    # === Signal logic ===
    if boom:
        if rsi > 90:
            signal = "SELL"
        else:
            signal = "BUY"
        reason += "Boom quarter"
        
    elif pd.notna(rsi) and pd.notna(sma):
        # Entry
        if (rsi < rsi_oversold) and (p > sma):
            signal = "BUY"
            reason += f"RSI {rsi:.1f} < {rsi_oversold} and Price > SMA200 ({p:.2f} > {sma:.2f})"        
        # Exit
        elif in_position:
            if p > sma * (1 + strongtrend_sma_gap):
                overbought_level = rsi_overbought_strongtrend
            else:
                overbought_level = rsi_overbought

            # === Tax sim logic ===
            if tax_mode == "rsi_sma":
                # Require BOTH RSI > overbought AND Price < SMA
                if (rsi > overbought_level) and (p < sma):
                    signal = "SELL"
                    reason += f"RSI {rsi:.1f} > {overbought_level} AND Price < SMA200 ({p:.2f} < {sma:.2f})"
            else:
                # Normal or SMA shift
                if (rsi > overbought_level) or (p < sma):
                    signal = "SELL"
                    if rsi > overbought_level and p < sma:
                        reason += f"RSI {rsi:.1f} > {overbought_level} and Price < SMA200 ({p:.2f} < {sma:.2f})"
                    elif rsi > overbought_level:
                        reason += f"RSI {rsi:.1f} > {overbought_level}"
                    elif p < sma:
                        reason += f"Price < SMA200 ({p:.2f} < {sma:.2f})"

            # ATR trailing stop
            if use_atr_stop and "ATR" in latest and not pd.isna(latest["ATR"]):
                if trailing_stop is None:
                    trailing_stop = p - atr_mult * latest["ATR"]
                else:
                    trailing_stop = max(trailing_stop, p - atr_mult * latest["ATR"])
                stop_level = trailing_stop
                if tax_mode == "atr_scale":
                    stop_level = stop_level * 0.85
                if p < stop_level:
                    signal = "SELL"
                    reason += f" Price hit ATR trailing stop ({p:.2f} < {stop_level:.2f})"

    return {
        "Symbol": ticker,
        "Date": latest.name.strftime("%Y-%m-%d"),
        "Price": round(p,2),
        "RSI": round(rsi,2) if pd.notna(rsi) else None,
        "SMA200": round(sma,2) if pd.notna(sma) else None,
        "ATR": round(latest.get("ATR", float("nan")),2) if use_atr_stop else None,
        "IsBoom": bool(boom),
        "Signal": signal,
        "Reason": reason
    }

# === Streamlit UI ===
st.set_page_config(page_title="Signal Dashboard", layout="wide")
st.title("📈 Signal Dashboard")

# Default config from sidebar
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

# === Tax Sim buttons ===
st.sidebar.subheader("TAX Sim")
tax_mode = "atr_scale"  # default
if st.sidebar.button("SMA Shift (15%)"):
    tax_mode = "sma_shift"
if st.sidebar.button("RSI + SMA Confirmation"):
    tax_mode = "rsi_sma"
if st.sidebar.button("ATR Scaling (default)"):
    tax_mode = "atr_scale"

symbols_input = st.text_area("Enter stock symbols (comma-separated)", 
                             value="HDFCBANK.NS,HERITGFOOD.NS,ADANIGREEN.NS,NIFTYBEES.NS,HDFCSML250.NS")
symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

if st.button("Generate Signals"):
    normal_results, tax_results = [], []
    for sym in symbols:
        normal_results.append(generate_signal(sym, default_cfg, tax_mode=None))
        tax_results.append(generate_signal(sym, default_cfg, tax
