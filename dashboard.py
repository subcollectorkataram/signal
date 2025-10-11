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
import numpy as np

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
def generate_signal(ticker, cfg, sim_mode="Normal"):
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

    # Demo placeholders
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

    # Boom quarter detection
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

    # === Apply TAX Simulation Mode Adjustments ===
    if sim_mode == "SMA Shift (15%)":
        rsi_overbought += 5
        rsi_overbought_strongtrend += 5
        sma = sma * 0.85 if pd.notna(sma) else sma
        reason += "[SMA Shift Mode] "
    elif sim_mode == "RSI + SMA Confirmation":
        reason += "[RSI+SMA Mode] "
    elif sim_mode == "ATR Scaling (default)":
        reason += "[ATR Scaling Mode] "

    # === Signal logic ===
    if boom:
        if rsi > 90:
            signal = "SELL"
        else:
            signal = "BUY"
        reason += "Boom quarter"

    elif pd.notna(rsi) and pd.notna(sma):
        if (rsi < rsi_oversold) and (p > sma):
            signal = "BUY"
            reason += f"RSI {rsi:.1f} < {rsi_oversold} and Price > SMA200 ({p:.2f} > {sma:.2f})"
        elif in_position:
            if p > sma * (1 + strongtrend_sma_gap):
                overbought_level = rsi_overbought_strongtrend
            else:
                overbought_level = rsi_overbought

            # --- SELL conditions based on sim_mode ---
            if sim_mode == "RSI + SMA Confirmation":
                if (rsi > overbought_level) and (p < sma):
                    signal = "SELL"
                    reason += f"RSI {rsi:.1f} > {overbought_level} AND Price < SMA200 ({p:.2f} < {sma:.2f})"
            else:
                if (rsi > overbought_level) or (p < sma):
                    signal = "SELL"
                    if rsi > overbought_level and p < sma:
                        reason += f"RSI {rsi:.1f} > {overbought_level} and Price < SMA200 ({p:.2f} < {sma:.2f})"
                    elif rsi > overbought_level:
                        reason += f"RSI {rsi:.1f} > {overbought_level}"
                    elif p < sma:
                        reason += f"Price < SMA200 ({p:.2f} < {sma:.2f})"

            # --- ATR trailing stop logic ---
            if use_atr_stop and "ATR" in latest and not pd.isna(latest["ATR"]):
                atr_stop_mult = atr_mult
                if sim_mode == "ATR Scaling (default)":
                    atr_stop_mult = atr_mult * 0.85
                if trailing_stop is None:
                    trailing_stop = p - atr_stop_mult * latest["ATR"]
                else:
                    trailing_stop = max(trailing_stop, p - atr_stop_mult * latest["ATR"])
                if p < trailing_stop:
                    signal = "SELL"
                    reason += f" Price hit ATR trailing stop ({p:.2f} < {trailing_stop:.2f})"

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


###==== Layered Entry 



# === Layered Entry Logic ===
def layered_entry_signal(ticker, cfg):
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

    # Demo placeholders
    in_position = True
    strongtrend_sma_gap = 0.05
    rsi_overbought_strongtrend = 80
    trailing_stop = None

    # Indicators
    price["SMA200"] = price["Price"].rolling(window=sma_window, min_periods=1).mean()
    price["SMA30"] = price["Price"].rolling(window=15, min_periods=1).mean()
    price["SMA50"] = price["Price"].rolling(window=50, min_periods=1).mean()
    price["SMA100"] = price["Price"].rolling(window=100, min_periods=1).mean()

    stock_quarterly = price["Price"].resample("QE").last()
    quarterly_returns = stock_quarterly.pct_change(fill_method=None)
    boom_quarters_idx = quarterly_returns[quarterly_returns > boom_threshold].index
    boom_quarters_set = set((q.year, q.quarter) for q in boom_quarters_idx)
    price["IsBoom"] = [(d.year, d.quarter) in boom_quarters_set for d in price.index]

    price["RSI"] = compute_rsi(price["Price"], period=rsi_period)
    use_atr_stop = cfg["use_atr_stop"]
    atr_mult = cfg["atr_mult"]
    if use_atr_stop:
        ohlc = yf.download(ticker, start=start, end=end,
                           progress=False, auto_adjust=True)[["High","Low","Close"]]
        hl = ohlc["High"] - ohlc["Low"]
        hp = (ohlc["High"] - ohlc["Close"].shift()).abs()
        lp = (ohlc["Low"] - ohlc["Close"].shift()).abs()
        tr = pd.concat([hl, hp, lp], axis=1).max(axis=1)
        price["ATR"] = tr.rolling(14).mean()

    latest = price.iloc[-1]
    live_price = yf.Ticker(ticker).info.get("currentPrice")
    p = live_price if live_price else latest["Price"]
    rsi = latest["RSI"]
    sma, sma100, sma50, sma30,boom = latest["SMA200"],latest["SMA100"], latest["SMA50"], latest["SMA30"],latest["IsBoom"]
    # boom = False  # Layered logic uses no boom



    in_position = True
    trailing_stop = None
    buy_signal = False
    sell_signal = False

    # Layered entry logic
    if boom:
        if rsi > 90:
            buy_signal = False
        else:
            buy_signal = True
    else:
        if pd.notna(rsi) and pd.notna(sma100) and pd.notna(sma50) and pd.notna(sma30):
            # Entry
            if (rsi < 30) and (p > sma100):
                buy_signal = True
            elif (p < sma100) and (p > sma30) and (rsi > 30 and rsi < 50):
                if p > sma30 * 1.01 and p > sma50:
                    buy_signal = True

            # Exit
            if in_position:
                if p > sma100 * 1.03:
                    overbought_level = 75
                else:
                    overbought_level = 70

                if (rsi > overbought_level) or (p < sma100) or (p < sma30 and rsi > 65):
                    sell_signal = True

                if use_atr_stop and not pd.isna(latest.get("ATR")):
                    if trailing_stop is None:
                        trailing_stop = p - 2.5 * latest["ATR"]
                    else:
                        trailing_stop = max(trailing_stop, p - 2.5 * latest["ATR"])
                    if p < trailing_stop:
                        sell_signal = True

    signal = "HOLD"
    reason = ""
    if buy_signal:
        signal = "BUY"
        reason = "Layered Entry Buy"
    elif sell_signal:
        signal = "SELL"
        reason = "Layered Entry Sell"

    return {
        "Symbol": ticker,
        "Date": latest.name.strftime("%Y-%m-%d"),
        "Price": round(p,2),
        "RSI": round(rsi,2) if pd.notna(rsi) else None,
        "SMA100": round(sma100,2) if pd.notna(sma100) else None,
        "SMA50": round(sma50,2) if pd.notna(sma50) else None,
        "SMA30": round(sma30,2) if pd.notna(sma30) else None,
        "ATR": round(latest.get("ATR", float("nan")),2) if use_atr_stop else None,
        "Signal": signal,
        "Reason": reason
    }






###==== Filter Signals



def filter_entry_signal(ticker, cfg):
    time.sleep(1)  # avoid too rapid API calls
    end = datetime.today().strftime("%Y-%m-%d")
    start = "2023-01-01"

    # Download price data
    prices = yf.download([ticker], start=start, end=end, progress=False, auto_adjust=True)["Close"].dropna()
    if ticker not in prices.columns:
        return {"Symbol": ticker, "Error": "Data not available"}

    price = prices[ticker].to_frame(name="Price")

    # === Config variables from sidebar ===
    sma_window = cfg["sma_window"]
    rsi_period = cfg["rsi_period"]
    rsi_overbought = cfg["rsi_overbought"]
    rsi_overbought_strongtrend = 80
    rsi_oversold = cfg["rsi_oversold"]
    strongtrend_sma_gap = 0.05
    use_atr_stop = cfg["use_atr_stop"]
    atr_period = cfg["atr_period"]
    atr_mult = cfg["atr_mult"]
    boom_threshold = cfg["boom_threshold"]

    # =====================
    # Helper functions
    # =====================
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def zlema(series, span):
        lag = int(max(1, (span - 1) // 2))
        de_lagged = series + (series - series.shift(lag))
        return de_lagged.ewm(span=span, adjust=False).mean()

    def kama(series, er_window=10, fast=2, slow=30):
        change = (series - series.shift(er_window)).abs()
        volatility = series.diff().abs().rolling(er_window).sum()
        er = change / (volatility + 1e-12)
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_series = pd.Series(index=series.index, dtype=float)
        if len(series) < er_window + 2:
            return series.ewm(span=slow, adjust=False).mean()
        kama_series.iloc[er_window] = series.iloc[:er_window + 1].mean()
        for i in range(er_window + 1, len(series)):
            prev = kama_series.iloc[i - 1]
            kama_series.iloc[i] = prev + sc.iloc[i] * (series.iloc[i] - prev)
        return kama_series.ffill().bfill()

    # =====================
    # Indicators
    # =====================
    price["RSI"] = compute_rsi(price["Price"], rsi_period)
    price["EMA50"] = price["Price"].ewm(span=50, adjust=False).mean()
    price["EMA100"] = price["Price"].ewm(span=100, adjust=False).mean()
    price["KAMA"] = kama(price["Price"], er_window=10, fast=2, slow=max(20, sma_window // 8))
    price["ZLEMA20"] = zlema(price["Price"], 20)

    price["HP"] = price["Price"] - price["KAMA"]
    hp_std = price["HP"].rolling(50, min_periods=10).std()
    price["HP_Z"] = price["HP"] / hp_std  # no nan handling

    returns = price["Price"].pct_change()
    vol_short = returns.rolling(10, min_periods=5).std()
    vol_long = returns.rolling(50, min_periods=20).std()
    price["VolRatio"] = vol_short / (vol_long + 1e-12)

    if use_atr_stop:
        ohlc = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)[["High", "Low", "Close"]]
        hl = ohlc["High"] - ohlc["Low"]
        hp = (ohlc["High"] - ohlc["Close"].shift()).abs()
        lp = (ohlc["Low"] - ohlc["Close"].shift()).abs()
        tr = pd.concat([hl, hp, lp], axis=1).max(axis=1)
        price["ATR"] = tr.rolling(atr_period, min_periods=1).mean()

    # =====================
    # Boom detection
    # =====================
    stock_quarterly = price["Price"].resample("QE").last()
    quarterly_returns = stock_quarterly.pct_change()
    boom_quarters_idx = quarterly_returns[quarterly_returns > boom_threshold].index
    boom_quarters_set = {(q.year, q.quarter) for q in boom_quarters_idx}
    price["IsBoom"] = [(d.year, d.quarter) in boom_quarters_set for d in price.index]

    # =====================
    # Latest bar
    # =====================
    latest = price.iloc[-1]
    live_price = yf.Ticker(ticker).info.get("currentPrice")
    p = live_price if live_price else latest["Price"]
    rsi = latest["RSI"]
    ema50, ema100, kama_, z20, hpz, volr, boom = (
        latest["EMA50"], latest["EMA100"], latest["KAMA"],
        latest["ZLEMA20"], latest["HP_Z"], latest["VolRatio"], latest["IsBoom"]
    )

    vol_trend_on = volr < 0.9  # simple initial guess
    CONF_ENTER = 0.6
    HPZ_PENALTY, HPZ_SEVERE = 1.5, 3.0
    VR_ENTER, VR_EXIT = 0.9, 1.2
    RSI_MID_LOW, RSI_MID_HIGH = 45, 60

    buy_signal = False
    sell_signal = False
    trailing_stop = None
    in_position = True  # assume currently holding for exit logic

    # =====================
    # Core Logic
    # =====================
    if boom:
        buy_signal = rsi < 90
        sell_signal = not buy_signal
    else:
        # Volatility regime
        if vol_trend_on:
            vol_trend_on = volr <= VR_EXIT
        else:
            vol_trend_on = volr <= VR_ENTER

        # Confidence scoring
        trend_align = 1.0 if (ema50 > ema100 > kama_) else 0.0
        dist_norm = np.clip((p - kama_) / (0.05 * kama_), -1.0, 1.0)
        rsi_mid = 1.0 if (RSI_MID_LOW <= rsi <= RSI_MID_HIGH) else 0.0
        rsi_reset = 1.0 if (rsi < rsi_oversold) else 0.0
        vshape_turn = 1.0 if (z20 > ema50 and rsi >= 40) else 0.0
        vol_inv = np.clip((1.5 - volr), 0.0, 1.0)

        conf = 0.30 * trend_align + 0.20 * dist_norm + 0.20 * vol_inv + 0.15 * rsi_mid + 0.15 * rsi_reset
        conf += 0.15 * vshape_turn
        conf = float(np.clip(conf, 0.0, 1.0))
        if abs(hpz) > HPZ_PENALTY:
            conf *= 0.75

        # Entry
        if vol_trend_on and conf >= CONF_ENTER:
            buy_signal = True

        # Exit
        overbought_level = rsi_overbought_strongtrend if p > kama_ * (1 + strongtrend_sma_gap) else rsi_overbought
        severe_noise = abs(hpz) > HPZ_SEVERE
        regime_weak = (not vol_trend_on) and (p < ema100)
        if in_position and (rsi > overbought_level or p < ema50 or severe_noise or regime_weak):
            sell_signal = True

        # ATR stop
        if use_atr_stop and "ATR" in latest and latest["ATR"] is not None:
            if trailing_stop is None:
                trailing_stop = p - atr_mult * latest["ATR"]
            else:
                trailing_stop = max(trailing_stop, p - atr_mult * latest["ATR"])
            if p < trailing_stop:
                sell_signal = True

    # =====================
    # Output
    # =====================
    signal = "HOLD"
    reason = ""
    if buy_signal:
        signal = "BUY"
        reason = "Filter Buy Signal"
    elif sell_signal:
        signal = "SELL"
        reason = "Filter Sell Signal"

    return {
        "Symbol": ticker,
        "Date": latest.name.strftime("%Y-%m-%d"),
        "Price": round(p, 2),
        "RSI": round(rsi, 2) if pd.notna(rsi) else None,
        "EMA50": round(ema50, 2),
        "EMA100": round(ema100, 2),
        "KAMA": round(kama_, 2),
        "ZLEMA20": round(z20, 2),
        "HP_Z": round(hpz, 2),
        "VolRatio": round(volr, 2),
        "ATR": round(latest.get("ATR", 0), 2) if use_atr_stop else None,
        "Signal": signal,
        "Reason": reason
    }

















# === Streamlit UI ===
st.set_page_config(page_title="Signal Dashboard", layout="wide")
st.title("📈 Signal Dashboard")

# Sidebar inputs
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

# === Tax Simulation Mode ===
sim_mode = st.sidebar.radio(
    "TAX Sim Mode",
    ["SMA Shift (15%)", "RSI + SMA Confirmation", "ATR Scaling (default)"],
    index=2
)

symbols_input = st.text_area("Enter stock symbols (comma-separated)",
                             value="NIFTYBEES.NS")
symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

if st.button("Generate Signals"):
    normal_results, sim_results, layered_results, filter_results = [], [], [], []
    for sym in symbols:
        normal_results.append(generate_signal(sym, default_cfg, sim_mode="Normal"))
        sim_results.append(generate_signal(sym, default_cfg, sim_mode=sim_mode))
        layered_results.append(layered_entry_signal(sym,default_cfg))
        filter_results.append(filter_entry_signal(sym,default_cfg))
        

    df_normal = pd.DataFrame(normal_results)
    df_sim = pd.DataFrame(sim_results)
    df_layer = pd.DataFrame(layered_results)
    df_filter = pd.DataFrame(filter_results)

    def highlight(row):
        color = {"BUY": "#d4f4dd", "SELL": "#f4d4d4", "HOLD": "#f4f4d4"}
        return [f"background-color: {color.get(row['Signal'], '')}" for _ in row]

    # === Display results ===
    if not df_normal.empty:
        st.subheader("Normal Signals")
        st.dataframe(df_normal.style.apply(highlight, axis=1), use_container_width=True)

    if not df_layer.empty:
        st.subheader("Layered Signals")
        st.dataframe(df_layer.style.apply(highlight, axis=1), use_container_width=True)

    if not df_filter.empty:
        st.subheader(f"Filter Signals")
        st.dataframe(df_filter.style.apply(highlight, axis=1), use_container_width=True)

    if not df_sim.empty:
        st.subheader(f"Signals — {sim_mode}")
        st.dataframe(df_sim.style.apply(highlight, axis=1), use_container_width=True)








