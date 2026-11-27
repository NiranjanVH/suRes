import argparse
from datetime import datetime, timezone
import os
import time
import numpy as np
import pandas as pd
import requests

COINDCX_FUTURES_CANDLES_URL = "https://public.coindcx.com/market_data/candlesticks"

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "9022153964:AAEf5GRv77AK6zsHLk59ZxNMwhuDfWAroIY",
)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "8970361774")

DEFAULTS = {
    "pair": "B-BTC_USDT",        # CoinDCX Futures instrument format (e.g., B-BTC_USDT, B-ETH_USDT, B-ZEC_USDT)
    "interval": "5m",            # Accepts 1m, 5m, 15m, 30m, 1h, 4h, 1d (or raw resolution like 5, 15, 60, 1D)
    "bars": 300,
    "pivot_len": 5,
    "range_len": 12,
    "max_range_pct": 1.0,
    "body_required": 0.55,
    "use_sr_breakout": True,
    "use_range_breakout": True,
    "entry_ref_mode": "Signal Close",
    "sl_buffer_ticks": 5,
    "tick_size": 0.1,
    "tp1_rr": 1.0,
    "tp2_rr": 2.0,
    "tp3_rr": 3.0,
}

INTERVAL_MAP = {
    "1m": ("1", 60),
    "3m": ("3", 180),
    "5m": ("5", 300),
    "15m": ("15", 900),
    "30m": ("30", 1800),
    "1h": ("60", 3600),
    "2h": ("120", 7200),
    "4h": ("240", 14400),
    "6h": ("360", 21600),
    "8h": ("480", 28800),
    "1d": ("1D", 86400),
}


def resolve_interval(interval: str):
    key = interval.strip()
    if key.lower() in INTERVAL_MAP:
        return INTERVAL_MAP[key.lower()]
    if key.isdigit():
        mins = int(key)
        return key, mins * 60
    if key.upper() == "1D":
        return "1D", 86400
    raise ValueError(f"Unsupported interval: {interval}")


def fetch_futures_closed_candles(pair: str, interval: str, bars: int = 300) -> pd.DataFrame:
    resolution, sec_per_bar = resolve_interval(interval)
    to_ts = int(time.time())
    from_ts = to_ts - (bars * sec_per_bar)

    resp = requests.get(
        COINDCX_FUTURES_CANDLES_URL,
        params={
            "pair": pair,
            "from": from_ts,
            "to": to_ts,
            "resolution": resolution,
            "pcode": "f",
        },
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()

    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not rows:
        raise RuntimeError(f"No futures candle data returned for {pair} ({interval}): {payload}")

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume", "time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    # Normalize timestamp to milliseconds if returned in seconds
    if df["time"].max() < 10_000_000_000:
        df["time"] = df["time"] * 1000

    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    # Drop the still-forming current candle so only closed candles are evaluated
    if len(df) > 1:
        df = df.iloc[:-1].reset_index(drop=True)
    return df


def compute_pivots(high: np.ndarray, low: np.ndarray, length: int):
    n = len(high)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)

    for idx in range(length, n - length):
        wh = high[idx - length : idx + length + 1]
        if high[idx] == np.max(wh) and np.sum(wh == high[idx]) == 1:
            ph[idx] = high[idx]

        wl = low[idx - length : idx + length + 1]
        if low[idx] == np.min(wl) and np.sum(wl == low[idx]) == 1:
            pl[idx] = low[idx]

    return ph, pl


def analyze_signals(df: pd.DataFrame, cfg: dict):
    pivot_len = cfg["pivot_len"]
    range_len = cfg["range_len"]
    max_range_pct = cfg["max_range_pct"]
    body_required = cfg["body_required"]
    use_sr = cfg["use_sr_breakout"]
    use_rng = cfg["use_range_breakout"]
    entry_mode = cfg["entry_ref_mode"]
    tick_size = cfg["tick_size"]
    buffer_amt = cfg["sl_buffer_ticks"] * tick_size

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    times = df["time"].to_numpy()

    ph, pl = compute_pivots(highs, lows, pivot_len)

    last_high = np.nan
    last_low = np.nan
    range_active = False
    acc_high = np.nan
    acc_low = np.nan

    signals = []

    for i in range(1, len(df)):
        pivot_idx = i - pivot_len
        if pivot_idx >= 0:
            if not np.isnan(ph[pivot_idx]):
                last_high = ph[pivot_idx]
            if not np.isnan(pl[pivot_idx]):
                last_low = pl[pivot_idx]

        if i >= range_len:
            r_high = np.max(highs[i - range_len : i])
            r_low = np.min(lows[i - range_len : i])
            r_width = r_high - r_low
            r_pct = (r_width / closes[i] * 100.0) if closes[i] > 0 else np.nan
            in_consolidation = (not np.isnan(r_pct)) and r_width > 0 and r_pct <= max_range_pct
        else:
            r_high = np.nan
            r_low = np.nan
            in_consolidation = False

        range_break_up = False
        range_break_down = False

        if range_active:
            range_break_up = (closes[i] > acc_high) and (closes[i - 1] <= acc_high)
            range_break_down = (closes[i] < acc_low) and (closes[i - 1] >= acc_low)

            if range_break_up or range_break_down:
                range_active = False
            elif in_consolidation:
                acc_high = max(acc_high, r_high)
                acc_low = min(acc_low, r_low)
            else:
                range_active = False
        elif in_consolidation:
            range_active = True
            acc_high = r_high
            acc_low = r_low

        candle_range = highs[i] - lows[i]
        body_ratio = (abs(closes[i] - opens[i]) / candle_range) if candle_range > 0 else 0.0
        strong_bull = (closes[i] > opens[i]) and (body_ratio >= body_required)
        strong_bear = (closes[i] < opens[i]) and (body_ratio >= body_required)

        bull_sr = use_sr and (not np.isnan(last_high)) and (closes[i] > last_high) and (closes[i - 1] <= last_high)
        bear_sr = use_sr and (not np.isnan(last_low)) and (closes[i] < last_low) and (closes[i - 1] >= last_low)

        bull_rng = use_rng and range_break_up
        bear_rng = use_rng and range_break_down

        buy_sig = strong_bull and (bull_sr or bull_rng)
        sell_sig = strong_bear and (bear_sr or bear_rng)

        if buy_sig or sell_sig:
            if buy_sig:
                side = "BUY"
                sl_price = lows[i] - buffer_amt
                reasons = []
                if bull_sr:
                    reasons.append("S/R Break")
                if bull_rng:
                    reasons.append("Range Break")
                candidates = [v for v, cond in [(last_high, bull_sr), (acc_high, bull_rng)] if cond and not np.isnan(v)]
                broken_lvl = max(candidates) if candidates else np.nan
                entry_price = broken_lvl if (entry_mode == "Broken Level" and not np.isnan(broken_lvl) and broken_lvl > sl_price) else closes[i]
                risk_amt = max(entry_price - sl_price, tick_size)
                tp1 = entry_price + risk_amt * cfg["tp1_rr"]
                tp2 = entry_price + risk_amt * cfg["tp2_rr"]
                tp3 = entry_price + risk_amt * cfg["tp3_rr"]
            else:
                side = "SELL"
                sl_price = highs[i] + buffer_amt
                reasons = []
                if bear_sr:
                    reasons.append("S/R Break")
                if bear_rng:
                    reasons.append("Range Break")
                candidates = [v for v, cond in [(last_low, bear_sr), (acc_low, bear_rng)] if cond and not np.isnan(v)]
                broken_lvl = min(candidates) if candidates else np.nan
                entry_price = broken_lvl if (entry_mode == "Broken Level" and not np.isnan(broken_lvl) and broken_lvl < sl_price) else closes[i]
                risk_amt = max(sl_price - entry_price, tick_size)
                tp1 = entry_price - risk_amt * cfg["tp1_rr"]
                tp2 = entry_price - risk_amt * cfg["tp2_rr"]
                tp3 = entry_price - risk_amt * cfg["tp3_rr"]

            signals.append({
                "bar_time": int(times[i]),
                "side": side,
                "reason": " + ".join(reasons),
                "close": float(closes[i]),
                "entry": float(entry_price),
                "sl": float(sl_price),
                "tp1": float(tp1),
                "tp2": float(tp2),
                "tp3": float(tp3),
                "is_latest_closed_bar": (int(times[i]) == int(times[-1])),
            })

    state = {
        "last_closed_time": int(times[-1]),
        "last_closed_close": float(closes[-1]),
        "resistance": None if np.isnan(last_high) else float(last_high),
        "support": None if np.isnan(last_low) else float(last_low),
        "range_active": bool(range_active),
    }
    return signals, state


def format_signal(sig: dict, pair: str, interval: str, decimals: int = 2) -> str:
    dt = datetime.fromtimestamp(sig["bar_time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tag = "FRESH FUTURES SIGNAL" if sig["is_latest_closed_bar"] else "LAST HISTORICAL FUTURES SIGNAL"
    return (
        f"[{tag}]\n"
        f"{sig['side']} FUTURES | {pair} ({interval})\n"
        f"Setup: {sig['reason']}\n"
        f"Candle Time: {dt}\n"
        f"Close: {sig['close']:.{decimals}f}\n"
        f"Limit: {sig['entry']:.{decimals}f}\n"
        f"SL: {sig['sl']:.{decimals}f}\n"
        f"TP1: {sig['tp1']:.{decimals}f}\n"
        f"TP2: {sig['tp2']:.{decimals}f}\n"
        f"TP3: {sig['tp3']:.{decimals}f}"
    )


def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    resp.raise_for_status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default=DEFAULTS["pair"])
    parser.add_argument("--interval", default=DEFAULTS["interval"])
    parser.add_argument("--tick", type=float, default=DEFAULTS["tick_size"])
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--show-history", action="store_true")
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    cfg["pair"] = args.pair
    cfg["interval"] = args.interval
    cfg["tick_size"] = args.tick

    df = fetch_futures_closed_candles(cfg["pair"], cfg["interval"], cfg["bars"])
    signals, state = analyze_signals(df, cfg)

    closed_dt = datetime.fromtimestamp(state["last_closed_time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"Futures Pair: {cfg['pair']} | Interval: {cfg['interval']} | Last Closed: {closed_dt}")
    print(f"Last Close: {state['last_closed_close']:.{args.decimals}f} | Res: {state['resistance']} | Sup: {state['support']}")

    fresh = [s for s in signals if s["is_latest_closed_bar"]]
    if fresh:
        for s in fresh:
            msg = format_signal(s, cfg["pair"], cfg["interval"], args.decimals)
            print(msg)
            if args.send:
                send_telegram(msg)
    else:
        print("No fresh BUY/SELL futures signal on the latest closed candle.")
        if args.show_history and signals:
            print(format_signal(signals[-1], cfg["pair"], cfg["interval"], args.decimals))


if __name__ == "__main__":
    main()
