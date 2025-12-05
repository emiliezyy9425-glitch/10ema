"""Live IBKR martingale trader derived from ``martingale_ibkr_backtester``.

This script mirrors the backtester's signal logic (10-day EMA cross) and
applies a simple martingale position-sizing scheme:

* Start at ``RISK_RESET_PCT`` of available equity per trade
* Double risk after a losing trade up to ``MARTINGALE_CAP_PCT``
* Reset risk to ``RISK_RESET_PCT`` after any winning trade

State (open position info and current risk level) is persisted to
``data/martingale_live_state.json`` so repeated invocations maintain the
martingale sequence. Trades and PnL snapshots are appended to
``data/trade_log_live.csv`` for review.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from pandas import Timestamp
from ib_insync import IB, MarketOrder, Stock, Trade, util

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_trading import connect_ibkr

# === HARD-CODED SAFE SETTINGS (no backtester needed) ===
CAPITAL = 500_000
COMMISSION_PER_SHARE = 0.0035
MARTINGALE_CAP_PCT = 16.0
RISK_RESET_PCT = 1.0
LIVE_TIMEFRAMES = ["30 mins", "1 hour", "4 hours", "1 day"]

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

PROJECT_ROOT = Path("/app")
DATA_DIR = PROJECT_ROOT / "data"
STATE_PATH = DATA_DIR / "martingale_live_state.json"
TRADE_LOG_PATH = DATA_DIR / "trade_log_live.csv"
TICKERS_FILE = PROJECT_ROOT / "tickers.txt"

# Match IBKR-compliant durations used in the backtester
DURATION_MAP = {
    "1 min": "30 D",
    "2 mins": "60 D",
    "3 mins": "90 D",
    "5 mins": "180 D",
    "15 mins": "365 D",
    "30 mins": "365 D",
    "1 hour": "365 D",
    "4 hours": "2 Y",
    "1 day": "3 Y",
}

# Restrict live trading to higher timeframes (exclude sub-30m intervals)
EXCLUDED_TIMEFRAMES = {"1 min", "2 mins", "3 mins", "5 mins", "15 mins"}
LIVE_TIMEFRAMES = [tf for tf in LIVE_TIMEFRAMES if tf not in EXCLUDED_TIMEFRAMES]


# --------------------------- Persistence ---------------------------
def load_tickers(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Ticker list not found: {path.resolve()}")
    tickers = [
        line.strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not tickers:
        raise ValueError("Ticker list is empty; add at least one symbol.")
    return tickers


def load_state() -> Dict[str, Dict[str, Any]]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        logging.exception("Failed to read state file; starting fresh.")
        return {}


def save_state(state: Dict[str, Dict[str, Any]]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


# --------------------------- Data helpers ---------------------------
def get_equity(ib: IB) -> float:
    summary = ib.accountSummary()
    for row in summary:
        if row.tag == "NetLiquidation":
            try:
                return float(row.value)
            except ValueError:
                continue
    return CAPITAL


def fetch_intraday(ib: IB, contract: Stock, timeframe: str) -> pd.DataFrame:
    end_dt = datetime.now()
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_dt,
        durationStr=DURATION_MAP.get(timeframe, "90 D"),
        barSizeSetting=timeframe,
        whatToShow="MIDPOINT",
        useRTH=True,
        formatDate=1,
    )
    df = util.df(bars)
    if df.empty:
        return df
    df_dates = pd.to_datetime(df["date"])
    df["date"] = (
        df_dates.dt.tz_localize("UTC") if df_dates.dt.tz is None else df_dates.dt.tz_convert("UTC")
    )
    return df.set_index("date")


def get_daily_ema10(ib: IB, contract: Stock) -> pd.Series:
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="3 Y",
        barSizeSetting="1 day",
        whatToShow="MIDPOINT",
        useRTH=True,
        formatDate=1,
    )
    df = util.df(bars)
    if df.empty:
        return pd.Series(dtype=float)

    df_dates = pd.to_datetime(df["date"])
    df["date"] = (
        df_dates.dt.tz_localize("UTC") if df_dates.dt.tz is None else df_dates.dt.tz_convert("UTC")
    )
    ema10 = df["close"].ewm(span=10, adjust=False).mean()
    return ema10.set_index(df["date"])


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    mapping = {
        "1 min": pd.Timedelta(minutes=1),
        "2 mins": pd.Timedelta(minutes=2),
        "3 mins": pd.Timedelta(minutes=3),
        "5 mins": pd.Timedelta(minutes=5),
        "15 mins": pd.Timedelta(minutes=15),
        "30 mins": pd.Timedelta(minutes=30),
        "1 hour": pd.Timedelta(hours=1),
        "4 hours": pd.Timedelta(hours=4),
        "1 day": pd.Timedelta(days=1),
    }
    return mapping.get(timeframe, pd.Timedelta(minutes=1))


# --------------------------- Trading logic ---------------------------
def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ema10" not in df:
        return df

    df = df.copy()
    df["prev_close"] = df["close"].shift(1)
    df["prev_ema"] = df["ema10"].shift(1)
    df["buy_signal"] = (df["prev_close"] < df["prev_ema"]) & (df["close"] > df["ema10"])
    df["sell_signal"] = (df["prev_close"] > df["prev_ema"]) & (df["close"] < df["ema10"])
    return df


def log_trade(entry: dict[str, Any]) -> None:
    TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not TRADE_LOG_PATH.exists()
    pd.DataFrame([entry]).to_csv(TRADE_LOG_PATH, mode="a", index=False, header=write_header)


def place_market_order(ib: IB, contract: Stock, action: str, quantity: int) -> Trade:
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(contract, order)
    ib.sleep(0.1)
    ib.waitOnUpdate(timeout=10)
    return trade


def execute_strategy_for_symbol(ib: IB, symbol: str, timeframe: str, state: Dict[str, Dict[str, Any]]) -> None:
    key = f"{symbol}:{timeframe}"
    state.setdefault(key, {"risk_pct": RISK_RESET_PCT, "position": 0})

    contract = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(contract)

    intraday = fetch_intraday(ib, contract, timeframe)
    if intraday.empty:
        logging.warning("No intraday data for %s (%s)", symbol, timeframe)
        return

    daily_ema = get_daily_ema10(ib, contract)
    if daily_ema.empty:
        logging.warning("No daily data for %s (%s)", symbol, timeframe)
        return

    df = intraday.copy()

    # Use only the most recent completed daily EMA value to avoid look-ahead bias
    last_daily_close = daily_ema.index[-1].normalize()
    now = datetime.now(tz=daily_ema.index[-1].tzinfo)
    if now.date() > last_daily_close.date():
        current_ema10 = daily_ema.iloc[-1]
    else:
        current_ema10 = daily_ema.iloc[-2]

    df["ema10"] = current_ema10
    df = compute_signals(df)
    latest = df.iloc[-1]
    latest_bar_start = latest.name.floor(timeframe_to_timedelta(timeframe))

    # Only act on fully closed bars
    bar_end = latest.name + timeframe_to_timedelta(timeframe)
    if datetime.now(tz=latest.name.tzinfo) < bar_end:
        return

    # Open/close logic mirrors the backtester: exit on opposite signal, then consider new entries
    position = state[key].get("position", 0)
    entry_price = state[key].get("entry_price", 0.0)
    shares = state[key].get("shares", 0)
    risk_pct = float(state[key].get("risk_pct", RISK_RESET_PCT))

    def update_risk(win: bool) -> float:
        return RISK_RESET_PCT if win else min(risk_pct * 2, MARTINGALE_CAP_PCT)

    if position != 0 and ((position == 1 and latest.sell_signal) or (position == -1 and latest.buy_signal)):
        equity = get_equity(ib)
        ticker = ib.reqTickers(contract)[0]
        exit_price = ticker.marketPrice()
        if exit_price is None or pd.isna(exit_price):
            exit_price = float(latest["close"])
        action = "SELL" if position == 1 else "BUY"
        trade = place_market_order(ib, contract, action, abs(shares))
        ib.sleep(1)
        actual_exit = trade.fills[-1].execution.price if trade.fills else exit_price
        pnl = position * (actual_exit - entry_price) * shares
        commission = shares * 2 * COMMISSION_PER_SHARE
        pnl -= commission
        win = pnl > 0
        risk_pct = update_risk(win)

        log_trade(
            {
                "timestamp": latest.name,
                "symbol": symbol,
                "timeframe": timeframe,
                "type": "EXIT",
                "action": action,
                "shares": shares,
                "entry": entry_price,
                "exit": actual_exit,
                "pnl": round(pnl, 2),
                "equity": round(equity + pnl, 2),
                "risk_pct": risk_pct,
                "win": win,
            }
        )

        state[key] = {
            "risk_pct": risk_pct,
            "position": 0,
            "last_entry_bar": state[key].get("last_entry_bar"),
            "last_exit_bar": latest_bar_start.isoformat(),
        }
        logging.info(
            "Exited %s %s | PnL: %.2f | Exit fill: %.4f | Risk now %.1f%%",
            symbol,
            timeframe,
            pnl,
            actual_exit,
            risk_pct,
        )

    # If flat, consider new entry on the latest completed bar
    position = state[key].get("position", 0)
    risk_pct = float(state[key].get("risk_pct", RISK_RESET_PCT))
    last_entry_time = state[key].get("last_entry_bar")
    last_exit_bar = state[key].get("last_exit_bar")
    symbol_positions = [
        v.get("position", 0)
        for k, v in state.items()
        if k.startswith(f"{symbol}:") and k != key
    ]
    if position == 0 and not any(symbol_positions):
        if last_exit_bar and Timestamp(last_exit_bar) >= latest_bar_start:
            return
        if last_entry_time is not None and Timestamp(last_entry_time) >= latest_bar_start:
            return
        if latest.buy_signal:
            action = "BUY"
            position = 1
        elif latest.sell_signal:
            action = "SELL"
            position = -1
        else:
            return

        equity = get_equity(ib)
        ticker = ib.reqTickers(contract)[0]
        price = ticker.marketPrice()
        if price is None or pd.isna(price):
            price = float(latest["close"])
        shares = max(int(round(equity * (risk_pct / 100) / price)), 1)
        trade = place_market_order(ib, contract, action, shares)
        ib.sleep(1)
        actual_entry = trade.fills[-1].execution.price if trade.fills else price

        state[key] = {
            "risk_pct": risk_pct,
            "position": position,
            "entry_price": actual_entry,
            "shares": shares,
            "last_entry_bar": latest_bar_start.isoformat(),
        }

        log_trade(
            {
                "timestamp": latest.name,
                "symbol": symbol,
                "timeframe": timeframe,
                "type": "ENTRY",
                "action": action,
                "shares": shares,
                "entry": actual_entry,
                "equity": round(equity, 2),
                "risk_pct": risk_pct,
            }
        )
        logging.info(
            "Entered %s on %s (%s) | Shares: %s | Entry fill: %.4f | Risk: %.1f%%",
            action,
            symbol,
            timeframe,
            shares,
            actual_entry,
            risk_pct,
        )


# --------------------------- Main ---------------------------
def main() -> None:
    tickers = load_tickers(TICKERS_FILE)
    ib = connect_ibkr(max_retries=3, initial_client_id=200)
    if ib is None or not ib.isConnected():
        raise RuntimeError("Failed to connect to IBKR")

    state = load_state()

    try:
        for symbol in tickers:
            for timeframe in LIVE_TIMEFRAMES:
                try:
                    execute_strategy_for_symbol(ib, symbol, timeframe, state)
                except Exception:
                    logging.exception("Error processing %s (%s)", symbol, timeframe)
    finally:
        save_state(state)
        ib.disconnect()
        logging.info("State saved to %s", STATE_PATH)


if __name__ == "__main__":
    main()
