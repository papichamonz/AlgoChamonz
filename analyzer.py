
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 – Simple Technical Prototype
- Downloads OHLCV data using yfinance
- Computes RSI(14), SMA20, SMA50
- Emits simple buy/sell/hold signals
- Saves a CSV with indicators

Usage:
  python analyzer.py --ticker AAPL --period 3mo --interval 1d
  python analyzer.py --ticker PETR4.SA --period 6mo --interval 1h

Requirements:
  pip install yfinance pandas

Notes:
  - This script avoids TA-Lib by implementing RSI in pure pandas.
  - For intraday data, Yahoo may have delays and limits.
"""

import argparse
from datetime import datetime, timezone
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency yfinance. Please run: pip install yfinance pandas") from e


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing in pure pandas."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use EMA with alpha=1/period as Wilder's smoothing approximation
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral at start


def generate_signals(df: pd.DataFrame) -> list:
    """Generate simple signals from latest row."""
    last = df.iloc[-1]
    signals = []

    # RSI-based
    if last["RSI14"] < 30:
        signals.append("📈 COMPRA (RSI < 30: sobrevendido)")
    elif last["RSI14"] > 70:
        signals.append("📉 VENDA (RSI > 70: sobrecomprado)")
    else:
        signals.append("➖ NEUTRO (RSI entre 30 e 70)")

    # Trend via moving averages
    if last["SMA20"] > last["SMA50"]:
        signals.append("📈 Tendência de alta (SMA20 > SMA50)")
    elif last["SMA20"] < last["SMA50"]:
        signals.append("📉 Tendência de baixa (SMA20 < SMA50)")
    else:
        signals.append("➖ Sem tendência clara (SMA20 ≈ SMA50)")

    return signals


def main():
    parser = argparse.ArgumentParser(description="Mini-TheAlgo • Fase 1 – Sinais técnicos básicos")
    parser.add_argument("--ticker", required=True, help="Ticker, ex: AAPL, PETR4.SA, VALE3.SA")
    parser.add_argument("--period", default="3mo", help="Período do Yahoo (1mo,3mo,6mo,1y,2y,5y,max) [default: 3mo]")
    parser.add_argument("--interval", default="1d", help="Intervalo (1d,1h,30m,15m,5m,1m) [default: 1d]")
    parser.add_argument("--out", default=None, help="Arquivo CSV de saída (default: indicators_<ticker>_<interval>.csv)")
    args = parser.parse_args()

    # Fetch
    print(f"Baixando {args.ticker} ({args.period}, {args.interval})...")
    df = yf.download(args.ticker, period=args.period, interval=args.interval, progress=False)
    if df.empty:
        raise SystemExit("Sem dados retornados. Verifique ticker/period/interval.")

    # Indicators
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["SMA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["RSI14"] = compute_rsi(df["Close"], period=14)

    # Signals
    signals = generate_signals(df)

    # Output
    now = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')
    last = df.iloc[-1]
    print("\n—— Resultado ———————————————————————————————————")
    print(f"Agora: {now}")
    print(f"Ticker: {args.ticker}")
    print(f"Últ. Preço (Close): {last['Close']:.4f}")
    print(f"RSI14: {last['RSI14']:.2f}  |  SMA20: {last['SMA20']:.4f}  |  SMA50: {last['SMA50']:.4f}")
    print("Sinais:")
    for s in signals:
        print(" - ", s)

    # Save CSV
    out = args.out or f"indicators_{args.ticker}_{args.interval}.csv"
    df.to_csv(out, index=True)
    print(f"\nCSV salvo: {out}")
    print("———————————————————————————————————————————————")

if __name__ == "__main__":
    main()
