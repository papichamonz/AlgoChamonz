#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 – Consolidation (Técnico + Padrões de Velas + Fundamentos + Notícias + (opcional) GPT)
-----------------------------------------------------------------------------------------------
Requisitos (ver requirements.txt sugerido):
  - yfinance, pandas, numpy, matplotlib
  - (opcional) TA-Lib para padrões de velas mais completos
  - (opcional) transformers/nltk via news_analyzer.py (já tens)

Uso básico:
  python phase3_consolidator.py --ticker AAPL --period 6mo --interval 1d --news_query "AAPL OR Apple Inc"

Com GPT (requer OPENAI_API_KEY no ambiente):
  python phase3_consolidator.py --ticker PETR4.SA --period 6mo --interval 1d --news_query "PETR4 OR Petrobras" \
      --openai_model "gpt-4.1-mini"

Saídas:
  - JSON consolidado em: reports/{TICKER}_{INTERVAL}_phase3.json
  - (opcional) relatório em texto (markdown) gerado pelo GPT em: reports/{TICKER}_{INTERVAL}_report.md
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------- Indicadores básicos -----------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI em pandas (sem TA-Lib)."""
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(50.0)

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    df["RSI14"] = rsi(close, 14)
    df["SMA20"] = close.rolling(window=20).mean()
    df["SMA50"] = close.rolling(window=50).mean()
    m, s, h = macd(close)
    df["MACD"] = m
    df["MACD_SIGNAL"] = s
    df["MACD_HIST"] = h
    return df

# --------------------------- Padrões de velas (TA-Lib?) ---------------------------

def _candles_talib(df: pd.DataFrame) -> Optional[Dict[str, int]]:
    try:
        import talib  # type: ignore
    except Exception:
        return None
    o, h, l, c = [df[col].astype(float).values for col in ["Open", "High", "Low", "Close"]]
    patterns = {
        "CDLENGULFING": talib.CDLENGULFING(o, h, l, c),
        "CDLHAMMER": talib.CDLHAMMER(o, h, l, c),
        "CDLSHOOTINGSTAR": talib.CDLSHOOTINGSTAR(o, h, l, c),
        "CDLDOJI": talib.CDLDOJI(o, h, l, c),
        "CDLMORNINGSTAR": talib.CDLMORNINGSTAR(o, h, l, c),
        "CDLEVENINGSTAR": talib.CDLEVENINGSTAR(o, h, l, c),
        "CDLHARAMI": talib.CDLHARAMI(o, h, l, c),
        "CDLPIERCING": talib.CDLPIERCING(o, h, l, c),
        "CDLDARKCLOUDCOVER": talib.CDLDARKCLOUDCOVER(o, h, l, c),
    }
    last_idx = -1
    detected = {}
    for name, series in patterns.items():
        val = int(series[last_idx])
        if val != 0:
            detected[name] = val
    return detected

def _is_doji(o, h, l, c, tol=0.0015):
    body = abs(c - o)
    rng = max(h, c, o) - min(l, c, o)
    if rng == 0:
        return False
    return (body / rng) <= tol

def _is_bullish_engulfing(prev_o, prev_c, o, c):
    return (prev_c < prev_o) and (c > o) and (c >= prev_o) and (o <= prev_c)

def _is_bearish_engulfing(prev_o, prev_c, o, c):
    return (prev_c > prev_o) and (c < o) and (c <= prev_o) and (o >= prev_c)

def _is_hammer(o, h, l, c):
    body = abs(c - o)
    lower_shadow = (min(c, o) - l)
    upper_shadow = (h - max(c, o))
    rng = h - l if (h - l) != 0 else 1e-9
    return (lower_shadow / rng > 0.5) and (body / rng < 0.3) and (upper_shadow / rng < 0.2)

def _candles_pandas(df: pd.DataFrame) -> Dict[str, int]:
    if len(df) < 2:
        return {}
    last = df.iloc[-1]
    prev = df.iloc[-2]
    o, h, l, c = float(last.Open), float(last.High), float(last.Low), float(last.Close)
    po, pc = float(prev.Open), float(prev.Close)

    detected: Dict[str, int] = {}
    if _is_doji(o, h, l, c):
        detected["DOJI"] = 1
    if _is_hammer(o, h, l, c):
        detected["HAMMER"] = 1
    if _is_bullish_engulfing(po, pc, o, c):
        detected["ENGULFING_BULL"] = 1
    if _is_bearish_engulfing(po, pc, o, c):
        detected["ENGULFING_BEAR"] = -1
    return detected

def detect_candles(df: pd.DataFrame) -> Dict[str, int]:
    d = _candles_talib(df)
    if d is None:
        d = _candles_pandas(df)
    return d

# ------------------------------- Fundamentos (Yahoo) -------------------------------

def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    t = yf.Ticker(ticker)
    info = {}
    try:
        # .info pode ser lento/instável; usar try/except
        info = t.info or {}
    except Exception:
        info = {}

    # Tentativa adicional com .fast_info
    try:
        fast = getattr(t, "fast_info", None) or {}
        # fast_info não é dict puro; convertemos campos úteis
        if hasattr(fast, "__dict__"):
            fast = fast.__dict__
        info = {**fast, **info}
    except Exception:
        pass

    fields = {
        "pe": ["trailingPE", "peTrailing", "pe_ratio"],
        "forwardPE": ["forwardPE"],
        "dividend_yield": ["dividendYield"],
        "profit_margin": ["profitMargins"],
        "roe": ["returnOnEquity"],
        "eps": ["trailingEps", "epsTrailingTwelveMonths"],
        "eps_growth_quarterly_yoy": ["earningsQuarterlyGrowth"],
        "revenue_growth_yoy": ["revenueGrowth"],
        "free_cashflow": ["freeCashflow"],
        "market_cap": ["marketCap"],
        "beta": ["beta"],
        "currency": ["currency"],
        "long_name": ["longName", "shortName"],
        "sector": ["sector"],
        "industry": ["industry"],
    }

    out: Dict[str, Any] = {}
    for k, keys in fields.items():
        val = None
        for kk in keys:
            if kk in info and info[kk] is not None:
                val = info[kk]
                break
        out[k] = float(val) if isinstance(val, (int, float)) else val
    return out

# ------------------------------ Notícias (integração) ------------------------------

@dataclass
class NewsSummary:
    avg_sentiment_score: Optional[float]
    counts: Dict[str, int]

def try_import_news_analyzer(path_hint: Optional[str] = None):
    """
    Carrega analyze_news() do teu 'news_analyzer (2).py' por importlib se o ficheiro existir localmente.
    """
    import importlib.util, sys, os
    candidates = []
    if path_hint:
        candidates.append(path_hint)
    # procura nomes comuns
    candidates += [
        "news_analyzer.py",
        "news_analyzer (2).py",
        os.path.join(os.getcwd(), "news_analyzer.py"),
        os.path.join(os.getcwd(), "news_analyzer (2).py"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            spec = importlib.util.spec_from_file_location("news_analyzer_dyn", p)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["news_analyzer_dyn"] = mod
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "analyze_news"):
                    return mod.analyze_news
    return None

def fetch_news_summary(query: Optional[str], limit: int, lookback: int, lang: str) -> NewsSummary:
    if not query:
        return NewsSummary(avg_sentiment_score=None, counts={"positive": 0, "neutral": 0, "negative": 0})
    analyze_news = try_import_news_analyzer()
    if analyze_news is None:
        # fallback: retorna estrutura neutra se módulo não está disponível
        return NewsSummary(avg_sentiment_score=None, counts={"positive": 0, "neutral": 0, "negative": 0})
    data = analyze_news(query=query, limit=limit, lookback=lookback, lang=lang)
    return NewsSummary(
        avg_sentiment_score=float(data.get("avg_sentiment_score", 0.0)),
        counts=data.get("counts", {"positive": 0, "neutral": 0, "negative": 0})
    )

# ------------------------------ Consolidação / Payload ------------------------------

def build_phase3_payload(ticker: str, df: pd.DataFrame, news_summary: NewsSummary, fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    last = df.iloc[-1]
    tech = {
        "close": float(last["Close"]),
        "rsi14": float(last.get("RSI14", float("nan"))),
        "sma20": float(last.get("SMA20", float("nan"))),
        "sma50": float(last.get("SMA50", float("nan"))),
        "macd": float(last.get("MACD", float("nan"))),
        "macd_signal": float(last.get("MACD_SIGNAL", float("nan"))),
        "macd_hist": float(last.get("MACD_HIST", float("nan"))),
        "trend": "alta" if last.get("SMA20", 0) > last.get("SMA50", 0) else "baixa",
        "candles": detect_candles(df),
    }
    payload = {
        "ticker": ticker,
        "technical": tech,
        "fundamental": fundamentals,
        "news_sentiment": {
            "avg_score": news_summary.avg_sentiment_score,
            "counts": news_summary.counts,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return payload

# ------------------------------ Geração de relatório (GPT opcional) ------------------------------

def generate_report_with_gpt(payload: Dict[str, Any], model: str) -> str:
    """
    Requer OPENAI_API_KEY no ambiente. Retorna markdown.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "# Relatório\n\n⚠️ OPENAI_API_KEY não definido. Conteúdo do payload:\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"

    try:
        from openai import OpenAI
    except Exception:
        return "# Relatório\n\n⚠️ SDK openai não instalado. Payload:\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"

    client = OpenAI(api_key=api_key)
    system = "Atua como analista financeiro objetivo. Sê claro, conciso e específico."
    user_prompt = (
        "Com base no JSON a seguir, escreve um parecer (2–4 parágrafos) e um sumário em bullet points. "
        "Destaca: tendência (SMA20 vs SMA50), RSI, padrões de candles (e o que significam), MACD, "
        "sentimento das notícias e principais fundamentos (P/L, Dividend Yield, crescimento de lucros/receitas). "
        "Inclui um risco/aviso no final. Adota PT-PT."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()

# ------------------------------------- Main -------------------------------------

def download_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    print(f"Baixando {ticker} ({period}, {interval})...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError("Falha ao baixar dados de preço.")
    df = df.dropna()
    return df

def ensure_reports_dir() -> str:
    out_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def main():
    ap = argparse.ArgumentParser(description="Phase 3 – Consolidador (técnico + padrões + fundamentos + notícias + GPT opcional)")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--period", default="6mo")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--news_query", default=None, help='Ex.: "AAPL OR Apple Inc" ou "PETR4 OR Petrobras"')
    ap.add_argument("--news_limit", type=int, default=10)
    ap.add_argument("--news_lookback", type=int, default=7)
    ap.add_argument("--news_lang", default="pt-PT")
    ap.add_argument("--openai_model", default=None, help='Ex.: "gpt-4.1-mini" (opcional)')
    args = ap.parse_args()

    # 1) Preços + indicadores + padrões
    df = download_data(args.ticker, args.period, args.interval)
    df = compute_indicators(df)
    candles = detect_candles(df)
    print(f"Padrões de velas (último candle): {candles or '{}'}")

    # 2) Fundamentos
    fundamentals = fetch_fundamentals(args.ticker)

    # 3) Notícias (via teu script se disponível)
    news_summary = fetch_news_summary(args.news_query, args.news_limit, args.news_lookback, args.news_lang)

    # 4) Payload consolidado
    payload = build_phase3_payload(args.ticker, df, news_summary, fundamentals)

    # 5) Guardar JSON e (opcional) relatório GPT
    out_dir = ensure_reports_dir()
    base = f"{args.ticker}_{args.interval}".replace("/", "-")
    json_path = os.path.join(out_dir, f"{base}_phase3.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON salvo em: {json_path}")

    if args.openai_model:
        report_md = generate_report_with_gpt(payload, args.openai_model)
        md_path = os.path.join(out_dir, f"{base}_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"📝 Relatório (GPT) salvo em: {md_path}")
    else:
        print("ℹ️ Sem modelo GPT fornecido (--openai_model). Apenas o JSON foi gerado.")

if __name__ == "__main__":
    main()
