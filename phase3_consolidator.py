#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 – Consolidation (Técnico + Padrões de Velas + Fundamentos + Notícias + (opcional) GPT)
-----------------------------------------------------------------------------------------------
"""

import argparse
import json
import os
from dataclasses import dataclass
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
    return rsi.bfill().fillna(50.0)  # usar bfill() para evitar FutureWarning

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

def _safe_scalar(x) -> float:
    """Extrai um escalar de objetos (Series, ndarray, etc.) de forma robusta."""
    try:
        arr = np.asarray(x).reshape(-1)
        if arr.size == 0 or arr[0] is None or (isinstance(arr[0], float) and np.isnan(arr[0])):
            return float("nan")
        return float(arr[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            return float("nan")

def _candles_pandas(df: pd.DataFrame) -> Dict[str, int]:
    if len(df) < 2:
        return {}
    last = df.iloc[-1]
    prev = df.iloc[-2]
    o, h, l, c = map(_safe_scalar, [last.Open, last.High, last.Low, last.Close])
    po, pc = map(_safe_scalar, [prev.Open, prev.Close])

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

# ------------------------------- Suporte / Resistência -------------------------------

def compute_support_resistance(df: pd.DataFrame, lookback: int = 20, levels: int = 3) -> Dict[str, List[float]]:
    """
    Calcula níveis de suporte e resistência usando máximos/mínimos recentes.
    Implementação robusta que garante arrays 1D e ignora NaN.
    """
    if len(df) == 0:
        return {"supports": [], "resistances": []}
    lookback = int(max(1, min(lookback, len(df))))
    window = df.tail(lookback)

    def _col_1d(col: str) -> np.ndarray:
        # Converte para array 1D float, ignorando NaN
        arr = np.asarray(window[col], dtype="float64")
        arr = arr.reshape(-1)  # força 1D mesmo se (n,1)
        return arr[~np.isnan(arr)]

    lows = _col_1d("Low")
    highs = _col_1d("High")

    if lows.size == 0 or highs.size == 0:
        return {"supports": [], "resistances": []}

    # ordenar e remover duplicados mantendo ordem
    lows_sorted = np.sort(lows)               # asc
    highs_sorted = np.sort(highs)[::-1]       # desc

    def _unique_preserve(seq: np.ndarray) -> List[float]:
        out = []
        seen = set()
        for v in seq:
            if v not in seen:
                out.append(float(v))
                seen.add(v)
        return out

    supports = _unique_preserve(lows_sorted)[:levels]
    resistances = _unique_preserve(highs_sorted)[:levels]
    return {"supports": supports, "resistances": resistances}

# ------------------------------- Fundamentos (Yahoo) -------------------------------

def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    try:
        fast = getattr(t, "fast_info", None) or {}
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
    import importlib.util, sys, os
    candidates = []
    if path_hint:
        candidates.append(path_hint)
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
        return NewsSummary(avg_sentiment_score=None, counts={"positive": 0, "neutral": 0, "negative": 0})
    data = analyze_news(query=query, limit=limit, lookback=lookback, lang=lang)
    return NewsSummary(
        avg_sentiment_score=float(data.get("avg_sentiment_score", 0.0)),
        counts=data.get("counts", {"positive": 0, "neutral": 0, "negative": 0})
    )

# ------------------------------ Consolidação / Payload ------------------------------

def build_phase3_payload(ticker: str, df: pd.DataFrame, news_summary: NewsSummary, fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    last = df.iloc[-1]
    sr = compute_support_resistance(df, lookback=20, levels=3)
    def _f(x): 
        try: 
            return float(x) 
        except Exception: 
            try: 
                return float(np.asarray(x).reshape(-1)[0]) 
            except Exception: 
                return float("nan")
    tech = {
        "close": _f(last.get("Close", np.nan)),
        "rsi14": _f(last.get("RSI14", np.nan)),
        "sma20": _f(last.get("SMA20", np.nan)),
        "sma50": _f(last.get("SMA50", np.nan)),
        "macd": _f(last.get("MACD", np.nan)),
        "macd_signal": _f(last.get("MACD_SIGNAL", np.nan)),
        "macd_hist": _f(last.get("MACD_HIST", np.nan)),
        "trend": "alta" if _f(last.get("SMA20", np.nan)) > _f(last.get("SMA50", np.nan)) else "baixa",
        "candles": detect_candles(df),
        "support_resistance": sr,
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
        "Destaca: tendência (SMA20 vs SMA50), RSI, padrões de candles, MACD, suportes/resistências, "
        "sentimento das notícias e fundamentos. Inclui um risco/aviso no final. Adota PT-PT."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    from openai import OpenAI
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()

# ------------------------------ Relatório estruturado (determinístico) ------------------------------

def map_candle_name(name: str) -> str:
    m = {
        "CDLENGULFING": "Engolfo",
        "CDLHAMMER": "Martelo",
        "CDLSHOOTINGSTAR": "Estrela Cadente",
        "CDLDOJI": "Doji",
        "CDLMORNINGSTAR": "Estrela da Manhã",
        "CDLEVENINGSTAR": "Estrela da Noite",
        "CDLHARAMI": "Harami",
        "CDLPIERCING": "Penetração",
        "CDLDARKCLOUDCOVER": "Nuvem Negra",
        "ENGULFING_BULL": "Engolfo (altista)",
        "ENGULFING_BEAR": "Engolfo (baixista)",
        "HAMMER": "Martelo",
        "DOJI": "Doji",
    }
    return m.get(name, name)

def compute_confidence(tech: dict, sentiment_avg: float, fundamentals: dict) -> float:
    score = 0.0
    if tech.get("sma20") and tech.get("sma50"):
        score += 0.15 if tech["sma20"] > tech["sma50"] else 0.05
    rsi_val = tech.get("rsi14")
    if rsi_val is not None and not (np.isnan(rsi_val)):
        score += 0.15 if 40 <= rsi_val <= 60 else 0.05
    macd, sig = tech.get("macd"), tech.get("macd_signal")
    if macd is not None and sig is not None and not (np.isnan(macd) or np.isnan(sig)):
        score += 0.15 if abs(macd - sig) < 0.5 * (abs(macd) + 1e-6) else 0.05
    score += 0.15 if (sentiment_avg or 0) > 0.05 else 0.05 if (sentiment_avg or 0) > -0.05 else 0.0
    fwdpe = fundamentals.get("forwardPE")
    margin = fundamentals.get("profit_margin")
    if isinstance(fwdpe, (int, float)):
        score += 0.05 if fwdpe > 60 else 0.10
    if isinstance(margin, (int, float)):
        score += 0.10 if margin > 0 else 0.02
    return float(max(0.0, min(1.0, score)))

def format_structured_report(payload: dict, locale: str = "pt-PT") -> str:
    tkr = payload["ticker"]
    tech = payload["technical"]
    fund = payload["fundamental"] or {}
    sent = payload["news_sentiment"] or {}
    candles = tech.get("candles") or {}
    candles_txt = ", ".join(f"{map_candle_name(k)} ({'▲' if v>0 else '▼'})" for k, v in candles.items()) if candles else "Nenhum relevante no último candle"

    trend_txt = "alta" if tech["sma20"] > tech["sma50"] else "baixa"
    conf = compute_confidence(tech, sent.get("avg_score"), fund)

    sr = tech.get("support_resistance", {})
    supports = ", ".join(str(round(x,2)) for x in sr.get("supports", []))
    resistances = ", ".join(str(round(x,2)) for x in sr.get("resistances", []))

    def fmt(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}" if isinstance(x, (int,float)) else str(x)

    md = []
    md.append(f"# {tkr} – Relatório Estruturado\n")
    md.append("## Sumário\n")
    md.append(f"- Tendência: **{trend_txt}** (SMA20 {fmt(tech['sma20'])} vs SMA50 {fmt(tech['sma50'])})")
    md.append(f"- RSI(14): **{fmt(tech['rsi14'])}**  •  MACD: {fmt(tech['macd'])} / sinal {fmt(tech['macd_signal'])}  •  Hist: {fmt(tech['macd_hist'])}")
    md.append(f"- Suportes: {supports or '—'}  •  Resistências: {resistances or '—'}")
    md.append(f"- Padrões de velas: **{candles_txt}**")
    md.append(f"- Sentimento (médio): **{fmt(sent.get('avg_score'))}**  •  (+/{sent.get('counts',{}).get('positive',0)} | 0/{sent.get('counts',{}).get('neutral',0)} | -/{sent.get('counts',{}).get('negative',0)})")
    md.append(f"- Confiança do diagnóstico: **{int(conf*100)}%**\n")

    md.append("## Métricas principais\n")
    md.append("| Métrica | Valor |")
    md.append("|---|---:|")
    md.append(f"| Fecho | {fmt(tech['close'])} |")
    md.append(f"| SMA20 / SMA50 | {fmt(tech['sma20'])} / {fmt(tech['sma50'])} |")
    md.append(f"| RSI(14) | {fmt(tech['rsi14'])} |")
    md.append(f"| MACD / Sinal / Hist | {fmt(tech['macd'])} / {fmt(tech['macd_signal'])} / {fmt(tech['macd_hist'])} |")
    md.append(f"| Suportes | {supports or '—'} |")
    md.append(f"| Resistências | {resistances or '—'} |")
    md.append(f"| P/L (trailing) / Forward P/E | {fmt(fund.get('pe'))} / {fmt(fund.get('forwardPE'))} |")
    md.append(f"| Dividend Yield | {fmt(fund.get('dividend_yield'))} |")
    md.append(f"| Margem líquida | {fmt(fund.get('profit_margin'))} |")
    md.append(f"| ROE | {fmt(fund.get('roe'))} |")
    md.append(f"| Beta | {fmt(fund.get('beta'))} |")
    md.append("")

    md.append("## Leitura\n")
    md.append(f"O ativo mostra **tendência de {trend_txt}**. O RSI em {fmt(tech['rsi14'])} sugere equilíbrio de momento. ")
    md.append("O MACD indica o **momento**; observar um eventual **cruzamento** para confirmação. ")
    if supports or resistances:
        md.append(f"Regiões técnicas relevantes: suporte em {supports or '—'} e resistência em {resistances or '—'}. ")
    if candles:
        md.append(f"Foram detetados padrões de velas: {candles_txt}. Tratar como **sinais de contexto**, não isolados. ")
    else:
        md.append("**Sem padrões de velas relevantes** no último candle. ")
    md.append("Nos fundamentos, atenção a P/E/forward P/E e à rentabilidade/fluxos. ")

    # Gatilhos
    triggers = []
    if tech['rsi14'] is not None and not np.isnan(tech['rsi14']):
        if tech['rsi14'] < 35: triggers.append("RSI < 35 (possível sobrevenda de curto prazo)")
        if tech['rsi14'] > 65: triggers.append("RSI > 65 (possível exaustão de curto prazo)")
    if tech['macd'] is not None and tech['macd_signal'] is not None:
        if (tech['macd'] - tech['macd_signal']) * (tech['macd_hist'] or 0) < 0:
            triggers.append("MACD a aproximar-se de cruzamento com a linha de sinal")
    if supports:
        triggers.append("Confirmar defesa do suporte com aumento de volume")
    if resistances:
        triggers.append("Observar ruptura de resistência com fecho acima")

    if triggers:
        md.append("\n## Gatilhos a monitorizar\n")
        for g in triggers:
            md.append(f"- {g}")

    md.append("\n## Aviso\nEste relatório é informativo e não constitui recomendação. Considere liquidez, custos e o seu perfil de risco.")
    return "\n".join(md)

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
    ap = argparse.ArgumentParser(description="Phase 3 – Consolidador (técnico + padrões + suporte/resistência + fundamentos + notícias + GPT opcional)")
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

    # 5) Guardar JSON e relatórios
    out_dir = ensure_reports_dir()
    base = f"{args.ticker}_{args.interval}".replace("/", "-")
    json_path = os.path.join(out_dir, f"{base}_phase3.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON salvo em: {json_path}")

    # Relatório estruturado (sem GPT)
    structured_md = format_structured_report(payload, locale="pt-PT")
    md_struct_path = os.path.join(out_dir, f"{base}_structured.md")
    with open(md_struct_path, "w", encoding="utf-8") as f:
        f.write(structured_md)
    print(f"📝 Relatório estruturado salvo em: {md_struct_path}")

    # (Opcional) Relatório GPT
    if args.openai_model:
        report_md = generate_report_with_gpt(payload, args.openai_model)
        md_path = os.path.join(out_dir, f"{base}_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"🤖 Relatório (GPT) salvo em: {md_path}")
    else:
        print("ℹ️ Sem modelo GPT fornecido (--openai_model).")

if __name__ == "__main__":
    main()
