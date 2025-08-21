
import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Ensure we can import our consolidator module from the same folder
APP_DIR = os.path.dirname(__file__)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from phase3_consolidator import (
    download_data, compute_indicators, fetch_fundamentals, fetch_news_summary,
    build_phase3_payload, format_structured_report
)

st.set_page_config(page_title="ALGO – Stock Dashboard", layout="wide")

st.title("📈 ALGO — Stock Dashboard")

with st.sidebar:
    st.header("Parâmetros")
    ticker = st.text_input("Ticker", value="AAPL")
    period = st.selectbox("Período", ["1mo","3mo","6mo","12mo","2y","5y","max"], index=2)
    interval = st.selectbox("Intervalo", ["1d","1h","1wk"], index=0)
    news_query = st.text_input("Query de notícias (opcional)", value="AAPL OR Apple Inc")
    news_limit = st.slider("Limite de notícias", 5, 30, 12)
    lookback = st.slider("Lookback notícias (dias)", 1, 30, 7)
    news_lang = st.text_input("Idioma Google News (hl)", value="pt-PT")
    openai_model = st.text_input("Modelo GPT (opcional)", value="")
    run_btn = st.button("Analisar", type="primary")

def plot_candles(df: pd.DataFrame, supports=None, resistances=None):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="OHLC"
    )])
    # Moving averages
    if 'SMA20' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', mode='lines'))
    if 'SMA50' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50', mode='lines'))

    # Horizontal lines for supports/resistances
    x0 = df.index.min()
    x1 = df.index.max()
    shapes = []
    for y in (supports or []):
        shapes.append(dict(type="line", xref="x", yref="y", x0=x0, x1=x1, y0=y, y1=y, line=dict(width=1, dash="dot")))
    for y in (resistances or []):
        shapes.append(dict(type="line", xref="x", yref="y", x0=x0, x1=x1, y0=y, y1=y, line=dict(width=1, dash="dash")))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, shapes=shapes, margin=dict(t=40, b=10, l=10, r=10))
    return fig

def plot_macd(df: pd.DataFrame):
    if not {'MACD','MACD_SIGNAL','MACD_HIST'}.issubset(df.columns):
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", mode='lines'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], name="Sinal", mode='lines'))
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], name="Histograma", opacity=0.4))
    fig.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10))
    return fig

if run_btn:
    try:
        df = download_data(ticker, period, interval)
        df = compute_indicators(df)

        fundamentals = fetch_fundamentals(ticker)
        news_summary = fetch_news_summary(news_query, news_limit, lookback, news_lang) if news_query else \
                       type("Obj", (), {"avg_sentiment_score": None, "counts": {"positive":0,"neutral":0,"negative":0}})()

        payload = build_phase3_payload(ticker, df, news_summary, fundamentals)
        tech = payload["technical"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fecho", f"{tech['close']:.2f}")
        rsi = tech.get("rsi14")
        col2.metric("RSI(14)", "—" if rsi is None or (isinstance(rsi,float) and np.isnan(rsi)) else f"{rsi:.1f}")
        col3.metric("Tendência", tech.get("trend","—"))
        macd_hist = tech.get("macd_hist")
        col4.metric("MACD Hist", "—" if macd_hist is None or (isinstance(macd_hist,float) and np.isnan(macd_hist)) else f"{macd_hist:.2f}")

        # Charts
        st.subheader("Gráfico")
        sr = tech.get("support_resistance", {})
        supports = sr.get("supports", [])
        resistances = sr.get("resistances", [])
        fig_c = plot_candles(df, supports, resistances)
        st.plotly_chart(fig_c, use_container_width=True)

        fig_m = plot_macd(df)
        if fig_m:
            st.plotly_chart(fig_m, use_container_width=True)

        # Structured report
        st.subheader("Relatório Estruturado")
        md = format_structured_report(payload, locale="pt-PT")
        st.markdown(md)

        # Downloads
        st.subheader("Downloads")
        st.download_button("Baixar JSON", data=json.dumps(payload, ensure_ascii=False, indent=2),
                           file_name=f"{ticker}_{interval}_phase3.json", mime="application/json")
        st.download_button("Baixar Relatório (MD)", data=md, file_name=f"{ticker}_{interval}_structured.md",
                           mime="text/markdown")

        # Optional GPT
        if openai_model.strip():
            st.info("Para relatório GPT, exporte OPENAI_API_KEY no ambiente antes de lançar o Streamlit.")
            from phase3_consolidator import generate_report_with_gpt
            try:
                gpt_md = generate_report_with_gpt(payload, openai_model.strip())
                st.subheader("Relatório (GPT)")
                st.markdown(gpt_md)
            except Exception as e:
                st.error(f"Falha ao gerar relatório GPT: {e}")

    except Exception as e:
        st.exception(e)
else:
    st.info("Preencha os parâmetros na barra lateral e clique **Analisar**.")
