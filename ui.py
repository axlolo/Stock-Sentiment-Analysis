"""invest_agent_ui.py – Streamlit dashboard with MPLFinance chart
Run:
    streamlit run invest_agent_ui.py
Requires:
    pip install streamlit yfinance mplfinance numpy pandas plotly (plus deps of invest_agent.py)
"""
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import mplfinance as mpf
from io import BytesIO
import matplotlib.pyplot as plt

# bring in your text‑synthesis helpersrom project import news_summary, reddit_summary

st.set_page_config(page_title="Invest Agent", layout="centered")

# ─────────────────────────── Helper: mplfinance plot ────────────────────── #

def plot_price_ta(df: pd.DataFrame, ticker: str) -> None:
    """Interactive Plotly candlestick with SMAs + Bollinger + RSI."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # ── indicators ──
    df = df.copy()
    df["SMA20"]  = df["Close"].rolling(20).mean()
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_up"] = df["SMA20"] + 2*std20
    df["BB_dn"] = df["SMA20"] - 2*std20

    delta = df["Close"].diff()
    gain  = np.where(delta>0, delta, 0.0)
    loss  = np.where(delta<0, -delta, 0.0)
    rs = (pd.Series(gain,index=df.index).rolling(14).mean() /
          pd.Series(loss,index=df.index).rolling(14).mean().replace(0,np.nan))
    df["RSI"] = 100 - (100/(1+rs))

    df.reset_index(inplace=True)  # x‑axis for plotly

    # ── build figure with two rows (price + RSI) ──
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7,0.3], vertical_spacing=0.03,
                        specs=[[{"secondary_y": True}], [{}]])

    # candles & volume
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Price"),
                  row=1,col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", opacity=0.3),
                  row=1,col=1, secondary_y=True)

    # SMAs & Bollinger
    for col,color in [("SMA20","#FFD54F"),("SMA50","#42A5F5"),("SMA200","#8E24AA")]:
        fig.add_trace(go.Scatter(x=df["Date"], y=df[col], name=col, line=dict(width=1,color=color)),
                      row=1,col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_up"], line=dict(width=0), name="BB_up", showlegend=False),
                  row=1,col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_dn"], line=dict(width=0), name="BB_dn", fill="tonexty",
                             fillcolor="rgba(200,200,200,0.2)", showlegend=False),
                  row=1,col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI‑14", line=dict(color="#ff9800")),
                  row=2,col=1)
    fig.add_hline(y=70,row=2,col=1,line=dict(color="red",dash="dot",width=1))
    fig.add_hline(y=30,row=2,col=1,line=dict(color="green",dash="dot",width=1))

    fig.update_yaxes(title_text="Price", row=1,col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1,col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="RSI", row=2,col=1)
    fig.update_layout(title=f"{ticker} – Interactive Price & Technicals", height=700,
                      xaxis_rangeslider_visible=False, legend_orientation="h")

    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────── UI layout ─────────────────────────────────── #

st.title("📈 Invest Agent Dashboard")

ticker = st.text_input("Enter ticker (e.g., AAPL, MSFT):", "AAPL").upper().strip()
colA, colB, colC, colD, colE = st.columns(5)

with colA:
    if st.button("Industry News"):
        st.markdown(news_summary(ticker, "market"))

with colB:
    if st.button("Company News"):
        st.markdown(news_summary(ticker, "company"))

with colC:
    if st.button("Reddit Sentiment"):
        st.markdown(reddit_summary(ticker))

with colD:
    if st.button("Run All"):
        st.markdown(news_summary(ticker, "market"))
        st.markdown(news_summary(ticker, "company"))
        st.markdown(reddit_summary(ticker))

with colE:
    if st.button("Plot Price & TA"):
        st.info("Fetching price history – may take a moment …")
        data = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
        if data.empty:
            st.error("No price data found for that ticker.")
        else:
            plot_price_ta(data, ticker)
