"""
Invest Agent – Streamlit dashboard
Features:
•  Search bar accepts *ticker* **OR** company name (auto‑resolves to ticker)
•  Full‑width interactive Plotly candlestick + SMAs + BB + RSI + Volume
•  Three summary buttons below the chart
-----------------------------------------------------------------------
Requires:
    pip install streamlit yfinance plotly numpy pandas requests
"""
from __future__ import annotations
import requests, pandas as pd, numpy as np, yfinance as yf, streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from project import news_summary, reddit_summary       # text layers

# ───────────────────────  Config  ──────────────────────── #
st.set_page_config(page_title="Invest Agent", layout="wide")
st.markdown("<h2 style='text-align:center;'>📈 Invest Agent Dashboard</h2>",
            unsafe_allow_html=True)

# ───────────  helper: resolve query → ticker symbol  ────── #
@st.cache_data(show_spinner=False, ttl=60*60)
def resolve_to_ticker(query: str) -> str | None:
    """Return an upper‑case ticker for a query (symbol or company)."""
    q = query.strip().upper()
    if len(q) <= 5 and q.isalnum():           # looks like a ticker
        return q

    # hit Yahoo Finance autocomplete API
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    try:
        resp = requests.get(url, params={"q": query, "quotesCount": 1}, timeout=6).json()
        items = resp.get("quotes", [])
        if items:
            return items[0]["symbol"].upper()
    except Exception:
        pass
    return None

# ─────────────  helper: build Plotly TA chart  ──────────── #
def plot_price_ta(df: pd.DataFrame, ticker: str) -> None:
    df = df.copy()
    # indicators
    df["SMA20"]  = df["Close"].rolling(20).mean()
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_up"] = df["SMA20"] + 2 * std20
    df["BB_dn"] = df["SMA20"] - 2 * std20
    delta = df["Close"].diff()
    rs = (pd.Series(np.where(delta>0, delta, 0.0), index=df.index).rolling(14).mean() /
          pd.Series(np.where(delta<0, -delta, 0.0), index=df.index).rolling(14).mean().replace(0,np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))
    df.reset_index(inplace=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3],
                        specs=[[{"secondary_y": True}], [{}]], vertical_spacing=0.05)

    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Price"),
                  row=1,col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume",
                         marker_color="rgba(160,160,160,0.3)"),
                  row=1,col=1, secondary_y=True)
    for col,color in [("SMA20","#FFD54F"),("SMA50","#42A5F5"),("SMA200","#8E24AA")]:
        fig.add_trace(go.Scatter(x=df["Date"], y=df[col], name=col,
                                 line=dict(color=color,width=1)), row=1,col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_up"], line=dict(width=0), showlegend=False), row=1,col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_dn"], fill="tonexty",
                             fillcolor="rgba(200,200,200,0.2)", line=dict(width=0),
                             showlegend=False), row=1,col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI‑14",
                             line=dict(color="#ff9800")), row=2,col=1)
    fig.add_hline(y=70,row=2,col=1,line=dict(color="red",dash="dot"))
    fig.add_hline(y=30,row=2,col=1,line=dict(color="green",dash="dot"))

    fig.update_yaxes(title_text="Price", row=1,col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1,col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="RSI", row=2,col=1)
    fig.update_layout(title=f"{ticker} – Price & Technicals (1 yr)",
                      height=720, xaxis_rangeslider_visible=False,
                      legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────  SEARCH BAR  ─────────────────── #
query = st.text_input("🔍 Enter ticker **or** company name", "AAPL").strip()
ticker = resolve_to_ticker(query)

if not ticker:
    st.error("Could not resolve that query to a ticker.")
    st.stop()

# ───────────  Show chart immediately  ─────────────-- #
with st.spinner("Fetching price history …"):
    data = yf.download(ticker, period="1y", interval="1d", auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.dropna(subset=["Open","High","Low","Close"]).astype(float)

if data.empty:
    st.error("No price data found.")
    st.stop()

plot_price_ta(data, ticker)

# ───────────  Summary buttons  ─────────── #
colA, colB, colC, colD = st.columns([1,1,1,1])

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