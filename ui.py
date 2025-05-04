"""
invest_agent_ui.py – dashboard with sidebar nav and advanced sentiment
Run:
    streamlit run invest_agent_ui.py
Requires:
    pip install streamlit yfinance plotly numpy pandas requests
"""

from __future__ import annotations
import streamlit as st, requests, yfinance as yf, numpy as np, pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from project import news_summary, reddit_summary

# --------------- PAGE & GLOBAL CSS ----------------
st.set_page_config(page_title="Invest Agent", layout="wide")
st.markdown("""
<style>
    .main .block-container { padding-top:0.5rem; }
    section[data-testid="collapsedControl"] {display:none!important;}
    [data-testid="stSidebar"]{transition:margin-left .35s;width:18rem!important;}
    body.sidebar-closed [data-testid="stSidebar"]{margin-left:-18rem;}
    [data-testid="stSidebar"] .block-container{padding-top:3.5rem;}
</style>
<script>
function toggleSidebar(){ document.body.classList.toggle('sidebar-closed'); }
</script>
""", unsafe_allow_html=True)

# -------- HEADER (logo + burger) ----------
h1,h2=st.columns([0.08,0.92])
with h1:
    st.image("logo.png", width=60)
    st.markdown('<button onclick="toggleSidebar()" style="font-size:1.8rem;">&#9776;</button>',unsafe_allow_html=True)
with h2:
    st.markdown("<h2 style='margin:0 0 .2rem 0;'>Invest Agent Dashboard</h2>",unsafe_allow_html=True)

# ---------- SIDEBAR nav ----------
with st.sidebar:
    st.title("Menu")
    if st.button("🏠 Main Dashboard"):   st.session_state.page="main"
    if st.button("🧠 Advanced Sentiment"): st.session_state.page="adv"

if "page" not in st.session_state: st.session_state.page="main"

# -------- Resolver ----------
@st.cache_data(ttl=3600)
def to_ticker(q:str)->str|None:
    q=q.strip().upper()
    if len(q)<=5 and q.isalnum(): return q
    try:
        r=requests.get("https://query1.finance.yahoo.com/v1/finance/search",
                       params={"q":q,"quotesCount":1},timeout=4).json()
        if r.get("quotes"): return r["quotes"][0]["symbol"].upper()
    except: pass
    try:
        sym=yf.Ticker(q).info.get("symbol"); return sym.upper() if sym else None
    except: return None

# -------- Chart helper ----------
def chart(df:pd.DataFrame,ticker:str,show_sma,show_bb,show_rsi):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3],vertical_spacing=0.04,
                      specs=[[{"secondary_y":True}],[{}]])
    fig.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="Price"),1,1)
    fig.add_trace(go.Bar(x=df.index,y=df.Volume,name="Vol",marker_color="rgba(160,160,160,0.3)"),1,1,secondary_y=True)
    if show_sma:
        for n,c in [(20,"#FFD54F"),(50,"#42A5F5"),(200,"#8E24AA")]:
            fig.add_trace(go.Scatter(x=df.index,y=df.Close.rolling(n).mean(),name=f"SMA{n}",
                                     line=dict(width=1,color=c)),1,1)
    if show_bb:
        sma20=df.Close.rolling(20).mean()
        std=df.Close.rolling(20).std()
        up=sma20+2*std; dn=sma20-2*std
        fig.add_trace(go.Scatter(x=df.index,y=up,line=dict(width=0),showlegend=False),1,1)
        fig.add_trace(go.Scatter(x=df.index,y=dn,fill="tonexty",fillcolor="rgba(200,200,200,0.2)",
                                 line=dict(width=0),showlegend=False),1,1)
    if show_rsi:
        delta=df.Close.diff(); gain=np.where(delta>0,delta,0); loss=np.where(delta<0,-delta,0)
        rs=pd.Series(gain,index=df.index).rolling(14).mean()/pd.Series(loss,index=df.index).rolling(14).mean().replace(0,np.nan)
        rsi=100-100/(1+rs)
        fig.add_trace(go.Scatter(x=df.index,y=rsi,name="RSI14",line=dict(color="#ff9800")),2,1)
        fig.add_hline(y=70,row=2,col=1,line=dict(color="red",dash="dot"))
        fig.add_hline(y=30,row=2,col=1,line=dict(color="green",dash="dot"))
    fig.update_layout(title=f"{ticker} – Price & TA",height=720,legend_orientation="h",
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)

# ================= MAIN PAGE =================
if st.session_state.page=="main":
    q=st.text_input("🔍 Enter ticker or company","AAPL")
    ticker=to_ticker(q)
    if not ticker: st.error("No ticker found."); st.stop()

    # ----- chart controls -----
    st.subheader("Chart Settings")
    colp,coli=st.columns(2)
    period=colp.selectbox("Period",["1d","5d","1mo","3mo","6mo","1y","2y","5y","max"],index=5)
    interval=coli.selectbox("Interval",["1m","5m","15m","1h","1d","1wk","1mo"],index=4)
    sma_chk,bb_chk,rsi_chk=st.columns(3)
    show_sma=sma_chk.checkbox("SMAs",value=True)
    show_bb =bb_chk.checkbox("Bollinger",value=True)
    show_rsi=rsi_chk.checkbox("RSI",value=True)

    with st.spinner("Fetching price…"):
        df=yf.download(ticker,period=period,interval=interval,auto_adjust=False)
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)
    df=df.dropna(subset=["Open","High","Low","Close","Volume"])
    if df.empty: st.error("No data."); st.stop()
    chart(df,ticker,show_sma,show_bb,show_rsi)

    c1,c2,c3,c4=st.columns(4)
    if c1.button("Industry News"): c1.markdown(news_summary(ticker,"market"))
    if c2.button("Company News"):  c2.markdown(news_summary(ticker,"company"))
    if c3.button("Reddit Sentiment"): c3.markdown(reddit_summary(ticker))
    if c4.button("Run All"):
        c4.markdown(news_summary(ticker,"market"))
        c4.markdown(news_summary(ticker,"company"))
        c4.markdown(reddit_summary(ticker))

# ================= ADVANCED SENTIMENT =================
elif st.session_state.page == "adv":
    st.subheader("Advanced Sentiment")

    adv_q = st.text_input("Ticker or company").strip()
    adv_tick = to_ticker(adv_q) if adv_q else None

    source = st.selectbox("Source",
                          ["Industry News", "Company News", "Reddit Sentiment"])
    n_results = st.number_input("# of results", 5, 50, 15)

    # Time‑frame dropdown
    win_label = st.selectbox("Time window",
                             ["Past hour", "Past day", "Past week",
                              "Past month", "Past year"], index=2)
    code_map = {"Past hour": "h", "Past day": "d", "Past week": "w",
                "Past month": "m", "Past year": "y"}
    timeframe = code_map[win_label]

    keyword = st.text_input("Optional keyword")

    if st.button("Run Analysis") and adv_tick:
        kw = keyword or None
        if source == "Industry News":
            st.markdown(news_summary(adv_tick, "market",
                                     num_results=n_results,
                                     timeframe=timeframe, keyword=kw))
        elif source == "Company News":
            st.markdown(news_summary(adv_tick, "company",
                                     num_results=n_results,
                                     timeframe=timeframe, keyword=kw))
        else:
            st.markdown(reddit_summary(adv_tick,
                                       num_results=n_results,
                                       keyword=kw))

    st.button("⬅ Back", on_click=lambda: st.session_state.update(page="main"))
