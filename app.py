import streamlit as st, pandas as pd, plotly.graph_objects as go
from datetime import date, timedelta
from pathlib import Path
from forecast import run_live_forecast, load_model, forecast_next_week
from pipeline import collect_all, build_feature_table

st.set_page_config(page_title="Oil Multi-Agent Price Forecaster", layout="wide")
st.title("🛢️ Oil Multi-Agent Price Forecaster")
st.caption("Agents collect prices, inventories, reserves, news sentiment, shipping/geopolitical signals, then forecast daily oil prices for the next week.")

with st.sidebar:
    st.header("Controls")
    years=st.slider("Historical years", 1, 5, 5)
    target=st.selectbox("Oil class target", ["WTI_close"], index=0)
    refresh=st.button("Collect data and forecast", type="primary")
    st.info("Set EIA_API_KEY, NEWS_API_KEY or SHIPPING_CSV as Hugging Face Secrets for richer live data.")

@st.cache_data(ttl=3600)
def cached_run(years):
    start=(date.today()-timedelta(days=365*years+30)).isoformat()
    return run_live_forecast(start=start)

try:
    df, fc, bundle, results = cached_run(years)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Model MAE", f"{bundle.get('mae',0):.2f}")
    c2.metric("Training rows", bundle.get("trained_rows",0))
    c3.metric("Last train date", bundle.get("last_train_date","n/a"))
    c4.metric("Latest WTI", f"{df['WTI_close'].iloc[-1]:.2f}")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["date"].tail(120), y=df["WTI_close"].tail(120), name="WTI actual"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast_price"], name="7-day forecast"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["upper_95"], name="95% upper", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["lower_95"], name="95% lower", line=dict(dash="dot")))
    fig.update_layout(height=520, title="WTI Daily Forecast with Confidence Interval")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Forecast table")
    st.dataframe(fc, use_container_width=True)
    st.download_button("Download forecast CSV", fc.to_csv(index=False), "daily_forecast.csv")
    st.subheader("Raw feature data")
    st.dataframe(df.tail(200), use_container_width=True)
    st.download_button("Download raw feature CSV", df.to_csv(index=False), "raw_feature_data.csv")
    st.subheader("Agent status")
    for name,res in results.items():
        st.write(f"**{name}** — {res.notes} Rows: {len(res.data)}")
except Exception as e:
    st.error(f"Unable to run live forecast: {e}")
    st.write("Run `python train_model.py` locally or in the Space terminal to refresh the model.")
