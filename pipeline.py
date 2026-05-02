from datetime import date, timedelta
import pandas as pd, numpy as np
from agents.price_agent import PriceAgent
from agents.inventory_agent import InventoryAgent
from agents.reserve_agent import ReserveAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.shipping_agent import ShippingAgent
from utils.config import load_config


def collect_all(start=None, end=None):
    cfg = load_config()
    agents = [
        PriceAgent(cfg["forecast"]["oil_symbols"]),
        InventoryAgent(), ReserveAgent(), NewsSentimentAgent(), ShippingAgent()
    ]
    results = {a.name: a.collect(start, end) for a in agents}
    return results


def build_feature_table(results: dict, target_col="WTI_close") -> pd.DataFrame:
    price = results["price_agent"].data.copy()
    if price.empty:
        raise ValueError("No price data available. Check internet access or upload data/raw_prices.csv.")
    df = price.copy()
    for name in ["inventory_agent", "news_sentiment_agent", "shipping_agent"]:
        d = results[name].data.copy()
        if not d.empty and "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"])
            df = df.merge(d, on="date", how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").ffill().fillna(0)
    price_cols = [c for c in df.columns if c.endswith("_close")]
    for c in price_cols:
        df[f"{c}_ret1"] = df[c].pct_change()
        df[f"{c}_ma5"] = df[c].rolling(5).mean()
        df[f"{c}_ma20"] = df[c].rolling(20).mean()
        df[f"{c}_vol10"] = df[c].pct_change().rolling(10).std()
    if "WTI_volume" in df.columns:
        df["WTI_volume_ma5"] = df["WTI_volume"].rolling(5).mean()
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


def make_supervised(df: pd.DataFrame, target_col="WTI_close", horizon=7):
    y_cols=[]
    out=df.copy()
    for h in range(1, horizon+1):
        col=f"target_t_plus_{h}"
        out[col]=out[target_col].shift(-h)
        y_cols.append(col)
    non_features = ["date"] + y_cols + [c for c in out.columns if c in ["headline","source","country"]]
    feature_cols = [c for c in out.columns if c not in non_features and pd.api.types.is_numeric_dtype(out[c])]
    out = out.dropna().reset_index(drop=True)
    return out[feature_cols], out[y_cols], feature_cols
