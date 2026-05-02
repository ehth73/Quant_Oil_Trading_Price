from pathlib import Path
import numpy as np, pandas as pd, joblib
from pipeline import collect_all, build_feature_table

ROOT=Path(__file__).resolve().parent

def load_model(path=ROOT/"models"/"oil_price_forecaster.joblib"):
    return joblib.load(path)

def forecast_next_week(df_features: pd.DataFrame, bundle: dict):
    latest=df_features.sort_values("date").iloc[-1:]
    X=latest.reindex(columns=bundle["feature_cols"], fill_value=0)
    pred=bundle["model"].predict(X)[0]
    sigma=np.array(bundle.get("residual_std", [1]*len(pred)))
    last_date=pd.to_datetime(latest["date"].iloc[0])
    dates=pd.bdate_range(last_date + pd.Timedelta(days=1), periods=bundle.get("horizon",7))
    return pd.DataFrame({"date":dates,"forecast_price":pred,"lower_80":pred-1.28*sigma,"upper_80":pred+1.28*sigma,"lower_95":pred-1.96*sigma,"upper_95":pred+1.96*sigma})

def run_live_forecast(start=None):
    bundle=load_model()
    results=collect_all(start=start)
    df=build_feature_table(results, target_col=bundle.get("target_col","WTI_close"))
    fc=forecast_next_week(df,bundle)
    (ROOT/"exports").mkdir(exist_ok=True)
    df.to_csv(ROOT/"exports"/"raw_feature_data.csv", index=False)
    fc.to_csv(ROOT/"exports"/"daily_forecast.csv", index=False)
    return df, fc, bundle, results
