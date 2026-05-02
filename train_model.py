from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from pipeline import collect_all, build_feature_table, make_supervised

ROOT=Path(__file__).resolve().parent


def fallback_training_data(days=900):
    rng=np.random.default_rng(42)
    dates=pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    shocks=rng.normal(0,1.2,len(dates))
    close=70+np.cumsum(shocks*0.18)+4*np.sin(np.arange(len(dates))/45)
    brent=close+3+rng.normal(0,.6,len(dates))
    df=pd.DataFrame({
        "date":dates,"WTI_open":close+rng.normal(0,.5,len(dates)),"WTI_high":close+1.2,"WTI_low":close-1.2,"WTI_close":close,"WTI_volume":rng.integers(100000,500000,len(dates)),
        "Brent_open":brent,"Brent_high":brent+1,"Brent_low":brent-1,"Brent_close":brent,"Brent_volume":rng.integers(80000,400000,len(dates)),
        "news_sentiment":rng.normal(0,.25,len(dates)),"news_count":rng.integers(1,20,len(dates)),
        "tanker_rate_index":100+rng.normal(0,10,len(dates)),"floating_storage_index":50+rng.normal(0,5,len(dates)),"chokepoint_risk_score":rng.random(len(dates))
    })
    return df


def train(use_live=True):
    if use_live:
        try:
            start=(date.today()-timedelta(days=365*5+30)).isoformat()
            results=collect_all(start=start)
            df=build_feature_table(results)
        except Exception as e:
            print("Live collection failed; using fallback synthetic bootstrap data:", e)
            df=fallback_training_data()
    else:
        df=fallback_training_data()
    # add engineered features even for fallback
    from pipeline import build_feature_table as _bft
    if "WTI_close_ret1" not in df.columns:
        class R: pass
        r=R(); r.data=df
        empty=R(); empty.data=pd.DataFrame()
        df=_bft({"price_agent":r,"inventory_agent":empty,"news_sentiment_agent":empty,"shipping_agent":empty})
    X,y,features=make_supervised(df,horizon=7)
    split=max(30,int(len(X)*0.82))
    X_train,X_test=X.iloc[:split],X.iloc[split:]
    y_train,y_test=y.iloc[:split],y.iloc[split:]
    rf=RandomForestRegressor(n_estimators=60, random_state=42, min_samples_leaf=3, n_jobs=-1)
    et=ExtraTreesRegressor(n_estimators=60, random_state=43, min_samples_leaf=3, n_jobs=-1)
    mlp=Pipeline([("scaler",StandardScaler()),("mlp",MLPRegressor(hidden_layer_sizes=(48,24), random_state=44, max_iter=250, early_stopping=True))])
    model=MultiOutputRegressor(VotingRegressor([("rf",rf),("et",et),("mlp",mlp)]))
    model.fit(X_train,y_train)
    preds=model.predict(X_test) if len(X_test) else model.predict(X_train)
    truth=y_test.values if len(X_test) else y_train.values
    residuals=truth-preds
    mae=float(mean_absolute_error(truth,preds))
    bundle={"model":model,"feature_cols":features,"target_col":"WTI_close","horizon":7,"residual_std":np.nanstd(residuals,axis=0).tolist(),"mae":mae,"trained_rows":len(X),"last_train_date":str(df["date"].max().date())}
    (ROOT/"models").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT/"models"/"oil_price_forecaster.joblib")
    df.to_csv(ROOT/"data"/"latest_feature_table.csv", index=False)
    print("Saved models/oil_price_forecaster.joblib", bundle)

if __name__ == "__main__":
    train(use_live=True)
