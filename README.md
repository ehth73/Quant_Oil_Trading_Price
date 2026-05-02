# Oil Multi-Agent Price Forecaster for Hugging Face Spaces

This project creates a deployable multi-agent oil analytics system that gathers daily and near-real-time signals, builds explainable ML features, and forecasts the next 7 business days of oil prices.

## What it includes

- `app.py` — Streamlit dashboard for forecasts, confidence intervals, raw data and downloads.
- `train_model.py` — trains an ensemble + neural-network model and saves `models/oil_price_forecaster.joblib`.
- `pipeline.py` — combines agent data into ML features.
- `forecast.py` — loads the `joblib` model and produces next-week forecasts.
- `agents/`
  - `price_agent.py`: WTI, Brent, heating oil, gasoline and gas futures via Yahoo/yfinance.
  - `inventory_agent.py`: EIA inventory connector using optional `EIA_API_KEY`.
  - `reserve_agent.py`: reserve dataset scaffold by major countries.
  - `news_sentiment_agent.py`: RSS-based oil/geopolitical/news sentiment.
  - `shipping_agent.py`: shipping CSV/API connector for tanker rates, floating storage and chokepoint risk.
- `data/` and `exports/` — raw and output folders.
- `.github/workflows/hf-sync.yml` — optional GitHub-to-Hugging-Face sync template.

## Model design

The default model is a multi-output 7-day forecaster using:

- Random Forest Regressor
- Extra Trees Regressor
- MLP Neural Network
- Voting ensemble wrapped in `MultiOutputRegressor`

Features include:

- multi-class oil prices: WTI, Brent, heating oil, gasoline, natural gas
- returns, moving averages and volatility
- volume features
- inventory features when an EIA key is provided
- news/geopolitical sentiment
- shipping/chokepoint features when provided
- calendar features

Confidence intervals are estimated from validation residual standard deviation. For investment-grade deployment, backtest regularly and replace this with conformal prediction or quantile models.

## Hugging Face deployment

1. Create a new Hugging Face Space.
2. Choose **Streamlit** as the SDK.
3. Upload all files in this repository.
4. Add optional Secrets:
   - `EIA_API_KEY`
   - `NEWS_API_KEY` if you extend the news agent
   - `SHIPPING_CSV` path or a shipping API key if you extend the shipping agent
5. The Space will run `app.py` automatically.

## Train or refresh the model

```bash
python train_model.py
```

This creates:

```text
models/oil_price_forecaster.joblib
data/latest_feature_table.csv
```

If live collection fails, the trainer falls back to synthetic bootstrap data so the app remains runnable. Replace it with real historical CSVs or API keys for production.

## Add shipping data

Create `data/shipping_signals.csv` with columns:

```csv
date,tanker_rate_index,floating_storage_index,chokepoint_risk_score
2025-01-01,103.2,48.1,0.20
```

## Important disclaimer

This is a research and educational forecasting system, not financial advice. Oil markets are affected by geopolitical shocks, OPEC decisions, macro data, weather, shipping disruptions and liquidity. Use robust backtesting, data-quality controls and human review before making investment decisions.
