import pandas as pd
try:
    import yfinance as yf
except Exception:
    yf = None
from .base import BaseAgent, AgentResult

class PriceAgent(BaseAgent):
    name = "price_agent"
    def __init__(self, symbols: dict[str, str]):
        self.symbols = symbols

    def collect(self, start=None, end=None) -> AgentResult:
        if yf is None:
            return AgentResult(self.name, pd.DataFrame(), "yfinance is not installed in this runtime; install requirements or use fallback training.")
        frames = []
        for label, symbol in self.symbols.items():
            try:
                df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
                if df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                out = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
                out.columns = ["date", f"{label}_open", f"{label}_high", f"{label}_low", f"{label}_close", f"{label}_volume"]
                frames.append(out)
            except Exception:
                continue
        if not frames:
            return AgentResult(self.name, pd.DataFrame(), "No price data returned. Check internet access/symbols.")
        data = frames[0]
        for f in frames[1:]:
            data = data.merge(f, on="date", how="outer")
        data = data.sort_values("date")
        return AgentResult(self.name, data, "Daily futures prices via yfinance.")
