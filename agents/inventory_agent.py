import os, requests, pandas as pd
from .base import BaseAgent, AgentResult

class InventoryAgent(BaseAgent):
    name = "inventory_agent"
    # EIA weekly US crude stocks example series. Users can modify/add country-specific series.
    SERIES = {"us_crude_stocks": "PET.WCESTUS1.W"}
    def collect(self, start=None, end=None) -> AgentResult:
        key = os.environ.get("EIA_API_KEY")
        if not key:
            return AgentResult(self.name, pd.DataFrame(columns=["date","us_crude_stocks"]), "Set EIA_API_KEY for real inventory feeds; empty fallback used.")
        frames=[]
        for col, sid in self.SERIES.items():
            url = "https://api.eia.gov/v2/seriesid/" + sid
            try:
                js = requests.get(url, params={"api_key": key}, timeout=20).json()
                rows = js.get("response", {}).get("data", [])
                df = pd.DataFrame(rows)
                if not df.empty:
                    value_col = "value" if "value" in df.columns else df.columns[-1]
                    frames.append(df.rename(columns={"period":"date", value_col: col})[["date", col]])
            except Exception:
                pass
        if not frames: return AgentResult(self.name, pd.DataFrame(), "No inventory data returned.")
        data=frames[0]
        for f in frames[1:]: data=data.merge(f,on="date",how="outer")
        data["date"] = pd.to_datetime(data["date"])
        return AgentResult(self.name, data, "Weekly inventory data from EIA API.")
