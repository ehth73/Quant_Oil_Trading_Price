import os, pandas as pd
from .base import BaseAgent, AgentResult

class ShippingAgent(BaseAgent):
    name = "shipping_agent"
    def collect(self, start=None, end=None) -> AgentResult:
        csv_path = os.environ.get("SHIPPING_CSV", "data/shipping_signals.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])
            return AgentResult(self.name, df, "Shipping signals loaded from CSV.")
        return AgentResult(self.name, pd.DataFrame(columns=["date","tanker_rate_index","floating_storage_index","chokepoint_risk_score"]), "No shipping CSV/API configured; empty fallback used.")
