import pandas as pd
from .base import BaseAgent, AgentResult

class ReserveAgent(BaseAgent):
    name = "reserve_agent"
    def collect(self, start=None, end=None) -> AgentResult:
        # Static explainability feature. Replace with API/CSV when reserve datasets are licensed.
        rows = [
            {"country":"Venezuela","proved_reserves_billion_bbl":303.8},
            {"country":"Saudi Arabia","proved_reserves_billion_bbl":267.2},
            {"country":"Canada","proved_reserves_billion_bbl":171.0},
            {"country":"Iran","proved_reserves_billion_bbl":208.6},
            {"country":"Iraq","proved_reserves_billion_bbl":145.0},
            {"country":"Russia","proved_reserves_billion_bbl":80.0},
            {"country":"United States","proved_reserves_billion_bbl":68.8},
        ]
        return AgentResult(self.name, pd.DataFrame(rows), "Static reserve table; update annually or connect API.")
