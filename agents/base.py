from dataclasses import dataclass
import pandas as pd

@dataclass
class AgentResult:
    name: str
    data: pd.DataFrame
    notes: str = ""

class BaseAgent:
    name = "base_agent"
    def collect(self, start: str | None = None, end: str | None = None) -> AgentResult:
        raise NotImplementedError
