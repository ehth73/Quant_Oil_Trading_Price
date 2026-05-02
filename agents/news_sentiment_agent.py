import re, pandas as pd
try:
    import feedparser
except Exception:
    feedparser = None
from datetime import datetime, timezone
from .base import BaseAgent, AgentResult

POSITIVE = {"rise","rises","rally","bullish","draw","shortage","cuts","sanction","disruption","demand","growth","attack","tension"}
NEGATIVE = {"fall","falls","bearish","surplus","build","weak","slowdown","glut","peace","ceasefire","oversupply","recession"}
FEEDS = [
    "https://news.google.com/rss/search?q=oil+prices+crude+inventory+OPEC+geopolitical+shipping&hl=en-SG&gl=SG&ceid=SG:en",
    "https://www.reuters.com/markets/commodities/rss",
]

def score_text(text: str) -> float:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words: return 0.0
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    return (pos - neg) / max(1, pos + neg)

class NewsSentimentAgent(BaseAgent):
    name = "news_sentiment_agent"
    def collect(self, start=None, end=None) -> AgentResult:
        if feedparser is None:
            return AgentResult(self.name, pd.DataFrame(columns=["date","headline","sentiment","source"]), "feedparser not installed; empty sentiment fallback used.")
        rows = []
        for feed in FEEDS:
            try:
                parsed = feedparser.parse(feed)
                for e in parsed.entries[:80]:
                    published = getattr(e, "published", None) or getattr(e, "updated", None)
                    dt = pd.to_datetime(published, errors="coerce") if published else pd.Timestamp.utcnow()
                    title = getattr(e, "title", "")
                    summary = getattr(e, "summary", "")
                    rows.append({"date": dt.date(), "headline": title, "sentiment": score_text(title + " " + summary), "source": feed})
            except Exception:
                pass
        if not rows:
            return AgentResult(self.name, pd.DataFrame(columns=["date","headline","sentiment","source"]), "RSS unavailable; returns empty sentiment.")
        raw = pd.DataFrame(rows)
        daily = raw.groupby("date", as_index=False).agg(news_sentiment=("sentiment","mean"), news_count=("headline","count"))
        daily["date"] = pd.to_datetime(daily["date"])
        return AgentResult(self.name, daily, "Daily oil news sentiment from RSS headlines.")
