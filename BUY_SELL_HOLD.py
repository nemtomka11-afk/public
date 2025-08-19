"""
crypto_news_signaler_binance.py

News-first crypto signaler tailored for Binance SPOT and top 3 coins (BTC, ETH, SOL),
with an optional "opportunity coin" when news sentiment + liquidity are exceptional.

Outputs concise lines ONLY (Buy/Sell/Hold + confidence + short why) suitable for piping to a bot.

Safety-first tweaks:
- News-first; tech is secondary multiplier.
- Credibility + recency weighting; minimum evidence gate.
- Dispersion & agreement checks to avoid acting on one-off headlines.
- Volume/liquidity filter on Binance spot before flagging opportunity coins.
- Cooldown to reduce churn.

Extendable modules: sources, sentiment, tech, notifiers.
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import math

import requests
import feedparser
from bs4 import BeautifulSoup

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except Exception:
    NEWSPAPER_AVAILABLE = False

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import pandas as pd
import numpy as np

try:
    import ccxt
    CCXT_AVAILABLE = True
except Exception:
    CCXT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crypto-news-signaler")

# =====================
# CONFIG
# =====================
DEFAULT_CONFIG = {
    "coins": [
        {"symbol": "BTC", "tickers": ["BTC/USDT"], "aliases": ["bitcoin"]},
        {"symbol": "ETH", "tickers": ["ETH/USDT"], "aliases": ["ethereum", "ether"]},
        {"symbol": "SOL", "tickers": ["SOL/USDT"], "aliases": ["solana"]},
    ],
    # optional candidates to surface as an extra opportunity if signal is very strong
    "opportunity_watchlist": [
        {"symbol": "LINK", "tickers": ["LINK/USDT"], "aliases": ["chainlink"]},
        {"symbol": "XRP", "tickers": ["XRP/USDT"], "aliases": []},
        {"symbol": "AVAX", "tickers": ["AVAX/USDT"], "aliases": ["avalanche"]},
        {"symbol": "ATOM", "tickers": ["ATOM/USDT"], "aliases": ["cosmos"]},
    ],
    "rss_feeds": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://www.theblock.co/feed",
    ],
    "sentiment": {
        "use_hf_finbert": True,
        "hf_model": "yiyanghkust/finbert-tone",
        "vader_weight": 0.15,
        "headline_boost": 0.1,  # small boost when title and body align
    },
    "aggregation": {
        "window_minutes": 180,                 # 3h window
        "recency_half_life_minutes": 60,       # 1h half-life
        "source_scores": {
            "coindesk.com": 1.0,
            "cointelegraph.com": 0.9,
            "decrypt.co": 0.85,
            "theblock.co": 0.9,
        },
        "min_weight_to_act": 0.35,             # require enough evidence
        "min_articles": 2,                      # avoid single-headline trades
        "agreement_floor": 0.15,               # median vs mean gap gate (lower=more agreement)
    },
    "technical": {
        "enabled": True,
        "exchange": "binance",                # SPOT only
        "ohlcv_timeframe": "1h",
        "sma_fast": 12,
        "sma_slow": 26,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "tech_multiplier_strength": 0.25,      # keep tech as a modifier
    },
    "opportunity": {
        "enabled": True,
        "news_score_gate": 0.6,                # very strong sentiment
        "binance_min_usdt_volume": 1_000_000,  # last 24h est. via OHLCV sum
    },
    "decision": {
        "buy_threshold": 0.28,                 # slightly stricter
        "sell_threshold": -0.28,
        "cooldown_minutes": 60,                # avoid flip-flopping
    },
    "runtime": {
        "poll_interval_seconds": 300,
    }
}

# =====================
# MODELS
# =====================
@dataclass
class ArticleData:
    url: str
    title: str
    published: datetime
    source: str
    text: str
    coin_mentions: Dict[str, float] = field(default_factory=dict)
    sentiment_score: Optional[float] = None
    title_sent: Optional[float] = None
    raw_sentiment: Dict[str, Any] = field(default_factory=dict)


def now_utc():
    return datetime.utcnow()


def parse_date(entry):
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime.utcfromtimestamp(time.mktime(entry.published_parsed))
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        return datetime.utcfromtimestamp(time.mktime(entry.updated_parsed))
    return now_utc()


# =====================
# FETCHERS
# =====================
class NewsFetcher:
    def __init__(self, config):
        self.config = config

    def fetch_rss(self) -> List[ArticleData]:
        arts = []
        for feed in self.config["rss_feeds"]:
            try:
                d = feedparser.parse(feed)
                for e in d.entries:
                    url = e.get("link") or e.get("id")
                    title = e.get("title", "")
                    published = parse_date(e)
                    source = domain_from_url(url)
                    arts.append(ArticleData(url=url, title=title, published=published, source=source, text=""))
            except Exception:
                logger.exception("RSS parse failed: %s", feed)
        # dedup
        dedup = {}
        for a in arts:
            if a.url and a.url not in dedup:
                dedup[a.url] = a
        return list(dedup.values())


# =====================
# EXTRACTION
# =====================

def extract_text(url: str) -> str:
    try:
        if NEWSPAPER_AVAILABLE:
            art = Article(url)
            art.download(); art.parse()
            t = art.text or ""
            if len(t) > 100:
                return t
    except Exception:
        pass
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        tag = soup.find("article")
        if tag:
            t = " ".join(p.get_text(" ", strip=True) for p in tag.find_all("p"))
            if len(t) > 80:
                return t
        t = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        return t[:20000]
    except Exception:
        return ""


# =====================
# DETECT COIN MENTIONS
# =====================

def detect_mentions(text: str, coins_cfg) -> Dict[str, float]:
    tl = (text or "").lower()
    out = {}
    for c in coins_cfg:
        aliases = [a.lower() for a in c.get("aliases", [])] + [c["symbol"].lower()]
        count = sum(tl.count(a) for a in aliases)
        if count > 0:
            out[c["symbol"]] = min(1.0, 0.5 + math.log1p(count)/10.0)
    return out


# =====================
# SENTIMENT
# =====================
class SentimentEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vader = SentimentIntensityAnalyzer()
        self.hf = None
        if HF_AVAILABLE and self.cfg["sentiment"].get("use_hf_finbert", True):
            try:
                self.hf = pipeline("sentiment-analysis", model=self.cfg["sentiment"].get("hf_model"))
            except Exception:
                self.hf = None

    def _hf_score(self, text: str) -> float:
        try:
            out = self.hf(text[:512])
            if isinstance(out, list):
                out = out[0]
            label = (out.get("label") or "").lower()
            score = float(out.get("score", 0.0))
            if "pos" in label:
                return +score
            if "neg" in label:
                return -score
            return 0.0
        except Exception:
            return 0.0

    def analyze(self, a: ArticleData) -> float:
        text = (a.title or "") + "\n" + (a.text or "")
        vader = self.vader.polarity_scores(text)["compound"]
        title_v = self.vader.polarity_scores(a.title or "")["compound"] if a.title else 0.0
        a.title_sent = title_v
        if self.hf is not None:
            hf = self._hf_score(text)
            w = self.cfg["sentiment"].get("vader_weight", 0.15)
            s = (1-w)*hf + w*vader
        else:
            s = vader
        # boost slightly when title and body align in sign
        boost = self.cfg["sentiment"].get("headline_boost", 0.1)
        if (title_v >= 0 and s >= 0) or (title_v <= 0 and s <= 0):
            s = s * (1 + boost*min(1.0, abs(title_v)))
        return max(-1.0, min(1.0, float(s)))


# =====================
# TECHNICALS (secondary)
# =====================
class TechEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = cfg["technical"].get("enabled", False) and CCXT_AVAILABLE
        self.exchange = None
        if self.enabled:
            try:
                eid = cfg["technical"].get("exchange", "binance")
                self.exchange = getattr(ccxt, eid)({
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                })
            except Exception:
                self.enabled = False

    def ohlcv(self, ticker: str, timeframe: str = "1h", limit: int = 200) -> Optional[pd.DataFrame]:
        if not self.enabled:
            return None
        try:
            data = self.exchange.fetch_ohlcv(ticker, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
            df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("datetime", inplace=True)
            return df
        except Exception:
            return None

    def indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        if df is None or df.empty:
            return {"tech_score": 0.0, "rsi": None, "sma_fast": None, "sma_slow": None}
        close = df["close"]
        sma_fast = close.rolling(window=self.cfg["technical"].get("sma_fast",12)).mean().iloc[-1]
        sma_slow = close.rolling(window=self.cfg["technical"].get("sma_slow",26)).mean().iloc[-1]
        delta = close.diff().dropna()
        up = delta.clip(lower=0).rolling(window=self.cfg["technical"].get("rsi_period",14)).mean()
        down = -1*delta.clip(upper=0).rolling(window=self.cfg["technical"].get("rsi_period",14)).mean()
        rs = up/(down.replace(0,np.nan))
        rsi = 100 - (100/(1+rs.iloc[-1])) if not rs.isna().all() else None
        tech = 0.0
        if sma_fast and sma_slow:
            cross = (sma_fast - sma_slow)/max(1e-9, sma_slow)
            tech += max(-0.6, min(0.6, cross*4))
        if rsi is not None:
            if rsi < self.cfg["technical"].get("rsi_oversold",30):
                tech += 0.2
            elif rsi > self.cfg["technical"].get("rsi_overbought",70):
                tech -= 0.2
        return {"tech_score": float(max(-1.0,min(1.0,tech))), "rsi": float(rsi) if rsi is not None else None,
                "sma_fast": float(sma_fast) if sma_fast else None, "sma_slow": float(sma_slow) if sma_slow else None}

    def est_24h_usdt_volume(self, ticker: str) -> float:
        # estimate via 1h OHLCV sum of volume * close
        df = self.ohlcv(ticker, timeframe="1h", limit=24)
        if df is None or df.empty:
            return 0.0
        return float((df["volume"]*df["close"]).sum())


# =====================
# DECISION ENGINE
# =====================
class DecisionEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.agg_cfg = cfg["aggregation"]
        self.dec_cfg = cfg["decision"]

    def domain_weight(self, domain: str) -> float:
        return self.agg_cfg.get("source_scores", {}).get(domain, 0.6)

    def time_decay(self, published: datetime) -> float:
        age_min = (now_utc() - published).total_seconds()/60.0
        half = self.agg_cfg.get("recency_half_life_minutes", 60)
        return 2 ** (-age_min/half)

    def aggregate(self, arts: List[ArticleData], coin: str) -> Dict[str, Any]:
        window = timedelta(minutes=self.agg_cfg.get("window_minutes",180))
        cutoff = now_utc() - window
        weights, scores, contribs = [], [], []
        for a in arts:
            if a.published < cutoff:
                continue
            m = a.coin_mentions.get(coin, 0.0)
            if m <= 0:
                continue
            w = m * self.domain_weight(a.source) * self.time_decay(a.published)
            s = a.sentiment_score or 0.0
            weights.append(w); scores.append(s)
            contribs.append({"url": a.url, "sent": s, "w": w, "title": a.title})
        if not weights:
            return {"news_score": 0.0, "weight_total": 0.0, "n": 0, "median": 0.0, "mean": 0.0, "contrib": []}
        wsum = sum(weights)
        mean = sum(w*s for w,s in zip(weights, scores))/wsum
        med = float(np.median(scores))
        return {"news_score": float(max(-1,min(1,mean))), "weight_total": float(wsum), "n": len(weights),
                "median": med, "mean": float(mean), "contrib": contribs}

    def final(self, news_score: float, tech_score: float) -> float:
        mult = 1 + self.cfg["technical"].get("tech_multiplier_strength",0.25)*(tech_score or 0.0)
        return float(max(-1.0, min(1.0, news_score * mult)))


# =====================
# NOTIFIER (concise)
# =====================
class ConciseNotifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_signal_time: Dict[str, datetime] = {}

    def _cooldown_ok(self, sym: str) -> bool:
        cd = self.cfg["decision"].get("cooldown_minutes", 60)
        last = self.last_signal_time.get(sym)
        if not last:
            return True
        return (now_utc() - last) >= timedelta(minutes=cd)

    def notify(self, sym: str, signal: str, conf: float, why: str):
        # Single line, minimal, human-friendly
        print(f"{sym}: {signal} ({int(conf*100)}%) — {why}")
        self.last_signal_time[sym] = now_utc()


# =====================
# HELPERS
# =====================
from urllib.parse import urlparse

def domain_from_url(u: str) -> str:
    try:
        return urlparse(u).netloc.replace("www.", "")
    except Exception:
        return ""


# =====================
# MAIN LOOP
# =====================

def run(config: Dict[str,Any]):
    fetcher = NewsFetcher(config)
    senti = SentimentEngine(config)
    tech = TechEngine(config)
    decide = DecisionEngine(config)
    out = ConciseNotifier(config)

    known: Dict[str, ArticleData] = {}

    while True:
        try:
            arts = fetcher.fetch_rss()
            for a in arts:
                if a.url in known:
                    continue
                a.text = extract_text(a.url)
                a.coin_mentions = detect_mentions(a.title + "\n" + a.text, config["coins"] + (config.get("opportunity_watchlist", []) if config.get("opportunity",{}).get("enabled", True) else []))
                if not a.coin_mentions:
                    continue
                a.sentiment_score = senti.analyze(a)
                known[a.url] = a

            # sliding window cleanup
            keep_for = timedelta(minutes=config["aggregation"]["window_minutes"]*3)
            cutoff = now_utc() - keep_for
            for k,v in list(known.items()):
                if v.published < cutoff:
                    known.pop(k, None)

            articles = list(known.values())

            # process core coins
            for coin in config["coins"]:
                sym = coin["symbol"]
                agg = decide.aggregate(articles, sym)
                # evidence gates
                if agg["weight_total"] < config["aggregation"]["min_weight_to_act"] or agg["n"] < config["aggregation"]["min_articles"]:
                    news_score = 0.0
                else:
                    # agreement check: if mean and median diverge too much, reduce score
                    gap = abs(agg["mean"] - agg["median"])  # dispersion proxy
                    penalty = max(0.0, gap - config["aggregation"].get("agreement_floor",0.15))
                    news_score = math.copysign(max(0.0, abs(agg["news_score"]) - penalty), agg["news_score"])

                # tech
                tech_res = {"tech_score": 0.0}
                if tech.enabled:
                    df = None
                    for t in coin.get("tickers", []):
                        df = tech.ohlcv(t, timeframe=config["technical"].get("ohlcv_timeframe","1h"), limit=200)
                        if df is not None:
                            break
                    tech_res = tech.indicators(df)

                final = decide.final(news_score, tech_res.get("tech_score",0.0))
                signal = "HOLD"
                if final >= config["decision"]["buy_threshold"]:
                    signal = "BUY"
                elif final <= config["decision"]["sell_threshold"]:
                    signal = "SELL"

                conf = abs(final)

                # concise explanation assembled from top contributor title/domain
                why = "news-driven"
                if agg["contrib"]:
                    top = sorted(agg["contrib"], key=lambda x: -x["w"])[0]
                    why = f"{domain_from_url(top['url'])} headline; tech {'supports' if tech_res.get('tech_score',0)>=0 else 'softens'}"

                # cooldown + only act if not HOLD or strong evidence
                if signal != "HOLD" and out._cooldown_ok(sym):
                    out.notify(sym, signal, conf, why)
                elif signal == "HOLD" and agg["weight_total"] >= config["aggregation"]["min_weight_to_act"] and out._cooldown_ok(sym):
                    out.notify(sym, signal, conf, "insufficient edge / mixed news")

            # scan opportunity
            if config.get("opportunity",{}).get("enabled", True):
                for coin in config.get("opportunity_watchlist", []):
                    sym = coin["symbol"]
                    agg = decide.aggregate(articles, sym)
                    if agg["n"] < 2:
                        continue
                    if agg["news_score"] < config["opportunity"]["news_score_gate"]:
                        continue
                    # liquidity filter on Binance spot
                    vol = 0.0
                    if tech.enabled:
                        vol = tech.est_24h_usdt_volume(coin["tickers"][0])
                    if vol < config["opportunity"]["binance_min_usdt_volume"]:
                        continue
                    # surface as a BUY idea with confidence tied to news_score
                    conf = min(1.0, agg["news_score"])  # 0.6..1.0
                    if out._cooldown_ok(sym):
                        why = f"opportunity: strong news + volume {int(vol):,} USDT"
                        out.notify(sym, "BUY", conf, why)

            time.sleep(config["runtime"].get("poll_interval_seconds", 300))
        except KeyboardInterrupt:
            break
        except Exception:
            logger.exception("Loop error; retrying soon...")
            time.sleep(10)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to JSON config override")
    args = parser.parse_args()
    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        try:
            with open(args.config, "r") as f:
                user_cfg = json.load(f)
            # shallow merge for brevity
            cfg.update(user_cfg)
            logger.info("Loaded config override: %s", args.config)
        except Exception:
            logger.exception("Failed to load config override; using defaults")
    run(cfg)
