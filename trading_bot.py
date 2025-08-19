"""
news_binance_trader.py

News-first trading bot for Binance Spot (BTC, ETH, SOL).
- Uses RSS news -> FinBERT (preferred) or VADER fallback for sentiment.
- Aggregates news per coin (3-hour window default) and emits BUY/SELL/HOLD.
- Executes trades automatically on Binance Spot (ccxt) only when confidence >= EXECUTION_CONFIDENCE.
- Starts trade size = BASE_TRADE_USDT (default $50). After profitable closed trades increases trade budget.
- Places market buy, then sets TP (limit) and SL (stop-limit). Polls for fills and cancels the loser.
- Persists state (trade sizes, cooldowns, open positions) in state_file (JSON).
- Logs actions to console and to file.

IMPORTANT SAFETY:
 - Use Testnet first. If using real keys, disable withdrawals and restrict IP (if possible).
 - Start with a small base trade size while testing.
"""

import os
import time
import math
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import requests
import feedparser
from bs4 import BeautifulSoup

# sentiment libs
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# market / exchange
import ccxt
import pandas as pd
import numpy as np

# ---------------------------
# CONFIG (tweak these)
# ---------------------------
CONFIG = {
    # tracked coins (symbol = base asset) mapped to Binance spot pair (quote USDT)
    "coins": {
        "BTC": {"pair": "BTC/USDT", "aliases": ["bitcoin"]},
        "ETH": {"pair": "ETH/USDT", "aliases": ["ethereum", "ether"]},
        "SOL": {"pair": "SOL/USDT", "aliases": ["solana"]},
    },
    # optional watchlist for "opportunity" surfacing
    "opportunity_watchlist": {
        # "LINK": {"pair": "LINK/USDT", "aliases": ["chainlink"]},
        # add if you want automatic opportunity scanning
    },
    # RSS feeds to monitor (edit or extend)
    "rss_feeds": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://www.theblock.co/feed",
    ],
    # sentiment model
    "sentiment": {
        "use_hf_finbert": True,  # if HF available we try to use FinBERT-like model
        "hf_model": "yiyanghkust/finbert-tone",
        "vader_weight": 0.12,  # small weight if combining
        "headline_boost": 0.08,
    },
    # aggregation / decision
    "aggregation": {
        "window_minutes": 180,
        "recency_half_life_minutes": 60,
        "min_articles": 2,
        "min_weight_to_act": 0.25,
        "agreement_floor": 0.15,
    },
    # technical (secondary)
    "technical": {
        "enabled": True,
        "ohlcv_timeframe": "1h",
        "sma_fast": 12,
        "sma_slow": 26,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "tech_multiplier_strength": 0.25,
    },
    # trading behavior
    "execution_confidence": 0.70,   # must be >= this to place real orders
    "base_trade_usdt": 50.0,        # initial $ per trade
    "min_trade_usdt": 10.0,         # minimum to place an order
    "max_trade_usdt": 2000.0,       # safety cap for automatic scaling
    "scale_up_on_win_usd": 10.0,    # add $ after a profitable closed trade
    "stop_loss_pct": 0.03,          # 3% stop loss
    "take_profit_pct": 0.06,        # 6% take profit
    "cooldown_minutes": 60,         # cooldown per coin after action
    "poll_interval_seconds": 300,   # how often to check news & markets
    # persistence/logs
    "state_file": "trader_state.json",
    "log_file": "trader.log",
    # Binance settings
    "binance": {
        "testnet": False,  # set True to use Binance Spot Testnet (recommended for testing)
        # When testnet=True, you must point to testnet URLs and use testnet keys
    }
}

# ---------------------------
# logging
# ---------------------------
logger = logging.getLogger("news-binance-trader")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)
fh = logging.FileHandler(CONFIG["log_file"])
fh.setFormatter(fmt)
logger.addHandler(fh)

# ---------------------------
# Utilities: time, domain, persistence
# ---------------------------
def now_utc():
    return datetime.utcnow()

def domain_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""

def load_state(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            logger.exception("Failed to load state file, creating fresh state.")
    # default state
    state = {
        "trade_usdt": {sym: CONFIG["base_trade_usdt"] for sym in CONFIG["coins"].keys()},
        "last_action": {},   # sym -> ISO timestamp
        "open_positions": {},  # sym -> {entry_price, amount, usdt_spent, sl, tp, orders: {...}}
        "trade_history": [],   # list of closed trade records
    }
    save_state(path, state)
    return state

def save_state(path: str, state: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)

# ---------------------------
# News fetching & extraction
# ---------------------------
def fetch_rss_articles(feeds: List[str]) -> List[Dict[str, Any]]:
    articles = []
    for feed in feeds:
        try:
            d = feedparser.parse(feed)
            for e in d.entries:
                url = e.get("link") or e.get("id") or ""
                title = e.get("title", "") or ""
                published = None
                if e.get("published_parsed"):
                    import time as _time
                    published = datetime.utcfromtimestamp(_time.mktime(e.published_parsed))
                else:
                    published = now_utc()
                source = domain_from_url(url)
                articles.append({"url": url, "title": title, "published": published.isoformat(), "source": source})
        except Exception:
            logger.exception("RSS fetch failed for %s", feed)
    # deduplicate by url
    seen = set()
    dedup = []
    for a in articles:
        if a["url"] and a["url"] not in seen:
            seen.add(a["url"])
            dedup.append(a)
    return dedup

def extract_text(url: str) -> str:
    # Try a simple requests + BeautifulSoup extractor; newspaper3k omitted to reduce heavy deps
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        # prefer <article>
        article_tag = soup.find("article")
        if article_tag:
            text = " ".join(p.get_text(" ", strip=True) for p in article_tag.find_all("p"))
            if len(text) > 80:
                return text
        # fallback gather p
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        return text[:30000]
    except Exception:
        return ""

# ---------------------------
# Sentiment engine (HF FinBERT preferred, VADER fallback)
# ---------------------------
class SentimentEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vader = SentimentIntensityAnalyzer()
        self.hf_pipe = None
        if HF_AVAILABLE and cfg["sentiment"].get("use_hf_finbert", True):
            try:
                logger.info("Loading HF sentiment model (may download first time): %s", cfg["sentiment"]["hf_model"])
                self.hf_pipe = pipeline("sentiment-analysis", model=cfg["sentiment"]["hf_model"])
            except Exception:
                logger.exception("HF model load failed; falling back to VADER")
                self.hf_pipe = None

    def analyze_text(self, title: str, body: str) -> Dict[str, Any]:
        """
        Returns normalized sentiment score [-1..1] and confidence [0..1].
        """
        text = (title or "") + "\n" + (body or "")
        if not text.strip():
            return {"score": 0.0, "confidence": 0.0, "raw": {}}

        # Vader baseline
        vader_compound = self.vader.polarity_scores(text)["compound"]

        if self.hf_pipe:
            try:
                # HF pipeline returns label and score, label like 'POSITIVE' or 'NEGATIVE' or 'NEUTRAL'
                out = self.hf_pipe(text[:512])
                if isinstance(out, list):
                    out = out[0]
                label = (out.get("label") or "").lower()
                score = float(out.get("score", 0.0))
                # map label to signed score
                if "pos" in label:
                    hf_score = score
                elif "neg" in label:
                    hf_score = -score
                else:
                    hf_score = 0.0
                # combine HF dominant + small VADER
                w_v = self.cfg["sentiment"].get("vader_weight", 0.12)
                combined = (1 - w_v) * hf_score + w_v * vader_compound
                # headline boost if headline agrees with body
                title_v = self.vader.polarity_scores(title or "")["compound"] if title else 0.0
                if (title_v >= 0 and combined >= 0) or (title_v <= 0 and combined <= 0):
                    combined = combined * (1 + self.cfg["sentiment"].get("headline_boost", 0.08) * min(1.0, abs(title_v)))
                return {"score": float(max(-1.0, min(1.0, combined))), "confidence": float(score), "raw": {"hf": out, "vader": vader_compound}}
            except Exception:
                logger.exception("HF sentiment failed mid-run; using VADER")
                return {"score": float(vader_compound), "confidence": abs(vader_compound), "raw": {"vader": vader_compound}}
        else:
            return {"score": float(vader_compound), "confidence": abs(vader_compound), "raw": {"vader": vader_compound}}

# ---------------------------
# Simple coin mention detector
# ---------------------------
def detect_coin_mentions(text: str, coins_cfg: Dict[str, Any]) -> Dict[str, float]:
    tl = (text or "").lower()
    out = {}
    for sym, meta in coins_cfg.items():
        aliases = [a.lower() for a in meta.get("aliases", [])] + [sym.lower()]
        count = sum(tl.count(a) for a in aliases)
        if count > 0:
            out[sym] = min(1.0, 0.5 + math.log1p(count) / 8.0)
    return out

# ---------------------------
# Tech engine (tiny sanity checks)
# ---------------------------
class TechEngine:
    def __init__(self, cfg, ccxt_exchange: Optional[ccxt.Exchange]):
        self.cfg = cfg
        self.exchange = ccxt_exchange
        self.enabled = cfg["technical"].get("enabled", True) and (ccxt_exchange is not None)

    def fetch_ohlcv_df(self, symbol_pair: str, timeframe: str="1h", limit: int=200) -> Optional[pd.DataFrame]:
        if not self.enabled:
            return None
        try:
            raw = self.exchange.fetch_ohlcv(symbol_pair, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
            df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("datetime", inplace=True)
            return df
        except Exception:
            logger.exception("Failed to fetch OHLCV for %s", symbol_pair)
            return None

    def compute_tech_score(self, df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return 0.0
        close = df["close"]
        sma_f = close.rolling(window=self.cfg["technical"].get("sma_fast",12)).mean().iloc[-1]
        sma_s = close.rolling(window=self.cfg["technical"].get("sma_slow",26)).mean().iloc[-1]
        tech = 0.0
        if sma_f and sma_s:
            cross = (sma_f - sma_s) / max(1e-9, sma_s)
            tech += max(-0.6, min(0.6, cross * 4))
        # RSI
        delta = close.diff().dropna()
        up = delta.clip(lower=0).rolling(window=self.cfg["technical"].get("rsi_period",14)).mean()
        down = -1*delta.clip(upper=0).rolling(window=self.cfg["technical"].get("rsi_period",14)).mean()
        rs = up/(down.replace(0,np.nan))
        rsi = None
        if not rs.isna().all():
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
            if rsi < self.cfg["technical"].get("rsi_oversold",30):
                tech += 0.2
            elif rsi > self.cfg["technical"].get("rsi_overbought",70):
                tech -= 0.2
        return float(max(-1.0, min(1.0, tech)))

# ---------------------------
# Decision aggregator
# ---------------------------
class DecisionEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def time_decay_weight(self, published_iso: str) -> float:
        try:
            published = datetime.fromisoformat(published_iso)
        except Exception:
            published = now_utc()
        age_min = (now_utc() - published).total_seconds() / 60.0
        half = self.cfg["aggregation"].get("recency_half_life_minutes", 60)
        return 2 ** (-age_min / half)

    def domain_weight(self, domain: str) -> float:
        # small credibility map; unknown default 0.6
        mapping = {
            "www.coindesk.com": 1.0,
            "coindesk.com": 1.0,
            "cointelegraph.com": 0.9,
            "decrypt.co": 0.85,
            "www.theblock.co": 0.9,
            "theblock.co": 0.9,
        }
        return mapping.get(domain, 0.6)

    def aggregate_for_coin(self, articles: List[Dict[str,Any]], coin_sym: str) -> Dict[str,Any]:
        window = timedelta(minutes=self.cfg["aggregation"].get("window_minutes",180))
        cutoff = now_utc() - window
        weighted_sum = 0.0
        weight_total = 0.0
        contribs = []
        for a in articles:
            # a: {url, title, published, source, text, mentions, sent}
            pub = datetime.fromisoformat(a["published"])
            if pub < cutoff:
                continue
            mention = a.get("mentions", {}).get(coin_sym, 0.0)
            if mention <= 0:
                continue
            domain = a.get("source") or domain_from_url(a.get("url",""))
            d_w = self.domain_weight(domain)
            t_w = self.time_decay_weight(a["published"])
            w = mention * d_w * t_w
            weighted_sum += (a.get("sent_score", 0.0)) * w
            weight_total += w
            contribs.append({"url": a["url"], "sent": a.get("sent_score",0.0), "w": w, "title": a.get("title","")})
        news_score = (weighted_sum/weight_total) if weight_total > 0 else 0.0
        news_score = max(-1.0, min(1.0, news_score))
        return {"news_score": news_score, "weight_total": weight_total, "n": len(contribs), "contribs": contribs}

    def final_score(self, news_score: float, tech_score: float) -> float:
        mult = 1 + self.cfg["technical"].get("tech_multiplier_strength",0.25) * (tech_score or 0.0)
        return float(max(-1.0, min(1.0, news_score * mult)))

# ---------------------------
# Executor: places orders on Binance Spot via ccxt
# - market buy using quoteOrderQty param (spend USDT)
# - creates TP limit order and SL stop-limit
# - polls order status and cancels the other when one filled
# ---------------------------
class TradeExecutor:
    def __init__(self, cfg, api_key: str, api_secret: str, state: Dict[str,Any]):
        self.cfg = cfg
        self.state = state
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = self._init_exchange()
        self.tech = TechEngine(cfg, self.exchange)

    def _init_exchange(self):
        opts = {"apiKey": self.api_key, "secret": self.api_secret, "enableRateLimit": True}
        if self.cfg["binance"].get("testnet", False):
            # ccxt binance testnet setup (user must provide testnet keys and proper URLs)
            ex = ccxt.binance(opts)
            ex.set_sandbox_mode(True)
        else:
            ex = ccxt.binance(opts)
        logger.info("Initialized ccxt Binance (testnet=%s)", self.cfg["binance"].get("testnet", False))
        return ex

    def fetch_balance(self):
        try:
            return self.exchange.fetch_balance()
        except Exception:
            logger.exception("fetch_balance failed")
            return {}

    def market_buy_by_usdt(self, pair: str, usdt_amount: float) -> Optional[Dict[str,Any]]:
        """
        Place a market buy spending approximately usdt_amount USDT using Binance 'quoteOrderQty' param.
        Returns executed trade info: {filled_qty (base), filled_price, cost_usdt, order_response}
        """
        if usdt_amount < self.cfg["min_trade_usdt"]:
            logger.info("USDT amount %.2f under min_trade_usdt", usdt_amount)
            return None
        try:
            params = {"quoteOrderQty": float(round(usdt_amount, 2))}
            logger.info("Placing market buy: pair=%s spending %s USDT", pair, params["quoteOrderQty"])
            order = self.exchange.create_order(pair, "market", "buy", None, None, params)
            # order may contain 'fills' with price info, else fetch last trade
            filled_qty = None
            filled_price = None
            cost = None
            if order:
                # Try to parse order response for executed qty & avg price
                # Binance ccxt's market buy with quoteOrderQty returns order with 'fills' sometimes
                if isinstance(order, dict) and order.get("status") in ("closed", "filled") and order.get("filled"):
                    # older ccxt style
                    filled_qty = float(order.get("filled"))
                    cost = float(order.get("cost"))
                    filled_price = cost / filled_qty if filled_qty else None
                elif isinstance(order, dict) and order.get("info"):
                    info = order["info"]
                    # try to inspect 'fills'
                    fills = info.get("fills") or []
                    if fills:
                        total_qty = 0.0
                        total_cost = 0.0
                        for f in fills:
                            q = float(f.get("qty", 0))
                            p = float(f.get("price", 0))
                            total_qty += q
                            total_cost += q * p
                        if total_qty > 0:
                            filled_qty = total_qty
                            filled_price = total_cost / total_qty
                            cost = total_cost
                # fallback: fetch trades for pair and find recent trades by timestamp (complex)
            logger.info("Market buy order placed; parsing result: qty=%s price=%s cost=%s", filled_qty, filled_price, cost)
            return {"filled_qty": filled_qty, "filled_price": filled_price, "cost": cost, "order": order}
        except Exception:
            logger.exception("Market buy failed for %s", pair)
            return None

    def market_sell_amount(self, pair: str, base_amount: float) -> Optional[Dict[str,Any]]:
        """
        Market sell a specified base asset amount.
        """
        try:
            logger.info("Placing market sell: pair=%s amount=%s", pair, base_amount)
            order = self.exchange.create_order(pair, "market", "sell", float(base_amount))
            # parse similar to buy
            return {"order": order}
        except Exception:
            logger.exception("Market sell failed for %s", pair)
            return None

    def place_limit_order(self, pair: str, side: str, amount: float, price: float) -> Optional[Dict[str,Any]]:
        try:
            logger.info("Placing limit %s for %s %s @ %s", side, amount, pair, price)
            order = self.exchange.create_order(pair, "limit", side, float(amount), float(price))
            return {"order": order}
        except Exception:
            logger.exception("Limit order failed")
            return None

    def place_stop_limit(self, pair: str, side: str, amount: float, stop_price: float, limit_price: float) -> Optional[Dict[str,Any]]:
        """
        Place stop-limit using params field (for Binance stopPrice)
        """
        try:
            params = {"stopPrice": float(stop_price)}
            logger.info("Placing stop-limit %s for %s %s stop %s limit %s", side, pair, amount, stop_price, limit_price)
            order = self.exchange.create_order(pair, "stop_limit", side, float(amount), float(limit_price), params)
            return {"order": order}
        except Exception:
            # Some ccxt versions require 'STOP_LOSS_LIMIT' or 'STOP_LOSS' as type; try alternative
            try:
                params = {"stopPrice": float(stop_price)}
                order = self.exchange.create_order(pair, "STOP_LOSS_LIMIT", side, float(amount), float(limit_price), params)
                return {"order": order}
            except Exception:
                logger.exception("Stop-limit order failed for %s", pair)
                return None

    # Basic monitor loop to wait until either TP or SL executes; cancels the other order
    def monitor_tp_sl(self, pair: str, base_amount: float, tp_order_id: Any, sl_order_id: Any, timeout_seconds: int = 60*60*6):
        """
        Polls open orders and trades to determine which filled. Cancels other order and returns result.
        Returns {'outcome': 'tp'|'sl'|'timeout'|'error', 'filled_price':..., 'filled_qty':...}
        """
        start = time.time()
        try:
            while True:
                elapsed = time.time() - start
                if elapsed > timeout_seconds:
                    logger.warning("TP/SL monitoring timed out after %s s", timeout_seconds)
                    # Cancel both to release funds
                    try:
                        self.exchange.cancel_order(tp_order_id)
                    except Exception: pass
                    try:
                        self.exchange.cancel_order(sl_order_id)
                    except Exception: pass
                    return {"outcome": "timeout"}
                # fetch open orders for the symbol
                try:
                    open_orders = self.exchange.fetch_open_orders(symbol=pair)
                except Exception:
                    logger.exception("fetch_open_orders failed; continuing")
                    open_orders = []
                # if one order missing => it was filled or canceled
                ids = [o.get("id") or (o.get("info") or {}).get("orderId") for o in open_orders]
                tp_present = tp_order_id in ids
                sl_present = sl_order_id in ids
                if not tp_present and sl_present:
                    # tp not present -> assume TP filled
                    logger.info("TP missing while SL present => TP likely filled")
                    # cancel SL
                    try:
                        self.exchange.cancel_order(sl_order_id)
                    except Exception:
                        pass
                    return {"outcome": "tp"}
                if not sl_present and tp_present:
                    logger.info("SL missing while TP present => SL likely filled")
                    try:
                        self.exchange.cancel_order(tp_order_id)
                    except Exception:
                        pass
                    return {"outcome": "sl"}
                # both missing -> maybe both executed? fetch trades to decide
                if not tp_present and not sl_present:
                    logger.info("Neither TP nor SL present -> check recent trades")
                    # fetch recent trades for the symbol and try to infer
                    # In interest of simplicity, return timeout (higher-level logic will inspect)
                    return {"outcome": "unknown"}
                time.sleep(5)
        except Exception:
            logger.exception("Error monitoring TP/SL")
            return {"outcome": "error"}

# ---------------------------
# Orchestration: main loop
# ---------------------------
def main_loop(api_key: str, api_secret: str, config: Dict[str, Any]):
    # load persistent state
    state = load_state(config["state_file"])

    # instantiate sentiment
    senti = SentimentEngine(config)
    # instantiate decision
    decider = DecisionEngine(config)
    # exchange & executor
    execer = TradeExecutor(config, api_key, api_secret, state)
    tech_engine = execer.tech  # uses same ccxt instance for market data if enabled

    logger.info("Bot started. Monitoring news every %s seconds.", config["poll_interval_seconds"])
    try:
        while True:
            # 1) fetch news
            raw_articles = fetch_rss_articles(config["rss_feeds"])
            articles = []
            for ra in raw_articles:
                # minimal extraction to save time; can fetch full text if needed
                text = extract_text(ra["url"]) if ra["url"] else ""
                mentions = detect_coin_mentions((ra.get("title","") + "\n" + text), {**config["coins"], **config.get("opportunity_watchlist", {})})
                if not mentions:
                    continue
                sent = senti.analyze_text(ra.get("title",""), text)
                articles.append({
                    "url": ra["url"], "title": ra["title"], "published": ra["published"], "source": ra["source"],
                    "text": text, "mentions": mentions, "sent_score": sent["score"], "sent_conf": sent["confidence"], "raw": sent["raw"]
                })

            # 2) For each core coin aggregate and decide
            for sym, meta in config["coins"].items():
                pair = meta["pair"]
                agg = decider.aggregate_for_coin(articles, sym)
                logger.debug("%s aggregate: %s", sym, agg)
                # evidence gates
                if agg["n"] < config["aggregation"]["min_articles"] or agg["weight_total"] < config["aggregation"]["min_weight_to_act"]:
                    news_score = 0.0
                else:
                    # penalize disagreement between mean and median
                    mean_score = agg["news_score"]
                    # compute median of contribs' sent
                    if agg["contribs"]:
                        med = float(np.median([c["sent"] for c in agg["contribs"]]))
                    else:
                        med = mean_score
                    gap = abs(mean_score - med)
                    penalty = max(0.0, gap - config["aggregation"].get("agreement_floor",0.15))
                    news_score = math.copysign(max(0.0, abs(mean_score) - penalty), mean_score)

                # technical check
                tech_score = 0.0
                if tech_engine.enabled:
                    df = tech_engine.fetch_ohlcv_df(pair, timeframe=config["technical"]["ohlcv_timeframe"], limit=200)
                    tech_score = tech_engine.compute_tech_score(df)

                final = decider.final_score(news_score, tech_score)
                logger.info("%s: news_score=%.3f tech_score=%.3f final=%.3f (n=%s, wt=%.3f)", sym, news_score, tech_score, final, agg["n"], agg["weight_total"])

                # Should we execute?
                # Only execute when |final| >= execution_confidence
                if abs(final) >= config["execution_confidence"]:
                    action = "BUY" if final > 0 else "SELL"
                    # cooldown check
                    last_act_iso = state["last_action"].get(sym)
                    cooldown_ok = True
                    if last_act_iso:
                        last_dt = datetime.fromisoformat(last_act_iso)
                        cooldown_ok = (now_utc() - last_dt) >= timedelta(minutes=config["cooldown_minutes"])
                    if not cooldown_ok:
                        logger.info("%s on cooldown - skipping automatic execution", sym)
                    else:
                        logger.info("High-confidence signal for %s: %s (%.2f) => consider execution", sym, action, final)
                        # Decide trade size (based on state)
                        trade_usdt = float(state["trade_usdt"].get(sym, config["base_trade_usdt"]))
                        balance = execer.fetch_balance()
                        free_usdt = float(balance.get("free", {}).get("USDT", balance.get("free", {}).get("usdt", 0.0) or 0.0) or 0.0)
                        if action == "BUY":
                            # only spend up to available USDT
                            spend = min(trade_usdt, free_usdt)
                            if spend < config["min_trade_usdt"]:
                                logger.warning("Insufficient USDT (have %.2f) to place trade for %s (needs >= %.2f). Skipping", free_usdt, sym, config["min_trade_usdt"])
                            else:
                                # market buy using quoteOrderQty param where supported
                                res = execer.market_buy_by_usdt(pair, spend)
                                if res and res.get("filled_qty"):
                                    # register open position
                                    filled_qty = res["filled_qty"]
                                    avg_price = res["filled_price"] or (res["cost"]/filled_qty if res.get("cost") and filled_qty else None)
                                    cost_usdt = res.get("cost") or spend
                                    entry_price = avg_price or execer.exchange.fetch_ticker(pair)["last"]
                                    # compute TP & SL prices
                                    tp_price = entry_price * (1 + config["take_profit_pct"])
                                    sl_price = entry_price * (1 - config["stop_loss_pct"])
                                    # place TP & SL orders
                                    tp = execer.place_limit_order(pair, "sell", filled_qty, tp_price)
                                    sl = execer.place_stop_limit(pair, "sell", filled_qty, sl_price, sl_price)
                                    # Save state
                                    state["open_positions"][sym] = {
                                        "entry_price": entry_price,
                                        "amount": filled_qty,
                                        "usdt_spent": cost_usdt,
                                        "tp_price": tp_price,
                                        "sl_price": sl_price,
                                        "tp_order": (tp["order"].get("id") if tp and tp.get("order") else None),
                                        "sl_order": (sl["order"].get("id") if sl and sl.get("order") else None),
                                        "opened_at": now_utc().isoformat()
                                    }
                                    state["last_action"][sym] = now_utc().isoformat()
                                    save_state(config["state_file"], state)
                                    logger.info("Opened position %s: qty=%s entry=%.6f tp=%.6f sl=%.6f", sym, filled_qty, entry_price, tp_price, sl_price)
                                    # monitor TP/SL (blocking wait but with long timeout)
                                    monitor_res = execer.monitor_tp_sl(pair, filled_qty, state["open_positions"][sym]["tp_order"], state["open_positions"][sym]["sl_order"])
                                    logger.info("Monitor outcome: %s", monitor_res)
                                    # After outcome, try to determine profit/loss and update trade history
                                    # For simplicity here we check current holdings or previous trades via fetch_my_trades or ticker
                                    # We'll try to infer PnL using last trade price vs entry
                                    # Close bookkeeping: assume when one executed, the position is closed
                                    # Attempt to find recent trades to deduce realized PnL (complex across exchanges).
                                    # We'll simply compute an approximate realized pct using current ticker when TP hit or SL hit.
                                    try:
                                        last_price = execer.exchange.fetch_ticker(pair)["last"]
                                        # if TP outcome, profit ~ tp - entry
                                        outcome = monitor_res.get("outcome")
                                        if outcome == "tp":
                                            realized = (state["open_positions"][sym]["tp_price"] - state["open_positions"][sym]["entry_price"]) * state["open_positions"][sym]["amount"]
                                            profit_pct = (state["open_positions"][sym]["tp_price"] / state["open_positions"][sym]["entry_price"]) - 1.0
                                        elif outcome == "sl":
                                            realized = (state["open_positions"][sym]["sl_price"] - state["open_positions"][sym]["entry_price"]) * state["open_positions"][sym]["amount"]
                                            profit_pct = (state["open_positions"][sym]["sl_price"] / state["open_positions"][sym]["entry_price"]) - 1.0
                                        else:
                                            # unknown/timeout: estimate using last price
                                            realized = (last_price - state["open_positions"][sym]["entry_price"]) * state["open_positions"][sym]["amount"]
                                            profit_pct = (last_price / state["open_positions"][sym]["entry_price"]) - 1.0
                                        # update trade history
                                        hist = {
                                            "symbol": sym, "entry_price": state["open_positions"][sym]["entry_price"],
                                            "exit_price": state["open_positions"][sym].get("tp_price") if monitor_res.get("outcome")=="tp" else state["open_positions"][sym].get("sl_price"),
                                            "amount": state["open_positions"][sym]["amount"],
                                            "usdt_spent": state["open_positions"][sym]["usdt_spent"],
                                            "realized_usd": realized,
                                            "profit_pct": profit_pct,
                                            "opened_at": state["open_positions"][sym]["opened_at"],
                                            "closed_at": now_utc().isoformat(),
                                            "outcome": monitor_res.get("outcome")
                                        }
                                        state["trade_history"].append(hist)
                                        # scale up budget on positive realized_usd
                                        if realized and realized > 0:
                                            prev = state["trade_usdt"].get(sym, config["base_trade_usdt"])
                                            new = min(config["max_trade_usdt"], prev + config["scale_up_on_win_usd"])
                                            state["trade_usdt"][sym] = new
                                            logger.info("Trade profitable: %.2f USD -> increasing next trade size for %s: %.2f", realized, sym, new)
                                        # remove open pos
                                        state["open_positions"].pop(sym, None)
                                        state["last_action"][sym] = now_utc().isoformat()
                                        save_state(config["state_file"], state)
                                    except Exception:
                                        logger.exception("Error finalizing trade bookkeeping")
                                else:
                                    logger.warning("Market buy did not return filled_qty; skipping state registration")
                        elif action == "SELL":
                            # SELL signals: try to close existing position if any OR SELL a small size if we have free base asset
                            pos = state["open_positions"].get(sym)
                            if pos:
                                # If position exists, cancel TP/SL and market-sell the base amount
                                amount = pos.get("amount")
                                if amount and amount > 0:
                                    sell_res = execer.market_sell_amount(pair, amount)
                                    if sell_res:
                                        logger.info("Market sold open position for %s amount %s", sym, amount)
                                        # we should cancel TP/SL orders
                                        try:
                                            if pos.get("tp_order"):
                                                execer.exchange.cancel_order(pos["tp_order"])
                                        except Exception: pass
                                        try:
                                            if pos.get("sl_order"):
                                                execer.exchange.cancel_order(pos["sl_order"])
                                        except Exception: pass
                                        # finalize history (approx)
                                        exit_price = execer.exchange.fetch_ticker(pair)["last"]
                                        realized = (exit_price - pos["entry_price"]) * amount
                                        profit_pct = (exit_price / pos["entry_price"]) - 1.0
                                        hist = {"symbol": sym, "entry_price": pos["entry_price"], "exit_price": exit_price, "amount": amount, "usdt_spent": pos["usdt_spent"], "realized_usd": realized, "profit_pct": profit_pct, "opened_at": pos.get("opened_at"), "closed_at": now_utc().isoformat(), "outcome": "manual_sell_signal"}
                                        state["trade_history"].append(hist)
                                        # scale up if profitable
                                        if realized and realized > 0:
                                            prev = state["trade_usdt"].get(sym, config["base_trade_usdt"])
                                            state["trade_usdt"][sym] = min(config["max_trade_usdt"], prev + config["scale_up_on_win_usd"])
                                        state["open_positions"].pop(sym, None)
                                        state["last_action"][sym] = now_utc().isoformat()
                                        save_state(config["state_file"], state)
                                    else:
                                        logger.warning("Market sell failed for %s", sym)
                                else:
                                    logger.info("Open position has zero amount; removing")
                                    state["open_positions"].pop(sym, None)
                                    save_state(config["state_file"], state)
                            else:
                                # no open pos: optionally short-sell not supported on spot; skip
                                logger.info("SELL signal but no open position for %s; skipping (spot can't short)", sym)
                        else:
                            logger.info("Action not handled: %s", action)
                else:
                    logger.debug("%s final below execution threshold (%.2f < %.2f) - no trade", sym, abs(final), config["execution_confidence"])

            # 3) optional: opportunity watchlist similar logic (omitted for brevity but can be added same as above)

            # Sleep
            logger.info("Iteration complete; sleeping %s seconds", config["poll_interval_seconds"])
            time.sleep(config["poll_interval_seconds"])
    except KeyboardInterrupt:
        logger.info("Interrupted by user; saving state and exiting.")
        save_state(config["state_file"], state)
    except Exception:
        logger.exception("Fatal error in main loop; saving state and exiting.")
        save_state(config["state_file"], state)

# ---------------------------
# CLI / Entrypoint
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News-driven Binance Spot trading bot")
    parser.add_argument("--key", help="Binance API Key (or set BINANCE_API_KEY env)", default=os.environ.get("BINANCE_API_KEY"))
    parser.add_argument("--secret", help="Binance API Secret (or set BINANCE_API_SECRET env)", default=os.environ.get("BINANCE_API_SECRET"))
    parser.add_argument("--testnet", action="store_true", help="Use Binance Testnet sandbox (recommended for testing)")
    args = parser.parse_args()
    if not args.key or not args.secret:
        logger.error("API key/secret not provided. Provide via --key/--secret or BINANCE_API_KEY/BINANCE_API_SECRET env vars.")
        raise SystemExit(1)
    CONFIG["binance"]["testnet"] = args.testnet
    main_loop(args.key, args.secret, CONFIG)
