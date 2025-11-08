import yfinance as yf
import pandas as pd
import time, random, math, requests
from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from yfinance.exceptions import YFRateLimitError  # newer versions
except Exception:  # fallback when not exported
    class YFRateLimitError(Exception):
        pass

def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("rate limit" in msg) or ("too many requests" in msg) or ("429" in msg) or ("yf ratelimit" in msg)

def _sleep_backoff(attempt: int, base: float = 1.6, jitter: float = 0.35, cap: float = 30.0):
    delay = min(cap, (base ** attempt) * (1.0 + random.random() * jitter))
    time.sleep(delay)

def _retry(fn, tries: int = 5):
    """Retry a callable on yfinance/network rate limits with exponential backoff."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            transient = isinstance(e, YFRateLimitError) or _is_rate_limit_error(e)
            attempt += 1
            if (not transient) or attempt >= tries:
                raise
            _sleep_backoff(attempt)


def get_option_chain(ticker: str, expiry: str):

    """
    Fetches the option chain (calls and puts) for a given stock ticker and expiry date.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL')
    expiry : str
        Expiration date in 'YYYY-MM-DD' format

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Calls DataFrame, Puts DataFrame
    """
    
    try:
        stock = yf.Ticker(ticker)  # <-- no session
        options = _retry(lambda: stock.options)
        if expiry not in options:
            raise ValueError(f"Expiration `{expiry}` not found. Available: {options}")

        chain = _retry(lambda: stock.option_chain(expiry))
        calls = chain.calls.copy()
        puts  = chain.puts.copy()

        # Labels
        calls["option_type"] = "call"
        puts["option_type"]  = "put"

        # Time to maturity (years)
        today = datetime.today().date()
        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        ttm = max((exp_date - today).days / 365.0, 0.0)
        calls["TTM"] = ttm
        puts["TTM"]  = ttm
        return calls, puts
    except Exception as e:
        print(f"[get_option_chain] {ticker} {expiry}: {e}")
        return pd.DataFrame(), pd.DataFrame() # Return two empty frames so pipeline doesn't break
    
def get_spot_price(ticker):
    """
    Fetches the current spot price for a stock ticker. Falls back to the most recent
    close if a live price is not available.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL').

    Returns
    -------
    float or None
        Spot price (live if available, else last close). Returns None if unavailable.
    """

    try:
        tk = yf.Ticker(ticker)  # <-- no session

        def _live():
            fi = tk.fast_info
            # keys vary across versions
            for key in ("last_price", "regular_market_price", "lastPrice", "regularMarketPrice"):
                v = fi.get(key)
                if v and v > 0:
                    return float(v)
            return None

        live = _retry(_live)
        if live is not None:
            return live

        def _hist():
            h = tk.history(period="5d", auto_adjust=False)
            if not h.empty:
                return float(h["Close"].iloc[-1])
            return None

        fallback = _retry(_hist)
        if fallback is not None:
            print(f"[{ticker}] Live price unavailable — using last close: {fallback:.4f}")
            return fallback

        print(f"[{ticker}] No live or historical data available.")
        return None
    except Exception as e:
        print(f"[get_spot_price] {ticker}: {e}")
        return None

def get_option_chains_all(ticker: str,
                                  max_workers: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches option chains (calls and puts) for every available expiry of a given ticker,
    performing API requests in parallel to reduce total fetch time.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL').
    max_workers : int, optional
        Maximum number of threads to use for concurrent fetching (default is 8).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - calls_df: DataFrame containing all calls across expiries, with added columns:
            * 'option_type' = 'call'
            * 'expiration'  = expiry date string 'YYYY-MM-DD'
            * 'TTM'         = time to maturity in years
        - puts_df: DataFrame containing all puts with the same added columns.
    """
    stock = yf.Ticker(ticker)  # <-- no session
    try:
        expiries = _retry(lambda: stock.options) or []
    except Exception as e:
        print(f"[{ticker}] Failed to list expiries: {e}")
        return pd.DataFrame(), pd.DataFrame()

    today = datetime.today().date()
    calls_accum, puts_accum = [], []

    def fetch_chain(expiry: str):
        try:
            chain = _retry(lambda: stock.option_chain(expiry))
            c = chain.calls.copy()
            p = chain.puts.copy()
        except Exception as e:
            print(f"[{ticker}] skip {expiry}: {e}")
            return None, None

        if c is not None and not c.empty:
            c["option_type"] = "call"
            c["expiration"]  = expiry
            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            c["TTM"] = max((exp_date - today).days / 365.0, 0.0)
        if p is not None and not p.empty:
            p["option_type"] = "put"
            p["expiration"]  = expiry
            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            p["TTM"] = max((exp_date - today).days / 365.0, 0.0)

        time.sleep(0.2)  # gentle pacing helps avoid bursts
        return c, p

    # Keep concurrency low; high fan-out triggers 429s
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        futures = [ex.submit(fetch_chain, e) for e in expiries]
        for fut in as_completed(futures):
            c, p = fut.result()
            if isinstance(c, pd.DataFrame) and not c.empty:
                calls_accum.append(c)
            if isinstance(p, pd.DataFrame) and not p.empty:
                puts_accum.append(p)

    all_calls = pd.concat(calls_accum, ignore_index=True) if calls_accum else pd.DataFrame()
    all_puts  = pd.concat(puts_accum,  ignore_index=True) if puts_accum  else pd.DataFrame()

    # Spot price (once per ticker)
    spot = get_spot_price(ticker)

    # Dividend yield: decimal preferred; fallback to rate/spot
    def _normalize_dividend_yield() -> float:
        try:
            info = _retry(lambda: stock.info)
        except Exception:
            info = {}

        dy_raw = info.get("dividendYield", None)  # usually decimal like 0.0123
        if dy_raw is not None:
            try:
                y = float(dy_raw)
                if math.isfinite(y):
                    return y / 100.0 if y > 1.5 else y  # guard against percent-like inputs
            except Exception:
                pass

        try:
            div_rate = info.get("trailingAnnualDividendRate", None)
            if div_rate is not None and spot:
                y = float(div_rate) / float(spot)
                if math.isfinite(y) and y >= 0:
                    return y
        except Exception:
            pass

        return 0.0

    dividend_yield = _normalize_dividend_yield()

    # Attach metadata
    for df in (all_calls, all_puts):
        if not df.empty:
            df["ticker"] = ticker
            df["dividendYield"] = dividend_yield
            df["spot_price"] = spot

    return all_calls, all_puts