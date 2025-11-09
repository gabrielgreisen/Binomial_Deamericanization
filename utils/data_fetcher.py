import yfinance as yf
import pandas as pd
import time, random, math
from datetime import datetime
from typing import Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- yfinance rate-limit compatibility shim (works across versions) ---
try:
    from yfinance.exceptions import YFRateLimitError  # newer versions
except Exception:  # fallback when not exported
    class YFRateLimitError(Exception):
        pass

# -------------------------
# Error classification
# -------------------------
def _is_rate_limit_error(e: Exception) -> bool:
    """
    Return True for yfinance/curl-cffi 429 'Too Many Requests' and similar
    transient throttling/network messages.
    """
    msg = str(e).lower()
    resp = getattr(e, "response", None)
    code = getattr(resp, "status_code", None)
    return (
        code == 429
        or "too many requests" in msg
        or "rate limit" in msg
        or "429" in msg
        or "yf ratelimit" in msg
        or "temporarily unavailable" in msg
        or "timed out" in msg
        or "timeout" in msg
        or "ssl" in msg
    )

def _is_permanent_ticker_error(e: Exception) -> bool:
    """Errors we shouldn't retry: delisted, invalid symbol, etc."""
    msg = str(e).lower()
    return any(s in msg for s in [
        "delisted", "de-listed",
        "no data found", "no timezone found",
        "not found in table", "invalid", "unknown symbol",
        "no option chain", "does not have any options",
        "symbol not found"
    ])

# -------------------------
# Backoff / retry
# -------------------------
def _sleep_backoff(attempt: int, base: float = 1.35, jitter: float = 0.25, cap: float = 12.0):
    delay = min(cap, (base ** attempt) * (1.0 + random.random() * jitter))
    time.sleep(delay)

def _retry(fn: Callable[[], object],
           tries: int = 4,
           base: float = 1.35,
           jitter: float = 0.25,
           cap: float = 12.0):
    """Retry only for transient / rate-limit looking errors."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if not _is_rate_limit_error(e) or attempt >= tries:
                raise
            _sleep_backoff(attempt, base, jitter, cap)

# -------------------------
# Single-expiry fetch
# -------------------------
def get_option_chain(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch the option chain for a ticker/expiry. Returns (calls_df, puts_df).
    Adds: 'option_type' and 'TTM'. Raises on transient 'options is None'.
    """
    stock = yf.Ticker(ticker)

    options = _retry(lambda: stock.options)
    if options is None:
        # transient hiccup from Yahoo → let caller retry
        raise RuntimeError("transient: options returned None")
    options = list(options)
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

# -------------------------
# Spot price with fallback
# -------------------------
def get_spot_price(ticker: str):
    """
    Return spot price (live via fast_info if possible, else recent close). None if unavailable.
    """
    tk = yf.Ticker(ticker)

    def _live():
        fi = tk.fast_info
        for key in ("last_price", "regular_market_price", "lastPrice", "regularMarketPrice"):
            v = fi.get(key)
            if v and v > 0:
                return float(v)
        return None

    try:
        live = _retry(_live)
        if live is not None:
            return live
    except Exception:
        pass

    def _hist():
        h = tk.history(period="5d", auto_adjust=False)
        if not h.empty:
            return float(h["Close"].iloc[-1])
        return None

    try:
        fallback = _retry(_hist)
        if fallback is not None:
            return fallback
    except Exception:
        pass

    return None

# -------------------------
# All-expiries fetch
# -------------------------
def get_option_chains_all(
    ticker: str,
    max_workers: int = 2,
    per_expiry_sleep: float = 0.08,
    retry_tries: int = 4,
    retry_base: float = 1.35,
    retry_cap: float = 12.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch calls & puts for **all available expiries** of `ticker`.
    Returns (all_calls_df, all_puts_df) with:
      - option_type, expiration, TTM, ticker, dividendYield (decimal), spot_price
    """
    stock = yf.Ticker(ticker)

    def _r(f):  # helper to carry retry params
        return _retry(f, tries=retry_tries, base=retry_base, cap=retry_cap)

    # List ALL expiries
    expiries = _r(lambda: stock.options)
    if expiries is None:
        # transient hiccup → let caller/multi_fetcher retry this ticker
        raise RuntimeError("transient: options returned None")
    expiries = list(expiries)

    # Ticker may legitimately have no options (not an error)
    if len(expiries) == 0:
        return pd.DataFrame(), pd.DataFrame()

    today = datetime.today().date()
    calls_accum, puts_accum = [], []

    def fetch_chain(expiry: str):
        try:
            chain = _r(lambda: stock.option_chain(expiry))
            c = chain.calls.copy()
            p = chain.puts.copy()
        except Exception as e:
            # Structured error output for easier tracking
            etype = type(e).__name__
            msg = str(e)
            if _is_permanent_ticker_error(e):
                print(f"[{datetime.now().isoformat()}][{ticker}][{expiry}] PERMANENT {etype}: {msg}")
            elif _is_rate_limit_error(e):
                print(f"[{datetime.now().isoformat()}][{ticker}][{expiry}] RATE_LIMIT {etype}: {msg}")
            else:
                print(f"[{datetime.now().isoformat()}][{ticker}][{expiry}] TRANSIENT {etype}: {msg}")
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

        if per_expiry_sleep > 0:
            time.sleep(per_expiry_sleep)
        return c, p

    # Conservative per-ticker concurrency
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
            info = _r(lambda: stock.info)
        except Exception:
            info = {}

        dy_raw = info.get("dividendYield", None)  # usually decimal like 0.0123
        if dy_raw is not None:
            try:
                y = float(dy_raw)
                if math.isfinite(y):
                    return y / 100.0 if y > 1.5 else y  # guard percent-like inputs
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
