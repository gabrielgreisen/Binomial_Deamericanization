import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        stock = yf.Ticker(ticker)

        if expiry not in stock.options:
            raise ValueError(f"Expiration `{expiry}` cannot be found. Available expirations are: {stock.options}")

        option_chain = stock.option_chain(expiry)
        calls = option_chain.calls.copy()
        puts = option_chain.puts.copy()

        # Add option type labels for downstream processing
        calls['option_type'] = 'call'
        puts['option_type'] = 'put'

        # Add Time To Maturity (TTM) in years
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        today = datetime.today().date()
        ttm = max((expiry_date - today).days/365.0, 0)

        calls['TTM'] = ttm
        puts['TTM'] = ttm


        return calls, puts
    except Exception as e:
        print(f"Error fetching option chain for {ticker} on {expiry}: {e}")
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
        tk = yf.Ticker(ticker)
        
        # Attempt to fetch live price
        live_price = tk.fast_info.get("last_price", None)
        if live_price and live_price > 0:
            return live_price

        # Fallback: most recent close
        hist = tk.history(period="1d")
        if not hist.empty:
            fallback_price = hist["Close"].iloc[-1]
            print(f"[{ticker}] Live price unavailable — using last close: {fallback_price:.2f}")
            return fallback_price

        print(f"[{ticker}] No live or historical data available.")
        return None

    except Exception as e:
        print(f"[{ticker}] Spot price fetch failed: {e}")
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
    stock = yf.Ticker(ticker)
    expiries = stock.options  # list of expiry date strings
    today = datetime.today().date()

    calls_accum = []
    puts_accum  = []

    def fetch_chain(expiry: str):
        """Fetch calls/puts for a single expiry and return (expiry, calls_df, puts_df)."""
        try:
            chain = stock.option_chain(expiry)
            calls = chain.calls.copy()
            puts  = chain.puts.copy()
        except Exception as e:
            # Return None on error so we can skip later
            return expiry, None, None

        # Tag each row with type and expiration
        calls['option_type']  = 'call'
        puts ['option_type']  = 'put'
        calls['expiration']   = expiry
        puts ['expiration']   = expiry

        # Compute time-to-maturity once
        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        ttm = max((exp_date - today).days / 365.0, 0.0)
        calls['TTM'] = ttm
        puts ['TTM'] = ttm

        return expiry, calls, puts

    # Fetch in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_chain, exp) for exp in expiries]
        for future in as_completed(futures):
            expiry, calls_df, puts_df = future.result()
            if calls_df is not None and not calls_df.empty:
                calls_accum.append(calls_df)
            if puts_df  is not None and not puts_df.empty:
                puts_accum.append(puts_df)

    # Concatenate results
    all_calls = pd.concat(calls_accum, ignore_index=True) if calls_accum else pd.DataFrame()
    all_puts  = pd.concat(puts_accum,  ignore_index=True) if puts_accum  else pd.DataFrame()

    # Fetch dividend yield and spot price for the company
    dividendYield = stock.info.get("dividendYield")
    if dividendYield is not None:
        dividendYield = dividendYield/100 # percentages on decimal basis
    all_calls["dividendYield"] = dividendYield
    all_puts["dividendYield"] = dividendYield

    all_calls["ticker"] = ticker
    all_puts["ticker"] = ticker

    spot_price = get_spot_price(ticker)
    all_calls["spot_price"] = spot_price
    all_puts["spot_price"] = spot_price

    return all_calls, all_puts