import time
import pandas as pd
from typing import Iterable, Tuple
from utils.data_fetcher import get_option_chains_all

def multi_fetcher(ticker_array: Iterable[str],
                  batch_size: int = 5,
                  inter_ticker_sleep: float = 0.4,
                  inter_batch_sleep: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch option chains for many tickers with polite throttling.
    Returns (calls_df, puts_df). Per-row columns already include:
      - ticker, option_type, expiration, TTM, dividendYield, spot_price
    """
    calls_list, puts_list = [], []
    tickers = [str(t).strip().upper() for t in ticker_array if str(t).strip()]

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        for t in batch:
            try:
                c, p = get_option_chains_all(t, max_workers=2)  # keep fan-out low
                if isinstance(c, pd.DataFrame) and not c.empty:
                    calls_list.append(c)
                if isinstance(p, pd.DataFrame) and not p.empty:
                    puts_list.append(p)
            except Exception as e:
                print(f"[multi_fetcher] {t}: failed -> {e}")
            time.sleep(inter_ticker_sleep)  # tiny pause between tickers
        if i + batch_size < len(tickers):
            time.sleep(inter_batch_sleep)   # slightly longer pause between batches

    calls = pd.concat(calls_list, ignore_index=True) if calls_list else pd.DataFrame()
    puts  = pd.concat(puts_list,  ignore_index=True) if puts_list  else pd.DataFrame()
    return calls, puts
