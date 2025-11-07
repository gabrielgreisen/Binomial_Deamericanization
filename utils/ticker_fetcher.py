import io
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

def get_all_us_listed_tickers():
    """
    Return a Python list containing EVERY U.S. LISTED ticker symbol
    (NASDAQ + Other Listed = NYSE, NYSE American, etc.), including ALL issue types
    (common, ETFs, ETNs, preferreds, units, rights, etc.).

    Data sources (official, refreshed intraday):
      - https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt
      - https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt
    """
    

    NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    OTHER_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    # Robust HTTP session with retries
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods={"GET"})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "TickerGrabber/1.0"})

    def read_pipe_table(url: str) -> pd.DataFrame:
        r = s.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="|", dtype=str).dropna(how="all", axis=1)
        # Drop the footer row like: "File Creation Time|MM/DD/YYYY|HH:MM"
        first_col = df.columns[0]
        footer = df[first_col].fillna("").str.contains("File Creation Time", na=False)
        return df.loc[~footer]

    nasdaq = read_pipe_table(NASDAQ_LISTED_URL)     # 'Symbol' column
    other  = read_pipe_table(OTHER_LISTED_URL)      # 'ACT Symbol' column

    tickers = (
        pd.concat(
            [
                nasdaq["Symbol"].astype(str).str.upper().str.strip(),
                other["ACT Symbol"].astype(str).str.upper().str.strip(),
            ],
            ignore_index=True,
        )
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return tickers
