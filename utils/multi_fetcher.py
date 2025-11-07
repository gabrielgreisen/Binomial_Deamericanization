import pandas as pd
from utils.data_fetcher import get_option_chains_all

def multi_fetcher(ticker_array):
    calls = pd.DataFrame()
    puts = pd.DataFrame()
    for ticker in ticker_array:
        ticker_call, ticker_put = get_option_chains_all(ticker)
        calls = pd.concat([calls, ticker_call], ignore_index=True)
        puts = pd.concat([puts, ticker_put], ignore_index=True)

    return calls, puts