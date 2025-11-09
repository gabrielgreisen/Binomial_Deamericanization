import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf
from typing import Iterable, Tuple, Dict

from utils.data_fetcher import get_option_chains_all, _is_rate_limit_error

# Error text hints we consider "permanent/invalid"
_PERM_HINTS = (
    "delisted", "de-listed",
    "no data found", "no timezone found",
    "not found in table", "invalid", "unknown symbol",
    "no option chain", "does not have any options",
    "symbol not found"
)

def _looks_permanent_msg(msg: str) -> bool:
    m = (msg or "").lower()
    return any(h in m for h in _PERM_HINTS)

def _confirm_permanent_via_info(ticker: str) -> bool:
    """
    Secondary probe. Return True only if metadata access also looks broken in a
    delisted/invalid way. Otherwise return False (treat as transient).
    """
    try:
        tk = yf.Ticker(ticker)
        fi = tk.fast_info
        _ = fi.get("last_price", None)
        try:
            info = tk.info
            if isinstance(info, dict) and info:
                return False
        except Exception as e_info:
            return _looks_permanent_msg(str(e_info))
        return False
    except Exception as e_fast:
        return _looks_permanent_msg(str(e_fast))

# -------------------------
# Simple file logger
# -------------------------
def _make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def _log(level: str, msg: str):
        ts = datetime.now().isoformat()
        line = f"{ts} [{level}] {msg}"
        print(line)
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
    return _log

def multi_fetcher(
    ticker_array: Iterable[str],
    *,
    retry_rounds: int = 10,           # significant retries across whole list
    inter_ticker_sleep: float = 0.15,
    max_workers_per_ticker: int = 1,  # forwarded into get_option_chains_all
    min_permanent_rounds: int = 2,    # require >= this many rounds of confirmed-permanent before removal
    log_file: str = ".yf_logs/errors.log"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch option chains for many tickers (ALL expiries) with conservative removal + structured logging:
      - Never remove on first error.
      - Only remove after >= min_permanent_rounds rounds where error looks permanent AND a metadata probe agrees.
      - Rate-limit/transient errors are retried up to retry_rounds.
      - Every failure, retry, removal, and success is logged to both stdout and log_file.

    Returns: (calls_df, puts_df)
    """
    log = _make_logger(Path(log_file))

    tickers = [str(t).strip().upper() for t in ticker_array if str(t).strip()]
    remaining = set(tickers)

    permanent_hint_counts: Dict[str, int] = {}
    calls_list, puts_list = [], []

    total = len(remaining)
    processed = 0

    for r in range(1, retry_rounds + 1):
        if not remaining:
            break

        round_list = list(remaining)
        log("INFO", f"round {r}/{retry_rounds} — remaining tickers: {len(round_list)}")

        for t in round_list:
            start = time.time()
            try:
                c, p = get_option_chains_all(
                    t,
                    max_workers=max_workers_per_ticker,
                    per_expiry_sleep=0.08,
                    retry_tries=6,
                    retry_base=1.35,
                    retry_cap=15.0,
                )
                if isinstance(c, pd.DataFrame) and not c.empty:
                    calls_list.append(c)
                if isinstance(p, pd.DataFrame) and not p.empty:
                    puts_list.append(p)
                remaining.discard(t)
                permanent_hint_counts.pop(t, None)
                processed += 1
                dt = time.time() - start
                log("OK", f"{processed}/{total} ticker={t} finished in {dt:.2f}s "
                          f"(calls_rows={len(c) if isinstance(c, pd.DataFrame) else 0}, "
                          f"puts_rows={len(p) if isinstance(p, pd.DataFrame) else 0})")

            except Exception as e:
                msg = str(e)

                if _is_rate_limit_error(e):
                    log("RETRY", f"ticker={t} rate-limited/transient -> retrying next round | err={type(e).__name__}: {msg}")
                elif _looks_permanent_msg(msg) and _confirm_permanent_via_info(t):
                    permanent_hint_counts[t] = permanent_hint_counts.get(t, 0) + 1
                    log("PERM?", f"ticker={t} permanent-looking {permanent_hint_counts[t]}/{min_permanent_rounds} | err={type(e).__name__}: {msg}")
                    if permanent_hint_counts[t] >= min_permanent_rounds:
                        remaining.discard(t)
                        log("REMOVE", f"ticker={t} removed as invalid/delisted after repeated confirmation")
                else:
                    log("RETRY", f"ticker={t} transient/unknown -> will retry | err={type(e).__name__}: {msg}")

            if inter_ticker_sleep > 0:
                time.sleep(inter_ticker_sleep)

    if remaining:
        log("WARN", f"finished retries; still unresolved (kept, not removed): {len(remaining)} tickers")
    unresolved_sample = list(sorted(remaining))[:10]
    if unresolved_sample:
        log("WARN", f"unresolved sample: {unresolved_sample}")

    calls = pd.concat(calls_list, ignore_index=True) if calls_list else pd.DataFrame()
    puts  = pd.concat(puts_list,  ignore_index=True) if puts_list  else pd.DataFrame()
    log("INFO", f"done. calls_rows={len(calls)}, puts_rows={len(puts)}")
    return calls, puts
