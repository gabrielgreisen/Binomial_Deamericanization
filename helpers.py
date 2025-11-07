def _d(msg, **k):
    items = " | ".join(f"{kk}={vv}" for kk, vv in k.items())
    print(f"[deAm] {msg}" + (f" :: {items}" if items else ""))

def _get_mid(row):
    """Return mid price from either 'midPrice' or 'mid_price' if present, else None."""
    v = row.get('midPrice', row.get('mid_price', None))
    try:
        return float(v) if v is not None else None
    except Exception:
        return None