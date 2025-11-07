import numpy as np
import QuantLib as ql
from utils.helpers import _get_mid, _d
from utils.quantLibHelpers import _setup_ts, _to_maturity
from binomial.binomial_deamericanization import _deamericanize_price_binomial

def _row_bs_iv_from_price(row, eval_date=None, iv_guess=0.25, use_deam=True):
    """
    Compute BS implied vol. If use_deam, first de-Americanize to a European price.
    Returns float IV or None.
    """
    S = float(row['spot_price']); K = float(row['strike']); T = float(row['TTM'])
    r = float(row['r']); q = float(row['dividendYield'])
    P_raw = _get_mid(row)
    opt_is_call = str(row.get('optionType', row.get('option_type'))).lower() == 'call'

    if P_raw is None or T <= 0 or S <= 0 or K <= 0:
        _d("IV: bad inputs", S=S, K=K, T=T, r=r, q=q, P=P_raw)
        return None

    if eval_date is None:
        eval_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = eval_date

    price_for_iv = P_raw
    if use_deam:
        P_eu = _deamericanize_price_binomial(row, eval_date=eval_date)
        if P_eu is None or not np.isfinite(P_eu) or P_eu <= 0:
            _d("IV: deAm failed", P_raw=P_raw, P_eu=P_eu)
            return None
        price_for_iv = float(P_eu)

    dc, r_ts, q_ts = _setup_ts(eval_date, r, q)
    cal = ql.NullCalendar()
    maturity = _to_maturity(eval_date, T)
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, cal, iv_guess, dc))
    proc = ql.BlackScholesMertonProcess(spot, q_ts, r_ts, vol_ts)
    ql_type = ql.Option.Call if opt_is_call else ql.Option.Put
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql_type, K), ql.EuropeanExercise(maturity))

    # Try QuantLib's built-in solver with wide bounds
    try:
        iv = opt.impliedVolatility(price_for_iv, proc, 1e-6, 2000, 1e-9, 12.0)
        iv = float(iv) if np.isfinite(iv) and iv > 0 else None
        if iv is not None:
            return iv
    except Exception as e:
        _d("IV: QL solver exception", err=str(e), price_for_iv=price_for_iv)

    # Fallback: bisection using analytic engine (monotone in vol)
    try:
        lo, hi = 1e-9, 12.0
        for it in range(120):
            mid = 0.5 * (lo + hi)
            vts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, cal, mid, dc))
            p = ql.BlackScholesMertonProcess(spot, q_ts, r_ts, vts)
            opt.setPricingEngine(ql.AnalyticEuropeanEngine(p))
            p_mid = opt.NPV()
            if not np.isfinite(p_mid):
                _d("IV: fallback p_mid not finite", it=it, mid=mid)
                return None
            if abs(p_mid - price_for_iv) < 1e-8:
                _d("IV: fallback converged", it=it, mid=mid)
                return mid
            if p_mid < price_for_iv:
                lo = mid
            else:
                hi = mid
        _d("IV: fallback no converge", lo=lo, hi=hi)
        return None
    except Exception as e:
        _d("IV: fallback exception", err=str(e))
        return None
