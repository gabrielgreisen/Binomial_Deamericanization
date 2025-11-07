# Deamericanize via binomial inversion
import numpy as np
import QuantLib as ql
from utils.helpers import _get_mid, _d
from utils.quantLibHelpers import _setup_ts, _to_maturity

def _deam_one_tree(row, eval_date, steps, tree, sigma_floor=1e-3):
    """
    Attempt de-Americanization using a single tree flavor. Returns (price or None, errstr or None).
    """
    S = float(row['spot_price'])
    K = float(row['strike'])
    T = float(row['TTM'])
    r = float(row['r'])
    q = float(row['dividendYield'])
    P = _get_mid(row)
    opt_is_call = str(row.get('optionType', row.get('option_type'))).lower() == 'call'

    # ---------- input guards ----------
    if P is None or T <= 0 or not np.isfinite(P) or P <= 0 or S <= 0 or K <= 0:
        return None, "bad inputs"
    if q > 1.0:  # defensive if someone passed percent
        q = q / 100.0
        _d("q looked like percent; converted to decimal", q=q)

    # ---------- no-arb bounds ----------
    df_r = np.exp(-r*T); df_q = np.exp(-q*T)
    if opt_is_call:
        lb, ub = max(0.0, S*df_q - K*df_r), S*df_q
    else:
        lb, ub = max(0.0, K*df_r - S*df_q), K*df_r
    if not (lb - 1e-8 <= P <= ub + 1e-8):
        return None, "no-arb violation"

    # ---------- trivial case: American call with ~zero dividend ----------
    if opt_is_call and abs(q) <= 1e-4:
        return float(P), None

    # ---------- engine & instruments ----------
    ql.Settings.instance().evaluationDate = eval_date
    dc, r_ts, q_ts = _setup_ts(eval_date, r, q)
    spot = ql.QuoteHandle(ql.SimpleQuote(S))

    # keep SimpleQuote so we can mutate σ during solve
    vol_sq = ql.SimpleQuote(max(0.30, sigma_floor))
    vol_h  = ql.QuoteHandle(vol_sq)
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, ql.NullCalendar(), vol_h, dc))
    process = ql.BlackScholesMertonProcess(spot, q_ts, r_ts, vol_ts)

    maturity = _to_maturity(eval_date, T)
    ql_type = ql.Option.Call if opt_is_call else ql.Option.Put
    am_opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql_type, K), ql.AmericanExercise(eval_date, maturity))
    eu_opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql_type, K), ql.EuropeanExercise(maturity))

    # Tree-specific adjustments
    t = tree.lower()
    use_steps = int(steps)
    if t == "lr" and use_steps % 2 == 0:
        use_steps += 1  # LR needs odd
    try:
        am_opt.setPricingEngine(ql.BinomialVanillaEngine(process, t, use_steps))
    except Exception as e:
        return None, f"engine init failed: {e}"

    # ---------- target function f(σ) with exception safety ----------
    class _Res:
        def __call__(self, sigma):
            s = max(float(sigma), sigma_floor)
            try:
                vol_sq.setValue(s)
                val = am_opt.NPV()  # can throw "negative probability" if p ∉ [0,1]
                if not np.isfinite(val):
                    _d("am NPV not finite", sigma=s, steps=use_steps, tree=t)
                    return np.nan
                return val - P
            except Exception as e:
                _d("am NPV exception", tree=t, steps=use_steps, sigma=s, err=str(e))
                return np.nan  # signal to bracket-expander to adjust

    f = _Res()

    # ---------- bracket with robust expansion ----------
    lo, hi = max(sigma_floor, 1e-3), 6.0
    f_lo, f_hi = f(lo), f(hi)
    
    expands = 0
    while (not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0) and expands < 12:
        lo *= 0.6
        hi *= 1.7
        lo = max(lo, sigma_floor)
        f_lo, f_hi = f(lo), f(hi)
        expands += 1
        _d("expand bracket", tree=t, expands=expands, lo=lo, f_lo=f_lo, hi=hi, f_hi=f_hi)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        return None, "no sigma bracket"

    # ---------- root solve ----------
    try:
        sigma_star = float(ql.Brent().solve(f, 1e-8, max(0.30, lo*1.5), lo, hi))
        _d("sigma*", tree=t, sigma_star=sigma_star)
    except Exception as e:
        return None, f"Brent failed: {e}"

    # ---------- European analytic price ----------
    try:
        vol_sq.setValue(max(sigma_star, sigma_floor))
        eu_opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        p_eu = float(eu_opt.NPV())
        _d("eu price", tree=t, p_eu=p_eu)
        return p_eu, None
    except Exception as e:
        return None, f"eu analytic failed: {e}"

def _deamericanize_price_binomial(row, eval_date=None, steps=400, tree="jr"):
    """
    Convert an American mid price to a European-equivalent price using a QuantLib binomial tree.

    Steps:
      1) Solve sigma* s.t. American_binomial(S,K,r,q,T,sigma*) = mid_price
      2) Return European_binomial(S,K,r,q,T,sigma*)

    Returns European price (float) or None if bracketing/inversion failed.
    """
    if eval_date is None:
        eval_date = ql.Date.todaysDate()

    # First try requested tree, then robust fallbacks
    tree_try = [tree.lower()]
    for t in ["jr", "trigeorgis", "tian", "crr", "lr"]:
        if t not in tree_try:
            tree_try.append(t)

    for t in tree_try:
        p, err = _deam_one_tree(row, eval_date, steps, t, sigma_floor=1e-3)
        if p is not None and np.isfinite(p) and p > 0:
            return p
        _d("tree attempt failed", tree=t, err=err)
    return None