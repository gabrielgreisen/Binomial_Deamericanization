import pandas as pd
import QuantLib as ql
from quantLibHelpers import _setup_ts, _to_maturity
from bs_iv import _row_bs_iv_from_price

def _calibrate_heston(group_df: pd.DataFrame, eval_date=None, init=None):
    """
    Calibrate a Heston model to one group (same instrument/day) using IVs from mid prices.
    Requires ≥ ~5 valid helpers across strikes/maturities.
    """
    if eval_date is None:
        eval_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = eval_date

    S = float(group_df['spot_price'].iloc[0])
    r = float(group_df['r'].iloc[0])
    q = float(group_df['dividendYield'].iloc[0])

    _, r_ts, q_ts = _setup_ts(eval_date, r, q)
    cal = ql.NullCalendar()
    spot_h = ql.QuoteHandle(ql.SimpleQuote(S))

    helpers = []
    for idx, row in group_df.iterrows():
        iv = _row_bs_iv_from_price(row, eval_date=eval_date, use_deam=True)
        if iv is None:
            continue
        K = float(row['strike']); T = float(row['TTM'])
        tenor = ql.Period((_to_maturity(eval_date, T) - eval_date), ql.Days)
        helpers.append(ql.HestonModelHelper(tenor, cal,S, K,
                                            ql.QuoteHandle(ql.SimpleQuote(iv)),r_ts,q_ts)
)

    if len(helpers) < 5:
        raise ValueError("Not enough valid options to calibrate Heston (need ≥ ~5 across strikes/maturities).")

    # Initial params (can be overridden with init)
    p = dict(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)
    if init: p.update(init)

    process = ql.HestonProcess(r_ts, q_ts, spot_h, p['v0'], p['kappa'], p['theta'], p['sigma'], p['rho'])
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    for h in helpers:
        h.setPricingEngine(engine)

    om = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    endc = ql.EndCriteria(500, 50, 1e-8, 1e-8, 1e-8)
    model.calibrate(helpers, om, endc)

    # Dump fitted params for visibility
    try:
        v0, kappa, theta, sigma, rho = model.params()
    except Exception:
        pass
    return model