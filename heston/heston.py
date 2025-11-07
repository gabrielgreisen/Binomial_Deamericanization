import pandas as pd
import QuantLib as ql
from utils.helpers import _d
from utils.quantLibHelpers import _to_maturity
from heston.heston_calibrator import _calibrate_heston

def _price_eu_heston(row, model: ql.HestonModel, eval_date=None):
    """
    Price a single European option under calibrated Heston.
    """
    if eval_date is None:
        eval_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = eval_date

    K = float(row['strike'])
    T = float(row['TTM'])
    maturity = _to_maturity(eval_date, T)
    ql_type = ql.Option.Call if str(row.get('optionType', row.get('option_type'))).lower() == 'call' else ql.Option.Put

    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql_type, K), ql.EuropeanExercise(maturity))
    opt.setPricingEngine(ql.AnalyticHestonEngine(model))
    p = float(opt.NPV())
    return p


def calibrate_and_price_heston_european(df: pd.DataFrame,
                                        group_cols=('ticker',),
                                        eval_date: ql.Date | None = None,
                                        init: dict | None = None) -> pd.Series:
    """
    Calibrate Heston per group from mid prices, then return ONLY the European-equivalent
    price for each row as 'V_EU_Heston'.
    """
    if eval_date is None:
        eval_date = ql.Date.todaysDate()

    results = pd.Series(index=df.index, dtype=float, name='V_EU_Heston')

    # Attach index to help correlate debug lines to inputs
    df = df.copy()
    df['idx'] = df.index

    for gkey, grp in df.groupby(list(group_cols), dropna=False):
        g = grp.copy()
        _d("=== group start ===", group=gkey, rows=len(g))
        try:
            model = _calibrate_heston(g, eval_date=eval_date, init=init)
        except Exception as e:
            _d("group calibration FAILED", group=gkey, err=str(e))
            continue

        prices = g.apply(lambda r: _price_eu_heston(r, model, eval_date), axis=1)
        results.loc[g.index] = prices.values
        _d("group priced", group=gkey)

    return results
