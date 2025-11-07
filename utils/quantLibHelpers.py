import QuantLib as ql

def _setup_ts(eval_date: ql.Date, r: float, q: float):
    """
    Set up flat (continuous-compounded) risk-free and dividend term structures for QuantLib.

    Parameters
    ----------
    eval_date : ql.Date
        QuantLib evaluation date to use for all curve objects.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Continuously compounded dividend yield.

    Returns
    -------
    (dc, r_ts, q_ts) : tuple
        dc   : ql.Actual365Fixed day counter.
        r_ts : ql.YieldTermStructureHandle (FlatForward at rate r).
        q_ts : ql.YieldTermStructureHandle (FlatForward at yield q).
    """

    dc = ql.Actual365Fixed()
    r_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(eval_date, ql.QuoteHandle(ql.SimpleQuote(float(r))),
                       dc, ql.Continuous, ql.NoFrequency))
    q_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(eval_date, ql.QuoteHandle(ql.SimpleQuote(float(q))),
                       dc, ql.Continuous, ql.NoFrequency))
    return dc, r_ts, q_ts

def _to_maturity(eval_date: ql.Date, T_years: float) -> ql.Date:
    """
    Convert a year fraction to a QuantLib maturity date using an Actual/365 convention.

    Parameters
    ----------
    eval_date : ql.Date
        Reference QuantLib date.
    T_years : float
        Time to maturity in years.

    Returns
    -------
    ql.Date
        Maturity date equal to eval_date + round(T_years * 365) days.
    """
    days = max(1, int(round(float(T_years) * 365.0)))
    return eval_date + days