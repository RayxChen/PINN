import numpy as np
from scipy.stats import norm
import QuantLib as ql

def BlackScholes(S, T, config):
    """
    Calculate the Black-Scholes price for a European call or put option.

    Parameters:
        S (float or np.ndarray): Stock price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        option_type (str): "call" for a call option, "put" for a put option.

    Returns:
        float or np.ndarray: Option price.
    """
    K = config["model"]["K"]
    r = config["model"]["r"]
    sigma = config["model"]["sigma"]
    option_type = config["model"]["option_type"]

    # Avoid division by zero in time to maturity
    T = np.maximum(T, 1e-10)
    S = np.maximum(S, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option_type. Use 'call' or 'put'.")
    
    return price


def Heston(S, t, v, config):
    """
    Price a European option (call/put) using the Heston model with QuantLib.

    Parameters:
        S (float): Stock price
        t (float): Time to maturity (in years)
        v (float): Initial variance
        config (dict): Configuration dictionary containing model parameters.
                       Required keys:
                       - K: Strike price
                       - r: Risk-free rate
                       - kappa: Mean reversion speed
                       - theta: Long-term variance
                       - sigma_v: Volatility of variance
                       - rho: Correlation between stock and variance
                       - option_type: Option type ('call' or 'put')

    Returns:
        float: Option price
    """
    # Extract parameters from config
            
    K = config["model"]["K"]           # Strike price
    r = config["model"]["r"]           # Risk-free rate
    kappa = config["model"]["kappa"]   # Mean reversion speed
    theta = config["model"]["theta"]   # Long-term variance
    sigma_v = config["model"]["sigma_v"] # Volatility of variance
    rho = config["model"]["rho"]       # Correlation
    option_type = config["model"]["option_type"].lower()  # Option type ('call' or 'put')

    # Handle terminal payoff explicitly when t = 0
    if t == 0:
        if option_type == "call":
            return max(S - K, 0)
        elif option_type == "put":
            return max(K - S, 0)

    # Determine QuantLib option type
    ql_option_type = ql.Option.Call if option_type == "call" else ql.Option.Put

    # Set up the evaluation date
    calendar = ql.NullCalendar()
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    # Calculate exercise date
    maturity_date = calendar.advance(today, int(t * 365), ql.Days)

    # Market data setup
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual360()))
    dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual360()))

    # Heston process
    heston_process = ql.HestonProcess(rate_handle, dividend_handle, spot_handle, v, kappa, theta, sigma_v, rho)
    heston_model = ql.HestonModel(heston_process)

    # Pricing engine
    engine = ql.AnalyticHestonEngine(heston_model)

    # Option setup
    exercise = ql.EuropeanExercise(maturity_date)
    payoff = ql.PlainVanillaPayoff(ql_option_type, K)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)

    # Price calculation
    price = option.NPV()
    return price