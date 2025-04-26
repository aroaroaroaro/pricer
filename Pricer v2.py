import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math

def generate_normal_random():
    """Génère un nombre aléatoire suivant une distribution normale standard."""
    u1 = random.random()
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

def geometric_brownian_motion(S0, mu, sigma, T, dt, steps):
    """Simule une trajectoire de mouvement brownien géométrique."""
    S = [S0]
    for _ in range(steps):
        dW = generate_normal_random() * math.sqrt(dt)
        S_next = S[-1] * math.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        S.append(S_next)
    return S

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    def norm_cdf(x):
        """Fonction de répartition cumulative de la distribution normale standard."""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if option_type == 'call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == 'put':
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

def implied_volatility(S, K, T, r, market_price, option_type='call'):
    def objective(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - market_price

    # Utilisation de la méthode de la bissection pour trouver la volatilité implicite
    tolerance = 1e-6
    lower_bound = 1e-6
    upper_bound = 5
    sigma = (lower_bound + upper_bound) / 2

    while abs(objective(sigma)) > tolerance:
        if objective(sigma) > 0:
            upper_bound = sigma
        else:
            lower_bound = sigma
        sigma = (lower_bound + upper_bound) / 2

    return sigma

def binomial_tree(S, K, T, r, sigma, steps, option_type='call', is_american=False):
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    stock_tree = [[0] * (steps + 1) for _ in range(steps + 1)]
    option_tree = [[0] * (steps + 1) for _ in range(steps + 1)]

    for i in range(steps + 1):
        for j in range(i + 1):
            stock_tree[j][i] = S * (u**j) * (d**(i-j))

    if option_type == 'call':
        option_tree[0][steps] = max(stock_tree[0][steps] - K, 0)
        option_tree[1][steps] = max(stock_tree[1][steps] - K, 0)
    elif option_type == 'put':
        option_tree[0][steps] = max(K - stock_tree[0][steps], 0)
        option_tree[1][steps] = max(K - stock_tree[1][steps], 0)

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            if is_american:
                if option_type == 'call':
                    option_tree[j][i] = max(stock_tree[j][i] - K, math.exp(-r * dt) * (p * option_tree[j][i + 1] + (1 - p) * option_tree[j + 1][i + 1]))
                elif option_type == 'put':
                    option_tree[j][i] = max(K - stock_tree[j][i], math.exp(-r * dt) * (p * option_tree[j][i + 1] + (1 - p) * option_tree[j + 1][i + 1]))
            else:
                option_tree[j][i] = math.exp(-r * dt) * (p * option_tree[j][i + 1] + (1 - p) * option_tree[j + 1][i + 1])

    return option_tree[0][0]

def option_pricer(S, K, T, r, sigma=None, market_price=None, option_type='call', is_american=False, steps=100):
    if sigma is None:
        if market_price is None:
            raise ValueError("market_price must be provided if sigma is not given")
        sigma = implied_volatility(S, K, T, r, market_price, option_type)

    if is_american:
        price = binomial_tree(S, K, T, r, sigma, steps, option_type, is_american)
    else:
        price = black_scholes(S, K, T, r, sigma, option_type)

    return price

def calculate_options(spot, strike, taux, maturite, volatilite):
    positions = ['Long Call', 'Long Put', 'Short Call', 'Short Put']
    results = []

    for position in positions:
        option_type = 'call' if 'Call' in position else 'put'
        price = option_pricer(spot, strike, maturite / 365.0, taux, volatilite, option_type=option_type)

        if 'Short' in position:
            price = -price

        results.append([position, price])

    df_results = pd.DataFrame(results, columns=['Position', 'Price'])
    return df_results

def plot_payoff(spot, strike, position):
    spot_range = np.linspace(spot * 0.5, spot * 1.5, 500)
    if 'Call' in position:
        payoff = np.maximum(spot_range - strike, 0)
    elif 'Put' in position:
        payoff = np.maximum(strike - spot_range, 0)

    if 'Short' in position:
        payoff = -payoff

    plt.figure(figsize=(10, 6))
    plt.plot(spot_range, payoff, label=f'{position} Payoff')
    plt.xlabel('Spot Price')
    plt.ylabel('Payoff')
    plt.title(f'{position} Payoff')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_greeks_3d(spot, strike, taux, maturite, volatilite, position):
    spot_range = np.linspace(spot * 0.5, spot * 1.5, 50)
    time_range = np.linspace(0.01, maturite / 365.0, 50)

    option_type = 'call' if 'Call' in position else 'put'

    delta = np.zeros((len(spot_range), len(time_range)))
    gamma = np.zeros((len(spot_range), len(time_range)))
    vega = np.zeros((len(spot_range), len(time_range)))
    theta = np.zeros((len(spot_range), len(time_range)))
    rho = np.zeros((len(spot_range), len(time_range)))

    for i, S in enumerate(spot_range):
        for j, T in enumerate(time_range):
            delta[i, j] = black_scholes_delta(S, strike, T, taux, volatilite, option_type)
            gamma[i, j] = black_scholes_gamma(S, strike, T, taux, volatilite)
            vega[i, j] = black_scholes_vega(S, strike, T, taux, volatilite)
            theta[i, j] = black_scholes_theta(S, strike, T, taux, volatilite, option_type)
            rho[i, j] = black_scholes_rho(S, strike, T, taux, volatilite, option_type)

    plot_3d_surface(spot_range, time_range, delta, f'Delta ({position})')
    plot_3d_surface(spot_range, time_range, gamma, f'Gamma ({position})')
    plot_3d_surface(spot_range, time_range, vega, f'Vega ({position})')
    plot_3d_surface(spot_range, time_range, theta, f'Theta ({position})')
    plot_3d_surface(spot_range, time_range, rho, f'Rho ({position})')

def black_scholes_delta(S, K, T, r, sigma, option_type='call'):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if option_type == 'call':
        return norm_cdf(d1)
    elif option_type == 'put':
        return norm_cdf(d1) - 1

def black_scholes_gamma(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))

def black_scholes_vega(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * norm_pdf(d1) * math.sqrt(T)

def black_scholes_theta(S, K, T, r, sigma, option_type='call'):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        return -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == 'put':
        return -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm_cdf(-d2)

def black_scholes_rho(S, K, T, r, sigma, option_type='call'):
    d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if option_type == 'call':
        return K * T * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == 'put':
        return -K * T * math.exp(-r * T) * norm_cdf(-d2)

def norm_cdf(x):
    """Fonction de répartition cumulative de la distribution normale standard."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    """Fonction de densité de probabilité de la distribution normale standard."""
    return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

def plot_3d_surface(X, Y, Z, title):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel(title)
    ax.set_title(f'3D Surface Plot of {title}')
    fig.colorbar(surf)
    plt.show()

# Paramètres
spot = 100
strike = 100
taux = 0.05
maturite = 30  # en jours
volatilite = 0.2

# Exemple d'utilisation
df_results = calculate_options(spot, strike, taux, maturite, volatilite)
print(df_results)

positions = ['Long Call', 'Long Put', 'Short Call', 'Short Put']
for position in positions:
    plot_payoff(spot, strike, position)
    plot_greeks_3d(spot, strike, taux, maturite, volatilite, position)
