import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import io
import zipfile

# Inclure toutes les fonctions de votre code ici...

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

def calculate_options_and_greeks(spot, strike, taux, maturite, volatilite):
    positions = ['Long Call', 'Long Put', 'Short Call', 'Short Put']
    results = []

    for position in positions:
        option_type = 'call' if 'Call' in position else 'put'
        T = maturite / 365.0
        price = option_pricer(spot, strike, T, taux, volatilite, option_type=option_type)
        d1 = (math.log(spot / strike) + (taux + 0.5 * volatilite**2) * T) / (volatilite * math.sqrt(T))
        d2 = d1 - volatilite * math.sqrt(T)

        delta = black_scholes_delta(spot, strike, T, taux, volatilite, option_type)
        gamma = black_scholes_gamma(spot, strike, T, taux, volatilite)
        vega = black_scholes_vega(spot, strike, T, taux, volatilite)
        theta = black_scholes_theta(spot, strike, T, taux, volatilite, option_type)
        rho = black_scholes_rho(spot, strike, T, taux, volatilite, option_type)

        if 'Short' in position:
            price = -price

        results.append([position, price, delta, gamma, vega, theta, rho])

    df_results = pd.DataFrame(results, columns=['Position', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho'])
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
    return plt

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

    plots = []
    plots.append(plot_3d_surface(spot_range, time_range, delta, f'Delta ({position})'))
    plots.append(plot_3d_surface(spot_range, time_range, gamma, f'Gamma ({position})'))
    plots.append(plot_3d_surface(spot_range, time_range, vega, f'Vega ({position})'))
    plots.append(plot_3d_surface(spot_range, time_range, theta, f'Theta ({position})'))
    plots.append(plot_3d_surface(spot_range, time_range, rho, f'Rho ({position})'))
    return plots

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
    st.pyplot(fig)
    return fig

def export_data(inputs, df_results, payoff_plot=None, greeks_plots=None, filename='export.zip'):
    # Create a DataFrame for inputs
    df_inputs = pd.DataFrame(inputs, index=['Value'])

    # Combine inputs and results into a single DataFrame
    df_combined = pd.concat([df_inputs, df_results], axis=1)

    # Save the combined DataFrame to CSV
    csv = df_combined.to_csv(index=False)
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("data.csv", csv)

        # Save payoff plot if provided
        if payoff_plot:
            payoff_buf = io.BytesIO()
            payoff_plot.savefig(payoff_buf, format='png')
            payoff_buf.seek(0)
            zip_file.writestr("payoff_plot.png", payoff_buf.getvalue())

        # Save greeks plots if provided
        if greeks_plots:
            for i, plot in enumerate(greeks_plots):
                greek_buf = io.BytesIO()
                plot.savefig(greek_buf, format='png')
                greek_buf.seek(0)
                zip_file.writestr(f"greeks_plot_{i}.png", greek_buf.getvalue())

    zip_buffer.seek(0)
    st.download_button(
        label="Download all data as ZIP",
        data=zip_buffer,
        file_name=filename,
        mime="application/zip"
    )

def main():
    st.title("Option Pricing and Visualization")

    # Paramètres par défaut
    spot = st.number_input("Spot Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    taux = st.number_input("Interest Rate", value=0.05)
    maturite = st.number_input("Maturity (days)", value=30)
    volatilite = st.number_input("Volatility", value=0.2)

    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs(["Calculate Option Prices and Greeks", "Payoff Visualization", "Greeks Visualization", "Export Data"])

    # Variables globales pour stocker les résultats et les graphiques
    global df_results, payoff_plot, greeks_plots
    df_results = pd.DataFrame()
    payoff_plot = None
    greeks_plots = []

    with tab1:
        st.header("Calculate Option Prices and Greeks")
        st.write("Calculate the prices and Greeks for different option positions.")
        if st.button("Calculate"):
            df_results = calculate_options_and_greeks(spot, strike, taux, maturite, volatilite)
            st.write(df_results)

    with tab2:
        st.header("Payoff Visualization")
        st.write("Visualize the payoff for different option positions.")
        position = st.selectbox("Select Position", ['Long Call', 'Long Put', 'Short Call', 'Short Put'])
        if st.button("Plot Payoff"):
            payoff_plot = plot_payoff(spot, strike, position)
            st.pyplot(payoff_plot)

    with tab3:
        st.header("Greeks Visualization")
        st.write("Visualize the Greeks for different option positions.")
        position = st.selectbox("Select Position", ['Long Call', 'Long Put', 'Short Call', 'Short Put'], key="greeks")
        if st.button("Plot Greeks"):
            greeks_plots = plot_greeks_3d(spot, strike, taux, maturite, volatilite, position)

    with tab4:
        st.header("Export Data")
        st.write("Download all inputs, results, and plots as a ZIP file.")
        inputs = {
            'Spot Price': spot,
            'Strike Price': strike,
            'Interest Rate': taux,
            'Maturity (days)': maturite,
            'Volatility': volatilite
        }
        if st.button("Export Data"):
            export_data(inputs, df_results, payoff_plot, greeks_plots, 'export.zip')

if __name__ == "__main__":
    main()
