import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import io
import zipfile

# Fonctions utilitaires
def generate_normal_random():
    u1 = random.random()
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

def geometric_brownian_motion(S0, mu, sigma, T, dt, steps):
    S = [S0]
    for _ in range(steps):
        dW = generate_normal_random() * math.sqrt(dt)
        S_next = S[-1] * math.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        S.append(S_next)
    return S

def geometric_brownian_motion_multiple_paths(S0, mu, sigma, T, steps, n_paths):
    dt = T / steps
    paths = np.zeros((steps + 1, n_paths))
    paths[0] = S0
    for t in range(1, steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == 'put':
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

# Greeks
def black_scholes_delta(S, K, T, r, sigma, option_type='call'):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if option_type == 'call':
        return norm_cdf(d1)
    else:
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
    else:
        return -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm_cdf(-d2)

def black_scholes_rho(S, K, T, r, sigma, option_type='call'):
    d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if option_type == 'call':
        return K * T * math.exp(-r * T) * norm_cdf(d2)
    else:
        return -K * T * math.exp(-r * T) * norm_cdf(-d2)

# Calcul complet
def calculate_options_and_greeks(spot, strike, taux, maturite, volatilite):
    positions = ['Long Call', 'Long Put', 'Short Call', 'Short Put']
    results = []
    for position in positions:
        option_type = 'call' if 'Call' in position else 'put'
        T = maturite / 365.0
        price = black_scholes(spot, strike, T, taux, volatilite, option_type)
        delta = black_scholes_delta(spot, strike, T, taux, volatilite, option_type)
        gamma = black_scholes_gamma(spot, strike, T, taux, volatilite)
        vega = black_scholes_vega(spot, strike, T, taux, volatilite)
        theta = black_scholes_theta(spot, strike, T, taux, volatilite, option_type)
        rho = black_scholes_rho(spot, strike, T, taux, volatilite, option_type)
        if 'Short' in position:
            price = -price
        results.append([position, price, delta, gamma, vega, theta, rho])
    return pd.DataFrame(results, columns=['Position', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho'])

# Plotting
def plot_payoff(spot, strike, position):
    spot_range = np.linspace(spot * 0.5, spot * 1.5, 500)
    if 'Call' in position:
        payoff = np.maximum(spot_range - strike, 0)
    else:
        payoff = np.maximum(strike - spot_range, 0)
    if 'Short' in position:
        payoff = -payoff
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spot_range, payoff, label=f'{position} Payoff')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Payoff')
    ax.set_title(f'{position} Payoff')
    ax.legend()
    ax.grid(True)
    return fig

def plot_greeks_3d(spot, strike, taux, maturite, volatilite, position):
    spot_range = np.linspace(spot * 0.5, spot * 1.5, 50)
    time_range = np.linspace(0.01, maturite / 365.0, 50)
    option_type = 'call' if 'Call' in position else 'put'
    delta = np.zeros((len(spot_range), len(time_range)))
    for i, S in enumerate(spot_range):
        for j, T in enumerate(time_range):
            delta[i, j] = black_scholes_delta(S, strike, T, taux, volatilite, option_type)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(spot_range, time_range)
    ax.plot_surface(X, Y, delta.T, cmap='viridis')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Delta')
    ax.set_title('Delta Surface')
    return fig

# Export function
def export_all(inputs, results_df, paths_df, payoff_plot=None, greeks_plot=None, brownian_plot=None):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
        # Inputs
        df_inputs = pd.DataFrame.from_dict(inputs, orient='index', columns=['Value'])
        zip_file.writestr("inputs.csv", df_inputs.to_csv())
        # Results
        zip_file.writestr("options_greeks.csv", results_df.to_csv(index=False))
        # Brownian paths
        zip_file.writestr("brownian_paths.csv", paths_df.to_csv(index=False))
        # Payoff plot
        if payoff_plot:
            buf = io.BytesIO()
            payoff_plot.savefig(buf, format='png')
            buf.seek(0)
            zip_file.writestr("payoff_plot.png", buf.getvalue())
        # Greeks plot
        if greeks_plot:
            buf = io.BytesIO()
            greeks_plot.savefig(buf, format='png')
            buf.seek(0)
            zip_file.writestr("greeks_plot.png", buf.getvalue())
        # Brownian plot
        if brownian_plot:
            buf = io.BytesIO()
            brownian_plot.savefig(buf, format='png')
            buf.seek(0)
            zip_file.writestr("brownian_plot.png", buf.getvalue())

    zip_buffer.seek(0)
    st.download_button("Download All as ZIP", data=zip_buffer, file_name="option_analysis.zip", mime="application/zip")

# Application Streamlit
def main():
    st.title("Option Pricing, Greeks and Brownian Simulation")

    spot = st.number_input("Spot Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    taux = st.number_input("Interest Rate", value=0.05)
    maturite = st.number_input("Maturity (days)", value=30)
    volatilite = st.number_input("Volatility", value=0.2)

    n_paths = st.number_input("Number of Brownian Paths", value=1000, min_value=1, step=1)
    T = st.number_input("Total Time for Brownian Simulation (years)", value=1.0)
    steps = st.number_input("Number of Steps", value=252)

    tab1, tab2, tab3, tab4 = st.tabs(["Option Pricing", "Payoff", "Greeks", "Brownian Simulation"])

    global results_df, payoff_fig, greeks_fig, brownian_paths, brownian_fig
    results_df = pd.DataFrame()
    payoff_fig = None
    greeks_fig = None
    brownian_paths = None
    brownian_fig = None

    with tab1:
        if st.button("Calculate Option Prices and Greeks"):
            results_df = calculate_options_and_greeks(spot, strike, taux, maturite, volatilite)
            st.write(results_df)

    with tab2:
        position = st.selectbox("Select Position for Payoff", ['Long Call', 'Long Put', 'Short Call', 'Short Put'])
        if st.button("Plot Payoff"):
            payoff_fig = plot_payoff(spot, strike, position)
            st.pyplot(payoff_fig)

    with tab3:
        position_greek = st.selectbox("Select Position for Greeks", ['Long Call', 'Long Put', 'Short Call', 'Short Put'])
        if st.button("Plot Greeks 3D"):
            greeks_fig = plot_greeks_3d(spot, strike, taux, maturite, volatilite, position_greek)
            st.pyplot(greeks_fig)

    with tab4:
        if st.button("Simulate Brownian Motions"):
            brownian_paths = geometric_brownian_motion_multiple_paths(spot, taux, volatilite, T, int(steps), int(n_paths))
            time_grid = np.linspace(0, T, int(steps)+1)
            brownian_fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(time_grid, brownian_paths, linewidth=0.5, alpha=0.6)
            ax.set_xlabel('Time (years)')
            ax.set_ylabel('Simulated Spot Price')
            ax.set_title(f'{n_paths} Geometric Brownian Motion Trajectories')
            ax.grid(True)
            st.pyplot(brownian_fig)

    if st.button("Export All Data"):
        inputs = {
            'Spot Price': spot,
            'Strike Price': strike,
            'Interest Rate': taux,
            'Maturity (days)': maturite,
            'Volatility': volatilite,
            'Number of Brownian Paths': n_paths,
            'Total Time': T,
            'Steps': steps
        }
        paths_df = pd.DataFrame(brownian_paths) if brownian_paths is not None else pd.DataFrame()
        export_all(inputs, results_df, paths_df, payoff_fig, greeks_fig, brownian_fig)

if __name__ == "__main__":
    main()
