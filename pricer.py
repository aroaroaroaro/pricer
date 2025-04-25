import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st

def black_scholes(spot, strike, taux, maturite, volatilite, option_type='call'):
    # Conversion de la maturité en années
    T = maturite / 365.0

    # Calcul des paramètres intermédiaires
    d1 = (np.log(spot / strike) + (taux + 0.5 * volatilite**2) * T) / (volatilite * np.sqrt(T))
    d2 = d1 - volatilite * np.sqrt(T)

    if option_type == 'call':
        price = spot * norm.cdf(d1) - strike * np.exp(-taux * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type == 'put':
        price = strike * np.exp(-taux * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    gamma = norm.pdf(d1) / (spot * volatilite * np.sqrt(T))
    vega = spot * norm.pdf(d1) * np.sqrt(T)
    theta = -(spot * norm.pdf(d1) * volatilite) / (2 * np.sqrt(T)) - taux * strike * np.exp(-taux * T) * norm.cdf(d2)
    rho = strike * T * np.exp(-taux * T) * norm.cdf(d2)

    return price, delta, gamma, vega, theta, rho

def calculate_options(spot, strike, taux, maturite, volatilite):
    positions = ['Long Call', 'Long Put', 'Short Call', 'Short Put']
    results = []

    for position in positions:
        option_type = 'call' if 'Call' in position else 'put'
        price, delta, gamma, vega, theta, rho = black_scholes(spot, strike, taux, maturite, volatilite, option_type)

        if 'Short' in position:
            price, delta, gamma, vega, theta, rho = -price, -delta, gamma, vega, -theta, -rho

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
    st.pyplot(plt)

def plot_greeks(spot, strike, taux, maturite, volatilite, position):
    spot_range = np.linspace(spot * 0.5, spot * 1.5, 500)
    prices = []
    deltas = []
    gammas = []
    vegas = []
    thetas = []
    rhos = []

    option_type = 'call' if 'Call' in position else 'put'

    for s in spot_range:
        price, delta, gamma, vega, theta, rho = black_scholes(s, strike, taux, maturite, volatilite, option_type)
        if 'Short' in position:
            price, delta, theta, rho = -price, -delta, -theta, -rho

        prices.append(price)
        deltas.append(delta)
        gammas.append(gamma)
        vegas.append(vega)
        thetas.append(theta)
        rhos.append(rho)

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(spot_range, prices, label=f'{position} Price')
    plt.xlabel('Spot Price')
    plt.ylabel('Price')
    plt.title(f'{position} Price')
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(spot_range, deltas, label=f'{position} Delta')
    plt.xlabel('Spot Price')
    plt.ylabel('Delta')
    plt.title(f'{position} Delta')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(spot_range, gammas, label=f'{position} Gamma')
    plt.xlabel('Spot Price')
    plt.ylabel('Gamma')
    plt.title(f'{position} Gamma')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(spot_range, vegas, label=f'{position} Vega')
    plt.xlabel('Spot Price')
    plt.ylabel('Vega')
    plt.title(f'{position} Vega')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(spot_range, thetas, label=f'{position} Theta')
    plt.xlabel('Spot Price')
    plt.ylabel('Theta')
    plt.title(f'{position} Theta')
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(spot_range, rhos, label=f'{position} Rho')
    plt.xlabel('Spot Price')
    plt.ylabel('Rho')
    plt.title(f'{position} Rho')
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(plt)

# Interface utilisateur avec Streamlit
st.title("Option Pricing avec Black-Scholes")

spot = st.number_input("Spot Price", value=200.0)
strike = st.number_input("Strike Price", value=200.0)
taux = st.number_input("Taux sans risque (annuel)", value=0.05)
maturite = st.number_input("Maturité (en jours)", value=30)
volatilite = st.number_input("Volatilité (annuelle)", value=0.2)

if st.button("Calculer"):
    df_results = calculate_options(spot, strike, taux, maturite, volatilite)
    st.write(df_results)

    positions = ['Long Call', 'Long Put', 'Short Call', 'Short Put']
    for position in positions:
        st.write(f"### {position}")
        plot_payoff(spot, strike, position)
        plot_greeks(spot, strike, taux, maturite, volatilite, position)
