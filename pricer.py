import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st
import io
import zipfile
from datetime import datetime, timedelta

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
    positions = ['Call', 'Put']
    results = []

    for position in positions:
        option_type = 'call' if position == 'Call' else 'put'
        price, delta, gamma, vega, theta, rho = black_scholes(spot, strike, taux, maturite, volatilite, option_type)
        results.append([position, price, delta, gamma, vega, theta, rho])

    df_results = pd.DataFrame(results, columns=['Position', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho'])
    return df_results

def plot_payoff(spot, strike, position):
    spot_range = np.linspace(spot * 0.5, spot * 1.5, 500)
    if position == 'Call':
        payoff = np.maximum(spot_range - strike, 0)
    elif position == 'Put':
        payoff = np.maximum(strike - spot_range, 0)

    plt.figure(figsize=(10, 6))
    plt.plot(spot_range, payoff, label=f'{position} Payoff')
    plt.xlabel('Spot Price')
    plt.ylabel('Payoff')
    plt.title(f'{position} Payoff')
    plt.legend()
    plt.grid(True)

    # Sauvegarde du graphique dans un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()  # Fermer la figure pour éviter les conflits
    return buf

def plot_greeks(spot, strike, taux, maturite, volatilite, position):
    spot_range = np.linspace(spot * 0.5, spot * 1.5, 500)
    deltas = []
    gammas = []
    vegas = []
    thetas = []
    rhos = []

    option_type = 'call' if position == 'Call' else 'put'

    for s in spot_range:
        _, delta, gamma, vega, theta, rho = black_scholes(s, strike, taux, maturite, volatilite, option_type)
        deltas.append(delta)
        gammas.append(gamma)
        vegas.append(vega)
        thetas.append(theta)
        rhos.append(rho)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(spot_range, deltas, label=f'{position} Delta')
    plt.xlabel('Spot Price')
    plt.ylabel('Delta')
    plt.title(f'{position} Delta')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(spot_range, gammas, label=f'{position} Gamma')
    plt.xlabel('Spot Price')
    plt.ylabel('Gamma')
    plt.title(f'{position} Gamma')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(spot_range, vegas, label=f'{position} Vega')
    plt.xlabel('Spot Price')
    plt.ylabel('Vega')
    plt.title(f'{position} Vega')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(spot_range, thetas, label=f'{position} Theta')
    plt.xlabel('Spot Price')
    plt.ylabel('Theta')
    plt.title(f'{position} Theta')
    plt.grid(True)

    plt.tight_layout()

    # Sauvegarde du graphique dans un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()  # Fermer la figure pour éviter les conflits
    return buf

# Interface utilisateur avec Streamlit
st.title("Option Pricing avec Black-Scholes")

spot = st.number_input("Spot Price", value=200.0)
strike = st.number_input("Strike Price", value=200.0)
taux = st.number_input("Taux sans risque (annuel)", value=0.05)
volatilite = st.number_input("Volatilité (annuelle)", value=0.2)

# Sélection de la date de maturité via un calendrier
today = datetime.today()
maturite_date = st.date_input("Date de Maturité", value=today + timedelta(days=30), min_value=today + timedelta(days=1))

if st.button("Calculer"):
    # Vérification que la date de maturité est bien sélectionnée
    if isinstance(maturite_date, datetime.date):
        maturite = (maturite_date - today).days
        df_results = calculate_options(spot, strike, taux, maturite, volatilite)
        st.write(df_results)

        positions = ['Call', 'Put']
        buffers = {}

        for position in positions:
            st.write(f"### {position}")
            buf_payoff = plot_payoff(spot, strike, position)
            buf_greeks = plot_greeks(spot, strike, taux, maturite, volatilite, position)
            buffers[f"{position}_payoff.png"] = buf_payoff
            buffers[f"{position}_greeks.png"] = buf_greeks

            # Afficher les graphiques dans l'application
            st.pyplot(plt)

        # Création d'un fichier ZIP contenant les résultats et les graphiques
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Ajouter le fichier CSV des résultats
            csv = df_results.to_csv(index=False)
            zip_file.writestr("option_pricing_results.csv", csv)

            # Ajouter les graphiques
            for file_name, buf in buffers.items():
                zip_file.writestr(file_name, buf.getvalue())

        zip_buffer.seek(0)
        st.download_button(
            label="Télécharger tous les résultats",
            data=zip_buffer,
            file_name="option_pricing_results.zip",
            mime="application/zip"
        )
    else:
        st.error("Veuillez sélectionner une date de maturité valide.")
