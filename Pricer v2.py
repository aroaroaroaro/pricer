import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import zipfile
from mpl_toolkits.mplot3d import Axes3D

# Fonctions Greeks Black-Scholes
def black_scholes_delta(S, K, T, r, sigma, option_type='call'):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return -norm.cdf(-d1)

def black_scholes_gamma(S, K, T, r, sigma):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def black_scholes_vega(S, K, T, r, sigma):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def black_scholes_theta(S, K, T, r, sigma, option_type='call'):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    first_term = -(S * norm.pdf(d1) * sigma) / (2*np.sqrt(T))
    if option_type == 'call':
        second_term = r * K * np.exp(-r*T) * norm.cdf(d2)
        return first_term - second_term
    else:
        second_term = r * K * np.exp(-r*T) * norm.cdf(-d2)
        return first_term + second_term

def black_scholes_rho(S, K, T, r, sigma, option_type='call'):
    from scipy.stats import norm
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    if option_type == 'call':
        return K * T * np.exp(-r*T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r*T) * norm.cdf(-d2)

# Fonction pour tracer la surface 3D d'un Greek
def plot_greek_3d(spot, strike, taux, maturite, volatilite, greek):
    spot_range = np.linspace(spot * 0.5, spot * 1.5, 50)
    time_range = np.linspace(0.01, maturite / 365.0, 50)
    values = np.zeros((len(spot_range), len(time_range)))
    for i, S in enumerate(spot_range):
        for j, T in enumerate(time_range):
            if greek == 'Delta':
                values[i, j] = black_scholes_delta(S, strike, T, taux, volatilite, 'call')
            elif greek == 'Gamma':
                values[i, j] = black_scholes_gamma(S, strike, T, taux, volatilite)
            elif greek == 'Theta':
                values[i, j] = black_scholes_theta(S, strike, T, taux, volatilite, 'call')
            elif greek == 'Rho':
                values[i, j] = black_scholes_rho(S, strike, T, taux, volatilite, 'call')
            elif greek == 'Vega':
                values[i, j] = black_scholes_vega(S, strike, T, taux, volatilite)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(spot_range, time_range)
    ax.plot_surface(X, Y, values.T, cmap='viridis')
    ax.set_xlabel('Prix Spot')
    ax.set_ylabel('Temps jusqu\'à échéance')
    ax.set_zlabel(greek)
    ax.set_title(f'Surface 3D de {greek}')
    plt.close(fig)
    return fig

# Simulation mouvement brownien
def simulate_brownian_motion(num_steps, num_paths):
    dt = 1/252
    dW = np.random.normal(0, np.sqrt(dt), size=(num_steps, num_paths))
    W = np.cumsum(dW, axis=0)
    return W

# --- App Streamlit ---
st.title("📈 Simulateur de Greeks et Mouvements Browniens")

spot = st.number_input("Prix Spot", value=100.0)
strike = st.number_input("Prix d'Exercice (Strike)", value=100.0)
taux = st.number_input("Taux d'Intérêt Annuel (%)", value=5.0) / 100
maturite = st.number_input("Maturité (en jours)", value=30)
volatilite = st.number_input("Volatilité Annuelle (%)", value=20.0) / 100
num_paths = st.number_input("Nombre de Trajectoires Browniennes", value=1000, step=100)

tabs = st.tabs(["1️⃣ Paramètres et Outputs", "2️⃣ Greeks 3D", "3️⃣ Simulation Brownienne", "4️⃣ Exporter Tout"])

# Variables globales pour éviter l'erreur
greek_figures = {}

with tabs[0]:
    st.header("Paramètres et Résultats des Options")
    positions = ['Long Call', 'Long Put', 'Short Call', 'Short Put']
    data = []
    for position in positions:
        sign = 1 if "Long" in position else -1
        delta = sign * black_scholes_delta(spot, strike, maturite/365, taux, volatilite, 'call' if "Call" in position else 'put')
        gamma = sign * black_scholes_gamma(spot, strike, maturite/365, taux, volatilite)
        vega = sign * black_scholes_vega(spot, strike, maturite/365, taux, volatilite)
        theta = sign * black_scholes_theta(spot, strike, maturite/365, taux, volatilite, 'call' if "Call" in position else 'put')
        rho = sign * black_scholes_rho(spot, strike, maturite/365, taux, volatilite, 'call' if "Call" in position else 'put')
        price = sign * 10  # Prix fictif pour exemple
        data.append([position, price, delta, gamma, vega, theta, rho])

    results_df = pd.DataFrame(data, columns=["Position", "Prix", "Delta", "Gamma", "Vega", "Theta", "Rho"])
    st.dataframe(results_df)

with tabs[1]:
    st.header("Surfaces 3D des Greeks")
    if st.button("Afficher les Surfaces 3D des Greeks"):
        with st.spinner('Génération des surfaces 3D en cours...'):
            for greek in ['Delta', 'Gamma', 'Theta', 'Rho', 'Vega']:
                fig = plot_greek_3d(spot, strike, taux, maturite, volatilite, greek)
                greek_figures[greek] = fig
                st.pyplot(fig)
        st.success("Affichage terminé ✅")

with tabs[2]:
    st.header("Simulation de Mouvements Browniens")
    if st.button("Simuler Brownien"):
        with st.spinner('Simulation en cours...'):
            W = simulate_brownian_motion(252, int(num_paths))
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(W)
            ax.set_title("Trajectoires du Mouvement Brownien")
            st.pyplot(fig)
            W_df = pd.DataFrame(W)
            st.session_state['W_df'] = W_df
        st.success("Simulation terminée ✅")

with tabs[3]:
    st.header("Exporter Tous les Résultats")
    if st.button("Exporter sous format ZIP"):
        with st.spinner("Création de l'archive ZIP..."):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zip_file:
                # Inputs
                inputs = {
                    'Prix Spot': spot,
                    'Strike': strike,
                    'Taux Intérêt': taux,
                    'Maturité (jours)': maturite,
                    'Volatilité': volatilite,
                    'Nombre de trajectoires': num_paths
                }
                inputs_df = pd.DataFrame(list(inputs.items()), columns=["Paramètre", "Valeur"])
                zip_file.writestr("inputs.csv", inputs_df.to_csv(index=False))

                # Outputs (table Greeks)
                zip_file.writestr("greeks_table.csv", results_df.to_csv(index=False))

                # Brownian Paths
                if 'W_df' in st.session_state:
                    zip_file.writestr("brownian_paths.csv", st.session_state['W_df'].to_csv(index=False))

                # Graphiques Greeks
                for greek, fig in greek_figures.items():
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format="png")
                    img_buf.seek(0)
                    zip_file.writestr(f"{greek.lower()}_surface.png", img_buf.getvalue())

            buf.seek(0)
            st.download_button(
                label="📥 Télécharger l'archive ZIP",
                data=buf,
                file_name="resultats_simulation.zip",
                mime="application/zip"
            )
        st.success("ZIP prêt ✅")
