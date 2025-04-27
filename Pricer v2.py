import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import zipfile
from mpl_toolkits.mplot3d import Axes3D

# Black-Scholes Greeks functions
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

# Function to plot Greeks
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

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(spot_range, time_range)
    ax.plot_surface(X, Y, values.T, cmap='viridis')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel(greek)
    ax.set_title(f'{greek} Surface')
    plt.close(fig)
    return fig

# Brownian motion simulation
def simulate_brownian_motion(num_steps, num_paths):
    dt = 1/252
    dW = np.random.normal(0, np.sqrt(dt), size=(num_steps, num_paths))
    W = np.cumsum(dW, axis=0)
    return W

# Streamlit app
st.title("Simulation Mouvements Browniens & Greeks 3D")

spot = st.number_input("Spot Price", value=100.0)
strike = st.number_input("Strike Price", value=100.0)
taux = st.number_input("Interest Rate (annual, %)", value=5.0) / 100
maturite = st.number_input("Maturity (days)", value=30)
volatilite = st.number_input("Volatility (annual, %)", value=20.0) / 100
num_paths = st.number_input("Nombre de trajectoires à simuler", value=1000, step=100)

tabs = st.tabs(["Brownian Motion", "Options Pricing", "Greeks 3D", "Export All"])

with tabs[0]:
    st.header("Mouvements Browniens")
    if st.button("Simuler"):
        with st.spinner('Simulation en cours...'):
            W = simulate_brownian_motion(252, int(num_paths))
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(W)
            ax.set_title("Mouvements Browniens simulés")
            st.pyplot(fig)
        st.success('Simulation terminée !')

with tabs[1]:
    st.header("Options Pricing Table")
    positions = ['Long Call', 'Long Put', 'Short Call', 'Short Put']
    data = []
    for position in positions:
        sign = 1 if "Long" in position else -1
        delta = sign * black_scholes_delta(spot, strike, maturite/365, taux, volatilite, 'call' if "Call" in position else 'put')
        gamma = sign * black_scholes_gamma(spot, strike, maturite/365, taux, volatilite)
        vega = sign * 0.01  # simplifié
        theta = sign * black_scholes_theta(spot, strike, maturite/365, taux, volatilite, 'call' if "Call" in position else 'put')
        rho = sign * black_scholes_rho(spot, strike, maturite/365, taux, volatilite, 'call' if "Call" in position else 'put')
        price = sign * 10  # simplifié
        data.append([position, price, delta, gamma, vega, theta, rho])

    results_df = pd.DataFrame(data, columns=["Position", "Price", "Delta", "Gamma", "Vega", "Theta", "Rho"])
    st.dataframe(results_df)

with tabs[2]:
    st.header("Greeks 3D")
    if st.button("Afficher les Greeks 3D"):
        with st.spinner('Calcul et génération des surfaces 3D...'):
            delta_fig = plot_greek_3d(spot, strike, taux, maturite, volatilite, 'Delta')
            gamma_fig = plot_greek_3d(spot, strike, taux, maturite, volatilite, 'Gamma')
            theta_fig = plot_greek_3d(spot, strike, taux, maturite, volatilite, 'Theta')
            rho_fig = plot_greek_3d(spot, strike, taux, maturite, volatilite, 'Rho')

            st.pyplot(delta_fig)
            st.pyplot(gamma_fig)
            st.pyplot(theta_fig)
            st.pyplot(rho_fig)
        st.success('Affichage terminé !')

with tabs[3]:
    st.header("Export Inputs & Outputs")
    if st.button("Exporter Tout"):
        with st.spinner('Préparation du fichier ZIP...'):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zip_file:
                # Sauvegarde Inputs
                inputs = {
                    'Spot': spot,
                    'Strike': strike,
                    'Interest Rate': taux,
                    'Maturity': maturite,
                    'Volatility': volatilite,
                    'Nombre de Trajectoires': num_paths
                }
                inputs_df = pd.DataFrame(list(inputs.items()), columns=["Paramètre", "Valeur"])
                zip_file.writestr("inputs.csv", inputs_df.to_csv(index=False))

                # Sauvegarde Outputs
                zip_file.writestr("greeks_results.csv", results_df.to_csv(index=False))

                # Sauvegarde Mouvements browniens
                W = simulate_brownian_motion(252, int(num_paths))
                np.save(io.BytesIO(), W)
                W_df = pd.DataFrame(W)
                zip_file.writestr("brownian_paths.csv", W_df.to_csv(index=False))

                # Sauvegarde Graphes Greeks
                for name, fig in [('delta_plot.png', delta_fig), ('gamma_plot.png', gamma_fig), ('theta_plot.png', theta_fig), ('rho_plot.png', rho_fig)]:
                    if fig:
                        img_buf = io.BytesIO()
                        fig.savefig(img_buf, format="png")
                        img_buf.seek(0)
                        zip_file.writestr(name, img_buf.getvalue())

            buf.seek(0)
            st.download_button(
                label="Télécharger le ZIP",
                data=buf,
                file_name="export_all.zip",
                mime="application/zip"
            )
        st.success('Fichier ZIP prêt à être téléchargé !')
