import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Monte Carlo Aandelen Simulator")


top_stocks = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Alphabet (Google)": "GOOGL",
    "Meta (Facebook)": "META",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Berkshire Hathaway": "BRK-B",
    "JPMorgan Chase": "JPM",
    "Johnson & Johnson": "JNJ",
    "Visa": "V",
    "Procter & Gamble": "PG",
    "Mastercard": "MA",
    "UnitedHealth": "UNH",
    "Home Depot": "HD",
    "ExxonMobil": "XOM",
    "Coca-Cola": "KO",
    "PepsiCo": "PEP",
    "Netflix": "NFLX",
    "Adobe": "ADBE"
}


selected_stock = st.selectbox(
    "Kies een bekend aandeel (optioneel)",
    ["-- Kies --"] + list(top_stocks.keys())
)

manual_ticker = st.text_input("Of voer zelf een ticker in (bijv. AAPL)")


if selected_stock != "-- Kies --":
    ticker = top_stocks[selected_stock]
else:
    ticker = manual_ticker.upper()

st.write(f"Gekozen ticker: **{ticker}**")


simulations = st.slider("Aantal simulaties", 100, 5000, 1000)
years = st.slider("Aantal jaren vooruit", 1, 10, 5)


if st.button("Start simulatie"):

    if ticker == "":
        st.warning("Voer een ticker in of kies een aandeel.")
    else:
        try:
            data = yf.download(ticker, period="10y", interval="1mo", auto_adjust=True)
            prices = data["Close"].dropna().values

            if len(prices) < 2:
                st.error("Onvoldoende data voor deze ticker.")
            else:
                returns = np.log(prices[1:] / prices[:-1])

                mu = np.mean(returns) * 12
                sigma = np.std(returns) * np.sqrt(12)

                S0 = prices[-1]
                dt = 1/12
                N = int(years / dt)

                z = np.random.standard_normal((N, simulations))

                paths = np.zeros((N + 1, simulations))
                paths[0] = S0

                for t in range(1, N + 1):
                    paths[t] = paths[t-1] * np.exp(
                        (mu - 0.5 * sigma**2) * dt +
                        sigma * np.sqrt(dt) * z[t-1]
                    )

                
                fig, ax = plt.subplots()
                ax.plot(paths, alpha=0.1)
                ax.plot(np.mean(paths, axis=1), linewidth=2, label="Gemiddelde")
                ax.set_title(f"Monte Carlo simulatie ({ticker})")
                ax.legend()

                st.pyplot(fig)

        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")


            # voer in de terminal in streamlit run montecarlo_simulator.py dan opent hij de website