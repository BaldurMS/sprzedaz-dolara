import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Kiedy sprzedać dolary?", layout="centered")

st.title("💵 Kiedy najlepiej sprzedać dolary (USD)?")

def pobierz_kurs_usd(dni=365):
    url = f"http://api.nbp.pl/api/exchangerates/rates/A/USD/last/{dni}/?format=json"
    response = requests.get(url)
    data = response.json()['rates']
    df = pd.DataFrame(data)
    df['effectiveDate'] = pd.to_datetime(df['effectiveDate'])
    df.rename(columns={'effectiveDate': 'data', 'mid': 'kurs'}, inplace=True)
    return df

def przewiduj_kurs(df, dni_do_przodu=30):
    df = df.copy()
    df['dni'] = (df['data'] - df['data'].min()).dt.days
    X = df[['dni']]
    y = df['kurs']
    model = LinearRegression()
    model.fit(X, y)
    przyszlosc = np.arange(df['dni'].max() + 1, df['dni'].max() + 1 + dni_do_przodu).reshape(-1, 1)
    prognozy = model.predict(przyszlosc)
    najlepszy_idx = np.argmax(prognozy)
    najlepszy_dzien = df['data'].min() + timedelta(days=int(przyszlosc[najlepszy_idx][0]))
    return najlepszy_dzien, prognozy[najlepszy_idx], przyszlosc, prognozy

with st.form("form"):
    data_zakupu = st.date_input("📅 Wybierz datę zakupu dolarów", value=datetime.today() - timedelta(days=60))
    submitted = st.form_submit_button("🔍 Analizuj")

if submitted:
    with st.spinner("Pobieranie danych..."):
        df = pobierz_kurs_usd()
        data_zakupu = pd.to_datetime(data_zakupu)

    st.subheader("📊 Analiza historyczna")
    df_po = df[df['data'] > data_zakupu]

    if df_po.empty:
        st.warning("Brak danych po dacie zakupu. Spróbuj późniejszą datę.")
    else:
        najlepszy = df_po.loc[df_po['kurs'].idxmax()]
        st.success(f"Najlepszy historyczny moment sprzedaży: **{najlepszy['data'].date()}** po kursie **{najlepszy['kurs']:.2f} PLN**")

    st.subheader("🧠 Prognoza AI")
    przyszla_data, kurs_pred, dni_pred, prognozy = przewiduj_kurs(df)
    st.info(f"AI przewiduje najlepszy kurs około: **{przyszla_data.date()}**, kurs: **{kurs_pred:.2f} PLN**")

    # Wykres
    st.subheader("📈 Wykres kursu i prognozy")
    fig, ax = plt.subplots()
    ax.plot(df['data'], df['kurs'], label='Kurs historyczny')
    ax.plot(
        [df['data'].min() + timedelta(days=int(i[0])) for i in dni_pred],
        prognozy, '--', label='Prognoza (AI)', color='orange'
    )
    ax.axvline(data_zakupu, color='gray', linestyle='--', label='Data zakupu')
    ax.plot([przyszla_data], [kurs_pred], 'ro', label='AI – max prognoza')
    ax.set_xlabel("Data")
    ax.set_ylabel("Kurs [PLN]")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
