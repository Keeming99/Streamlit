import streamlit as st
from PIL import Image
import pandas as pd
import requests
import json

#---------------------------------#
# Title

image = Image.open('logo.png')

st.image(image, width = 390)

st.title("KM's Currency Converter App")
st.markdown("""
This app interconverts the value of foreign currencies!

""")

#---------------------------------#
# Sidebar + Main panel
st.sidebar.header('Input Options')

## Sidebar - Currency price unit
currency_list = ['AUD', 'BGN', 'BRL', 'CAD', 'CHF', 'CNY', 'CZK', 'DKK', 'GBP', 'HKD', 'HRK', 'HUF', 'IDR', 'ILS', 'INR', 'ISK', 'JPY', 'KRW', 'MXN', 'MYR', 'NOK', 'NZD', 'PHP', 'PLN', 'RON', 'RUB', 'SEK', 'SGD', 'THB', 'TRY', 'USD', 'ZAR']
base_price_unit = st.sidebar.selectbox('Select base currency for conversion', currency_list)
symbols_price_unit = st.sidebar.selectbox('Select target currency to convert to', currency_list)

# Amount to convert
amount = st.sidebar.number_input("Amount", min_value=0.0, value=1.0)

# Retrieving currency data from freecurrencyapi.com
@st.cache_data
def load_data(base, symbols):
    api_key = 'fca_live_PN7RiBu3mrIixFlTUvozMqjxYkV3IWYrK28bsboH'  # Replace with your actual API key
    url = f'https://api.freecurrencyapi.com/v1/latest?apikey={api_key}&base_currency={base}&currencies={symbols}'
    response = requests.get(url)
    data = response.json()
    
    if 'data' in data:
        rates_df = pd.DataFrame(data['data'].items(), columns=['converted_currency', 'price'])
        base_currency = pd.Series(base, name='base_currency')
        df = pd.concat([base_currency, rates_df], axis=1)
        return df
    else:
        return pd.DataFrame(columns=['base_currency', 'converted_currency', 'price'])

df = load_data(base_price_unit, symbols_price_unit)

st.header('Currency conversion')

if not df.empty:
    st.write(df)
    conversion_rate = df.loc[df['converted_currency'] == symbols_price_unit, 'price'].values[0]
    converted_amount = amount * conversion_rate
    st.write(f"{amount} {base_price_unit} is equal to {converted_amount:.2f} {symbols_price_unit}")
else:
    st.write("No data available for the selected currencies.")


#---------------------------------#
#---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python libraries:** streamlit, pandas, pillow, requests, json
* **Data source:** [freecurrencyapi.com](https://freecurrencyapi.com/) which provides free foreign exchange rates.
""")

