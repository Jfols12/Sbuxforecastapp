
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from io import StringIO

# Title
st.title("Starbucks Revenue Forecasting App")
st.markdown("""
This app forecasts Starbucks revenue using a SARIMAX model, integrates live CPI data from FRED,
and evaluates additional insights like store count trends. 
""")

# Load data
df = pd.read_csv("starbucks_financials_expanded.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# User input: CPI multiplier
st.sidebar.header("User Input")
cpi_multiplier = st.sidebar.slider("CPI Impact Multiplier", 0.5, 2.0, 1.0, step=0.1)

# Live data scraping from FRED for latest CPI
st.subheader("Live CPI Data from FRED")
fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
try:
    response = requests.get(fred_url)
    fred_data = pd.read_csv(StringIO(response.text))
    fred_data['DATE'] = pd.to_datetime(fred_data['DATE'])
    fred_data.set_index('DATE', inplace=True)
    latest_cpi = fred_data.iloc[-1].values[0]
    st.write(f"Latest CPI (FRED): {latest_cpi:.2f}")
except:
    st.warning("Failed to fetch CPI data from FRED.")
    latest_cpi = 1.0

# Modeling: SARIMAX (ARIMAX)
y = df['revenue']
X = df[['cpi']].copy()
X['cpi'] *= cpi_multiplier

# Fit model
model = SARIMAX(y, exog=X, order=(1,1,1), seasonal_order=(0,0,0,0))
results = model.fit(disp=False)

# Forecast next 4 quarters
future_dates = pd.date_range(start=df.index[-1] + pd.offsets.QuarterEnd(), periods=4, freq='Q')
future_cpi = [latest_cpi * cpi_multiplier] * 4
future_exog = pd.DataFrame({'cpi': future_cpi}, index=future_dates)
forecast = results.get_forecast(steps=4, exog=future_exog)
forecast_mean = forecast.predicted_mean

# Plot forecast vs actual
st.subheader("Revenue Forecast")
fig, ax = plt.subplots()
y.plot(label='Historical Revenue', ax=ax)
forecast_mean.plot(label='Forecasted Revenue', ax=ax)
plt.legend()
plt.ylabel("Revenue (Millions)")
st.pyplot(fig)

# New insight: Store count over time
st.subheader("Store Count Insight")
fig2, ax2 = plt.subplots()
df['store_count'].plot(ax=ax2)
plt.title("Quarterly Store Count")
plt.ylabel("# Stores")
st.pyplot(fig2)

# AI-generated summary
summary = (
    f"As of the latest forecast, Starbucks revenue is projected to grow over the next four quarters, "
    f"driven in part by macroeconomic stability reflected in CPI values. With {df['store_count'].iloc[-1]} "
    f"stores currently in operation, continued expansion may signal a risk of revenue overstatement if not matched by proportional income growth."
)

st.subheader("AI-Generated Executive Summary")
st.write(summary)

# Optional Enhancement: Simple Risk Flag
avg_growth = forecast_mean.pct_change().mean()
if avg_growth > 0.05 and df['store_count'].diff().mean() > 10:
    st.warning("⚠️ Potential Risk Flag: Rapid revenue growth coinciding with store expansion may warrant further review.")
