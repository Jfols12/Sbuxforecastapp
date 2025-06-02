
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from fredapi import Fred

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

# Sidebar user inputs
st.sidebar.header("User Inputs")
cpi_multiplier = st.sidebar.slider("CPI Impact Multiplier", 0.5, 2.0, 1.0, step=0.1)
store_growth_rate = st.sidebar.slider("Expected Quarterly Store Growth Rate (%)", 0.0, 10.0, 2.0, step=0.5) / 100

# Live CPI data from FRED using API
st.subheader("Live CPI Data from FRED")
fred = Fred(api_key="841ad742e3a4e4bb2f3fcb90a3d078fb")
try:
    cpi_series = fred.get_series("CPIAUCSL")
    latest_cpi = cpi_series.iloc[-1]
    st.write(f"Latest CPI (FRED): {latest_cpi:.2f}")
except Exception as e:
    st.warning(f"Failed to fetch CPI data from FRED. Error: {e}")
    latest_cpi = 1.0

# Modeling: SARIMAX (ARIMAX)
y = df['revenue']
X = df[['cpi', 'store_count']].copy()
X['cpi'] *= cpi_multiplier

# Fit model
model = SARIMAX(y, exog=X, order=(1,1,1), seasonal_order=(0,0,0,0))
results = model.fit(disp=False)

# Forecast next 4 quarters
future_dates = pd.date_range(start=df.index[-1] + pd.offsets.QuarterEnd(), periods=4, freq='Q')
future_cpi = [latest_cpi * cpi_multiplier] * 4

last_store_count = df['store_count'].iloc[-1]
future_store_count = [last_store_count * ((1 + store_growth_rate) ** i) for i in range(1, 5)]

future_exog = pd.DataFrame({
    'cpi': future_cpi,
    'store_count': future_store_count
}, index=future_dates)

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
    f"driven in part by macroeconomic stability reflected in CPI values and a projected quarterly store growth rate of {store_growth_rate*100:.1f}%. "
    f"With {int(last_store_count)} stores currently in operation, expansion could influence revenue trends but may also increase risk if not matched by proportional income growth."
)

st.subheader("AI-Generated Executive Summary")
st.write(summary)

# Optional Enhancement: Simple Risk Flag
avg_growth = forecast_mean.pct_change().mean()
if avg_growth > 0.05 and df['store_count'].diff().mean() > 10:
    st.warning("⚠️ Potential Risk Flag: Rapid revenue growth coinciding with store expansion may warrant further review.")
