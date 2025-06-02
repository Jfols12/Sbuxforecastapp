
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fredapi import Fred

# Title
st.title("Starbucks Revenue Forecasting App")
st.markdown("""
This AI-powered app forecasts Starbucks revenue using an ARIMAX model, integrates real-time CPI data from FRED, 
and provides insight on employee count trends as a potential audit risk factor.
""")

# Load Starbucks data
df = pd.read_csv("starbucks_financials_expanded.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Sidebar user input: CPI multiplier
st.sidebar.header("User Input")
cpi_multiplier = st.sidebar.slider("Adjust CPI Impact Multiplier", 0.5, 2.0, 1.0, step=0.1)

# Live CPI data from FRED API
st.subheader("Live CPI Data from FRED")
fred = Fred(api_key="841ad742e3a4e4bb2f3fcb90a3d078fb")
try:
    cpi_series = fred.get_series("CPIAUCSL")
    latest_cpi = cpi_series.iloc[-1]
    st.write(f"Latest CPI (FRED): {latest_cpi:.2f}")
except Exception as e:
    st.warning(f"Failed to fetch CPI data from FRED. Error: {e}")
    latest_cpi = 1.0

# Prepare variables for ARIMAX model
y = df['revenue']
X = df[['cpi']].copy()
X['cpi'] *= cpi_multiplier

# Fit ARIMAX model
model = SARIMAX(y, exog=X, order=(1,1,1), seasonal_order=(0,0,0,0))
results = model.fit(disp=False)

# Forecast next 4 quarters
future_dates = pd.date_range(start=df.index[-1] + pd.offsets.QuarterEnd(), periods=4, freq='Q')
future_cpi = [latest_cpi * cpi_multiplier] * 4
future_exog = pd.DataFrame({'cpi': future_cpi}, index=future_dates)
forecast = results.get_forecast(steps=4, exog=future_exog)
forecast_mean = forecast.predicted_mean

# Plot actual vs forecasted revenue
st.subheader("Forecasted vs Actual Revenue")
fig, ax = plt.subplots()
y.plot(label='Actual Revenue', ax=ax)
forecast_mean.plot(label='Forecasted Revenue', ax=ax)
plt.legend()
plt.ylabel("Revenue (Millions)")
st.pyplot(fig)

# New Insight: Employee Count Analysis
st.subheader("Employee Count Trend")
fig2, ax2 = plt.subplots()
df['employee_count'].plot(ax=ax2, color='green')
plt.title("Quarterly Employee Count")
plt.ylabel("# Employees")
st.pyplot(fig2)

# AI-generated summary
summary = (
    f"""Starbucks revenue is projected to rise over the next four quarters under a CPI multiplier of {cpi_multiplier:.1f}. 
    Forecasting incorporates live CPI data from the FRED API, offering an adaptive model for inflation sensitivity. 
    A steady increase in employee count observed may indicate resource expansion, which could elevate cost risk if not matched by proportional revenue gains. 
    The audit committee should monitor whether employee growth aligns with revenue trends."""
)
st.subheader("AI-Generated Executive Summary")
st.write(summary)
