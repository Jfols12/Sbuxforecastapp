import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fredapi import Fred
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("starbucks_financials_expanded.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

data = load_data()

# FRED API - Scraping CPI data
st.markdown("### 📈 CPI Data Integration from FRED")
st.markdown("Live CPI data is scraped directly from the Federal Reserve Economic Data (FRED) API to enrich the revenue forecast model.")

fred = Fred(api_key='841ad742e3a4e4bb2f3fcb90a3d078fb')
cpi_data = fred.get_series('CPIAUCSL')
cpi_data = cpi_data.resample('Q').mean()
cpi_data.name = 'cpi_fred'

# Show scraped CPI values
st.dataframe(cpi_data.tail(8).rename("CPI (Quarterly Avg)").to_frame(), use_container_width=True)

# Merge CPI
data = data.merge(cpi_data, left_index=True, right_index=True, how='left')
data['cpi'] = data['cpi_fred'].fillna(data['cpi'])

# User input for transactions
st.title("Starbucks Revenue Forecasting App")
st.markdown("Forecasting revenue and flagging audit risks using ARIMAX, CPI, and transaction input.")

future_txn = st.slider("Select expected transactions for the next 4 quarters:", min_value=800, max_value=1200, value=1000, step=10)

# Forecasting
train = data.iloc[:-1]
exog_vars = ['transactions', 'cpi']
model = SARIMAX(train['revenue'], exog=train[exog_vars], order=(1,1,1), seasonal_order=(1,0,1,4))
results = model.fit(disp=False)

# Prepare forecast inputs for 4 periods
future_dates = [data.index[-1] + pd.DateOffset(months=3*(i+1)) for i in range(4)]
future_exog = pd.DataFrame({
    'transactions': [future_txn] * 4,
    'cpi': [data['cpi'].iloc[-1]] * 4
}, index=future_dates)

forecast = results.get_forecast(steps=4, exog=future_exog)
forecasted_revenue = forecast.predicted_mean

# Combine with original data
forecast_df = pd.DataFrame({'revenue': forecasted_revenue}, index=future_dates)
combined_revenue = pd.concat([data['revenue'], forecast_df['revenue']])

# Visualization
fig, ax = plt.subplots()
data['revenue'].plot(label='Actual Revenue', ax=ax)
forecast_df['revenue'].plot(ax=ax, label='Forecasted Revenue', style='ro--')
ax.legend()
ax.set_title("Revenue Forecast vs Actual")
ax.set_ylabel("Revenue (in millions)")
st.pyplot(fig)

# Store count analysis
st.subheader("Store Count Trend")
fig2, ax2 = plt.subplots()
data['store_count'].plot(ax=ax2)
ax2.set_title("Starbucks Store Count Over Time")
ax2.set_ylabel("Number of Stores")
st.pyplot(fig2)

# Risk flagging
risk_flag = any(abs(forecast_df['revenue'] - data['revenue'].iloc[-1]) > 800)

# AI-generated summary
summary = (
    f"Based on the ARIMAX model incorporating {future_txn} expected transactions and live-scraped CPI data, "
    f"the forecasted revenue for the next four quarters ranges from ${forecast_df['revenue'].min():.2f} million to ${forecast_df['revenue'].max():.2f} million. "
    f"This projection reflects both macroeconomic conditions and operational input. Store count has continued its upward trajectory, "
    f"supporting top-line growth capacity. "
)

if risk_flag:
    summary += "\n\n**Risk Flag:** One or more forecasts show a significant deviation from recent actual revenues, raising potential concerns of revenue recognition or operational misalignment. Audit attention is advised."

st.subheader("AI-Generated Audit Summary")
st.markdown(summary)

# Footer
st.caption("Built for ITEC 3155 / ACTG 4155 - Final Project")
