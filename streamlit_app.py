
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

# FRED API
fred = Fred(api_key='841ad742e3a4e4bb2f3fcb90a3d078fb')
cpi_data = fred.get_series('CPIAUCSL')
cpi_data = cpi_data.resample('Q').mean()
cpi_data.name = 'cpi_fred'

# Merge CPI
data = data.merge(cpi_data, left_index=True, right_index=True, how='left')
data['cpi'] = data['cpi_fred'].fillna(data['cpi'])

# User input for transactions
st.title("Starbucks Revenue Forecasting App")
st.markdown("Forecasting revenue and flagging audit risks using ARIMAX, CPI, and transaction input.")

future_txn = st.number_input("Enter expected transactions for next quarter:", min_value=800, max_value=1200, value=1000)

# Forecasting
train = data.iloc[:-1]  # leave one period for comparison
exog_vars = ['transactions', 'cpi']
model = SARIMAX(train['revenue'], exog=train[exog_vars], order=(1,1,1), seasonal_order=(1,0,1,4))
results = model.fit(disp=False)

# Prepare forecast input
last_row = data.iloc[-1:]
future_exog = pd.DataFrame({
    'transactions': [future_txn],
    'cpi': [last_row['cpi'].values[0]]
}, index=[data.index[-1] + pd.DateOffset(months=3)])

forecast = results.get_forecast(steps=1, exog=future_exog)
forecasted_revenue = forecast.predicted_mean.values[0]

# Visualization
fig, ax = plt.subplots()
data['revenue'].plot(label='Actual Revenue', ax=ax)
ax.plot(future_exog.index, [forecasted_revenue], label='Forecasted Revenue', marker='o')
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
risk_flag = False
if abs(forecasted_revenue - last_row['revenue'].values[0]) > 800:
    risk_flag = True

# AI-generated summary
summary = (
    f"Based on the ARIMAX model incorporating {future_txn} expected transactions and current CPI levels, "
    f"the forecasted revenue for the upcoming quarter is ${forecasted_revenue:.2f} million. "
    f"This projection reflects both economic trends and consumer activity assumptions. Store count has continued its upward trajectory, "
    f"supporting top-line growth capacity. "
)

if risk_flag:
    summary += "\n\n**Risk Flag:** The forecast shows a significant deviation from recent actual revenues, raising a potential concern of revenue recognition or operational misalignment. Audit attention is advised."

st.subheader("AI-Generated Audit Summary")
st.markdown(summary)

# Footer
st.caption("Built for ITEC 3155 / ACTG 4155 - Final Project")
