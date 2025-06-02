
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.api import OLS, add_constant
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

# App header
st.title("Audit Risk Evaluation: Starbucks Revenue Forecasting")
st.markdown("This tool assists auditors in evaluating the **risk of revenue overstatement** at Starbucks using ARIMAX forecasting, transaction input, CPI, and store count analysis.")

# User input
future_txn = st.slider("Select expected transactions for the next 4 quarters:", min_value=800, max_value=1200, value=1000, step=10)

# Forecast inputs
future_dates = [data.index[-1] + pd.DateOffset(months=3*(i+1)) for i in range(4)]

# ARIMAX model: transactions + CPI
train_base = data.iloc[:-1]
exog_vars_base = ['transactions', 'cpi']
model_base = SARIMAX(train_base['revenue'], exog=train_base[exog_vars_base], order=(1,1,1), seasonal_order=(1,0,1,4))
results_base = model_base.fit(disp=False)

future_exog_base = pd.DataFrame({
    'transactions': [future_txn] * 4,
    'cpi': [data['cpi'].iloc[-1]] * 4
}, index=future_dates)

forecast_base = results_base.get_forecast(steps=4, exog=future_exog_base)
forecasted_revenue_base = forecast_base.predicted_mean
forecast_df_base = pd.DataFrame({'revenue': forecasted_revenue_base}, index=future_dates)

# ARIMAX model: transactions + CPI + store_count
train_ext = data.iloc[:-1]
exog_vars_ext = ['transactions', 'cpi', 'store_count']
model_ext = SARIMAX(train_ext['revenue'], exog=train_ext[exog_vars_ext], order=(1,1,1), seasonal_order=(1,0,1,4))
results_ext = model_ext.fit(disp=False)

future_exog_ext = pd.DataFrame({
    'transactions': [future_txn] * 4,
    'cpi': [data['cpi'].iloc[-1]] * 4,
    'store_count': [data['store_count'].iloc[-1]] * 4
}, index=future_dates)

forecast_ext = results_ext.get_forecast(steps=4, exog=future_exog_ext)
forecasted_revenue_ext = forecast_ext.predicted_mean
forecast_df_ext = pd.DataFrame({'revenue': forecasted_revenue_ext}, index=future_dates)

# Visualization - Base Model
st.subheader("Revenue Forecast vs Actual (Base Model: Transactions + CPI)")
fig1, ax1 = plt.subplots()
data['revenue'].plot(label='Actual Revenue', ax=ax1)
forecast_df_base['revenue'].plot(ax=ax1, label='Forecasted Revenue (Base)', style='ro--')
ax1.legend()
ax1.set_title("Base ARIMAX Forecast")
ax1.set_ylabel("Revenue (in millions)")
st.pyplot(fig1)

# Visualization - Extended Model
st.subheader("Revenue Forecast vs Actual (Extended Model: Transactions + CPI + Store Count)")
fig2, ax2 = plt.subplots()
data['revenue'].plot(label='Actual Revenue', ax=ax2)
forecast_df_ext['revenue'].plot(ax=ax2, label='Forecasted Revenue (Extended)', style='go--')
ax2.legend()
ax2.set_title("Extended ARIMAX Forecast")
ax2.set_ylabel("Revenue (in millions)")
st.pyplot(fig2)

# Store count coefficient analysis
st.subheader("Store Count Coefficient Analysis")
coef = results_ext.params
pvalues = results_ext.pvalues
store_coef = coef.get('store_count', np.nan)
store_pval = pvalues.get('store_count', np.nan)
st.markdown(f"**Coefficient for store_count:** {store_coef:.4f}")
st.markdown(f"**P-value:** {store_pval:.4f}")
if store_pval < 0.05:
    st.success("Store count is statistically significant in explaining revenue.")
else:
    st.warning("Store count is NOT statistically significant in this model.")

# Store count trend
st.subheader("📊 Store Count Trend Over Time")
fig3, ax3 = plt.subplots()
data['store_count'].plot(ax=ax3, color='purple')
ax3.set_title("Store Count Trend")
ax3.set_ylabel("Number of Stores")
st.pyplot(fig3)

# Risk flagging
risk_flag = any(abs(forecast_df_ext['revenue'] - data['revenue'].iloc[-1]) > 800)

# AI-generated summary
summary = (
    f"This audit-focused app helps assess the **risk of revenue overstatement** at Starbucks by forecasting revenue using expected transactions ({future_txn}), CPI from FRED, and store count.\n"
    f"The extended ARIMAX model projects next quarter revenues between ${forecast_df_ext['revenue'].min():.2f}M and ${forecast_df_ext['revenue'].max():.2f}M.\n"
    f"The trend in store count has been upward, which generally supports higher revenue potential. However, over-expansion could inflate revenue expectations without corresponding demand.\n"
    f"Store count's coefficient is {store_coef:.2f}, with a p-value of {store_pval:.4f}, indicating it {'is' if store_pval < 0.05 else 'is NOT'} statistically significant in this model."
)

if risk_flag:
    summary += "\n\n**Risk Flag:** Significant variance detected in revenue forecasts, suggesting potential overstatement or operational volatility that merits audit attention."

st.subheader("AI-Generated Audit Summary")
st.markdown(summary)

# Footer
st.caption("Built for ITEC 3155 / ACTG 4155 - Final Project")
