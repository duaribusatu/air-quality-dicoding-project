import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Dashboard
st.title('PM2.5 Concentration Dashboard - Guanyuan Air Quality Analysis')

# Load Data
data = pd.read_csv('all_data.csv')

# Buat kolom 'date' dari 'year', 'month', 'day', dan 'hour'
data['date'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])

# Bagian 1: Monthly PM2.5 Concentration
st.subheader('Monthly PM2.5 Concentration')

monthly_data = data[['month', 'PM2.5']]
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
ordered_monthdf = pd.DataFrame(months, columns=['month'])
map_dict = {i+1: j for i, j in enumerate(months)}

monthly_data['month'] = monthly_data['month'].map(map_dict)
monthly_average = monthly_data.groupby('month').median()
monthly_average = pd.merge(ordered_monthdf, monthly_average, left_on='month', right_index=True)
monthly_average = np.round(monthly_average, 1)
monthly_average = monthly_average.set_index('month')

# Visualisasi Monthly Data
st.write('Monthly average of PM2.5 concentration:')
fig, ax = plt.subplots(figsize=(12, 5))
with plt.style.context('ggplot'):
    monthly_average.plot(kind='bar', legend=False, ax=ax)
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('PM2.5 concentration (ug/m^3)', fontsize=14)
    ax.set_title('Monthly average PM2.5 concentration', fontsize=16)
    plt.grid(axis='x')
    st.pyplot(fig)

# Bagian 2: Daily Trend of PM2.5 Concentration
st.subheader('Daily Trend of PM2.5 Concentration')

daily_data = data[['date', 'PM2.5']]
daily_data = daily_data.set_index('date').resample('D').median()

# Decompose Trend
decomposition = seasonal_decompose(daily_data, model='additive')

# Visualisasi Daily Trend
st.write('Daily trend of PM2.5 concentration:')
fig, ax = plt.subplots(figsize=(12, 5))
with plt.style.context('fivethirtyeight'):
    decomposition.trend.plot(ax=ax, style='k-', linewidth=.9)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('PM2.5 concentration (ug/m^3)', fontsize=14)
    ax.set_title('Daily trend of PM2.5 concentration', fontsize=16)
    plt.grid(axis='x')
    st.pyplot(fig)

# Bagian 3: Hourly PM2.5 Concentration
st.subheader('Hourly PM2.5 Concentration')

hourly_data = data[['hour', 'PM2.5']]
hrs = ['12 AM', '1 AM', '2 AM', '3 AM', '4 AM', '5 AM', '6 AM', '7 AM', '8 AM', '9 AM', '10 AM',
       '11 AM', '12 PM', '1 PM', '2 PM', '3 PM', '4 PM', '5 PM', '6 PM', '7 PM',
       '8 PM', '9 PM', '10 PM', '11 PM']
hour_dict = {i: j for i, j in enumerate(hrs)}

hourly_data = hourly_data.groupby('hour').median().reset_index()
hourly_data['hour'] = hourly_data['hour'].map(hour_dict)
hourly_data = hourly_data.set_index('hour')

# Visualisasi Hourly Data
st.write('Average PM2.5 concentration by the hour of the day:')
fig, ax = plt.subplots(figsize=(12, 8))
with plt.style.context('ggplot'):
    hourly_data.plot(kind='barh', legend=False, ax=ax)
    ax.set_ylabel('Hours', fontsize=14)
    ax.set_xlabel('PM2.5 concentration (ug/m^3)', fontsize=14)
    ax.set_title('Hourly PM2.5 concentration', fontsize=16)
    plt.grid(axis='y')
    st.pyplot(fig)

# Bagian 4: Wind Direction vs PM2.5
st.subheader('PM2.5 Concentration by Wind Direction')

wind_dir = data[['wd', 'PM2.5']]
wind_dir = wind_dir.groupby('wd').median()

# Visualisasi Wind Direction Data
st.write('PM2.5 concentration by wind direction:')
fig, ax = plt.subplots(figsize=(12, 5))
with plt.style.context('ggplot'):
    wind_dir.plot(kind='bar', legend=False, ax=ax)
    ax.set_xlabel('Wind direction', fontsize=14)
    ax.set_ylabel('PM2.5 concentration (ug/m^3)', fontsize=14)
    ax.set_title('PM2.5 concentration by wind direction', fontsize=16)
    plt.grid(axis='x')
    st.pyplot(fig)

# Bagian 5: Correlation Heatmap
st.subheader('Correlation Between Variables')

correlation_data = data[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']]

# Visualisasi Heatmap
st.write('Correlation matrix heatmap:')
fig, ax = plt.subplots(figsize=(13, 9))
sns.heatmap(correlation_data.corr(), cmap=plt.cm.Reds, annot=True, ax=ax)
plt.title('Correlation matrix of variables', fontsize=16)
st.pyplot(fig)