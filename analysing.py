
#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pylab
from termcolor import colored as cl
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import math
# setting the size of the figures displayed
plt.rcParams['figure.figsize'] = (20,8)

#simple moving average mean
def get_sma(prices, rate):
    return prices.rolling(rate).mean()

#simple moving average standard deviation
def get_sma_std(prices, rate):
    return prices.rolling(rate).std()

#Data Analysis
def data_analysis(df):
    print(f'\nThe Head of the data set: \n{df.head()}\n')
    print(f'Data Types: \n{df.dtypes}\n')
    print(f'Shape of Dataset: \n{df.shape}\n')
    print(f'Dataset Info: \n{df.info()}\n')
    print(f'Null count: \n{df.isnull().sum()}\n')
    df.drop(df.index[-1], inplace=True)
    print(f'Null Count: \n{df.isnull().sum()}\n')
    print(f'Duplicated Sum: \n{df.duplicated().sum()}\n')
    
    return df
    
def data_preprocessing(df):
    k_data = df[df['Vol.'].astype(str).str.contains('K')]
    df = df[df["Vol."].str.contains("K") == False]
    k_data['Vol.'] = k_data['Vol.'].str.replace('K', '')
    k_data['Vol.'] = k_data['Vol.'].apply(pd.to_numeric)
    k_data['Vol.'] = [(i/1000) for i in k_data['Vol.']]
    df = pd.concat([df, k_data], join="inner")
    
    df.rename(columns={"Vol.":"Volume"}, inplace=True)
    
    df['Volume'] = df['Volume'].str.replace("M", '')
    df['Volume'] = df['Volume'].str.replace('-','0')
    df = df.astype(float)
    
    print(f'Dataset Describe: \n{df.describe()}\n')
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")

    df.sort_index(ascending=True, inplace=True)
    print(f'The Head of the data set: \n{df.head()}\n')
    
    print(f'Start Date of TimeSeries \n{df.index.min()}\n')
    print(f'End Date of TimeSeries: \n{df.index.max()}\n')
    print(f'Days Inbetween: \n{df.index.max() - df.index.min()}\n')
    
    # setting the time series frequency to business days
    df = df.asfreq("b")
    # forward fill missing values
    df = df.fillna(method='ffill')
    
    return df
    
    
    
def graph_plot(df):
    # Line plot of the Price, High and Low variables
    df[['Open','Price','High','Low']].plot()
    plt.title('Price, High and Low Over Time', size=22)
    plt.show()
    
    
    # Line plot of the price variable
    plt.plot(df.index, df['Price'])
    plt.xlabel("Date", size=10)
    plt.ylabel("Price", size=10)
    plt.title("Price Over Time", size=24)
    plt.show()
          
    
    # Line plot of the price and change % variables
    plt.plot(df.index, df['Price'], label='Price')
    plt.plot(df.index, df['Change %']*100, label='Change %')
    plt.xlabel("Date", size=10)
    plt.ylabel("Price", size=10)
    plt.title("Price & Change % Over Time", size=22)
    plt.legend()
    plt.show()
    
    
    # QQ Plot for the price
    stats.probplot(df['Price'], plot=pylab)
    plt.title("Price QQ Plot", size=24)
    plt.show()
    
    # Box Plot of Price
    sns.boxplot(x = df['Price'])
    plt.title('Price BoxPlot', size=24)
    plt.show()
    
          
    sm = get_sma(df['Price'], 7) # Get 3 day SMA
    sm_std = get_sma_std(df['Price'], 7)
    
    # Plot SMA
    plt.plot(df['Price'], label='Original')
    plt.plot(sm, color='red', label='Rolling mean')
    plt.plot(sm_std, color='green', label='Rolling std')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.title('Rolling Mean & Standard Deviation', size=22)
    plt.legend(loc='best')
    plt.show()
    
def series_decomposition(df):
    # Applying seasonal decomposition
    print('\nTimeseries Decomposition\n')
    df_decomposed = seasonal_decompose(df, model='additive')
    # Plotting the trend, seasonal and residual
    df_decomposed.plot()
    plt.show()
    

# This function makes the time series stationary.
def stationary(df):
    # ADF Test to fins the p-value
    print(cl("\nStationarity Test\n", attrs=['bold']))
    result = adfuller(df.Price.values, autolag='AIC')
    if result[1] > 0.05:
        print('ADF for the original price values')
        dfoutput = pd.Series(
            result[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "Lags Used",
                "Number of Observations Used",
            ],
        )
        for key, value in result[4].items():
            dfoutput["Critical Value (%s)" % key] = value
        print(dfoutput)
        
        result = adfuller(np.diff(df.Price.values), autolag='AIC')
       
    if result[1] < 0.05:
        print('\nADF for the price values after difference:')
        dfoutput = pd.Series(
            result[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "Lags Used",
                "Number of Observations Used",
            ],
        )
        for key, value in result[4].items():
            dfoutput["Critical Value (%s)" % key] = value
        print(dfoutput)
        difference = df.Price.diff()
        df['difference'] = difference
    else:
        print('Your time series is not stationary, you may need to make another difference')
        
    print(f'\nThe Head of the data set: \n{df.head()}\n')
    return df

def stationary_plot(df):
    # Line plot of the stationary variable
    plt.plot(df.index, df['difference'])
    plt.xlabel("Date", size=10)
    plt.ylabel("Price", size=10)
    plt.title("Series Stationary Plot", size=22)
    plt.show()
    
def acf_pacf_plot(df):
    df.dropna(inplace=True)
    #Autocorrelation plot
    fig, ax = plt.subplots(2, 1)
    plot_acf(df['difference'], lags=30, ax=ax[0], auto_ylims=True)
    plot_pacf(df['difference'], lags=30, method='ywm', ax=ax[1], auto_ylims=True)
    plt.show()
    return df

def analysing_report(df):
    
    df = data_analysis(df)
    df = data_preprocessing(df)
    graph_plot(df)
    series_decomposition(df['Price'])
    df = stationary(df)
    stationary_plot(df)
    df = acf_pacf_plot(df)
    
    return df
    









    