import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from arch import arch_model
from arch.__future__ import reindexing
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler
from pmdarima.arima import auto_arima
from random import gauss
from random import seed


def check_seasonality(df)->None:
    #df.index = pd.to_datetime(df.index)

    n_periods = 24
    result = seasonal_decompose(
             df,
             model='multiplicative',
             period=n_periods
    )
    fig, axs = plt.subplots(2,2, figsize=(20,10))
    # Observed
    axs[0,0].plot(result.observed)
    axs[0,0].set_title('Observed')
    # Trend
    axs[0,1].plot(result.trend)
    axs[0,1].set_title('Trend')
    # Residual
    axs[1,0].plot(result.resid)
    axs[1,0].set_title('Resid')
    # Seasonal
    axs[1,1].plot(result.seasonal)
    axs[1,1].set_title('Seasonal')

    plt.show()
    return None


def get_stationarity(df)->None:    
    # rolling statistics
    rolling_mean = df.rolling(window=12).mean()
    rolling_std = df.rolling(window=12).std()
    
    # rolling statistics plot
    fig = plt.figure(figsize=(15,7))
    # Original
    original = plt.plot(
                        df, 
                        color='blue',
                        label='Original'
    )
    mean = plt.plot(
                    rolling_mean,
                    color='red',
                    label='Rolling Mean'
    )
    std = plt.plot(
                   rolling_std,
                   color='black',
                   label='Rolling Std'
    )
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Dickey–Fuller test:
    result = adfuller(df)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
    return None

def FirstOrderDiff(df)->None:
    column_name = df.columns[0]
    df[column_name+'_diff'] = df[column_name].diff(periods=1)
    df = df.dropna()

    fig1 = plt.figure(figsize=(6.3,4))
    plt.plot(
        df[column_name],
        label=column_name
    )
    plt.title(
        column_name +
        'dataset with First-order difference',     
    )
    plt.legend()
    plt.show()

    # Autocorrection plots
    #1 acf
    plot_acf(
        df[column_name+'_diff'],
        lags=30)

    #2 pacf 
    plot_pacf(
        df[column_name+'_diff'],
        lags=30)

    return None


def arimamodel(df, exogenous=None):
    column_name = df.columns[0]
    autoarima_model = auto_arima(
                       df[column_name],
                       start_p=1, # auto-regressive (AR)
                       start_q=2, # moving average
                       test="adf", # ADF Augmented Dickey-Fuller test.
                       trace=True,
                       seasonal=True,
                       d= None,
                       max_d=4,
                       max_p=4,
                       exogenous=exogenous,
                       random_state=42
    )
    return autoarima_model

def get_arima_orders(df)->tuple:
    max_q = 8
    arima_model = arimamodel(df)
    parameters = arima_model.get_params().get('order')
  
    return parameters 



# split into train/test
def perform_GARCH(df, parameters):
    P,O,Q = parameters
    n_test = 600  #80/20 training vs testing
    train, test = df[:-n_test], df[-n_test:]
    # define model
    model = arch_model(
                train,
                mean='Zero',
                vol='GARCH',
                p=P,
                o=O,
                q=Q
    ) 
    # show this one
    model_fit = model.fit()
    yhat = model_fit.forecast(horizon=n_test)

    N = len(df)
    var = [i*0.01 for i in range(0,N)]
    fig = plt.figure(figsize=(10,5))
    actual = plt.plot(var[-n_test:],label='Actual')
    focasted = plt.plot(yhat.variance.values[-1, :], label='Forecasted')
    plt.title('Actual vs Forecast Variance')
    plt.legend(loc='best')
    plt.show()

    return None



def perform_arima_garch(df):
    check_seasonality(df)
    get_stationarity(df)
    FirstOrderDiff(df)
    arimamodel(df, exogenous=None)
    params = get_arima_orders(df)
    if params[0]==0 and params[1]==0:
        print('Errors in time series data is uncorrelated which means the errors are random')
    else:
        perform_GARCH(params)




