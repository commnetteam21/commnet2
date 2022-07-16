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
    fig, axs = plt.subplots(4,1, figsize=(20,10))
    # Observed
    axs[0].plot(result.observed)
    axs[0].set_title('Observed')
    # Trend
    axs[1].plot(result.trend)
    axs[1].set_title('Trend')
    # Residual
    axs[2].plot(result.resid)
    axs[2].set_title('Resid')
    # Seasonal
    axs[3].plot(result.seasonal)
    axs[3].set_title('Seasonal')

    plt.show()
    return fig


def get_stationarity(df)->None:    
    # rolling statistics
    rolling_mean = df.rolling(window=12).mean()
    rolling_std = df.rolling(window=12).std()
    
    # rolling statistics plot
    fig = plt.figure(figsize=(10,3.5))
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

    fig1 = plt.figure(figsize=(10,4))
    plt.plot(
        df[column_name],
        label=column_name,
        color='blue'
    )
    plt.plot(
        df[column_name+'_diff'],
        label='1st Order Difference',
        color='orange' 
    )
    plt.title(
        column_name +
        'dataset with First-order difference',     
    )
    plt.legend()
    plt.show()
    return None

def acf_plot(df)->None:
    # Autocorrection plots
    column_name = df.columns[0]
    df[column_name+'_diff'] = df[column_name].diff(periods=1)
    df = df.dropna()
    plot_acf(
        df[column_name+'_diff'],
        lags=30)
    return None

def pacf_plot(df)->None:
    column_name = df.columns[0]
    df[column_name+'_diff'] = df[column_name].diff(periods=1)
    df = df.dropna()
    plot_pacf(
        df[column_name+'_diff'],
        lags=30)

    return None


def arimamodel(TSarray, exogenous=None):
    # column_name = df.columns[0]
    autoarima_model = auto_arima(
                       TSarray,
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


def MA(rand_e_list, b_list, mean=0):
    """
    Return a time series that follows an MA model of order q=len(theta).
    Args:
        rand_e_list: list of random variables that represents residuals/erros. 
        b_list: list of the weights/coefficients of this model
        mean: time serie offset
    """
    b_list = list(b_list)
    N = len(rand_e_list) # 
    b_with_inter = np.array([1] + b_list)
    b_with_inter = b_with_inter[::-1] # invert order to have (a{N-1}, a{N-2}, a{N-q}, a0)
    q = len(b_with_inter)
    # store all the values of the time series
    X = []
    # MA time series is the sum of the q last obs of stochastic variable E multiplied by the coefficients b_i
    for i in range(N-q):
        X.append(np.dot(b_with_inter, rand_e_list[i:i+q])+mean)
     
    return np.array(X)


def get_arima_orders(df)->tuple:
    column_name = df.columns[0]
    # generate 500 set of points
    #N = 500
    b_coeff = []
    X_MA = []
    q_max = 8  
    weight = 1
    # N fixed stochastic variable as white noise series
    rand_e_list = df[column_name].values

    for q in range(1, q_max+1, 2):
        print(f'>> Create MA time series of order {q}')
        b_coeff.append(weight*np.random.random(q))
        # use a new set of growing theta 
        X_MA.append(MA(rand_e_list, b_coeff[-1]))

    for i in range(q_max//2):
        arima_model = arimamodel(X_MA[i])
        parameters = arima_model.get_params().get('order')
  
    return parameters


# split into train/test
def perform_GARCH(df, parameters):
    column_name = df.columns[0]
    
    P,O,Q = parameters
    n_test = 600  #80/20 training vs testing
    train, test = df[column_name][:-n_test], df[column_name][-n_test:]
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
    fig = plt.figure(figsize=(8,4))
    plt.plot(var[-n_test:],label='Actual')
    plt.plot(yhat.variance.values[-1, :], label='Forecasted')
    plt.title('Actual vs Forecast Variance')
    plt.legend(loc='best')
    plt.show()

    return None










