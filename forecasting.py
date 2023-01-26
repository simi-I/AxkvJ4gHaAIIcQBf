#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# setting the size of the figures displayed
plt.rcParams['figure.figsize'] = (20,8)
from fbprophet import Prophet
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from analysing import *


def train_test_split(df):
    df_train = df.loc['2020']
    df_test = df.loc['2021']
    
    # Number of training and testing data points
    print("The number of training data: {} ".format(len(df_train)))
    print("The number of testing data: {} ".format(len(df_test)))
    
    return df_train, df_test

def forecast_accuracy(forecast, actual, model_name):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    mae = np.mean(np.abs(forecast - actual))    # MAE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    
    accuracy = {'Model': model_name, 'MAPE': mape, 'MAE': mae, 'RMSE':rmse}
    
    return accuracy


#Moving Average Model
def arma_model(df_train, df_test):
    
    print(cl("\nMoving Average Model Training and Testing\n", attrs=['bold']))
    
    model_name = 'ARMA Model'
    history = [x for x in df_train['Price']]
    
    predictions = []

    for i in range(len(df_test)):
        model = ARIMA(history, order=(0,1,7))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = df_test['Price'][i]
        history.append(obs)
    
    forecast = pd.Series(predictions, index=df_test.index)
    
    
    #Data Plot
    plt.plot(df_train['Price'], label='Train')
    plt.plot(df_test['Price'], label='Test')
    plt.plot(forecast, label='Predicted')
    plt.xlabel("Date", size=10)
    plt.ylabel("Price", size=10)
    plt.ylim(bottom=0)
    plt.legend()
    plt.title('Train, Test and Predicted data points using ARMA', size=20)
    plt.show()
    
    model_accuracy = forecast_accuracy(forecast, df_test.Price, model_name)
    
    return forecast, model_accuracy, model_name
    
    
#ARIMA Model
def arima_model(df_train, df_test):
    
    print(cl("\nARIMA Model Training and Testing\n", attrs=['bold']))
    
    model_name = 'ARIMA Model'
   
    arima_model = pm.auto_arima(df_train['Price'], start_p=1, start_q=1, test='adf', max_p=6, max_q=6, 
                          m=1, d=None, seasonal=False, start_P=0, D=1, trace=True, error_action='ignore',  
                          suppress_warnings=True, stepwise=False)
    
    print(arima_model.aic())
    
    # model summary
    print(arima_model.summary())

    # diagnostic plots of best model
    arima_model.plot_diagnostics()
    plt.show()

    # prediction using the model
    prediction = []
    
    for i in df_test['Price']:
        predict = arima_model.predict(n_periods=1)[0]
        prediction.append(predict)
        arima_model.update(i)
    
    forecast = pd.Series(prediction, index=df_test.index)
    
    #Data Plot
    plt.plot(df_train['Price'], label='Train')
    plt.plot(df_test['Price'], label='Test')
    plt.plot(forecast, label='Predicted')
    plt.xlabel("Date", size=10)
    plt.ylabel("Price", size=10)
    plt.ylim(bottom=0)
    plt.legend()
    plt.title('Train, Test and Predicted data points using ARIMA', size=20)
    plt.show()
    
    model_accuracy = forecast_accuracy(forecast, df_test.Price, model_name)
    
    return forecast, model_accuracy, model_name
    
#Sarima
def sarima_model(df_train, df_test):
    
    print(cl("\nSARIMA Model Training and Testing\n", attrs=['bold']))
    
    model_name = 'SARIMA Model'
    sarima_model = pm.auto_arima(df_train['Price'], start_p=1, start_q=1, test='adf', max_p=6, max_q=6, 
                  m=1, d=None, seasonal=True, start_P=0, D=1, trace=True, error_action='ignore',  
                  suppress_warnings=True, stepwise=False)
    print(sarima_model.aic())
    
    # model summary
    print(sarima_model.summary())

    # diagnostic plots of best model
    sarima_model.plot_diagnostics()
    plt.show()

    # prediction using the model    
    prediction = []
    for i in df_test['Price']:
        predict = sarima_model.predict(n_periods=1)[0]
        prediction.append(predict)
        sarima_model.update(i)
    
    forecast = pd.Series(prediction, index=df_test.index)

    #Data Plot
    plt.plot(df_train['Price'], label='Train')
    plt.plot(df_test['Price'], label='Test')
    plt.plot(forecast, label='Predicted')
    plt.xlabel("Date", size=10)
    plt.ylabel("Price", size=10)
    plt.ylim(bottom=0)
    plt.legend()
    plt.title('Train, Test and Predicted data points using SARIMA', size=20)
    plt.show()
    
    model_accuracy = forecast_accuracy(forecast, df_test.Price, model_name)
    
    return forecast, model_accuracy, model_name
    
#Sarimax    
def sarimax_model(df_train, df_test):
    
    print(cl("\nSARIMAX Model Training and Testing\n", attrs=['bold']))
    
    model_name = 'SARIMAX Model'
    sarimax_model = pm.auto_arima(df_train['Price'], seasonal=True,  trace=True, error_action="ignore", 
                                 suppress_warnings=True)
    
    
    # model summary
    print(sarimax_model.summary())

    # diagnostic plots of best model
    sarimax_model.plot_diagnostics()
    plt.show()

    # prediction using the model
    prediction = []
    for i in df_test['Price']:
        predict = sarimax_model.predict(n_periods=1)[0]
        prediction.append(predict)
        sarimax_model.update(i)
    
    forecast = pd.Series(prediction, index=df_test.index)
    
    #Data Plot
    plt.plot(df_train['Price'], label='Train')
    plt.plot(df_test['Price'], label='Test')
    plt.plot(forecast, label='Predicted')
    plt.xlabel("Date", size=10)
    plt.ylabel("Price", size=10)
    plt.ylim(bottom=0)
    plt.legend()
    plt.title('Train, Test and Predicted data points using SARIMAX', size=20)
    plt.show()
    model_accuracy = forecast_accuracy(forecast, df_test.Price, model_name)
    
    return forecast, model_accuracy, model_name
    
    
def holt_winter(df, df_train, df_test):
    
    print(cl("\nHolt Winter Model Training and Testing\n", attrs=['bold']))
    
    model_name = 'Holt Winter Model'
    # Set the value of Alpha and define x as the time period
    x = 12
    alpha = 1/(2*x)
    
    # Single exponential smoothing of the visitors data set
    df['HWES1'] = SimpleExpSmoothing(df['Price']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues      
    df[['Price','HWES1']].plot(title='Holt Winters Single Exponential Smoothing graph')

    
    # Double exponential smoothing of visitors data set ( Additive and multiplicative)
    df['HWES2_ADD'] = ExponentialSmoothing(df['Price'],trend='add').fit().fittedvalues
    df['HWES2_MUL'] = ExponentialSmoothing(df['Price'],trend='mul').fit().fittedvalues
    df[['Price','HWES2_ADD','HWES2_MUL']].plot(title='Holt Winters graph: Additive Trend and Multiplicative Trend')
    plt.ylim(bottom=0)
    plt.show()
    
    # Fit the model
    fitted_model = ExponentialSmoothing(df_train['Price'],trend='mul',seasonal='mul',seasonal_periods=7).fit()
    print(fitted_model.summary())
    
    # prediction size
    n = df_test.shape[0]
    
    #Prediction
    test_predictions = fitted_model.forecast(n)

    #Data Plot
    df_train['Price'].plot(legend=True,label='TRAIN')
    df_test['Price'].plot(legend=True,label='TEST')
    test_predictions.plot(legend=True,label='PREDICTION')
    plt.ylim(bottom=0)
    plt.title('Train, Test and Predicted data points using Holt Winters Exponential Smoothing', size=20)
    plt.show()
    
    model_accuracy = forecast_accuracy(test_predictions, df_test.Price, model_name)
    
    return test_predictions, model_accuracy, model_name

def facebook_prophet(df):
    
    
    print(cl("\nFacebook Prophet Model Training and Testing\n", attrs=['bold']))
    
    
    model_name = 'Facebook Prophet Model'
    #Create dataframe for prophet
    prophet_df = df['Price']

    prophet_df = prophet_df.rename_axis('ds').reset_index()
    prophet_df = prophet_df.rename(columns={'Price':'y'})
    
    #Training and Testing split
    train = prophet_df[prophet_df.ds.astype(str).str.contains("2020")]
    test = prophet_df[prophet_df.ds.astype(str).str.contains('2021')]

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    
    fb_model = Prophet(interval_width=0.95,daily_seasonality = True).fit(train)

    forecast = fb_model.predict(test[['ds']])
    
    # Facebook prophet plot components
    fb_model.plot_components(forecast)
    plt.show()
    
    #Data Plot
    plt.plot(train['ds'], train['y'], label='Train')
    plt.plot(test['ds'], test['y'], label='Test')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted')
    plt.xlabel("Date", size=10)
    plt.ylabel("Price", size=10)
    plt.legend()
    plt.ylim(bottom=0)
    plt.title('Train, Test and Predicted data points using Facebook Prophet', size=20)
    plt.show()
    # Facebook Prophet accuracy metric
    model_accuracy = forecast_accuracy(forecast.yhat, test.y, model_name)
    
    return forecast.yhat, model_accuracy, model_name
    
    
def get_bollinger_band(prices, rate=20):
    print(cl("\n\nBollinger Strategy\n", attrs=['bold']))
    
    sma = get_sma(prices, rate)
    std = get_sma_std(prices, rate)
    bollinger_up = sma + std * 2 # Calculate top band
    bollinger_down = sma - std * 2
    
    plt.plot(prices, label='Closing Prices')
    plt.plot(bollinger_up, label='Bollinger Up', c='g')
    plt.plot(bollinger_down, label='Bollinger Down', c='r')
    plt.ylim(bottom=0)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Closing Prices')
    plt.title(' Bollinger Bands', size=20)
    plt.legend(loc='best')
    plt.show()
    
    return bollinger_down,  bollinger_up


def implement_bb_strategy(prices, bollinger_down,  bollinger_up):
    
    buy_signal = [] #buy list
    sell_signal = [] #sell list
    hold_signal = [] # hold list
    bb_signal = []
    
    for i in range(len(prices)):
        if prices[i] > bollinger_up[i]: #Then you should sell 
            buy_signal.append(np.nan)
            sell_signal.append(prices[i])
            hold_signal.append(np.nan)
            bb_signal.append(1)
        elif prices[i] < bollinger_down[i]: #Then you should buy
            sell_signal.append(np.nan)
            buy_signal.append(prices[i])
            hold_signal.append(np.nan)
            bb_signal.append(-1)
        else:
            buy_signal.append(np.nan)
            sell_signal.append(np.nan)
            hold_signal.append(prices[i]) #Then you should hold
            bb_signal.append(0)
            
    plt.plot(prices, label='Predicted Prices')
    plt.plot(bollinger_up, label='Bollinger Up', c='b')
    plt.plot(bollinger_down, label='Bollinger Down', c='orange')
    plt.scatter(prices.index, buy_signal, marker = '^', color = 'green', label = 'Buy', s = 200)
    plt.scatter(prices.index, np.absolute(sell_signal), marker = 'v', color = 'red', label = 'Sell', s = 200)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Predicted Prices')
    plt.ylim(bottom=0)
    plt.title('Bollinger Bands Strategy', size=20)
    plt.legend(loc='best')
    plt.show()
            
    return buy_signal, sell_signal, bb_signal
    
    
    
def price_position(df,bb_signal):
    position = []
    for i in range(len(bb_signal)):
        if bb_signal[i] > 1:
            position.append(0)
        else:
            position.append(1)

    for i in range(len(df)):
        if bb_signal[i] == 1:
            position[i] = 1
        elif bb_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]
    return position
    

    
def percentage_backtest(prediction, position):
    
    df_ret = pd.DataFrame(np.diff(prediction)).rename(columns = {0:'returns'})
    bb_strategy_ret = []

    for i in range(len(df_ret)):
        try:
            returns = df_ret['returns'][i]*position['bb_position'][i]
            bb_strategy_ret.append(returns)
        except:
            pass

    bb_strategy_ret_df = pd.DataFrame(bb_strategy_ret).rename(columns = {0:'bb_returns'})

    investment_value = 100000
    number_of_stocks = math.floor(investment_value/prediction[-1])
    bb_investment_ret = []

    for i in range(len(bb_strategy_ret_df['bb_returns'])):
        returns = number_of_stocks * bb_strategy_ret_df['bb_returns'][i]
        bb_investment_ret.append(returns)

    bb_investment_ret_df = pd.DataFrame(bb_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(bb_investment_ret_df['investment_returns']), 2)
    profit_percentage = math.floor((total_investment_ret/investment_value)*100)
    print(cl('Profit gained from the BB strategy by investing $100k : {}'.format(total_investment_ret), attrs = ['bold']))
    print(cl('Profit percentage of the BB strategy : {}%'.format(profit_percentage), attrs = ['bold']))
       
    
def forecasting_report(df):
    
    model_acc_list = []
        
    df_train, df_test = train_test_split(df)
    
    arma_forecast, arma_model_acc, arma_model_name = arma_model(df_train, df_test)
    model_acc_list.append(arma_model_acc)
    
    arima_forecast, arima_model_acc, arima_model_name = arima_model(df_train, df_test)
    model_acc_list.append(arima_model_acc)
    
    sarima_forecast, sarima_model_acc, sarima_model_name = sarima_model(df_train, df_test)
    model_acc_list.append(sarima_model_acc)
    
    sarimax_forecast, sarimax_model_acc, sarimax_model_name = sarimax_model(df_train, df_test)
    model_acc_list.append(sarimax_model_acc)
    
    hw_forecast, hw_model_acc, hw_model_name = holt_winter(df, df_train, df_test)
    model_acc_list.append(hw_model_acc)
    
    fbp_forecast, fbp_model_acc, fbp_model_name = facebook_prophet(df)
    model_acc_list.append(fbp_model_acc)
    
    model_acc_df = pd.DataFrame(model_acc_list)
    
    print('\nForecasting Accuracy')
    print(model_acc_df)
    
    
    bollinger_down,  bollinger_up = get_bollinger_band(df_test['Price'])
    
    buy_price, sell_price, bb_signal = implement_bb_strategy(sarimax_forecast, bollinger_down,  bollinger_up)
    
    position = price_position(sarimax_forecast, bb_signal)
    
    position = pd.DataFrame(position).rename(columns = {0:'bb_position'}).set_index(df_test.index)
    
    percentage_backtest(sarimax_forecast, position)
    
    
    
    
    
    
    
    
    
    
    
    