

''' LSTM neural network '''

''' LSTM is a neural network technique to predict sequences.

LSTMs are widely used for sequence prediction problems and have proven to be extremely effective. The reason they work so well is because LSTM is able to store past information that is important, and forget the information that is not.

It has three main gates:

Input gate: The input gate adds information to the cell state
Forget gate: It removes the information that is no longer required by the model
Output gate: Output Gate at LSTM selects the information to be shown as output '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fastai
from fastai.tabular import  add_datepart
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import influxdb
from pandas import DataFrame
import numpy


warnings.filterwarnings('ignore')

host = 'localhost'
port = 8086
user = ''
password = ''
dbname = 'stock'

''' DATA PREPROCESSING '''

sns.set_style('whitegrid')

# raw_data = pd.read_csv("MSFT.csv")
# data = raw_data.copy()

client = influxdb.InfluxDBClient(host, port, user, password, dbname)
q = 'SELECT open,high,low,close,volume FROM "stockforecast";'
data = DataFrame(client.query(q).get_points())

'''Taking only 6 years data'''
data['time'] = pd.to_datetime(data.time,format='%Y-%m-%d')
data.index = data['time']
#sorting
data = data.sort_index(ascending=True, axis=0)
data = data['2012-01-01' :]

# Viusalizing the target variable (CLOSE)
plt.figure(figsize = (15,6))
plt.plot(data.close)
plt.show()

'''For our problem statement, we do not have a set of independent variables. We have only the dates instead. Let's use 
the date column to extract features like â€“ day, month, year, day of the week etc.'''

new_data = pd.DataFrame(data = data , columns=['time','close'])

# Feature creation based on dates using fastai
add_datepart(new_data,'time')
new_data.drop('timeElapsed', axis=1, inplace=True)  #elapsed will be the time stamp

# Encoding the main categorical variables: â€˜Is_month_endâ€™, â€˜Is_month_startâ€™, â€˜Is_quarter_endâ€™, â€˜Is_quarter_startâ€™, â€˜Is_year_endâ€™, and â€˜Is_year_startâ€™
cols = ['timeIs_month_end', 'timeIs_month_start', 'timeIs_quarter_end', 'timeIs_quarter_start', 'timeIs_year_end', 'timeIs_year_start']
encoder = LabelEncoder()
for x in cols:
  new_data[x] = encoder.fit_transform(new_data[x].values)

# Moving averages
new_data['Monthly_moving_average'] = new_data['close'].rolling(window=30).mean()# window = 30  means monthly average
# Replacing the NaN values with 0
new_data['Monthly_moving_average'] = new_data['Monthly_moving_average'].fillna(0)

# Expanding moving average
new_data['Expanding_mean'] = new_data['close'].expanding(2).mean()
#Filling NaN value
new_data['Expanding_mean'] = new_data['Expanding_mean'].fillna(0)

''' Splitting the dataset '''
#Training set
train = new_data[ : '2017-01-01']
#Validation set
val = new_data['2017-01-01' : ]

'''
LSTM model
'''

# scaling the data
data_close = new_data['close'].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_close.reshape(-1,1))

"""
Will take past 90 days of data to predict the 91th day stock price.
"""

x_train, y_train = [],[]
for i in range(90,len(train)):
    x_train.append(scaled_data[i-90:i])
    y_train.append(scaled_data[i])
x_train, y_train = np.array(x_train), np.array(y_train) # Converting to numpy array
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1)) # Reshaping

model = Sequential()

#1st Layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))

#2nd layer
model.add(LSTM(units=50))
model.add(Dropout(0.3))

#Output layer
model.add(Dense(1))

#Model compilation
adam = Adam(lr = 0.01, beta_1 = 0.99, epsilon = 1e-7)
model.compile(loss = 'mean_squared_error', optimizer = adam)

early_stopping = EarlyStopping(monitor = 'loss', patience = 2)
#Training
model.fit(x_train, y_train, epochs = 10, batch_size=1, callbacks =[early_stopping], verbose=1)

# Creating a seperate dataframe
inputs = new_data['close'][len(new_data) - len(val) - 90:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

#Generating validation data
x_val = []
for i in range(90,inputs.shape[0]):
    x_val.append(inputs[i-90:i])

x_val = np.array(x_val)

x_val = np.reshape(x_val, (x_val.shape[0],x_val.shape[1],1))

#Predicting validation data
predictions = model.predict(x_val)
predictions = scaler.inverse_transform(predictions)

# Calculating the root-mean squared error
lstm_forecast = pd.DataFrame(predictions, index = val.index, columns = ['Prediction'])
rmse = np.sqrt(mean_squared_error(val['close'], lstm_forecast['Prediction']))
print(rmse)


plt.figure(figsize = (15,7))
plt.plot(train['close'])
plt.plot(val['close'])
plt.plot(lstm_forecast['Prediction'])
plt.show()

