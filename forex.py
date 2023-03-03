import yfinance as yf
import pandas as pd 
import numpy as np
import math 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

forex_data = yf.download('EURUSD=X',start='2018-01-02', end='2021-9-01')
forex_data.index = pd.to_datetime(forex_data.index)

forex_data.shape


plt.figure(figsize=(16,8))


plt.title('close price history')

plt.plot(forex_data['Close'])
plt.xlabel('date')
plt.ylabel('price')
plt.show()

data = forex_data.filter(['Close'])
dateset = data.values

training_date_len = math.ceil(len(dateset)*0.8)
training_date_len

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dateset)

scaled_data

# create training dateset
train_data = scaled_data[0:training_date_len,:]
x_train=[]
y_train=[]
for i in range (60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()

x_train, y_train = np.array(x_train), np.array(y_train)
x_train.shape

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape

# build lstm model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error')

# train model
model.fit(x_train,y_train,batch_size=1,epochs=1)

test_data = scaled_data[training_date_len - 60:,:]
x_test = []
y_test=dateset[training_date_len:,:]

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test= np.array(x_test)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(predictions-y_test)**2)

train = data[:training_date_len]
valid = data[training_date_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('close price')
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
