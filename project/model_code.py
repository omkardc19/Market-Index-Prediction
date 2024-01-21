# -*- coding: utf-8 -*-
"""modeltrain.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X57f5RycCmgOWSv_sN4iDCCJF2o6lLQa

### Market index Prediction
"""

### Keras and Tensorflow >2.0

import pandas as pd

df=pd.read_csv('STOCK_INDEX.csv')

df.head()

df.tail()

df1=df.reset_index()['Close']

df1

df1= df1.dropna()
df1.info()

import matplotlib.pyplot as plt
plt.plot(df1)

### LSTM are sensitive to the scale of the data. so we apply MinMax scaler

import numpy as np

df1

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

print(df1)

##splitting dataset into train and test split
training_size=int(len(df1)*0.80)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

training_size,test_size

train_data

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 50
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

print(X_train.shape), print(y_train.shape)

print(X_test.shape), print(ytest.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

y_train =y_train.reshape(y_train.shape[0] , 1)
ytest = ytest.reshape(ytest.shape[0], 1)

### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(50,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=32,verbose=1)

model.save('MODEL.h5')

import tensorflow as tf

tf.__version__

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)



##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

##Transformback to original form
y_train=scaler.inverse_transform(y_train)
ytest=scaler.inverse_transform(ytest)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))

### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))

### Plotting
# shift train predictions for plotting
look_back=50
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# from here begins for next n days

data=pd.read_csv('sample_input.csv')

data.info()
data.fillna(method='ffill', inplace=True)

data1=data.reset_index()['Close']
data1.info()

data1.info()

data1 = data1.interpolate(method='spline', order=3)
data1.fillna(method='ffill', inplace=True) # Fill missing values forward
data1.fillna(method='bfill', inplace=True) # Fill missing values backward
# data1=data1.ffill()

import matplotlib.pyplot as plt
plt.plot(data1)

len(data1)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data1=scaler.fit_transform(np.array(data1).reshape(-1,1))

# len(test dta)-50 in next line instead of 341

x_input=data1.reshape(1,-1)
x_input.shape

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

temp_input

# demonstrate prediction for next 2 days
from numpy import array

lst_output=[]
n_steps=50
i=0
while(i<2):

    if(len(temp_input)>50):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1


print(lst_output)

lst_output_2=scaler.inverse_transform(lst_output).tolist()
lst_output_2

day_new=np.arange(1,51)
day_pred=np.arange(51,53)

import matplotlib.pyplot as plt

len(data1)



plt.plot(day_new,scaler.inverse_transform(data1))
plt.plot(day_pred,scaler.inverse_transform(lst_output))

data3=data1.tolist()
data3.extend(lst_output)
plt.plot(data3)

data3=scaler.inverse_transform(data3).tolist()

plt.plot(data3)
