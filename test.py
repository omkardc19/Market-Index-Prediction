import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.models import load_model


def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv('sample_input.csv')
     
    actual_close = np.loadtxt('sample_close.txt')
    
    pred_close = predict_func(df)

    for i in pred_close:
        print(i)
    
    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close-pred_close))


    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close
    
    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr)-np.array(pred_prev)
    actual_dir = np.array(actual_curr)-np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100

    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')
    

def predict_func(data):
    """
    Modify this function to predict closing prices for next 2 samples.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you 
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for closing price of next 2 samples
    """
    
    model = load_model('MODEL_1.h5')

    data = data.interpolate(method='spline', order=3)
    data.fillna(method='ffill', inplace=True) # Fill missing values forward
    data.fillna(method='bfill', inplace=True) # Fill missing values backward

    # data.fillna(method='ffill', inplace=True)
    data1=data.reset_index()['Close']

    from sklearn.preprocessing import MinMaxScaler

    scaler=MinMaxScaler(feature_range=(0,1))
    data1=scaler.fit_transform(np.array(data1).reshape(-1,1))

    x_input=data1.reshape(1,-1)
    x_input.shape
    
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    # demonstrate prediction for next 10 days
    from numpy import array

    lst_output=[]
    n_steps=50
    i=0
    while(i<2):
    
        if(len(temp_input)>50):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            # print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            # print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    

    # print(lst_output)

    lst_output_2=scaler.inverse_transform(lst_output).tolist()
    # for i in lst_output_2:
    #     print(i)

    return lst_output_2

    # return [7214.200195, 7548.899902]
    

if __name__== "__main__":
    evaluate()