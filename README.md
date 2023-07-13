Name: Omkar Chavan
Roll No.: 210280

# Market Index Prediction using LSTM Model

-This project focuses on predicting the closing prices of a stock based on the last 50 days of data using a Long Short-Term Memory (LSTM) model. The LSTM model is a type of recurrent neural network (RNN) that is well-suited for sequential data analysis, making it an excellent choice for time series forecasting tasks such as stock price prediction.

##Libraries required to run the code:
1. Pandas
2. Numpy
3. scikit-learn
4. tensorflow
5. keras


for this run code in terminal (to download in system, if not present already)
Commands for libraries to be installed if code throws an error:

pip install pandas
pip install numpy
pip install scikit-learn
pip install tensorflow
pip install keras

## Instructions
Python version: 3.11.1
1. Input Data:
   - Make sure that you have the 'sample_input.csv' file in same directory, which contains the historical data of the stock index of last 50 days to import in evaluate code().
   - Make sure the 'sample_close.txt' file is available in same directory, containing the actual closing prices for evaluation in evaluate() code.

2. Pre-Trained Model:
   - The code in .py file assumes that the pre-trained model of .h5 format named 'MODEL.h5' is present in same directory as the code.
   - Ensure that the model file is correctly named and saved in the same directory as the code.

3. Running the Code:
   - Entire directory should be opened in a code editor(like VS code) using using command like "Open with code", as the code has imported the input files like        'sample_input.csv','sample_close.text' using their names and not path. 
   - Execute the 'evaluate()' function in the code to perform the stock index prediction.
   - The code will load the input data, preprocess it, and make predictions for the next 2 days' closing prices.
   - The mean square error and directional accuracy will be calculated and displayed.

Note: It is important to keep the file names & locations, dependencies intact for the code to run without errors.


