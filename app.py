# pip install yfinance pycaret

import yfinance as yf
import pandas as pd
from pycaret.regression import *
from flask import Flask, render_template, request

'''stock_data = yf.download('AAPL')

stock_data = pd.DataFrame(stock_data)

print(stock_data.columns)'''

def download_stock_data(stock_name):
    stock_data = yf.download(stock_name)
    #print(stock_data)
    return stock_data

def preprocessing_stock_data(stock_data):
    #pd.DataFrame(stock_data)
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.reset_index(inplace=True)
    pd.DataFrame(stock_data)
    #print(stock_data.columns)
    return stock_data

def train_best_model(stock_data):
    reg = setup(data=stock_data, target='Close', 
                train_size=0.7, session_id=123)
    best_model = compare_models(sort='R2')
    return best_model

def predict_stock_price(stock_data, best_model):
    predictions = predict_model(best_model, data=stock_data)
    return predictions

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_name = request.form['stock_name']
    
    stock_data = download_stock_data(stock_name)
    
    stock_data = preprocessing_stock_data(stock_data)
    
    best_model = train_best_model(stock_data)
    
    predicts = predict_stock_price(stock_data, best_model)
    
    predicted_prices = predicts.tail(7)['Close']
    
    return render_template('index.html', 
                           stock_name=stock_name,
                           predicted_prices=predicted_prices)

if __name__ == '__main__':
    app.run(debug=True)