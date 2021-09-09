#import libraries
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Initialize the flask App
app = Flask(__name__)
model_catboost_br = pickle.load(open('catboost_br.h5', 'rb'))
model_lr = pickle.load(open('lr.h5', 'rb'))

def predict_price(location, sqft, bath, bhk):

    X = pd.read_csv('X.csv')
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    a = 0.2 * model_catboost_br['catboost'].predict([x])[0] + 0.4 * model_catboost_br['br'].predict([x])[0] + 0.4 * model_lr.predict([x])[0]
    return a




# default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')



# To use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    final_features = [x for x in request.form.values()]
    # final_features = [np.array(features)]
    output = predict_price(final_features[0],float(final_features[1]),int(final_features[2]),int(final_features[3]))
    prediction_text = f'House Price: {output}\n'

    return render_template('prediction.html', prediction_text=prediction_text)



@app.route('/PREDICTION')
def PREDICTION():
    '''
    For GOING TO PREDICTION PAGE
    '''
    return render_template('prediction.html')



if __name__ == "__main__":
    app.run(debug=True)
