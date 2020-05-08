# 1. Train our Model
# 2. Create web app using Flask
# 3. Commit the code in Github
# 4. Create the account in Heroku (PaaS)
# 5. Link the github to heroku
# 6. deploy the model
# 7. web app is ready

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

model= pickle.load(open('model.pkl', 'rb'))
tfidf= pickle.load(open('transform.pickle', 'rb'))
app= Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method== 'POST':
        message= request.form['message']
        data= [message]
        vect= tfidf.transform(data).toarray()
        my_prediction= model.predict(vect)
    return render_template('result.html', prediction= my_prediction)

if __name__ == '__main__':
    app.run(debug= True)
