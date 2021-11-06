from flask import Flask, render_template, request, jsonify, session
import sklearn
import pickle 
import pandas as pd
import numpy as np 


#session.clear()
app = Flask(__name__)


pipe = pickle.load(open('models/insurance_lr_pipe.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
            age = int(request.form.get('age'))
            sex = str(request.form.get('sex'))
            bmi = float(request.form.get('bmi'))
            children = int(request.form.get('children'))
            smoker = str(request.form.get('smoker'))
            region = str(request.form.get('region'))

            df = pd.DataFrame([[age, sex, bmi, children, smoker, region]])

            df.columns = [ 
                'age',
                'sex',
                'bmi',
                'children',
                'smoker',
                'region'
            ]

            prediction = abs(round(pipe.predict(df)[0], 2))
            return render_template('output.html', prediction_text='Based on the informations provided your Annual Medical Charge is $ {}'.format(prediction))




if __name__ == "__main__":
    app.run(debug=True)
