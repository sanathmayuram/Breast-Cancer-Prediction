from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

logistic_model = pickle.load(open('logistic.pkl', 'rb'))
standed_scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/', methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        ID = float(request.form.get('ID'))
        radius_mean = float(request.form.get('radius_mean'))
        texture_mean = float(request.form.get('texture_mean'))
        smoothness_mean = float(request.form.get('smoothness_mean'))
        compactness_mean = float(request.form.get('compactness_mean'))
        symmetry_mean = float(request.form.get('symmetry_mean'))
        fractal_dimension_mean = float(request.form.get('fractal_dimension_mean'))
        radius_se = float(request.form.get('radius_se'))
        texture_se = float(request.form.get('texture_se'))
        smoothness_se = float(request.form.get('smoothness_se'))
        compactness_se = float(request.form.get('compactness_se'))
        concave_points_se = float(request.form.get('concave_points_se'))
        symmetry_se = float(request.form.get('symmetry_se'))
        fractal_dimension_se = float(request.form.get('fractal_dimension_se'))
        smoothness_worst = float(request.form.get('smoothness_worst'))
        symmetry_worst = float(request.form.get('symmetry_worst'))
        fractal_dimension_worst = float(request.form.get('fractal_dimension_worst'))

        new_data_scaled =  standed_scaler.transform([[ID, radius_mean, texture_mean, smoothness_mean,
       compactness_mean, symmetry_mean, fractal_dimension_mean,
       radius_se, texture_se, smoothness_se, compactness_se,
       concave_points_se, symmetry_se, fractal_dimension_se,
       smoothness_worst, symmetry_worst, fractal_dimension_worst]])
        
        result = 'Benign' if logistic_model.predict(new_data_scaled) == [0] else 'Malignant'

        return render_template("home.html",results=result)
    else:
        return render_template("home.html")
    

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)