from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)
app = Flask(__name__)

# Load the pre-trained XGBoost model
model = pickle.load(open("model.pkl", "rb"))

# Load the ColumnTransformer
ct = joblib.load("column")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    # Get user inputs from the form
    inputs = [float(request.form.get('longitude')),
              float(request.form.get('lat')),
              float(request.form.get('cloud_cover')),
              float(request.form.get('dirunal_temperature_range')),
              float(request.form.get('frost_day_frequency')),
              float(request.form.get('evapotranspiration')),
              float(request.form.get('precipitation')),
              float(request.form.get('tmn')),
              float(request.form.get('tmp')),
              float(request.form.get('tmx')),
              float(request.form.get('vap')),
              float(request.form.get('wet_day')),
              float(request.form.get('elevation')),
              float(request.form.get('dominant_land_cover')),
              request.form.get('country'),
              request.form.get('region')]
    # Convert categorical inputs using the pre-trained ColumnTransformer
    ct = ColumnTransformer([("ohe", OneHotEncoder(handle_unknown="ignore"), [14, 15])], remainder="passthrough")
    inputs = ct.fit_transform([inputs])
    inputs = sc.fit_transform(inputs)
    # Make the prediction
    prediction = model.predict(inputs)

    # Display the prediction result
    if prediction[0]==0:
        return render_template("negative.html")
    else:
        return render_template("positive.html")
    #return f"Prediction: {prediction[0]}"


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
