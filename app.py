from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from src.WineQualityPrediction.pipeline.PredictionPipeline import PredictionPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # HTML form page

@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successful!"

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Retrieve form data
        form_data = {
            "fixed_acidity": request.form.get('fixed_acidity'),
            "volatile_acidity": request.form.get('volatile_acidity'),
            "citric_acid": request.form.get('citric_acid'),
            "residual_sugar": request.form.get('residual_sugar'),
            "chlorides": request.form.get('chlorides'),
            "free_sulfur_dioxide": request.form.get('free_sulfur_dioxide'),
            "total_sulfur_dioxide": request.form.get('total_sulfur_dioxide'),
            "density": request.form.get('density'),
            "pH": request.form.get('ph'),
            "sulphates": request.form.get('sulphates'),
            "alcohol": request.form.get('alcohol'),
        }

        # Convert form data to a NumPy array
        data = np.array(list(form_data.values()), dtype=float).reshape(1, -1)

        # Predict the wine quality
        obj = PredictionPipeline()
        predict = obj.predict(data)

        # Render the result page
        return render_template('result.html', wine_quality=predict[0])

    except Exception as e:
        print(f"Error: {e}")
        return render_template('result.html', wine_quality="Error in prediction.")

if __name__ == '__main__':
    app.run(debug=True)
