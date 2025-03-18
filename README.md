# Wine Quality Prediction Project

## Overview
This project is a web-based application for predicting the likelihood of diabetes in individuals using the Indian Diabetes Prediction Dataset. It leverages a machine learning model and is built with Flask for the backend. The application allows users to input health-related parameters and receive predictions on diabetes risk.

## Features
- **User-friendly interface** for inputting health parameters.
- **Real-time predictions** of diabetes likelihood using a trained machine learning model.
- **Model performance metrics** available for evaluation.

## Dataset
The Indian Diabetes Prediction Dataset is used in this project. It contains health-related parameters like glucose levels, BMI, and age, which are key factors in predicting diabetes.

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nasir015/WineQualityPrediction.git
   cd WineQualityPrediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```bash
   python app.py
   ```

5. **Access the application:**
   Open your browser and navigate to `http://127.0.0.1:5000`.

## Usage
1. Enter health-related parameters in the web application.
2. Click the "Predict" button to get a prediction.
3. View the prediction result and confidence level.

## Model Performance
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       163
           1       0.95      1.00      0.97       126
           2       0.93      0.80      0.86        46
           3       0.82      0.75      0.78        24
           4       0.99      0.96      0.98       129
           5       0.96      1.00      0.98       120

    accuracy                           0.97       608
    macro avg      0.94      0.92      0.93       608
    weighted avg   0.97      0.97      0.97       608


## Contributing
We welcome contributions! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License
This project is licensed under the [LICENSE](LICENSE) file. Please check the terms before using this project.

## Security
For any security concerns, please refer to the [SECURITY.md](SECURITY.md) file.

## Acknowledgements
Special thanks to the contributors and open-source libraries used in this project.

## Contact
Email: nasir.uddin.6314@gmail.com
WhatsApp: +8801793502127

