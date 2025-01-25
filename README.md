# Indian Diabetes Prediction Project

## Overview
This project is a web-based application for predicting the likelihood of diabetes in individuals using the Indian Diabetes Prediction Dataset. It leverages a machine learning model and is built with Flask for the backend. The application allows users to input health-related parameters and receive predictions on diabetes risk.

## Features
- **User-friendly interface** for inputting health parameters.
- **Real-time predictions** of diabetes likelihood using a trained machine learning model.
- **Model performance metrics** available for evaluation.

## Dataset
The Indian Diabetes Prediction Dataset is used in this project. It contains health-related parameters like glucose levels, BMI, and age, which are key factors in predicting diabetes.

## Project Structure
```
├── .git
├── .github
├── artifacts
│   └── model_evaluation
│       └── metrics.json         # Contains classification report of the model
├── config
├── log
├── research
├── src
├── templates                   # HTML templates for the Flask app
├── app.py                      # Main Flask application
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Dockerfile                  # For containerizing the application
├── LICENSE
├── main.py                     # Entry point for training or running the model
├── params.yaml                 # Parameters for the machine learning pipeline
├── README.md                   # Documentation for the project
├── requirements.txt            # Dependencies required for the project
├── schema.yaml                 # Schema for validating input data
├── SECURITY.md
├── setup.py
├── template.py
└── WineQualityPrediction.egg-info
```

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/indian-diabetes-prediction.git
   cd indian-diabetes-prediction
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
The classification report for the trained model can be found in `artifacts/model_evaluation/metrics.json`. It includes metrics such as:
- Accuracy
- Precision
- Recall
- F1-score

## Contributing
We welcome contributions! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License
This project is licensed under the [LICENSE](LICENSE) file. Please check the terms before using this project.

## Security
For any security concerns, please refer to the [SECURITY.md](SECURITY.md) file.

## Acknowledgements
Special thanks to the contributors and open-source libraries used in this project.

## Contact
For any questions or feedback, please open an issue on the GitHub repository.

