import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
from src.WineQualityPrediction.entity.config_entity import ModelTrainerConfig
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import optuna
from xgboost import XGBClassifier
from src.WineQualityPrediction.utils.common import save_model
log_path = 'log\log_file.log'

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer class with a given configuration.

        Parameters:
            config (ModelTrainerConfig): An object containing the configuration 
                                        parameters for training, including paths 
                                        to datasets, model hyperparameters, and 
                                        save paths.
        """
        self.config = config
        self.X = pd.read_csv(self.config.X_data_path)
        self.y = pd.read_csv(self.config.y_data_path)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.3, random_state=42)
        self.y_train = self.y_train.squeeze()
        self.y_test = self.y_test.squeeze()
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
        # Standardize the data
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)




    def objective(self,trial):
        logger(log_path, logging.INFO, "Optuna optimization started....")
        classifier_name = trial.suggest_categorical(
            'classifier', ['DecisionTree', 'RandomForest', 'XGBoost', 'SVM']
        )
        logger(log_path, logging.INFO, f"Classifier: {classifier_name}")

        if classifier_name == 'DecisionTree':
            model = DecisionTreeClassifier(
                max_depth=trial.suggest_int('dt_max_depth', 1, 20),
                min_samples_split=trial.suggest_int('dt_min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('dt_min_samples_leaf', 1, 20),
                max_features=trial.suggest_categorical('dt_max_features', [None, 'sqrt', 'log2']),
                criterion=trial.suggest_categorical('dt_criterion', ['gini', 'entropy']),
                random_state=42,
            )
        
        elif classifier_name == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int('rf_n_estimators', 50, 500),
                max_depth=trial.suggest_int('rf_max_depth', 2, 20),
                min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('rf_min_samples_leaf', 1, 20),
                max_features=trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
                bootstrap=trial.suggest_categorical('rf_bootstrap', [True, False]),
                random_state=42,
            )

        elif classifier_name == 'XGBoost':
            model = XGBClassifier(
                n_estimators=trial.suggest_int('xgb_n_estimators', 50, 500),
                max_depth=trial.suggest_int('xgb_max_depth', 2, 20),
                min_child_weight=trial.suggest_int('xgb_min_child_weight', 1, 20),
                subsample=trial.suggest_float('xgb_subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
                learning_rate=trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
                gamma=trial.suggest_float('xgb_gamma', 0, 10),
                random_state=42,
            )

        elif classifier_name == 'SVM':
            kernel = trial.suggest_categorical('svm_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            gamma = (
                trial.suggest_categorical('svmcategorical_gamma', ['scale', 'auto'])
                if kernel in ['linear', 'sigmoid']
                else trial.suggest_float('svmfloat_gamma', 0.001, 10)
            )
            model = SVC(
                C=trial.suggest_float('svm_C', 1e-3, 1e3),
                kernel=kernel,
                degree=trial.suggest_int('svm_degree', 2, 5) if kernel == 'poly' else 3,
                gamma=gamma,
                coef0=trial.suggest_float('svm_coef0', 0.0, 1.0) if kernel in ['poly', 'sigmoid'] else 0.0,
                random_state=42,
            )
        score = cross_val_score(model, self.X_train, self.y_train, cv=5,n_jobs=-1).mean()
        logger(log_path, logging.INFO, f"Score: {score}")
        logger(log_path, logging.INFO, "Optuna optimization completed....")
        return score

    def train_best_model(self):
        """
        Runs the Optuna study, trains the best model, evaluates it, saves results as CSV, 
        and exports the trained model.
        """
        logger(log_path, logging.INFO, "Training the best model started ....")
        # Run Optuna optimization
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        logger(log_path, logging.INFO, "Optuna study started....")
        study.optimize(self.objective, n_trials=50,timeout=3600)
        logger(log_path, logging.INFO, "Optuna study completed....")
        # Retrieve the best model and parameters
        best_params = study.best_params
        best_model_name = best_params.pop('classifier')
        logger(log_path, logging.INFO, f"Best model: {best_model_name}")
        logger(log_path, logging.INFO, f"Best parameters: {best_params}")
        # Map the best model name to its class
        model_mapping = {
            'DecisionTree': DecisionTreeClassifier,
            'RandomForest': RandomForestClassifier,
            'XGBoost': XGBClassifier,
            'SVM': SVC
        }

        # Instantiate the best model with the best parameters
        logger(log_path, logging.INFO, "Instantiating the best model....")
        model_class = model_mapping[best_model_name]
        model = model_class(**{key.split('_', 1)[1]: value for key, value in best_params.items()})

        # Train the best model
        model.fit(self.X_train, self.y_train)
        logger(log_path, logging.INFO, "Model training completed....")

        # Evaluate performance
        logger(log_path, logging.INFO, "Evaluating model performance....")
        train_accuracy = accuracy_score(self.y_train, model.predict(self.X_train))
        test_accuracy = accuracy_score(self.y_test, model.predict(self.X_test))
        f1 = f1_score(self.y_test, model.predict(self.X_test), average='weighted')
        recall = recall_score(self.y_test, model.predict(self.X_test), average='weighted')
        precision = precision_score(self.y_test, model.predict(self.X_test), average='weighted')
        logger(log_path, logging.INFO, f"Train Accuracy: {train_accuracy}")
        logger(log_path, logging.INFO, f"Test Accuracy: {test_accuracy}")
        logger(log_path, logging.INFO, f"F1 Score: {f1}")
        logger(log_path, logging.INFO, f"Recall: {recall}")
        logger(log_path, logging.INFO, f"Precision: {precision}")
        # Save results as CSV
        results = pd.DataFrame(
            {
                'Train Accuracy': [train_accuracy],
                'Test Accuracy': [test_accuracy],
                'F1 Score': [f1],
                'Recall': [recall],
                'Precision': [precision]
            }
        )
        results_csv_path = os.path.join(self.config.root_dir, 'model_performance.csv')
        results.to_csv(results_csv_path, index=False)
        
        # create a csv file to store the model name and the best parameters
        model_name = [best_model_name]
        model_parameters = [best_params]
        model_results = pd.DataFrame(list(zip(model_name, model_parameters)), columns =['Model Name', 'Model Parameters'])
        model_results_csv_path = os.path.join(self.config.root_dir, 'model_results.csv')
        model_results.to_csv(model_results_csv_path, index=False)
        # Save the model
        save_model(model, self.config.root_dir, self.config.model_name)



        return model, train_accuracy, test_accuracy

