import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
import joblib
from src.WineQualityPrediction.entity.config_entity import ModelTrainerConfig
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.WineQualityPrediction.config.configuration import ConfigurationManager

log_path = 'log\log_file.log'

import os
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

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
        self.train_data = pd.read_csv(self.config.train_data_path)
        self.test_data = pd.read_csv(self.config.test_data_path)
        self.X_train = self.train_data.drop([self.config.target_column], axis=1)
        self.X_test = self.test_data.drop([self.config.target_column], axis=1)
        self.y_train = self.train_data[self.config.target_column]
        self.y_test = self.test_data[self.config.target_column]
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def objective(self, trial):
        classifier_name = trial.suggest_categorical(
            'classifier', ['DecisionTree', 'RandomForest', 'XGBoost', 'SVM', 'LogisticRegression']
        )

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
                trial.suggest_categorical('svm_gamma_categorical', ['scale', 'auto'])
                if kernel in ['linear', 'sigmoid']
                else trial.suggest_float('svm_gamma_float', 0.001, 10)
            )
            model = SVC(
                C=trial.suggest_float('svm_C', 1e-3, 1e3),
                kernel=kernel,
                degree=trial.suggest_int('svm_degree', 2, 5) if kernel == 'poly' else 3,
                gamma=gamma,
                coef0=trial.suggest_float('svm_coef0', 0.0, 1.0) if kernel in ['poly', 'sigmoid'] else 0.0,
                random_state=42,
            )

        elif classifier_name == 'LogisticRegression':
            penalty = trial.suggest_categorical('lr_penalty', ['l1', 'l2'])
            model = LogisticRegression(
                C=trial.suggest_float('lr_C', 1e-3, 1e3),
                penalty=penalty,
                solver='liblinear' if penalty == 'l1' else 'lbfgs',
                random_state=42,
            )

        score = cross_val_score(model, self.X_train, self.y_train, cv=5).mean()
        return score

    def train_best_model(self):
        """
        Runs the Optuna study, trains the best model, evaluates it, saves results as CSV, 
        and exports the trained model.
        """
        # Run Optuna optimization
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(self.objective, n_trials=100)

        # Retrieve the best model and parameters
        best_params = study.best_params
        best_model_name = best_params.pop('classifier')

        # Map the best model name to its class
        model_mapping = {
            'DecisionTree': DecisionTreeClassifier,
            'RandomForest': RandomForestClassifier,
            'XGBoost': XGBClassifier,
            'SVM': SVC,
            'LogisticRegression': LogisticRegression,
        }

        # Instantiate the best model with the best parameters
        model_class = model_mapping[best_model_name]
        model = model_class(**{key.split('_', 1)[1]: value for key, value in best_params.items()})

        # Train the best model
        model.fit(self.X_train, self.y_train)

        # Evaluate performance
        train_accuracy = accuracy_score(self.y_train, model.predict(self.X_train))
        test_accuracy = accuracy_score(self.y_test, model.predict(self.X_test))

        # Save results as CSV
        results = pd.DataFrame(
            {'Metric': ['Train Accuracy', 'Test Accuracy'], 'Value': [train_accuracy, test_accuracy]}
        )
        results_csv_path = os.path.join(self.config.output_dir, 'model_performance.csv')
        results.to_csv(results_csv_path, index=False)

        # Save the model
        model_path = os.path.join(self.config.output_dir, 'best_model.pkl')
        joblib.dump(model, model_path)

    

        return model, train_accuracy, test_accuracy
'''
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
        self.train_data = pd.read_csv(self.config.train_data_path)
        self.test_data = pd.read_csv(self.config.test_data_path)
        self.X_train = self.train_data.drop([self.config.target_column], axis=1)
        self.X_test = self.test_data.drop([self.config.target_column], axis=1)
        self.y_train = self.train_data[[self.config.target_column]]
        self.y_test = self.test_data[[self.config.target_column]]


    def objective(self,trial):
        classifier_name = trial.suggest_categorical('classifier', ['DecisionTree', 'RandomForest', 'XGBoost', 'SVM', 'LogisticRegression'])

        if classifier_name == 'DecisionTree':
            # Define hyperparameters to search
            max_depth = trial.suggest_int('dt_max_depth', 1, 20)
            min_samples_split = trial.suggest_int('dt_min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('dt_min_samples_leaf', 1, 20)
            max_features = trial.suggest_categorical('dt_max_features', [None, 'sqrt', 'log2'])
            criterion = trial.suggest_categorical('dt_criterion', ['gini', 'entropy'])

            # Create Decision Tree model
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                criterion=criterion,
                random_state=42
            )

        elif classifier_name == 'RandomForest':
            # Define hyperparameters to search
            n_estimators = trial.suggest_int('rf_n_estimators', 50, 500)
            max_depth = trial.suggest_int('rf_max_depth', 2, 20)
            min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 20)
            max_features = trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None])
            bootstrap = trial.suggest_categorical('rf_bootstrap', [True, False])

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=42
            )

        elif classifier_name == 'XGBoost':
            # Define hyperparameters to search
            n_estimators = trial.suggest_int('xgb_n_estimators', 50, 500)
            max_depth = trial.suggest_int('xgb_max_depth', 2, 20)
            min_child_weight = trial.suggest_int('xgb_min_child_weight', 1, 20)
            subsample = trial.suggest_float('xgb_subsample', 0.5, 1.0)
            colsample_bytree = trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0)
            learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3)
            gamma = trial.suggest_float('xgb_gamma', 0, 10)

            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma,
                random_state=42
            )

        elif classifier_name == 'SVM':
            # Define hyperparameters to search
            C = trial.suggest_float('svm_C', 1e-3, 1e3)
            kernel = trial.suggest_categorical('svm_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            
            # Use separate parameter names for categorical and float gamma
            if kernel in ['linear', 'sigmoid']:
                gamma = trial.suggest_categorical('svm_gamma_categorical', ['scale', 'auto'])
            else:
                gamma = trial.suggest_float('svm_gamma_float', 0.001, 10)

            degree = trial.suggest_int('svm_degree', 2, 5) if kernel == 'poly' else 3
            coef0 = trial.suggest_float('svm_coef0', 0.0, 1.0) if kernel in ['poly', 'sigmoid'] else 0.0

            model = SVC(
                C=C,
                kernel=kernel,
                degree=degree,
                gamma=gamma,
                coef0=coef0,
                random_state=42
            )
        elif classifier_name == 'LogisticRegression':
            # Define hyperparameters to search
            C = trial.suggest_float('lr_C', 1e-3, 1e3)
            penalty = trial.suggest_categorical('lr_penalty', ['l1', 'l2'])
            solver = 'liblinear' if penalty == 'l1' else 'lbfgs'

            model = LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                random_state=42
            )
    
        score = cross_val_score(model, self.X_train, self.y_train, cv=5).mean()

        return score
        

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=100)


    # best model name 
    best_model_name = study.best_params['classifier']
    best_model_params = study.best_params

'''


if __name__ == '__main__':
    config_manager = ConfigurationManager()
    model_trainer_config = config_manager.get_model_trainer_config()
    model_trainer = ModelTrainer(config=model_trainer_config)

    # Train the best model and get the results
    model, train_accuracy, test_accuracy = model_trainer.train_best_model()

    print(f'Train accuracy: {train_accuracy}')
    print(f'Test accuracy: {test_accuracy}')
