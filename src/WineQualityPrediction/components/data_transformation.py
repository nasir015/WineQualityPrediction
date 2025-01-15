import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
from src.WineQualityPrediction.entity.config_entity import DataTransformationConfig

log_path = "log\log_file.log"

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation instance with the provided configuration.
        
        Args:
            config (DataTransformationConfig): Configuration object containing paths
                                                and other parameters for data transformation.
        """
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):
        logger(log_path,logging.INFO,"Train Test Splitting Started....")
        """
        Splits the dataset into training and testing sets (75% train, 25% test).
        Saves the resulting dataframes as CSV files in the specified directory.
        
        - Loads the dataset from the path defined in `config.data_path`.
        - Splits the data into training and testing sets.
        - Saves the resulting sets as `train.csv` and `test.csv` in the root directory defined in `config.root_dir`.
        
        Logs the shape of the train and test datasets and prints it to the console.
        """
        data = pd.read_csv(self.config.data_path)
        logger(log_path,logging.INFO,"Data Loaded Successfully....")

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)
        logger(log_path,logging.INFO,"Data Split Successfully....")

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)
        logger(log_path,logging.INFO,"Data Saved Successfully....")
        
        logger(log_path,logging.INFO,f'test shape is {test.shape}')
        logger(log_path,logging.INFO,f'train shape is {train.shape}')

        print(train.shape)
        print(test.shape)