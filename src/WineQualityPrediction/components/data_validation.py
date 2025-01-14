import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import pandas as pd
from typing import Optional
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
from src.WineQualityPrediction.entity.config_entity import DataValidationConfig

log_path = 'log/log_file.log'



class DataValidation:
    """
    A class to validate data based on the provided schema.

    Attributes:
        config (DataValidationConfig): Configuration object containing paths and schema for validation.

    Methods:
        validate_all_columns() -> bool:
            Validates all columns in the dataset against the predefined schema.

    Raises:
        Exception: Any exception raised during the validation process is re-raised for further handling.
    """

    def __init__(self, config: DataValidationConfig):
        """
        Initializes the DataValidation class with the given configuration.

        Args:
            config (DataValidationConfig): Configuration object with paths and schema.

        Attributes:
            config (DataValidationConfig): Stores the configuration for validation.
        """
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        Validates that all columns in the dataset match the predefined schema.

        Reads the data from the specified CSV file, checks if all columns in the dataset 
        exist in the schema, and writes the validation status to a file.

        Returns:
            bool: True if all columns are valid, False otherwise.

        Raises:
            Exception: If there is an error during file reading or validation.

        Workflow:
            - Load the dataset from the provided path.
            - Compare dataset columns against schema keys.
            - Write the validation status (True/False) to the specified status file.
        """
        try:
            # Initialize validation status
            validation_status: Optional[bool] = None
            logger(log_path, logging.INFO, "Data validation started...")

            # Read data from the provided CSV file
            data = pd.read_csv(self.config.data_file)
            logger(log_path, logging.INFO, "Data loaded successfully...")
            all_cols = list(data.columns)

            # Get all schema keys
            all_schema = self.config.all_schema.keys()
            logger(log_path, logging.INFO, "Schema keys loaded successfully...")

            # Validate each column in the dataset
            logger(log_path, logging.INFO, "Column validation started...")
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.status_file, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    break
                else:
                    validation_status = True

            # Write the final status if all columns are valid
            if validation_status:
                with open(self.config.status_file, 'w') as f:
                    f.write(f"Validation status: {validation_status}")
            
            logger(log_path, logging.INFO, "Data validation completed successfully...")

            return validation_status

        except Exception as e:
            # Re-raise exception for external handling
            logger(log_path, logging.ERROR, f"Data validation failed: {CustomException(e,sys)}")
            raise CustomException(e,sys)
        
