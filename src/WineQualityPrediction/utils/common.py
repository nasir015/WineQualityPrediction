import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import logging
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from typing import Dict
from box.exceptions import BoxValueError

# create a file utils.log in LogFile folder
log_path = 'log\log_file.log'
os.makedirs(os.path.dirname(log_path), exist_ok=True)

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the YAML file to be read.

    Raises:
        ValueError: If the YAML file is empty or has invalid content.
        CustomException: For other errors during file reading or parsing.

    Returns:
        ConfigBox: A `ConfigBox` object containing the parsed YAML data.

    Example:
        >>> from pathlib import Path
        >>> config_path = Path("config.yaml")
        >>> config = read_yaml(config_path)
        >>> print(config.some_key)  # Access values using dot notation
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger(log_path, logging.INFO, f"Yaml file read successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise CustomException(e, sys)




@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger(log_path, logging.INFO, f"Directory created at {path}")
        



def save_json(path: Path, data: Dict) -> None:
    """
    Save JSON data to a file.

    Args:
        path (Path): The file path where the JSON data will be saved.
        data (Dict): The data to be serialized and saved as JSON.

    Raises:
        ValueError: If the data cannot be serialized to JSON.
        IOError: If the file cannot be written.
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger(log_path, logging.INFO, f"JSON data saved to {path}")
    except TypeError as e:
        logger(log_path, logging.ERROR, f"Data provided cannot be serialized to JSON: {e}")
        raise CustomException(e, sys)
    except IOError as e:
        logger(log_path, logging.ERROR, f"Error writing JSON data to {path}: {e}")
        raise CustomException(e, sys)
    

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file and returns its content as a ConfigBox object.

    Args:
        path (Path): The file path to the JSON file.

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
        json.JSONDecodeError: If the JSON file contains invalid JSON.
        ValueError: If the loaded content is empty or invalid.

    Returns:
        ConfigBox: A `ConfigBox` object allowing attribute-style access to the data.

    Example:
        >>> from pathlib import Path
        >>> json_path = Path("config.json")
        >>> config = load_json(json_path)
        >>> print(config.database.host)  # Access values using dot notation
    """
    try:
        # Ensure the file exists and is readable
        if not path.is_file():
            raise FileNotFoundError(f"JSON file not found at path: {path}")

        with open(path, "r") as file:
            content = json.load(file)
            if not content:  # Check if the JSON file is empty or contains invalid data
                raise ValueError("The JSON file is empty or contains invalid data.")

        logger(log_path, logging.INFO, f"JSON file loaded successfully from {path}")
        return ConfigBox(content)
    except FileNotFoundError as e:
        logger(log_path, logging.ERROR, f"JSON file not found at path: {path}")
        raise CustomException(e, sys)
    except json.JSONDecodeError as e:
        logger(log_path, logging.ERROR, f"Invalid JSON format in file: {path}")
        raise CustomException(e, sys)
    except Exception as e:
        logger(log_path, logging.ERROR, f"Error loading JSON file: {e}")
        raise CustomException(e, sys)
    

@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    """
    Save data to a binary file using Joblib.

    Args:
        data (Any): The data to be serialized and saved as a binary file.
        path (Path): The file path where the binary file will be saved.

    Raises:
        ValueError: If the data is None or invalid.
        IOError: If the file cannot be written due to permissions or other issues.

    Example:
        >>> from pathlib import Path
        >>> from sklearn.linear_model import LogisticRegression
        >>> model = LogisticRegression()
        >>> save_bin(model, Path("model.pkl"))
    """
    try:
        if data is None:
            raise ValueError("The data to be saved cannot be None.")

        joblib.dump(value=data, filename=path)
        logger(log_path, logging.INFO, f"Binary file saved at {path}")
    except ValueError as e:
        logger(log_path, logging.ERROR, f"Invalid data provided for saving binary file: {e}")
        raise CustomException(e, sys)
    except IOError as e:
        logger(log_path, logging.ERROR, f"Error writing binary file to {path}: {e}")
        raise CustomException(e, sys)
    except Exception as e:
        logger(log_path, logging.ERROR, f"Error saving binary file: {e}")
        raise CustomException(e, sys)


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load binary data from a file using Joblib.

    Args:
        path (Path): Path to the binary file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.
        ValueError: If the file contains invalid data.

    Returns:
        Any: The object stored in the file.

    Example:
        >>> from pathlib import Path
        >>> model_path = Path("model.pkl")
        >>> model = load_bin(model_path)
    """
    try:
        if not path.is_file():
            raise FileNotFoundError(f"Binary file not found at: {path}")
        
        data = joblib.load(path)
        logger(log_path, logging.INFO, f"Binary file loaded successfully from: {path}")
        return data
    except FileNotFoundError as e:
        logger(log_path, logging.ERROR, f"Error loading binary file: {e}")
        raise CustomException(e, sys)
    except IOError as e:
        logger(log_path, logging.ERROR, f"Error reading binary file at {path}: {e}")
        raise CustomException(e, sys)
    except ValueError as e:
        logger(log_path, logging.ERROR, f"Invalid data in binary file: {e}")
        raise CustomException(e, sys)
    except Exception as e:
        logger(log_path, logging.ERROR, f"An unexpected error occurred while loading binary file: {e}")
        raise CustomException(e, sys)