import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urllib.request as request
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
from src.WineQualityPrediction.entity.config_entity import DataIngestionConfig
import zipfile
import logging
log_path = 'log\log_file.log'


class DataIngestion:
    """
    Handles the ingestion of data, including downloading and extracting files.

    Attributes:
        config (DataIngestionConfig): Configuration object containing paths and URLs.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion instance with configuration.

        Args:
            config (DataIngestionConfig): Configuration for data ingestion.
        """
        self.config = config

    def download_file(self) -> None:
        """
        Downloads the file from the source URL to the local path if it doesn't already exist.
        """
        try:
            if not os.path.exists(self.config.local_data_file):
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL,
                    filename=self.config.local_data_file
                )
                logger(log_path,logging.INFO,f"Downloaded file from {self.config.source_URL} to {filename}")
            else:
                logger(log_path,logging.INFO,f"File already exists at {self.config.local_data_file}")
        except Exception as e:
            logger(log_path,logging.ERROR,f"Error downloading file: {CustomException(e,sys)}")
            raise CustomException(e,sys)

    def extract_zip_file(self) -> None:
        """
        Extracts the ZIP file located at `self.config.local_data_file` 
        to the directory specified by `self.config.unzip_dir`.
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger(log_path,logging.INFO,f"Extracted ZIP file to {unzip_path}")
        except zipfile.BadZipFile as e:
            logger(log_path,logging.ERROR,f"Error extracting the ZIP file: {CustomException(e,sys)}")
            raise CustomException(e,sys)
        except Exception as e:
            logger(log_path,logging.ERROR,f"Error extracting the ZIP file: {CustomException(e,sys)}")
            raise CustomException(e,sys)
