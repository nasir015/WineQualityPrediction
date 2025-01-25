from src.WineQualityPrediction.constants import *
from src.WineQualityPrediction.utils.common import read_yaml, create_directories
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException

from src.WineQualityPrediction.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig , ModelTrainerConfig, ModelEvaluationConfig


class ConfigurationManager:
    """
    Manages the configuration for the project by reading YAML files
    and providing structured configurations.

    Attributes:
        config (ConfigBox): Parsed configuration file.
        params (ConfigBox): Parsed parameters file.
        schema (ConfigBox): Parsed schema file.
    """

    def __init__(self,
                 config_filepath: Path = CONFIG_FILE_PATH,
                 params_filepath: Path = PARAMS_FILE_PATH,
                 schema_filepath: Path = SCHEMA_FILE_PATH):
        """
        Initializes the ConfigurationManager by loading YAML files and creating directories.

        Args:
            config_filepath (Path): Path to the main configuration file.
            params_filepath (Path): Path to the parameters configuration file.
            schema_filepath (Path): Path to the schema configuration file.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        # Ensure the artifacts root directory exists
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Provides the configuration for the data ingestion component.

        Returns:
            DataIngestionConfig: Configuration object for data ingestion.
        """
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Provides the configuration for the data ingestion component.

        Returns:
            DataIngestionConfig: Configuration object for data ingestion.
        """
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            data_file =config.data_file,
            status_file=config.STATUS_FILE,
            all_schema=schema
        )
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Provides the configuration for the data transformation component.

        Returns:
            DataTransformationConfig: Configuration object for data transformation.
        """
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            target_column=self.schema.TARGET_COLUMN.name
        )
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Provides the configuration for the model trainer.

        Returns:
            ModelTrainerConfig: Configuration for the model trainer.
        """
        config = self.config.model_trainer
        schema = self.schema.TARGET_COLUMN
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            X_data_path=config.X_data_path,
            y_data_path=config.y_data_path,
            model_name=config.model_name,
            target_column=schema.name
        )
        return model_trainer_config   
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config=self.config.model_evaluation
        params=self.params.ElasticNet
        schema=self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config=ModelEvaluationConfig(
            root_dir=config.root_dir,
            X_test=config.X_test,
            y_test=config.y_test,
            model_path = config.model_path,
            model_name = config.model_name,
            metric_file_name = config.metric_file_name,
            target_column = schema.name,
            mlflow_uri="https://dagshub.com/nasir.uddin.6314/WineQualityPrediction.mlflow"


        )
        return model_evaluation_config    
