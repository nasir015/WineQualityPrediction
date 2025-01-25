from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataValidationConfig:
    root_dir: Path 
    data_file: str
    status_file: Path
    all_schema: dict

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    target_column: str

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    y_data_path: Path
    X_data_path: Path
    model_name: str
    target_column: str


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    X_test: Path
    y_test: Path
    model_path: Path
    model_name: str
    metric_file_name: str
    target_column: str
    mlflow_uri: str