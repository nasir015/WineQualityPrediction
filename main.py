import os 
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.WineQualityPrediction.utils.my_logging import logger
from src.WineQualityPrediction.utils.my_exception import CustomException
from src.WineQualityPrediction.pipeline.DataIngestionPipeline import DataIngestionTrainingPipeline
from src.WineQualityPrediction.pipeline.DataValidationPipeline import DataValidationTrainingPipeline
from src.WineQualityPrediction.pipeline.DataTransformationPipeline import DataTransformationTrainingPipeline
from src.WineQualityPrediction.pipeline.ModelTrainingPipeline import ModelTrainingPipeline
from src.WineQualityPrediction.pipeline.ModelEvaluationPipeline import ModelEvaluationPipeline

log_path = 'log\log_file.log'


Stage_name = "Data Ingestion Stage"

try:
    logger(log_path, logging.INFO, f">>>>>>>>>>Starting {Stage_name}<<<<<<<<<<<")
    pipeline = DataIngestionTrainingPipeline()
    pipeline.initiate_data_ingestion()
    logger(log_path, logging.INFO, f">>>>>>>>>>{Stage_name} completed successfully.<<<<<<<<<<<")
except CustomException as ce:
    logger(log_path, logging.ERROR, f"CustomException occurred: {CustomException(ce, sys)}")
    raise CustomException(ce, sys)


Stage_name = "Data Validation Stage"

try:
    logger(log_path, logging.INFO, f">>>>>>>>>>Starting {Stage_name}<<<<<<<<<<<")
    pipeline = DataValidationTrainingPipeline()
    pipeline.initiate_data_validation()
    logger(log_path, logging.INFO, f">>>>>>>>>>{Stage_name} completed successfully.<<<<<<<<<<<")
except CustomException as ce:
    logger(log_path, logging.ERROR, f"CustomException occurred: {CustomException(ce, sys)}")
    raise CustomException(ce, sys)

STAGE_NAME = "Data Transformation stage"
try:
   logger(log_path, logging.INFO, f">>>>>> Starting stage {STAGE_NAME} <<<<<<")
   data_ingestion = DataTransformationTrainingPipeline()
   data_ingestion.initiate_data_transformation()
   logger(log_path, logging.INFO, f">>>>>> {STAGE_NAME} completed successfully. <<<<<<")
except Exception as e:
    logger(log_path, logging.ERROR, f"An unexpected error occurred: {CustomException(e, sys)}")
    raise CustomException(e, sys)


Stage_name = "Model Training Stage"

try:
    logger(log_path, logging.INFO, f">>>>>>>>>>Starting {Stage_name}<<<<<<<<<<<")
    pipeline = ModelTrainingPipeline()
    pipeline.initiate_model_training()
    logger(log_path, logging.INFO, f">>>>>>>>>>{Stage_name} completed successfully.<<<<<<<<<<<")
except CustomException as ce:
    logger(log_path, logging.ERROR, f"CustomException occurred: {CustomException(ce, sys)}")
    raise CustomException(ce, sys)

Stage_name = "Model Evaluation Stage"

try:
    logger(log_path, logging.INFO, f">>>>>>>>>>Starting {Stage_name}<<<<<<<<<<<")
    pipeline = ModelEvaluationPipeline()
    pipeline.initiate_model_evaluation()
    logger(log_path, logging.INFO, f">>>>>>>>>>{Stage_name} completed successfully.<<<<<<<<<<<")
except CustomException as ce:
    logger(log_path, logging.ERROR, f"CustomException occurred: {CustomException(ce, sys)}")
    raise CustomException(ce, sys)