import os
import sys
from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object


from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer




class TrainPipeline:
    def __init__(self) :
        pass
    

    def run_pipeline(self):
        try : 
            logging.info("Starting the training pipeline.")

            # Data Ingestion
            data_ingestion = DataIngestion()
            train_data_path , test_data_path = data_ingestion.initiate_data_ingestion()

            # Data Transformation
            data_transformation = DataTransformation()
            preprocessor_obj_file_path = data_transformation.get_data_transformer_object()
            train_arr, test_arr, _ = data_transformation.
    