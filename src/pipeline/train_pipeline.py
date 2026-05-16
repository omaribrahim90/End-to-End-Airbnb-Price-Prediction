import os
import sys
from src.exception import CustomException
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
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            # Model Training
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info("Training pipeline completed successfully.")
        except Exception as e:
            logging.error(f"Error occurred in training pipeline: {e}")
            raise CustomException(e, sys)
        
if __name__=="__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline()
    
    