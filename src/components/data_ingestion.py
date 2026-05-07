from src.exception import CustomException
from src.logger import logging
import os 
import pandas as pd
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


# just set the paths were file will be stored after the data ingestion
@dataclass
class DataIngestionConfig:
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_data_path: str=os.path.join(_project_root,'artifacts',"train.csv")
    test_data_path: str=os.path.join(_project_root,'artifacts',"test.csv")
    raw_data_path: str=os.path.join(_project_root,'artifacts',"data.csv")


class DataIngestion:
    def __init__(self) :
        self.ingestion_config = DataIngestionConfig() # we can new get access to the paths that we set above 

    
    def initiate_data_ingestion(self): # this is the main method that will be called to start the data ingestion process
        logging.info("Entered the data ingestion method or component")

        try :
            # Get the project root directory and construct the data path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            data_path = os.path.join(project_root, "Notebook_Experiments", "Data", "Airbnb_Data.csv")
            df = pd.read_csv(data_path)
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True) # will create a directory if it does not exist
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) # this will save the raw data to the specified 
            
            logging.info("Train test split initiated")
            train_set , test_set = train_test_split(df, test_size=0.2, random_state=42) # this will split the data into train and test set

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) # this will save the train data to the specified path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()


           
            
