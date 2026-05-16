from dataclasses import dataclass
import os
import sys

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation




@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', "model.pkl")



class ModelTrainer: 
    def __init__(self) :
        self.model_trainer_config=ModelTrainerConfig()



    def initiate_model_trainer(self,train_array,test_array):

        try : 
            logging.info("Splitting training and testing input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
            }

            model_report: dict = evaluate_model(models, X_train, y_train, X_test, y_test)

            logging.info("Model performance report (r2_score on test set):")
            print("\n===== Model Performance Report (r2_score) =====")
            for model_name, score in sorted(model_report.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"{model_name}: r2_score = {score}")
                print(f"{model_name:<20} : {score:.4f}")
            print("===============================================\n")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found (r2_score < 0.6)", sys)

            logging.info(f"Best model found on both training and testing dataset is {best_model_name} with r2_score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
        

                

    