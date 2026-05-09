import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object










@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")



class DataTransformation:
    def __init__(self) :
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):
        try :
            logging.info("Data transformation initiated")

            numerical_col = [
                "amenities", "accommodates", "bathrooms", "latitude", "longitude",
                "number_of_reviews", "review_scores_rating", "bedrooms", "beds"
            ]

            # low cardinality (<= 30 unique values) → OneHotEncoder
            low_card_col = [
                "room_type", "bed_type", "cancellation_policy", "cleaning_fee",
                "city", "host_has_profile_pic", "host_identity_verified", "instant_bookable"
            ]

            # high cardinality (> 30 unique values) → TargetEncoder
            high_card_col = [
                "property_type", "host_response_rate"
            ]

            num_pip = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",  StandardScaler()),
                ]
            )

            low_card_pip = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )

            high_card_pip = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", TargetEncoder()),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num",       num_pip,       numerical_col),
                    ("low_card",  low_card_pip,  low_card_col),
                    ("high_card", high_card_pip, high_card_col),
                ]
            )

            logging.info("Preprocessing pipeline created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "log_price"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df, target_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
