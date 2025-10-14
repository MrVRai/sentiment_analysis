import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Import the transformation and model training components
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw_reviews.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")
        try:
            # Path to your Amazon Fine Food Reviews CSV file
            input_csv = 'artifacts/Reviews.csv' 
            if not os.path.exists(input_csv):
                logging.error(f"Input file not found: {input_csv}")
                raise FileNotFoundError(f"Input file not found: {input_csv}")

            df = pd.read_csv(input_csv)
            logging.info("Successfully read the dataset as a DataFrame")

            # --- SENTIMENT ANALYSIS SPECIFIC PREPROCESSING ---
            # 1. Droping unnecessay columns
            df.dropna(subset=['Text', 'Score'], inplace=True)
            
            # 2. Filter out neutral (3-star) reviews
            df_binary = df[df['Score'] != 3].copy()
            logging.info("Filtered out neutral 3-star reviews")

            # 3. Create the binary 'sentiment' column (1 for Positive, 0 for Negative)
            df_binary['sentiment'] = np.where(df_binary['Score'] > 3, 1, 0)
            
            # 4. Select only the necessary columns for the project
            df_final = df_binary[['Text', 'sentiment']]
            logging.info("Created binary sentiment column and selected final columns")


            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df_final.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Saved the preprocessed raw data")

            logging.info("Initiating train-test split")
            train_set, test_set = train_test_split(df_final, test_size=0.2, random_state=42, stratify=df_final['sentiment'])

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion is complete")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
