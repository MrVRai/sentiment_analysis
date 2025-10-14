import sys
from src.exception import CustomException
from src.logger import logging

# Import the components you've already built
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        # Initialize the components
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        '''
        This method executes the full training pipeline.
        '''
        try:
            logging.info("--- Starting the Training Pipeline ---")
            
            # Step 1: Data Ingestion
            logging.info("Executing Data Ingestion...")
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            logging.info("Data Ingestion completed successfully.")
            
            # Step 2: Data Transformation
            logging.info("Executing Data Transformation...")
            X_train_arr, y_train, X_test_arr, y_test = self.data_transformation.initiate_data_transformation(
                train_path=train_data_path, 
                test_path=test_data_path
            )
            logging.info("Data Transformation completed successfully.")

            # Step 3: Model Training
            logging.info("Executing Model Training...")
            f1_score = self.model_trainer.initiate_model_trainer(
                X_train_arr=X_train_arr,
                y_train=y_train,
                X_test_arr=X_test_arr,
                y_test=y_test
            )
            logging.info(f"Model Training completed successfully. Final F1 Score: {f1_score:.4f}")
            
            logging.info("--- Training Pipeline has finished successfully. ---")
            
        except Exception as e:
            logging.error("An error occurred during the training pipeline execution.")
            raise CustomException(e, sys)

# This block allows you to run the entire pipeline from the command line
if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()