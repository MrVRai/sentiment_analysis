import sys, os
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.components.dl_data_transformation import DLDataTranformation
from src.components.dl_model_trainer import DLModelTrainer


@dataclass
class DLTrainPipelineConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DLTrainPipeline:
    def __init__(self):
        self.config = DLTrainPipelineConfig()
        self.data_tranformation = DLDataTranformation()
        self.model_trainer = DLModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting DL pipeline")
            X_train, y_train, X_test, y_test = (
                self.data_tranformation.initiate_dl_tranformation(self.config.train_data_path, self.config.test_data_path)
            )

            self.model_trainer.initiate_dl_model_trainer(X_train, y_train, X_test, y_test)

            logging.info("DL pipeline finished")

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    pipeline = DLTrainPipeline()
    pipeline.run_pipeline()