import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from scipy.sparse import csr_matrix


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train_arr, y_train, X_test_arr, y_test):
        """
        This function trains the model on the vectorized data and saves it.
        """
        try:
            logging.info("Starting model training process.")
            # Convert to CSR format for compatibility
            X_train_arr = csr_matrix(X_train_arr)
            X_test_arr = csr_matrix(X_test_arr)
            model = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0, random_state=42)
            
            logging.info("Training the classification model")
            model.fit(X_train_arr, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            logging.info("Saved the trained model object to 'artifacts/model.pkl'")

            logging.info("Evaluating the model on the test set.")
            predictions = model.predict(X_test_arr)

            # Calculate F1 score for evaluation
            score = f1_score(y_test, predictions)
            logging.info(f"Calculated F1 score: {score}")

            return score

        except Exception as e:
            raise CustomException(e, sys)