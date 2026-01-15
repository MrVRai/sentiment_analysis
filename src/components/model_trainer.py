import os
import sys
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from scipy.sparse import csr_matrix



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    metrics_file_path = os.path.join("artifacts", "model_metrics.csv")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train_arr, y_train, X_test_arr, y_test):
        """
        This function train multiple models, compare them, and save the best-performing one.
        """
        try:
            logging.info("Starting model comparison and training process.")
            # Convert to CSR format for compatibility
            X_train_arr = csr_matrix(X_train_arr)
            X_test_arr = csr_matrix(X_test_arr)
            models = {
                "logistic_regression": LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42
                ),
                "linear_svm": LinearSVC(
                    class_weight="balanced",
                    random_state=42
                )
            }
            
            metrics = []
            best_model = None
            best_f1 = -1
            best_model_name = None

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")
                model.fit(X_train_arr, y_train)

                predictions = model.predict(X_test_arr)

                f1 = f1_score(y_test, predictions)
                precision = precision_score(y_test, predictions)
                recall = recall_score(y_test, predictions)

                logging.info(
                    f"{model_name} -> F1: {f1}, Precision: {precision}, Recall: {recall}"
                )

                metrics.append({
                    "model": model_name,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall
                })

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_model_name = model_name

            # Save metrics table
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(
                self.model_trainer_config.metrics_file_path,
                index=False
            )

            logging.info("Saved model comparison metrics.")


            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(
                f"Best model: {best_model_name} with F1 score: {best_f1}"
            )

            return best_model_name, best_f1

            # save_object(
            #     file_path=self.model_trainer_config.trained_model_file_path,
            #     obj=model
            # )
            # logging.info("Saved the trained model object to 'artifacts/model.pkl'")

            # logging.info("Evaluating the model on the test set.")
            # predictions = model.predict(X_test_arr)

            # # Calculate F1 score for evaluation
            # score = f1_score(y_test, predictions)
            # logging.info(f"Calculated F1 score: {score}")

            # return score

        except Exception as e:
            raise CustomException(e, sys)