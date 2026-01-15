import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        try:
            logging.info("Loading model and vectorizer at startup.")

            self.model_path = "artifacts/model.pkl"
            self.vectorizer_path = "artifacts/vectorizer.pkl"
            self.label_mapping = {0: "Negative", 1: "Positive"}

            # LOAD ONCE
            self.model = load_object(self.model_path)
            self.vectorizer = load_object(self.vectorizer_path)

            logging.info("Model and vectorizer loaded successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        """
        Perform sentiment prediction.
        """
        try:
            data_vectorized = self.vectorizer.transform(features)
            predictions = self.model.predict(data_vectorized)

            return [self.label_mapping[pred] for pred in predictions]

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, text: str):
        self.text = text

    def get_data_for_prediction(self):
        return [self.text]
