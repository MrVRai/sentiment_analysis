import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Paths for our sentiment analysis model and vectorizer
            model_path = 'artifacts/model.pkl'
            vectorizer_path = 'artifacts/vectorizer.pkl'
            
            # Load the trained model and vectorizer objects
            model = load_object(file_path=model_path)
            vectorizer = load_object(file_path=vectorizer_path)
            
            # The vectorizer expects a list of strings
            data_vectorized = vectorizer.transform(features)
            
            # Make the prediction
            prediction = model.predict(data_vectorized)

            # Map numerical prediction back to a readable label
            if prediction[0] == 1:
                return "Positive"
            else:
                return "Negative"
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    '''
    This class is responsible for handling the input data from the user (e.g., from a web form).
    For our project, the only input is the text of the review.
    '''
    def __init__(self, text: str):
        self.text = text

    def get_data_for_prediction(self):
        '''
        This method returns the input data in the list format
        that our TfidfVectorizer expects.
        '''
        try:
            # The vectorizer's .transform method needs an iterable (like a list) of strings
            return [self.text]
        
        except Exception as e:
            raise CustomException(e, sys)