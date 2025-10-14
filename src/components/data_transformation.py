import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class LemmaTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    def __call__(self, doc):
        # First, tokenize the document (preserve_line=True avoids sentence tokenization)
        tokens = word_tokenize(doc.lower(), preserve_line=True)
        # Then, lemmatize each token
        lemmas = [self.lemmatizer.lemmatize(t) for t in tokens]
        # Remove stop words after lemmatization
        return [lemma for lemma in lemmas if lemma not in self.stop_words]
    


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "vectorizer.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating the data transformer object (TfidfVectorizer for text).
        '''
        try:
            # TF-IDF Vectorizer converts text into numerical features
            # max_features=5000 means it will only consider the top 5000 most frequent words
            vectorizer = TfidfVectorizer(
                stop_words=None,
                tokenizer=LemmaTokenizer(),
                ngram_range=(1, 2),
                max_features=10000
            )
            logging.info("TfidfVectorizer object created.")
            return vectorizer
        
        except Exception as e:
            raise CustomException(e, sys)
            
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Drop rows with missing text values, if any
            train_df.dropna(subset=['Text'], inplace=True)
            test_df.dropna(subset=['Text'], inplace=True)

            logging.info("Obtaining vectorizer object")
            vectorizer_obj = self.get_data_transformer_object()

            target_column_name = "sentiment"
            input_feature_column_name = "Text"

            # Separate input features and target features
            X_train = train_df[input_feature_column_name]
            y_train = train_df[target_column_name]
            X_test = test_df[input_feature_column_name]
            y_test = test_df[target_column_name]


            logging.info("Applying vectorizer on training and testing text.")
            X_train_arr = vectorizer_obj.fit_transform(X_train)
            X_test_arr = vectorizer_obj.transform(X_test)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=vectorizer_obj
            )
            logging.info("Saved vectorizer object.")

            return (
                X_train_arr,
                y_train,
                X_test_arr,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)