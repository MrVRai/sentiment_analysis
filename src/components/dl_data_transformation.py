import os 
import sys
from dataclasses import dataclass

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DLDataTranformationConfig:
    tokenizer_path: str = os.path.join("artifacts","dl_tokenizer.pkl")
    max_len: int = 100
    vocab_size: int = 10000

class DLDataTranformation:
    def __init__(self):
        self.config = DLDataTranformationConfig()
    
    def initiate_dl_tranformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Preparing text tokenizer for DL pipeline")

            tokenizer = Tokenizer(num_words=self.config.vocab_size)
            tokenizer.fit_on_texts(train_df["Text"])

            X_train = tokenizer.texts_to_sequences(train_df["Text"])
            X_test = tokenizer.texts_to_sequences(test_df["Text"])

            X_train = pad_sequences(X_train, maxlen = self.config.max_len, padding="post")
            X_test = pad_sequences(X_test, maxlen = self.config.max_len, padding="post")

            y_train = train_df["sentiment"].values
            y_test = test_df["sentiment"].values

            save_object(self.config.tokenizer_path, tokenizer)

            logging.info("DL transformation completed")

            return X_train,y_train,X_test,y_test
        
        except Exception as e:
            raise CustomException(e, sys)