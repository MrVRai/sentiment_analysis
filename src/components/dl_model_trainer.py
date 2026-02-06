import os, sys
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Attention, GlobalAveragePooling1D, GRU
from tensorflow.keras import Input
from tensorflow.keras import Model

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.metrics import f1_score

@dataclass
class DLModelTrainerConfig:
    model_type: str = "gru"
    model_path: str = os.path.join("artifacts", "dl_model.h5")
    epochs: int = 3
    batch_size: int = 32
    metrics_path: str = os.path.join("artifacts", "dl_model_metrics.csv")

class DLModelTrainer:
    def __init__(self):
        self.config = DLModelTrainerConfig()

    def build_model(self, vocab_size = 10000, max_len = 100):

        inputs = Input(shape=(max_len,))
        x = Embedding(vocab_size,128)(inputs)

        x = Bidirectional(GRU(64, return_sequences=True))(x)

        attention = Attention()([x,x])
        x = GlobalAveragePooling1D()(attention)

        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(inputs, outputs)

        # model = Sequential([
        #     Embedding(vocab_size, 128, input_length=max_len),
        #     Bidirectional(LSTM(64)),
        #     Dense(1, activation="sigmoid")
        # ])

        model.compile(
            loss = "binary_crossentropy",
            optimizer = "adam",
            metrics=["accuracy"]
        )
        return model
        
    def initiate_dl_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Training deep learning model")

            results = []
            best_f1 = -1
            best_model = None
            best_model_name = None

            for model_type in ["lstm","gru"]:
                self.config.model_type = model_type

                logging.info(f"Training {model_type.upper()} model")

                model = self.build_model()
                model.fit(
                    X_train,
                    y_train, 
                    validation_data=(X_test,y_test),
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size,
                    verbose=1
                )

                preds = model.predict(X_test)
                preds = (preds>0.5).astype(int)

                f1 = f1_score(y_test, preds)

                logging.info(f"{model_type.upper()} F1 Score: {f1:.4f}")

                results.append({
                    "model" : model_type,
                    "f1_Score" : f1
                })

                if (f1 > best_f1):
                    best_f1 = f1
                    best_model = model
                    best_model_name = model_type

            df = pd.DataFrame(results)
            df.to_csv(self.config.metrics_path, index = False)

            logging.info("Saved model comparison metrics")

            best_model.save(self.config.model_path)
            logging.info(
                f"Best model: {best_model_name} with F1 score: {best_f1}"
            )
            logging.info("DL model trainig completed")
        
        except Exception as e:
            raise CustomException(e,sys)