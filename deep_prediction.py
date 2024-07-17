print("it's loading")
import os
print("os loading")
import sys
print("sys loading")
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
print("sys finish loading")
import tensorflow as tf
print("tensorflow loading")

from tensorflow.keras.layers import LSTM, GRU, Dense, Activation, LeakyReLU, Dropout
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import warnings

from data_extractor import load_data
from create_csv import write_custom_csv, write_emodb_csv, write_tess_ravdess_csv
from emotion_recognition import EmotionRecognizer
from utils import get_first_letters, AVAILABLE_EMOTIONS, extract_feature, get_dropout_str

import numpy as np
import pandas as pd
import random

# Filter out the ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class DeepEmotionRecognizer(EmotionRecognizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.n_rnn_layers = kwargs.get("n_rnn_layers", 2)
        self.n_dense_layers = kwargs.get("n_dense_layers", 2)
        self.rnn_units = kwargs.get("rnn_units", 128)
        self.dense_units = kwargs.get("dense_units", 128)
        self.cell = kwargs.get("cell", LSTM)

        self.dropout = kwargs.get("dropout", 0.3)
        self.dropout = self.dropout if isinstance(self.dropout, list) else [self.dropout] * (self.n_rnn_layers + self.n_dense_layers)
        self.output_dim = len(self.emotions)

        self.optimizer = kwargs.get("optimizer", "adam")
        self.loss = kwargs.get("loss", "categorical_crossentropy")

        self.batch_size = kwargs.get("batch_size", '1')  # Changed to 'auto'
        self.epochs = kwargs.get("epochs", 1000)  # Increased to 1000
        
        self.model_name = ""
        self._update_model_name()

        self.model = None
        self._compute_input_length()
        self.model_created = False

    # ... (other methods remain the same)

    def train(self, override=False):
        if not self.model_created:
            self.create_model()

        if not override:
            model_name = self._model_exists()
            if model_name:
                self.model.load_weights(model_name)
                self.model_trained = True
                if self.verbose > 0:
                    print("[*] Model weights loaded")
                return
        
        if not os.path.isdir("results"):
            os.mkdir("results")

        if not os.path.isdir("logs"):
            os.mkdir("logs")

        model_filename = self._get_model_filename()

        self.checkpointer = ModelCheckpoint(model_filename, save_best_only=True, verbose=1)
        self.tensorboard = TensorBoard(log_dir=os.path.join("logs", self.model_name))

        # Adjust batch size if it's 'auto'
        #if self.batch_size == 'auto':
        #    batch_size = min(64, self.X_train.shape[1])  # Default to 32 or smaller if dataset is smaller
       # else:
        batch_size = min(self.batch_size, self.X_train.shape[1])

        try:
            self.history = self.model.fit(self.X_train, self.y_train,
                            batch_size=batch_size,
                            epochs=self.epochs,
                            validation_data=(self.X_test, self.y_test),
                            callbacks=[self.checkpointer, self.tensorboard],
                            verbose=self.verbose)
        except KeyboardInterrupt:
            print("Training interrupted. Saving current state...")
            self.model.save_weights(self._get_model_filename())
            tf.keras.backend.clear_session()
        
        self.model_trained = True
        if self.verbose > 0:
            print("[+] Model trained")

    # ... (other methods remain the same)

if __name__ == "__main__":
    rec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'],
                                epochs=50000, verbose=1, batch_size=32)
    try:
        rec.train(override=False)
        print("Test accuracy score:", rec.test_score() * 100, "%")
    except KeyboardInterrupt:
        print("Process interrupted by user. Saving current state.")

    # initialize instance
    deeprec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], 
                                    n_rnn_layers=2, n_dense_layers=2, rnn_units=64, dense_units=64,
                                    batch_size='auto', epochs=1000)
    try:
        # train the model
        deeprec.train()
        # get the accuracy
        print(deeprec.test_score())
        # predict angry audio sample
        prediction = deeprec.predict('data/validation/Actor_10/03-02-05-02-02-02-10_angry.wav')
        
        print(f"Prediction: {prediction}")
        print("\n\n")
        
        print(deeprec.predict_proba("data/emodb/wav/16a01Wb.wav"))
        print(deeprec.confusion_matrix(percentage=True, labeled=True))
    except KeyboardInterrupt:
        print("Process interrupted by user. Saving current state.")
