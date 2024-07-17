"""
A script to grid search all parameters provided in parameters.py
including both classifiers and regressors.
Note that the execution of this script may take hours to search the 
best possible model parameters for various algorithms, feel free
to edit parameters.py on your need ( e.g remove some parameters for 
faster search )
"""

import pickle

from parameters import classification_grid_parameters, regression_grid_parameters



from utils import extract_feature, AVAILABLE_EMOTIONS
from create_csv import write_emodb_csv, write_tess_ravdess_csv, write_custom_csv

from sklearn.metrics import accuracy_score, make_scorer, fbeta_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as pl
from time import time
from utils import get_best_estimators, get_audio_config
import numpy as np
import tqdm
import os
import random
import pandas as pd



import numpy as np
import pandas as pd
import pickle
import tqdm
import os

from utils import get_label, extract_feature, get_first_letters
from collections import defaultdict


class AudioExtractor:
    """A class that is used to featurize audio clips, and provide
    them to the machine learning algorithms for training and testing"""
    def __init__(self, audio_config=None, verbose=1, features_folder_name="features", classification=True,
                    emotions=['sad', 'neutral', 'happy'], balance=True):
        """
        Params:
            audio_config (dict): the dictionary that indicates what features to extract from the audio file,
                default is {'mfcc': True, 'chroma': True, 'mel': True, 'contrast': False, 'tonnetz': False}
                (i.e mfcc, chroma and mel)
            verbose (bool/int): verbosity level, 0 for silence, 1 for info, default is 1
            features_folder_name (str): the folder to store output features extracted, default is "features".
            classification (bool): whether it is a classification or regression, default is True (i.e classification)
            emotions (list): list of emotions to be extracted, default is ['sad', 'neutral', 'happy']
            balance (bool): whether to balance dataset (both training and testing), default is True
        """
        self.audio_config = audio_config if audio_config else {'mfcc': True, 'chroma': True, 'mel': True, 'contrast': False, 'tonnetz': False}
        self.verbose = verbose
        self.features_folder_name = features_folder_name
        self.classification = classification
        self.emotions = emotions
        self.balance = balance
        # input dimension
        self.input_dimension = None

    def _load_data(self, desc_files, partition, shuffle):
        self.load_metadata_from_desc_file(desc_files, partition)
        # balancing the datasets ( both training or testing )
        if partition == "train" and self.balance:
            self.balance_training_data()
        elif partition == "test" and self.balance:
            self.balance_testing_data()
        else:
            if self.balance:
                raise TypeError("Invalid partition, must be either train/test")
        if shuffle:
            self.shuffle_data_by_partition(partition)

    def load_train_data(self, desc_files=["train_speech.csv"], shuffle=False):
        """Loads training data from the metadata files `desc_files`"""
        self._load_data(desc_files, "train", shuffle)
        
    def load_test_data(self, desc_files=["test_speech.csv"], shuffle=False):
        """Loads testing data from the metadata files `desc_files`"""
        self._load_data(desc_files, "test", shuffle)

    def shuffle_data_by_partition(self, partition):
        if partition == "train":
            self.train_audio_paths, self.train_emotions, self.train_features = shuffle_data(self.train_audio_paths,
            self.train_emotions, self.train_features)
        elif partition == "test":
            self.test_audio_paths, self.test_emotions, self.test_features = shuffle_data(self.test_audio_paths,
            self.test_emotions, self.test_features)
        else:
            raise TypeError("Invalid partition, must be either train/test")

    def load_metadata_from_desc_file(self, desc_files, partition):
        """Read metadata from a CSV file & Extract and loads features of audio files
        Params:
            desc_files (list): list of description files (csv files) to read from
            partition (str): whether is "train" or "test"
        """
        # empty dataframe
        df = pd.DataFrame({'path': [], 'emotion': []})
        for desc_file in desc_files:
            # concat dataframes
            df = pd.concat((df, pd.read_csv(desc_file)), sort=False)
        if self.verbose:
            print("[*] Loading audio file paths and its corresponding labels...")
        # get columns
        audio_paths, emotions = list(df['path']), list(df['emotion'])
        # if not classification, convert emotions to numbers
        if not self.classification:
            # so naive and need to be implemented
            # in a better way
            if len(self.emotions) == 3:
                self.categories = {'sad': 1, 'neutral': 2, 'happy': 3}
            elif len(self.emotions) == 5:
                self.categories = {'angry': 1, 'sad': 2, 'neutral': 3, 'ps': 4, 'happy': 5}
            else:
                raise TypeError("Regression is only for either ['sad', 'neutral', 'happy'] or ['angry', 'sad', 'neutral', 'ps', 'happy']")
            emotions = [ self.categories[e] for e in emotions ]
        # make features folder if does not exist
        if not os.path.isdir(self.features_folder_name):
            os.mkdir(self.features_folder_name)
        # get label for features
        label = get_label(self.audio_config)
        # construct features file name
        n_samples = len(audio_paths)
        first_letters = get_first_letters(self.emotions)
        name = os.path.join(self.features_folder_name, f"{partition}_{label}_{first_letters}_{n_samples}.npy")
        if os.path.isfile(name):
            # if file already exists, just load then
            if self.verbose:
                print("[+] Feature file already exists, loading...")
            features = np.load(name)
        else:
            # file does not exist, extract those features and dump them into the file
            features = []
            append = features.append
            for audio_file in tqdm.tqdm(audio_paths, f"Extracting features for {partition}"):
                feature = extract_feature(audio_file, **self.audio_config)
                if self.input_dimension is None:
                    self.input_dimension = feature.shape[0]
                append(feature)
            # convert to numpy array
            features = np.array(features)
            # save it
            np.save(name, features)
        if partition == "train":
            try:
                self.train_audio_paths
            except AttributeError:
                self.train_audio_paths = audio_paths
                self.train_emotions = emotions
                self.train_features = features
            else:
                if self.verbose:
                    print("[*] Adding additional training samples")
                self.train_audio_paths += audio_paths
                self.train_emotions += emotions
                self.train_features = np.vstack((self.train_features, features))
        elif partition == "test":
            try:
                self.test_audio_paths
            except AttributeError:
                self.test_audio_paths = audio_paths
                self.test_emotions = emotions
                self.test_features = features
            else:
                if self.verbose:
                    print("[*] Adding additional testing samples")
                self.test_audio_paths += audio_paths
                self.test_emotions += emotions
                self.test_features = np.vstack((self.test_features, features))
        else:
            raise TypeError("Invalid partition, must be either train/test")

    def _balance_data(self, partition):
        if partition == "train":
            emotions = self.train_emotions
            features = self.train_features
            audio_paths = self.train_audio_paths
        elif partition == "test":
            emotions = self.test_emotions
            features = self.test_features
            audio_paths = self.test_audio_paths
        else:
            raise TypeError("Invalid partition, must be either train/test")
        
        count = []
        if self.classification:
            for emotion in self.emotions:
                count.append(len([ e for e in emotions if e == emotion]))
        else:
            # regression, take actual numbers, not label emotion
            for emotion in self.categories.values():
                count.append(len([ e for e in emotions if e == emotion]))
        # get the minimum data samples to balance to
        minimum = min(count)
        if minimum == 0:
            # won't balance, otherwise 0 samples will be loaded
            print("[!] One class has 0 samples, setting balance to False")
            self.balance = False
            return
        if self.verbose:
            print("[*] Balancing the dataset to the minimum value:", minimum)
        d = defaultdict(list)
        if self.classification:
            counter = {e: 0 for e in self.emotions }
        else:
            counter = { e: 0 for e in self.categories.values() }
        for emotion, feature, audio_path in zip(emotions, features, audio_paths):
            if counter[emotion] >= minimum:
                # minimum value exceeded
                continue
            counter[emotion] += 1
            d[emotion].append((feature, audio_path))

        emotions, features, audio_paths = [], [], []
        for emotion, features_audio_paths in d.items():
            for feature, audio_path in features_audio_paths:
                emotions.append(emotion)
                features.append(feature)
                audio_paths.append(audio_path)
        
        if partition == "train":
            self.train_emotions = emotions
            self.train_features = features
            self.train_audio_paths = audio_paths
        elif partition == "test":
            self.test_emotions = emotions
            self.test_features = features
            self.test_audio_paths = audio_paths
        else:
            raise TypeError("Invalid partition, must be either train/test")

    def balance_training_data(self):
        self._balance_data("train")

    def balance_testing_data(self):
        self._balance_data("test")
        

def shuffle_data(audio_paths, emotions, features):
    """ Shuffle the data (called after making a complete pass through 
        training or validation data during the training process)
    Params:
        audio_paths (list): Paths to audio clips
        emotions (list): Emotions in each audio clip
        features (list): features audio clips
    """
    p = np.random.permutation(len(audio_paths))
    audio_paths = [audio_paths[i] for i in p] 
    emotions = [emotions[i] for i in p]
    features = [features[i] for i in p]
    return audio_paths, emotions, features


def load_data(train_desc_files, test_desc_files, audio_config=None, classification=True, shuffle=True,
                balance=True, emotions=['sad', 'neutral', 'happy']):
    # instantiate the class
    audiogen = AudioExtractor(audio_config=audio_config, classification=classification, emotions=emotions,
                                balance=balance, verbose=0)
    # Loads training data
    audiogen.load_train_data(train_desc_files, shuffle=shuffle)
    # Loads testing data
    audiogen.load_test_data(test_desc_files, shuffle=shuffle)
    # X_train, X_test, y_train, y_test
    return {
        "X_train": np.array(audiogen.train_features),
        "X_test": np.array(audiogen.test_features),
        "y_train": np.array(audiogen.train_emotions),
        "y_test": np.array(audiogen.test_emotions),
        "train_audio_paths": audiogen.train_audio_paths,
        "test_audio_paths": audiogen.test_audio_paths,
        "balance": audiogen.balance,
    }

class EmotionRecognizer:
    """A class for training, testing and predicting emotions based on
    speech's features that are extracted and fed into `sklearn` or `keras` model"""
    def __init__(self, model=None, **kwargs):
        """
        Params:
            model (sklearn model): the model used to detect emotions. If `model` is None, then self.determine_best_model()
                will be automatically called
            emotions (list): list of emotions to be used. Note that these emotions must be available in
                RAVDESS_TESS & EMODB Datasets, available nine emotions are the following:
                    'neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps' ( pleasant surprised ), 'boredom'.
                Default is ["sad", "neutral", "happy"].
            tess_ravdess (bool): whether to use TESS & RAVDESS Speech datasets, default is True
            emodb (bool): whether to use EMO-DB Speech dataset, default is True,
            custom_db (bool): whether to use custom Speech dataset that is located in `data/train-custom`
                and `data/test-custom`, default is True
            tess_ravdess_name (str): the name of the output CSV file for TESS&RAVDESS dataset, default is "tess_ravdess.csv"
            emodb_name (str): the name of the output CSV file for EMO-DB dataset, default is "emodb.csv"
            custom_db_name (str): the name of the output CSV file for the custom dataset, default is "custom.csv"
            features (list): list of speech features to use, default is ["mfcc", "chroma", "mel"]
                (i.e MFCC, Chroma and MEL spectrogram )
            classification (bool): whether to use classification or regression, default is True
            balance (bool): whether to balance the dataset ( both training and testing ), default is True
            verbose (bool/int): whether to print messages on certain tasks, default is 1
        Note that when `tess_ravdess`, `emodb` and `custom_db` are set to `False`, `tess_ravdess` will be set to True
        automatically.
        """
        # emotions
        self.emotions = kwargs.get("emotions", ["sad", "neutral", "happy"])
        # make sure that there are only available emotions
        self._verify_emotions()
        # audio config
        self.features = kwargs.get("features", ["mfcc", "chroma", "mel"])
        self.audio_config = get_audio_config(self.features)
        # datasets
        self.tess_ravdess = kwargs.get("tess_ravdess", True)
        self.emodb = kwargs.get("emodb", True)
        self.custom_db = kwargs.get("custom_db", True)

        if not self.tess_ravdess and not self.emodb and not self.custom_db:
            self.tess_ravdess = True
    
        self.classification = kwargs.get("classification", True)
        self.balance = kwargs.get("balance", True)
        self.override_csv = kwargs.get("override_csv", True)
        self.verbose = kwargs.get("verbose", 1)

        self.tess_ravdess_name = kwargs.get("tess_ravdess_name", "tess_ravdess.csv")
        self.emodb_name = kwargs.get("emodb_name", "emodb.csv")
        self.custom_db_name = kwargs.get("custom_db_name", "custom.csv")

        self.verbose = kwargs.get("verbose", 1)

        # set metadata path file names
        self._set_metadata_filenames()
        # write csv's anyway
        self.write_csv()

        # boolean attributes
        self.data_loaded = False
        self.model_trained = False

        # model
        if not model:
            self.determine_best_model()
        else:
            self.model = model

    def _set_metadata_filenames(self):
        """
        Protected method to get all CSV (metadata) filenames into two instance attributes:
        - `self.train_desc_files` for training CSVs
        - `self.test_desc_files` for testing CSVs
        """
        train_desc_files, test_desc_files = [], []
        if self.tess_ravdess:
            train_desc_files.append(f"train_{self.tess_ravdess_name}")
            test_desc_files.append(f"test_{self.tess_ravdess_name}")
        if self.emodb:
            train_desc_files.append(f"train_{self.emodb_name}")
            test_desc_files.append(f"test_{self.emodb_name}")
        if self.custom_db:
            train_desc_files.append(f"train_{self.custom_db_name}")
            test_desc_files.append(f"test_{self.custom_db_name}")

        # set them to be object attributes
        self.train_desc_files = train_desc_files
        self.test_desc_files  = test_desc_files

    def _verify_emotions(self):
        """
        This method makes sure that emotions passed in parameters are valid.
        """
        for emotion in self.emotions:
            assert emotion in AVAILABLE_EMOTIONS, "Emotion not recognized."

    def get_best_estimators(self):
        """Loads estimators from grid files and returns them"""
        return get_best_estimators(self.classification)

    def write_csv(self):
        """
        Write available CSV files in `self.train_desc_files` and `self.test_desc_files`
        determined by `self._set_metadata_filenames()` method.
        """
        for train_csv_file, test_csv_file in zip(self.train_desc_files, self.test_desc_files):
            # not safe approach
            if os.path.isfile(train_csv_file) and os.path.isfile(test_csv_file):
                # file already exists, just skip writing csv files
                if not self.override_csv:
                    continue
            if self.emodb_name in train_csv_file:
                write_emodb_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file, verbose=self.verbose)
                if self.verbose:
                    print("[+] Generated EMO-DB CSV File")
            elif self.tess_ravdess_name in train_csv_file:
                write_tess_ravdess_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file, verbose=self.verbose)
                if self.verbose:
                    print("[+] Generated TESS & RAVDESS DB CSV File")
            elif self.custom_db_name in train_csv_file:
                write_custom_csv(emotions=self.emotions, train_name=train_csv_file, test_name=test_csv_file, verbose=self.verbose)
                if self.verbose:
                    print("[+] Generated Custom DB CSV File")

    def load_data(self):
        """
        Loads and extracts features from the audio files for the db's specified
        """
        if not self.data_loaded:
            result = load_data(self.train_desc_files, self.test_desc_files, self.audio_config, self.classification,
                                emotions=self.emotions, balance=self.balance)
            self.X_train = result['X_train']
            self.X_test = result['X_test']
            self.y_train = result['y_train']
            self.y_test = result['y_test']
            self.train_audio_paths = result['train_audio_paths']
            self.test_audio_paths = result['test_audio_paths']
            self.balance = result["balance"]
            if self.verbose:
                print("[+] Data loaded")
            self.data_loaded = True

    def train(self, verbose=1):
        """
        Train the model, if data isn't loaded, it 'll be loaded automatically
        """
        if not self.data_loaded:
            # if data isn't loaded yet, load it then
            self.load_data()
        if not self.model_trained:
            self.model.fit(X=self.X_train, y=self.y_train)
            self.model_trained = True
            if verbose:
                print("[+] Model trained")

    def predict(self, audio_path):
        """
        given an `audio_path`, this method extracts the features
        and predicts the emotion
        """
        feature = extract_feature(audio_path, **self.audio_config).reshape(1, -1)
        return self.model.predict(feature)[0]

    def predict_proba(self, audio_path):
        """
        Predicts the probability of each emotion.
        """
        if self.classification:
            feature = extract_feature(audio_path, **self.audio_config).reshape(1, -1)
            proba = self.model.predict_proba(feature)[0]
            result = {}
            for emotion, prob in zip(self.model.classes_, proba):
                result[emotion] = prob
            return result
        else:
            raise NotImplementedError("Probability prediction doesn't make sense for regression")

    def grid_search(self, params, n_jobs=2, verbose=1):
        """
        Performs GridSearchCV on `params` passed on the `self.model`
        And returns the tuple: (best_estimator, best_params, best_score).
        """
        score = accuracy_score if self.classification else mean_absolute_error
        grid = GridSearchCV(estimator=self.model, param_grid=params, scoring=make_scorer(score),
                            n_jobs=n_jobs, verbose=verbose, cv=3)
        grid_result = grid.fit(self.X_train, self.y_train)
        return grid_result.best_estimator_, grid_result.best_params_, grid_result.best_score_

    def determine_best_model(self):
        """
        Loads best estimators and determine which is best for test data,
        and then set it to `self.model`.
        In case of regression, the metric used is MSE and accuracy for classification.
        Note that the execution of this method may take several minutes due
        to training all estimators (stored in `grid` folder) for determining the best possible one.
        """
        if not self.data_loaded:
            self.load_data()
        
        # loads estimators
        estimators = self.get_best_estimators()

        result = []

        if self.verbose:
            estimators = tqdm.tqdm(estimators)

        for estimator, params, cv_score in estimators:
            if self.verbose:
                estimators.set_description(f"Evaluating {estimator.__class__.__name__}")
            detector = EmotionRecognizer(estimator, emotions=self.emotions, tess_ravdess=self.tess_ravdess,
                                        emodb=self.emodb, custom_db=self.custom_db, classification=self.classification,
                                        features=self.features, balance=self.balance, override_csv=False)
            # data already loaded
            detector.X_train = self.X_train
            detector.X_test  = self.X_test
            detector.y_train = self.y_train
            detector.y_test  = self.y_test
            detector.data_loaded = True
            # train the model
            detector.train(verbose=0)
            # get test accuracy
            accuracy = detector.test_score()
            # append to result
            result.append((detector.model, accuracy))

        # sort the result
        # regression: best is the lower, not the higher
        # classification: best is higher, not the lower
        result = sorted(result, key=lambda item: item[1], reverse=self.classification)
        best_estimator = result[0][0]
        accuracy = result[0][1]
        self.model = best_estimator
        self.model_trained = True
        if self.verbose:
            if self.classification:
                print(f"[+] Best model determined: {self.model.__class__.__name__} with {accuracy*100:.3f}% test accuracy")
            else:
                print(f"[+] Best model determined: {self.model.__class__.__name__} with {accuracy:.5f} mean absolute error")

    def test_score(self):
        """
        Calculates score on testing data
        if `self.classification` is True, the metric used is accuracy,
        Mean-Squared-Error is used otherwise (regression)
        """
        y_pred = self.model.predict(self.X_test)
        if self.classification:
            return accuracy_score(y_true=self.y_test, y_pred=y_pred)
        else:
            return mean_squared_error(y_true=self.y_test, y_pred=y_pred)

    def train_score(self):
        """
        Calculates accuracy score on training data
        if `self.classification` is True, the metric used is accuracy,
        Mean-Squared-Error is used otherwise (regression)
        """
        y_pred = self.model.predict(self.X_train)
        if self.classification:
            return accuracy_score(y_true=self.y_train, y_pred=y_pred)
        else:
            return mean_squared_error(y_true=self.y_train, y_pred=y_pred)

    def train_fbeta_score(self, beta):
        y_pred = self.model.predict(self.X_train)
        return fbeta_score(self.y_train, y_pred, beta, average='micro')

    def test_fbeta_score(self, beta):
        y_pred = self.model.predict(self.X_test)
        return fbeta_score(self.y_test, y_pred, beta, average='micro')

    def confusion_matrix(self, percentage=True, labeled=True):
        """
        Computes confusion matrix to evaluate the test accuracy of the classification
        and returns it as numpy matrix or pandas dataframe (depends on params).
        params:
            percentage (bool): whether to use percentage instead of number of samples, default is True.
            labeled (bool): whether to label the columns and indexes in the dataframe.
        """
        if not self.classification:
            raise NotImplementedError("Confusion matrix works only when it is a classification problem")
        y_pred = self.model.predict(self.X_test)
        matrix = confusion_matrix(self.y_test, y_pred, labels=self.emotions).astype(np.float32)
        if percentage:
            for i in range(len(matrix)):
                matrix[i] = matrix[i] / np.sum(matrix[i])
            # make it percentage
            matrix *= 100
        if labeled:
            matrix = pd.DataFrame(matrix, index=[ f"true_{e}" for e in self.emotions ],
                                    columns=[ f"predicted_{e}" for e in self.emotions ])
        return matrix

    def draw_confusion_matrix(self):
        """Calculates the confusion matrix and shows it"""
        matrix = self.confusion_matrix(percentage=False, labeled=False)
        #TODO: add labels, title, legends, etc.
        pl.imshow(matrix, cmap="binary")
        pl.show()

    def get_n_samples(self, emotion, partition):
        """Returns number data samples of the `emotion` class in a particular `partition`
        ('test' or 'train')
        """
        if partition == "test":
            return len([y for y in self.y_test if y == emotion])
        elif partition == "train":
            return len([y for y in self.y_train if y == emotion])

    def get_samples_by_class(self):
        """
        Returns a dataframe that contains the number of training 
        and testing samples for all emotions.
        Note that if data isn't loaded yet, it'll be loaded
        """
        if not self.data_loaded:
            self.load_data()
        train_samples = []
        test_samples = []
        total = []
        for emotion in self.emotions:
            n_train = self.get_n_samples(emotion, "train")
            n_test = self.get_n_samples(emotion, "test")
            train_samples.append(n_train)
            test_samples.append(n_test)
            total.append(n_train + n_test)
        
        # get total
        total.append(sum(train_samples) + sum(test_samples))
        train_samples.append(sum(train_samples))
        test_samples.append(sum(test_samples))
        return pd.DataFrame(data={"train": train_samples, "test": test_samples, "total": total}, index=self.emotions + ["total"])

    def get_random_emotion(self, emotion, partition="train"):
        """
        Returns random `emotion` data sample index on `partition`.
        """
        if partition == "train":
            index = random.choice(list(range(len(self.y_train))))
            while self.y_train[index] != emotion:
                index = random.choice(list(range(len(self.y_train))))
        elif partition == "test":
            index = random.choice(list(range(len(self.y_test))))
            while self.y_train[index] != emotion:
                index = random.choice(list(range(len(self.y_test))))
        else:
            raise TypeError("Unknown partition, only 'train' or 'test' is accepted")

        return index


def plot_histograms(classifiers=True, beta=0.5, n_classes=3, verbose=1):
    """
    Loads different estimators from `grid` folder and calculate some statistics to plot histograms.
    Params:
        classifiers (bool): if `True`, this will plot classifiers, regressors otherwise.
        beta (float): beta value for calculating fbeta score for various estimators.
        n_classes (int): number of classes
    """
    # get the estimators from the performed grid search result
    estimators = get_best_estimators(classifiers)

    final_result = {}
    for estimator, params, cv_score in estimators:
        final_result[estimator.__class__.__name__] = []
        for i in range(3):
            result = {}
            # initialize the class
            detector = EmotionRecognizer(estimator, verbose=0)
            # load the data
            detector.load_data()
            if i == 0:
                # first get 1% of sample data
                sample_size = 0.01
            elif i == 1:
                # second get 10% of sample data
                sample_size = 0.1
            elif i == 2:
                # last get all the data
                sample_size = 1
            # calculate number of training and testing samples
            n_train_samples = int(len(detector.X_train) * sample_size)
            n_test_samples = int(len(detector.X_test) * sample_size)
            # set the data
            detector.X_train = detector.X_train[:n_train_samples]
            detector.X_test = detector.X_test[:n_test_samples]
            detector.y_train = detector.y_train[:n_train_samples]
            detector.y_test = detector.y_test[:n_test_samples]
            # calculate train time
            t_train = time()
            detector.train()
            t_train = time() - t_train
            # calculate test time
            t_test = time()
            test_accuracy = detector.test_score()
            t_test = time() - t_test
            # set the result to the dictionary
            result['train_time'] = t_train
            result['pred_time'] = t_test
            result['acc_train'] = cv_score
            result['acc_test'] = test_accuracy
            result['f_train'] = detector.train_fbeta_score(beta)
            result['f_test'] = detector.test_fbeta_score(beta)
            if verbose:
                print(f"[+] {estimator.__class__.__name__} with {sample_size*100}% ({n_train_samples}) data samples achieved {cv_score*100:.3f}% Validation Score in {t_train:.3f}s & {test_accuracy*100:.3f}% Test Score in {t_test:.3f}s")
            # append the dictionary to the list of results
            final_result[estimator.__class__.__name__].append(result)
        if verbose:
            print()
    visualize(final_result, n_classes=n_classes)
    


def visualize(results, n_classes):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - results: a dictionary of lists of dictionaries that contain various results on the corresponding estimator
      - n_classes: number of classes
    """

    n_estimators = len(results)

    # naive predictor
    accuracy = 1 / n_classes
    f1 = 1 / n_classes
    # Create figure
    fig, ax = pl.subplots(2, 4, figsize = (11,7))
    # Constants
    bar_width = 0.4
    colors = [ (random.random(), random.random(), random.random()) for _ in range(n_estimators) ]
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                x = bar_width * n_estimators
                # Creative plot code
                ax[j//3, j%3].bar(i*x+k*(bar_width), results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([x-0.2, x*2-0.2, x*3-0.2])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.2, x*3))
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))
    # Set additional plots invisibles
    ax[0, 3].set_visible(False)
    ax[1, 3].axis('off')
    # Create legend
    for i, learner in enumerate(results.keys()):
        pl.bar(0, 0, color=colors[i], label=learner)
    pl.legend()
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()


# emotion classes you want to perform grid search on
emotions = ['sad', 'neutral', 'happy']
# number of parallel jobs during the grid search
n_jobs = 4

best_estimators = []

for model, params in classification_grid_parameters.items():
    if model.__class__.__name__ == "KNeighborsClassifier":
        # in case of a K-Nearest neighbors algorithm
        # set number of neighbors to the length of emotions
        params['n_neighbors'] = [len(emotions)]
    d = EmotionRecognizer(model, emotions=emotions)
    d.load_data()
    best_estimator, best_params, cv_best_score = d.grid_search(params=params, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions} {best_estimator.__class__.__name__} achieved {cv_best_score:.3f} cross validation accuracy score!")

print(f"[+] Pickling best classifiers for {emotions}...")
pickle.dump(best_estimators, open(f"grid/best_classifiers.pickle", "wb"))

best_estimators = []

for model, params in regression_grid_parameters.items():
    if model.__class__.__name__ == "KNeighborsRegressor":
        # in case of a K-Nearest neighbors algorithm
        # set number of neighbors to the length of emotions
        params['n_neighbors'] = [len(emotions)]
    d = EmotionRecognizer(model, emotions=emotions, classification=False)
    d.load_data()
    best_estimator, best_params, cv_best_score = d.grid_search(params=params, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions} {best_estimator.__class__.__name__} achieved {cv_best_score:.3f} cross validation MAE score!")

print(f"[+] Pickling best regressors for {emotions}...")
pickle.dump(best_estimators, open(f"grid/best_regressors.pickle", "wb"))



# Best for SVC: C=0.001, gamma=0.001, kernel='poly'
# Best for AdaBoostClassifier: {'algorithm': 'SAMME', 'learning_rate': 0.8, 'n_estimators': 60}
# Best for RandomForestClassifier: {'max_depth': 7, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 40}
# Best for GradientBoostingClassifier: {'learning_rate': 0.3, 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 70, 'subsample': 0.7}
# Best for DecisionTreeClassifier: {'criterion': 'entropy', 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
# Best for KNeighborsClassifier: {'n_neighbors': 5, 'p': 1, 'weights': 'distance'}
# Best for MLPClassifier: {'alpha': 0.005, 'batch_size': 256, 'hidden_layer_sizes': (300,), 'learning_rate': 'adaptive', 'max_iter': 500}
