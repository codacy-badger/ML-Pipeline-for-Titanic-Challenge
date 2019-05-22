"""
Summary:     A collections of functions to generate features.
Description:
Author:      Kunyu He, CAPP'20
"""

import logging
import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.externals import joblib


#----------------------------------------------------------------------------#
INPUT_DIR = "../data/"
OUTPUT_DIR = "../processed_data/"
LOG_DIR = "../logs/featureEngineering/"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# logging
logger= logging.getLogger('featureEngineering')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
logger.addHandler(ch)
fh = logging.FileHandler(LOG_DIR + time.strftime("%Y%m%d-%H%M%S") + '.log')
logger.addHandler(fh)

pd.set_option('mode.chained_assignment', None)


#----------------------------------------------------------------------------#
def read_data(file_name, drop_na=False):
    """
    Read credit data in the .csv file and data types from the .json file.

    Inputs:
        - data_file (string): name of the data file.
        - drop_na (bool): whether to drop rows with any missing values

    Returns:
        (DataFrame) clean data set with correct data types

    """
    data = pd.read_csv(INPUT_DIR + file_name)

    if drop_na:
        data.dropna(axis=0, inplace=True)

    return data


def ask():
    """
    Ask user for choice of an imputer and a scaler.

    """
    scaler_index = int(input(("\nUp till now we support:\n"
                              "\t1. StandardScaler\n"
                              "\t2. MinMaxScaler\n"
                              "Please input a scaler index (1 or 2):\n")))

    return scaler_index


class FeaturePipeLine:
    """
    Preprocess pipeline for a data set from CSV file. Modify the class
    variables to fill in missing values, combine multinomial variables to ones
    with less levels and binaries, and apply one-hot-encoding. Then split data
    into features and traget, drop rows with missing labels and some columns.
    At last, apply scaling.

    """
    TO_DESCRETIZE = {'Age': 5}
    RIGHT_INCLUSIVE = {'Age': True}

    TO_FILL_NA = {'Cabin': "None",
                  'Embarked': "Unknown"}

    TO_COMBINE = {}
    TO_BINARIES = {'Sex': 'auto',
                   'Cabin': 'auto'}
    TO_ONE_HOT = {'Pclass', 'Embarked', 'Age'}

    TARGET = 'Survived'
    TO_DROP = ['PassengerId', 'Ticket', 'Name']

    SCALERS = [StandardScaler, MinMaxScaler]
    SCALER_NAMES = ["Standard Scaler", "MinMax Scaler"]

    def __init__(self, file_name, ask_user=True, verbose=True,
                 drop_na=False, test=False):
        """
        Construct a preprocessing pipeline given name of the data file.

        Inputs:
            - file_name (string): name of the data file
            - verbose (bool): whether to make extended printing in
                preprocessing
            - drop_na (bool): whether to drop rows with missing values

        """
        logger.info("**-----------------------------------------------**")
        logger.info("Creating the preprocessing pipeline for '{}'.".format(\
            file_name))
        self.data = read_data(file_name, drop_na)
        self.verbose = verbose
        self.test = test
        logger.info("Finished reading cleaned data.")

        if not self.test:
            if ask_user:
                self.scaler_index = ask() - 1
            else:
                self.scaler_index = 0
            self.scaler = self.SCALERS[self.scaler_index]()
            logger.info("Training data preprocessing. Pipeline using {}.".\
                        format(self.SCALER_NAMES[self.scaler_index]))
        else:
            self.scaler = joblib.load(INPUT_DIR + 'fitted_scaler.pkl')
            logger.info("Test data preprocessing. Pre-fitted scaler loaded.")

        self.X = None
        self.y = None

    def discretize(self):
        """
        Discretizes continuous variables into multinomials.

        """
        logger.info("\n\n**-----------------------------------------------**")
        logger.info("Start to discretizes continuous variables:")

        for var, n in self.TO_DESCRETIZE.items():
            self.data[var] = pd.cut(self.data[var], n,
                                    right=self.RIGHT_INCLUSIVE[var]).cat.codes

            if self.verbose:
                if not self.data[var].isnull().sum():
                    logger.info(("\tThere are missing values in '{}', "
                        "discretized it into {} bins, where '-1' indicates "
                        "that the value is missing.".format(var, n + 1)))
                else:
                    logger.info("\tDiscretized '{}' into {} bins.".\
                                format(var, n))

        return self

    def fill_na(self):
        """
        Fill in missing data with desired entry.

        """
        logger.info("\n\n**-----------------------------------------------**")
        logger.info("Start to fill in missing values:")

        for var, fill in self.TO_FILL_NA.items():
            self.data[var].fillna(value=fill, inplace=True)

            if self.verbose:
                logger.info("\tFilled missing values in '{}' with '{}'.".\
                      format(var, fill))

            if fill == "None":
                to_combine = [col for col in list(self.data[var].unique())
                              if col != "None"]
                self.TO_COMBINE[var] = {"Yes": to_combine}
                logger.info("\t\t'{}' added to 'TO_COMBINE'".format(var))

        return self

    def to_combine(self):
        """
        Combine some unecessary levels of multinomials.

        """
        logger.info("\n\n**-----------------------------------------------**")
        logger.info("Start to combine unnecessary levels of multinomials.")

        for var, dict_combine in self.TO_COMBINE.items():
            for combined, lst_combine in dict_combine.items():
                self.data.loc[self.data[var].isin(lst_combine), var] = combined

            if self.verbose:
                logger.info("\tCombinations of levels on '{}'.".format(var))

        return self

    def to_binary(self):
        """
        Trasform variables to binaries.

        """
        logger.info("\n\n**-----------------------------------------------**")
        logger.info(("Start to transform the following variables: {} to "
                     "binaries.").format(list(self.TO_BINARIES.keys())))

        for var, cats in self.TO_BINARIES.items():
            enc = OrdinalEncoder(categories=cats)
            self.data[var] = enc.fit_transform(np.array(self.data[var]).\
                                               reshape(-1, 1))

        return self

    def one_hot(self):
        """
        Ccreates binary/dummy variables from multinomials, drops the original
        and inserts the dummies back.

        """
        logger.info("\n\n**-----------------------------------------------**")
        logger.info(("Start to apply one-hot-encoding to the following "
                     "categorical variables: {}\n").format(self.TO_ONE_HOT))

        for var in self.TO_ONE_HOT:
            dummies = pd.get_dummies(self.data[var], prefix=var)
            self.data.drop(var, axis=1, inplace=True)
            self.data = pd.concat([self.data, dummies], axis=1)

        return self

    def split(self):
        """
        Drop rows with missing labels, drop some columns that are not relevant
        or have too many missing values, split the features (X) and targert (y)
        Write columns names to "feature_names.txt" in the output directory.

        """
        logger.info("\n\n**-----------------------------------------------**")

        self.data.dropna(axis=0, subset=[self.TARGET], inplace=True)
        self.y = self.data[self.TARGET]
        self.data.drop(self.TO_DROP, axis=1, inplace=True)
        self.X = self.data
        logger.info("Finished extracting the features (X) and targert (y).")

        file_name = 'feature_names.txt'
        with open(OUTPUT_DIR + file_name, 'w') as file:
            file.write(",".join(self.X.columns))
            logger.info("Feature names wrote to '{}' under directory '{}'".\
                        format(file_name, OUTPUT_DIR))

    def scale(self):
        """
        Fit and transform the scaler on the training data and return the
        scaler data to scale test data.

        """
        logger.info("\n\n**-----------------------------------------------**")

        if not self.test:
            self.scaler.fit(self.X.values.astype(float))
            joblib.dump(self.scaler, INPUT_DIR + 'fitted_scaler.pkl')
            logger.info(("Training data preprocessing. Fitted scaler dumped to "
                         "'{}' under directory '{}'.").format('fitted_scaler.pkl',
                                                              INPUT_DIR))

        self.X = self.scaler.transform(self.X.values.astype(float))
        logger.info("Finished scaling the feature matrix.")

        return self

    def save_data(self):
        """
        Saves the feature matrix and target as numpy arrays in the output
        directory.

        """
        logger.info("\n\n**-----------------------------------------------**")
        extension = ["_train.npz", "_test.npz"][int(self.test)]

        np.savez(OUTPUT_DIR + "X" + extension, data=self.X)
        np.savez(OUTPUT_DIR + "y" + extension, data=self.y.values.astype(float))

        logger.info(("Saved the resulting NumPy matrices to directory '{}'. "
                     "Features are in 'X{}' and target is in 'y{}'.").\
                     format(OUTPUT_DIR, extension, extension))

    def preprocess(self):
        """
        Finish preprocessing the data file.

        """
        self.discretize().fill_na().to_combine().to_binary().one_hot().split()
        self.scale().save_data()

        logger.info("Finished procssing {} data.".\
                    format(["training", "test"][int(self.test)]))
        logger.info("\n*****************************************************")
        logger.info("*****************************************************\n")


#----------------------------------------------------------------------------#
if __name__ == "__main__":

    training_pipeline = FeaturePipeLine(TRAIN_FILE, test=False)
    training_pipeline.preprocess()

    test_pipeline = FeaturePipeLine(TRAIN_FILE, test=True)
    test_pipeline.preprocess()
