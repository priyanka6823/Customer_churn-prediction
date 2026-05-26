import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import os
import seaborn as sns
import logging
import pickle
import sys

from logging_code import setup_logging
logger = setup_logging("feature_scaling")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from all_models import common


def fs(x_train, y_train, x_test, y_test):
    try:
        logger.info("=== FEATURE SCALING STARTED ===")
        logger.info(f"x_train shape: {x_train.shape}, columns: {list(x_train.columns)}")
        logger.info(f"x_test shape: {x_test.shape}, columns: {list(x_test.columns)}")

        # ------------------ SCALING ------------------
        sc = StandardScaler() # x - mean / std
        sc.fit(x_train)

        x_train_sc = sc.transform(x_train)
        x_test_sc = sc.transform(x_test)

        #  Convert to DataFrame (IMPORTANT FIX)
        x_train_sc = pd.DataFrame(x_train_sc, columns=x_train.columns, index=x_train.index)
        x_test_sc = pd.DataFrame(x_test_sc, columns=x_test.columns, index=x_test.index)

        logger.info(f"x_train AFTER scaling shape and columns: {x_train_sc.shape}:\n : {x_train_sc.columns}")
        logger.info(f"x_test AFTER scaling shape: {x_test_sc.shape}:\n : {x_test_sc.columns}")

        # ------------------ SAVE SCALER ------------------
        os.makedirs("models", exist_ok=True)

        with open('scaler.pkl', 'wb') as f:
            pickle.dump(sc, f)

        logger.info("Scaler saved as scaler.pkl")

        # ------------------ RUN MULTIPLE MODELS ------------------
        logger.info("=========== TRAINING ALL MODELS ===========")
        common(x_train_sc, y_train, x_test_sc, y_test)

        # ------------------ LOGISTIC REGRESSION ------------------
        reg = LogisticRegression(max_iter=1000)
        reg.fit(x_train_sc, y_train)

        y_pred = reg.predict(x_test_sc)

        # ------------------ SAVE MODEL ------------------
        with open('model.pkl', 'wb') as f:
            pickle.dump(reg, f)

        # ------------------ METRICS ------------------
        logger.info(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")
        logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        logger.info("=== FEATURE SCALING COMPLETED ===")

        return x_train_sc, y_train, x_test_sc, y_test

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error in line no: {er_line.tb_lineno} due to: {er_msg}")

        return x_train, y_train, x_test, y_test