import pandas as pd
import numpy as np
import logging
import sys
import sklearn
from logging_code import setup_logging
logger = setup_logging("filter_method")
from sklearn.feature_selection import VarianceThreshold
def filter_features(x_train, x_test, threshold=0.99):
    try:
        # -----------------------------
        # BEFORE LOGGING
        # -----------------------------
        logger.info(f"BEFORE Train Shape: {x_train.shape}")
        logger.info(f"BEFORE Test Shape: {x_test.shape}")
        logger.info(f"BEFORE Train Columns: {list(x_train.columns)}")

        # -----------------------------
        # 1. Constant Features
        # -----------------------------
        constant_cols = [col for col in x_train.columns if x_train[col].nunique() == 1]
        logger.info(f"Constant Columns Removed: {constant_cols}")

        x_train = x_train.drop(columns=constant_cols)
        x_test = x_test.drop(columns=constant_cols)

        # -----------------------------
        # 2. Quasi-Constant
        # -----------------------------
        quasi_cols = []
        for col in x_train.columns:
            top_freq = x_train[col].value_counts(normalize=True).values[0]
            if top_freq > threshold:
                quasi_cols.append(col)

        logger.info(f"Quasi-Constant Columns Removed: {quasi_cols}")

        x_train = x_train.drop(columns=quasi_cols)
        x_test = x_test.drop(columns=quasi_cols)

        # -----------------------------
        # 3. Duplicate Columns
        # -----------------------------
        duplicate_mask = x_train.T.duplicated()
        duplicate_cols = x_train.columns[duplicate_mask]

        logger.info(f"Duplicate Columns Removed: {list(duplicate_cols)}")

        x_train = x_train.drop(columns=duplicate_cols)
        x_test = x_test.drop(columns=duplicate_cols)

        # -----------------------------
        # 4. Correlation
        # -----------------------------
        corr_matrix = x_train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        corr_cols = [col for col in upper.columns if any(upper[col] > 0.85)]
        logger.info(f"Highly Correlated Columns Removed: {corr_cols}")

        x_train = x_train.drop(columns=corr_cols)
        x_test = x_test.drop(columns=corr_cols)

        # -----------------------------
        # AFTER LOGGING
        # -----------------------------
        logger.info(f"AFTER Train Shape: {x_train.shape}")
        logger.info(f"AFTER Test Shape: {x_test.shape}")
        logger.info(f"AFTER Train Columns: {list(x_train.columns)}")

        return x_train, x_test

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error in line no: {er_line.tb_lineno} due to: {er_msg}")
