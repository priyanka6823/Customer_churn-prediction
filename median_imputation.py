import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import logging
import warnings
warnings.filterwarnings("ignore")
from logging_code import setup_logging
logger = setup_logging("median_imputation")
def handle_missing_value(x_train, x_test):


    try:

        logger.info(f"Before Handling Null values x_train shape: {x_train.shape}")
        logger.info(f"x_train Missing:\n{x_train.isnull().sum()}")

        logger.info(f"Before Handling Null values x_test shape: {x_test.shape}")
        logger.info(f"x_test Missing:\n{x_test.isnull().sum()}")


        num_cols = x_train.select_dtypes(exclude='object')
        cat_cols = x_train.select_dtypes(include='object')


        for i in num_cols:
            if x_train[i].isnull().sum() > 0:
                #median_value = x_train[col].median()
                x_train[i + '_median'] = x_train[i].median()
                x_test[i + '_median'] = x_test[i].median()
                x_train = x_train.drop([i], axis=1)
                x_test = x_test.drop([i], axis=1)

                #x_train[col] = X_train[col].fillna(median_value)
                #x_test[col] = X_test[col].fillna(median_value)

                #logger.info(f"Filled Numerical Column '{col}' with Median: {median_value}")




        logger.info(f"After Handling Null values x_train shape: {x_train.shape}")
        logger.info(f"x_train Missing:\n{x_train.isnull().sum()}")

        logger.info(f"After Handling Null values x_test shape: {x_test.shape}")
        logger.info(f"x_test Missing:\n{x_test.isnull().sum()}")

        return x_train, x_test

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error in line no: {er_line.tb_lineno} due to: {er_msg}")