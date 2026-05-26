import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import seaborn as sns
import logging
from logging_code import setup_logging
logger= setup_logging("var_trans")
import sys
from scipy.stats import yeojohnson
from scipy import stats
from sklearn.preprocessing import QuantileTransformer

def v_trans (x_train_num ,x_test_num):
    try:
        logger.info("=== VARIABLE TRANSFORMATION STARTED ===")
        logger.info("=== VARIABLE TRANSFORMATION STARTED ===")
        logger.info(f"Before Train Column Name : {x_train_num.shape} and columns: {x_train_num.columns}")
        logger.info(f"Before Test Column Name : {x_test_num.shape} and columns: {x_train_num.columns}")

        def fun(data,var):

                plt.figure(figsize=(10, 3))
                plt.subplot(1, 3, 1)
                plt.title("Outliers")
                sns.boxplot(x=data[var])

                plt.subplot(1, 3, 2)
                plt.title("Normal Distribution")
                data[var].plot(kind='kde')

                plt.subplot(1, 3, 3)
                plt.title("probplot")
                stats.probplot(data[var], dist="norm", plot=plt)
                plt.show()



        for i in  x_train_num.columns:

            if 'SeniorCitizen' != i:
                if i=='tenure':

                    x_train_num[i+'_yeo'], lam = yeojohnson(x_train_num[i])
                    x_test_num[i+'_yeo'] = yeojohnson(x_test_num[i], lmbda=lam)
                    x_train_num = x_train_num.drop(i, axis=1)
                    x_test_num = x_test_num.drop(i, axis=1)
                    fun(x_train_num, i+'_yeo')

                else:
                    qt = QuantileTransformer(output_distribution='normal', random_state=0)
                    x_train_num[i+'_qt'] = qt.fit_transform(x_train_num[[i]]).ravel()
                    x_test_num[i+'_qt'] = qt.fit_transform(x_test_num[[i]]).ravel()
                    x_train_num = x_train_num.drop(i, axis=1)
                    x_test_num = x_test_num.drop(i, axis=1)
                    fun(x_train_num, i+'_qt')

        logger.info(f"After Train Column Name : {x_train_num.shape} and columns: {x_train_num.columns}")
        logger.info(f"After Test Column Name : {x_test_num.shape} and columns: {x_train_num.columns}")

        return x_train_num , x_test_num

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

        return x_train_num, x_test_num



def hanling_outliers(x_train_num , x_test_num):
    try:
        logger.info("=== handling outliers STARTED ===")

        logger.info(f"Before Train Column Name : {x_train_num.columns}")
        logger.info(f"Before Test Column Name : {x_test_num.columns}")

        def fun1(data,var):
            plt.figure(figsize=(10, 3))
            sns.boxplot(x=data[var])
            plt.title(var)
            plt.show()

        for col in x_train_num.columns:
            if 'SeniorCitizen' != col:
                Q1 = x_train_num[col].quantile(0.25)
                Q3 = x_train_num[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_limit = Q1 - 1.5 * IQR
                upper_limit = Q3 + 1.5 * IQR
                x_train_num[col+'_trim'] = np.where(
                    x_train_num[col] > upper_limit, upper_limit,
                    np.where(x_train_num[col] < lower_limit, lower_limit, x_train_num[col])
                )
                x_test_num[col+'_trim'] = np.where(
                    x_test_num[col] > upper_limit, upper_limit,
                    np.where(x_test_num[col] < lower_limit, lower_limit, x_test_num[col])
                )

                fun1(x_train_num, col+'_trim')
                x_train_num = x_train_num.drop(col, axis=1)
                x_test_num = x_test_num.drop(col, axis=1)

        logger.info(f"After Train Columns: {x_train_num.columns}")
        logger.info(f"After Test Columns: {x_test_num.columns}")
        logger.info("=== handling outliers completed ===")

        return x_train_num, x_test_num


    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

        return x_train_num, x_test_num
