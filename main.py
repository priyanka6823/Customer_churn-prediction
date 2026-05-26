'''
In this file I am going to call all related functions for data cleaning and model
development
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import seaborn as sns
import logging
from logging_code import setup_logging
logger = setup_logging("main")
import sys
from sklearn.model_selection import train_test_split
from median_imputation import handle_missing_value
from var_tras import  v_trans
from var_tras import hanling_outliers
from ca_to_num import cat_to_num
from filter_method import filter_features
from imblearn.over_sampling import SMOTE
from feature_scaling import fs
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from all_models import common


class TELCO:

        def __init__(self,path):
            try:
                self.path = path
                self.df = pd.read_csv(self.path)
                logger.info(f"Total data size: {self.df.shape}")
                self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
                logger.info(f"Null values:\n{self.df.isnull().sum()}")
                self.df.drop(['customerID'], axis=1, inplace=True)
                self.df['Churn']=self.df['Churn'].map({'Yes':1,'No':0})
                self.x=self.df.drop('Churn',axis=1)
                self.y=self.df['Churn']
                self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size=0.2,random_state=45)
                logger.info(f"Train data size: {len(self.x_train)}:{len(self.y_train)}\n:total train data: {self.x_train.shape}")
                logger.info(f"Test data size: {len(self.x_test)}:{len(self.y_test)}\n:total test data: {self.x_test.shape}")
            except Exception as e:
                er_type, er_msg, er_line = sys.exc_info()
                logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

        def missing_values(self):
            try:
                 logger.info("=== MISSING VALUE HANDLING ===")

                 logger.info(f" Before Handling Null values x_train columns names and shape:\n{self.x_train.shape}:{self.x_train.columns}:{self.x_train.isnull().sum()}")
                 logger.info(f" Before Handling Null values x_test columns names and shape:\n{self.x_test.shape}\n:{self.x_train.columns}:{self.x_test.isnull().sum()}")

                 self.x_train, self.x_test = handle_missing_value(self.x_train, self.x_test)

                 logger.info(f"After Handling NUll values x_train Columns names and shape : {self.x_train.shape} \n : {self.x_train.columns} : {self.x_train.isnull().sum()}")
                 logger.info(f"After Handling NUll values x_test Columns names and shape : {self.x_test.shape} \n : {self.x_test.columns} : {self.x_test.isnull().sum()}")
            except Exception as e:
                er_type, er_msg, er_line = sys.exc_info()
                logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
        def data_seperation(self):
            try:
                logger.info("=== DATA SEPARATION ===")
                self.x_train_num = self.x_train.select_dtypes(exclude='object')
                self.x_test_num = self.x_test.select_dtypes(exclude='object')

                self.x_train_cat = self.x_train.select_dtypes(include='object')
                self.x_test_cat = self.x_test.select_dtypes(include='object')
                #logger.info(f"{self.x_train_num}:\n{self.x_train_num.shape}")
                #logger.info(f"{self.x_test_num}:\n{self.x_test_num.shape}")
                #logger.info(f"=========================================================")
                #logger.info(f"{self.x_train_cat}:\n{self.x_train_cat.shape}")
                #logger.info(f"{self.x_test_cat}:\n{self.x_test_cat.shape}")
                logger.info(f"Numerical Columns ({len(self.x_train_num.columns)}): {list(self.x_train_num.columns)}")
                logger.info(f"Numerical Shape: {self.x_train_num.shape}")

                logger.info(f"Categorical Columns ({len(self.x_train_cat.columns)}): {list(self.x_train_cat.columns)}")
                logger.info(f"Categorical Shape: {self.x_train_cat.shape}")

            except Exception as e:
                er_type, er_msg, er_line = sys.exc_info()
                logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

        def variable_transformation(self):
                    try:
                        logger.info("===  VARIABLE TRANSFORMATION ===")
                        logger.info(f"Train rows BEFORE: {self.x_train_num.shape} : \n : {self.x_train_num.columns}")
                        logger.info(f"Test rows BEORE: {self.x_test_num.shape}: \n : {self.x_train_num.columns}")

                        self.x_train_num, self.x_test_num = v_trans(self.x_train_num,self.x_test_num)

                        logger.info(f"Train rows AFTER : {self.x_train_num.shape} : \n : {self.x_train_num.columns}")
                        logger.info(f"Test rows AFTER : {self.x_test_num.shape} : \n : {self.x_train_num.columns}")
                        logger.info("=== VARIABLE TRANSFORMATION COMPLETED ===")

                        logger.info("===  handling outliers ===")
                        logger.info(f"Train rows BEFORE: {self.x_train_num.shape} : \n : {self.x_train_num.columns}")
                        logger.info(f"Test rows BEORE: {self.x_test_num.shape}: \n : {self.x_train_num.columns}")

                        self.x_train_num, self.x_test_num = hanling_outliers(self.x_train_num, self.x_test_num)

                        logger.info(f"Train rows AFTER : {self.x_train_num.shape} : \n : {self.x_train_num.columns}")
                        logger.info(f"Test rows AFTER : {self.x_test_num.shape} : \n : {self.x_train_num.columns}")
                        logger.info("=== handling outliers COMPLETED ===")

                    except Exception as e:
                        er_type, er_msg, er_line = sys.exc_info()
                        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

        def categorical_numerical(self):

            try:
                logger.info("=== CATEGORICAL ENCODING ===")
                #logger.info(f'before x_train shape: {self.x_train_cat_enc.shape}:\n : {self.x_train_cat_enc.columns}')
                #logger.info(f'before x_test shape: {self.x_test_cat_enc.shape}:\n : {self.x_test_cat_enc.columns}')

                self.x_train_cat_enc, self.x_test_cat_enc = cat_to_num(self.x_train_cat, self.x_test_cat)

                logger.info("Categorical encoding completed")
                logger.info(f"Encoded Categorical x_Train_cat Shape: {self.x_train_cat_enc.shape}:\n : {self.x_train_cat_enc.columns}")
                logger.info(f"Encoded Categorical x_Test_ cat Shape: {self.x_test_cat_enc.shape}:\n : {self.x_test_cat_enc.columns}")

            except Exception as e:
                                er_type, er_msg, er_line = sys.exc_info()
                                logger.info(f"Error in line no: {er_line.tb_lineno} due to: {er_msg}")
                                logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

        def combine_data(self):
            try:
                logger.info("=== COMBINING DATA ===")

                #  Reset index (prevents row increase)
                self.x_train_num = self.x_train_num.reset_index(drop=True)
                self.x_train_cat_enc = self.x_train_cat_enc.reset_index(drop=True)
                self.x_test_num = self.x_test_num.reset_index(drop=True)
                self.x_test_cat_enc = self.x_test_cat_enc.reset_index(drop=True)

                self.x_train_final = pd.concat([self.x_train_num, self.x_train_cat_enc], axis=1)
                self.x_test_final = pd.concat([self.x_test_num, self.x_test_cat_enc], axis=1)
                #final validataion
                assert self.x_train_final.shape[0] == self.y_train.shape[0], "Train mismatch "
                assert self.x_test_final.shape[0] == self.y_test.shape[0], "Test mismatch "

                logger.info(f"Final Train Shape: {self.x_train_final.shape}")
                logger.info(f"Final Test Shape: {self.x_test_final.shape}")

            except Exception as e:
                er_type, er_msg, er_line = sys.exc_info()
                logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

        def feature_selection(self, top_n=15):
            try:
                logger.info("=== FEATURE SELECTION STARTED ===")

                logger.info(f"X_train BEFORE FS: {self.x_train_final.shape}:\n : {self.x_train_final.columns}")
                logger.info(f"X_test BEFORE FS: {self.x_test_final.shape}:\n : {self.x_test_final.columns}")


                self.x_train_final, self.x_test_final = filter_features(
                    self.x_train_final,
                    self.x_test_final
                )

                logger.info(f"x_train AFTER FS: {self.x_train_final.shape}:\n : {self.x_train_final.columns}")
                logger.info(f"x_test AFTER FS: {self.x_test_final.shape}:\n : {self.x_test_final.columns}")

                logger.info("=== FEATURE SELECTION COMPLETED ===")

                return self.x_train_final, self.x_test_final


            except Exception as e:
             er_type, er_msg, er_line = sys.exc_info()
             logger.error(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

        def data_balancing(self):
            try:

                logger.info("_________Before Balancing__________")

                logger.info(f"Good (1) count BEFORE: {sum(self.y_train == 1)}")
                logger.info(f"Bad (0) count BEFORE: {sum(self.y_train == 0)}")

                logger.info(f"X_train shape BEFORE: {self.x_train_final.shape}")
                logger.info(f"y_train shape BEFORE: {self.y_train.shape}")

                logger.info("---------------Applying SMOTE-------------------------")

                sm = SMOTE(random_state=42)
                self.x_train_final, self.y_train = sm.fit_resample(self.x_train_final, self.y_train)

                logger.info("---------------After Balancing-------------------------")

                logger.info(f"Good (1) count AFTER: {sum(self.y_train == 1)}")
                logger.info(f"Bad (0) count AFTER: {sum(self.y_train == 0)}")

                logger.info(f"X_train shape AFTER: {self.x_train_final.shape}")
                logger.info(f"y_train shape AFTER: {self.y_train.shape}")

            except Exception as e:
                er_type, er_msg, er_line = sys.exc_info()
                logger.error(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
        def fs(self):
            try:
                logger.info("=== FEATURE SCALING STARTED ===")

                # BEFORE SCALING
                logger.info(f"x_train BEFORE scaling shape: {self.x_train_final.shape}")
                logger.info(f"x_test BEFORE scaling shape: {self.x_test_final.shape}")


                self.x_train_final, self.y_train, self.x_test_final, self.y_test = fs(self.x_train_final,self.y_train,self.x_test_final ,self.y_test)

                # AFTER SCALIN
                logger.info(f"x_train AFTER scaling shape: {self.x_train_final.shape}")
                logger.info(f"x_test AFTER scaling shape: {self.x_test_final.shape}")

       
                logger.info("=== FEATURE SCALING COMPLETED ===")
            except Exception as e:
                er_type, er_msg, er_line = sys.exc_info()
                logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")





if __name__ == "__main__":
    try:
        obj = TELCO('updated_telco.csv')
        obj.missing_values()
        obj.data_seperation()
        obj.variable_transformation()
        obj.categorical_numerical()
        obj.combine_data()
        obj.feature_selection()
        obj.data_balancing()
        obj.fs()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
