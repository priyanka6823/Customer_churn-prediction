import pandas as pd
import sys
from logging_code import setup_logging
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

logger = setup_logging("cat_to_num")


def cat_to_num(x_train_cat, x_test_cat):
    try:
        logger.info("=== CATEGORICAL ENCODING STARTED ===")

        x_train = x_train_cat.copy()
        x_test = x_test_cat.copy()

        logger.info(f"Train Cat Shape BEFORE: {x_train.shape}")
        logger.info(f"Test Cat Shape BEFORE: {x_test.shape}")

        # -----------------------------------
        # 1. Binary Columns → Label Encoding
        # -----------------------------------
        binary_cols = ['PaperlessBilling', 'Partner', 'Dependents']

        for col in binary_cols:
            if col in x_train.columns:
                le = LabelEncoder()
                x_train[col] = le.fit_transform(x_train[col])
                x_test[col] = le.transform(x_test[col])

        # -----------------------------------
        # 2. Ordinal Column → Contract
        # -----------------------------------
        if 'Contract' in x_train.columns:
            contract_order = [['Month-to-month', 'One year', 'Two year']]

            ord_encoder = OrdinalEncoder(categories=contract_order)

            x_train[['Contract']] = ord_encoder.fit_transform(x_train[['Contract']])
            x_test[['Contract']] = ord_encoder.transform(x_test[['Contract']])

        # -----------------------------------
        # 3. Nominal Columns → OneHotEncoder
        # -----------------------------------
        nominal_cols = [col for col in x_train.columns
                        if col not in ['Contract'] + binary_cols]

        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

        x_train_ohe = ohe.fit_transform(x_train[nominal_cols])
        x_test_ohe = ohe.transform(x_test[nominal_cols])

        ohe_cols = ohe.get_feature_names_out(nominal_cols)

        x_train_ohe = pd.DataFrame(x_train_ohe, columns=ohe_cols, index=x_train.index)
        x_test_ohe = pd.DataFrame(x_test_ohe, columns=ohe_cols, index=x_test.index)

        # Drop original nominal cols
        x_train = x_train.drop(nominal_cols, axis=1)
        x_test = x_test.drop(nominal_cols, axis=1)

        # Combine all
        x_train_final = pd.concat([x_train, x_train_ohe], axis=1)
        x_test_final = pd.concat([x_test, x_test_ohe], axis=1)

        logger.info(f"Train Cat Shape AFTER: {x_train_final.shape}")
        logger.info(f"Test Cat Shape AFTER: {x_test_final.shape}")
        logger.info("=== CATEGORICAL ENCODING COMPLETED ===")

        return x_train_final, x_test_final

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
        return pd.DataFrame(), pd.DataFrame()