import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import xgboost as xgb

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path, sep=';')



def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        df[target],
                                                        test_size=0.15,
                                                        random_state=8)
    return X_train, X_test, y_train, y_test


def drop_duplicate(df):
    # remove any duplicate records
    return df.drop_duplicates()


def bmi_feature(df, var1, var2):
    # function captures bmi determined by weight/(height/100)**2
    return df[var1]/(df[var2]/100)**2

def remove_outlier(df, var):
    # only keep blood pressure values within a range
    out_filter = ((df[var] > 405) | (df[var] < 10))
    return df[~out_filter]


def train_scaler(df, output_path):
    scaler = PowerTransformer(method = 'yeo-johnson')
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler



def scale_features(df, scaler):
    scaler = joblib.load(scaler)
    return scaler.transform(df)



def train_model(df, target, output_path):
    # initialise the model
    xgboost_model = xgb.XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints=None,
              learning_rate=0.01, max_delta_step=0, max_depth=5,
              min_child_weight=2, monotone_constraints=None,
              n_estimators=780, n_jobs=0, num_parallel_tree=1,
              objective='reg:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=0.2, tree_method=None,
              validate_parameters=False, verbosity=None)

    # train the model
    xgboost_model.fit(df, target)

    # save the model
    joblib.dump(xgboost_model, output_path)

    return None



def predict(df, model):
    model = joblib.load(model)
    return model.predict(df)
