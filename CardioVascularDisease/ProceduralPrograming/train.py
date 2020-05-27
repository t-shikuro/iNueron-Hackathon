import numpy as np

import preprocessing_functions as pf
import config

import warnings
warnings.simplefilter(action='ignore')

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
data = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_test, y_train, y_test = pf.divide_train_test(data, config.TARGET)

# remove duplicate
X_train = pf.drop_duplicate(X_train)

# Engineer BMI column
X_train['bmi'] = pf.bmi_feature(X_train, 'weight', 'height')

# Remove extreme outliers for ap_hi
X_train['ap_hi'] = pf.remove_outlier(X_train, 'ap_hi')

# Remove extreme outliers for ap_lo
X_train['ap_lo'] = pf.remove_outlier(X_train, 'ap_lo')

# train scaler and save
scaler = pf.train_scaler(X_train[config.FEATURES],
                         config.OUTPUT_SCALER_PATH)

# scale train set
X_train = scaler.transform(X_train[config.FEATURES])

# train model and save
pf.train_model(X_train,
               (y_train),
               config.OUTPUT_MODEL_PATH)

print('Finished training')
