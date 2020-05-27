import preprocessing_functions as pf
import config

# =========== scoring pipeline =========

# impute categorical variables
def predict(data):

    # remove duplicate
    data = pf.drop_duplicate(data)

    # Engineer BMI column
    data['bmi'] = pf.bmi_feature(data, 'weight', 'height')

    # Remove extreme outliers for ap_hi
    data['ap_hi'] = pf.remove_outlier(data, 'ap_hi')

    # Remove extreme outliers for ap_lo
    data['ap_lo'] = pf.remove_outlier(data, 'ap_lo')

    # scale variables
    data = pf.scale_features(data[config.FEATURES],
                             config.OUTPUT_SCALER_PATH)

    # make predictions
    predictions = pf.predict(data, config.OUTPUT_MODEL_PATH)

    return predictions

# ======================================

# small test that scripts are working ok

if __name__ == '__main__':

    from math import sqrt
    import numpy as np

    from sklearn.metrics import accuracy_score

    import warnings
    warnings.simplefilter(action='ignore')

    # Load data
    data = pf.load_data(config.PATH_TO_DATASET)
    X_train, X_test, y_train, y_test = pf.divide_train_test(data,
                                                            config.TARGET)

    pred = predict(X_test)

    # evaluate
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()
