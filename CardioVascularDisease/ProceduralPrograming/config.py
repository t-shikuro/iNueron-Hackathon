# ====   PATHS ===================

PATH_TO_DATASET = "cardio_train.csv"
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'xgboost.pkl'



# ======= PARAMETERS ===============

# ======= FEATURE GROUPS =============

TARGET = 'cardio'

# selected features for training
FEATURES = ['age', 'ap_hi', 'cholesterol', 'bmi', 'ap_lo', 'weight', 'gluc',
           'active', 'active', 'height', 'gender',
           'smoke', 'alco']
