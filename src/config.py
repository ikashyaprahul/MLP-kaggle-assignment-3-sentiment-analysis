# Data paths
TRAIN_DATA_PATH = "/kaggle/input/mlp-term-3-2025-kaggle-assignment-3/train.csv"
TEST_DATA_PATH = "/kaggle/input/mlp-term-3-2025-kaggle-assignment-3/test.csv"
SAMPLE_SUBMISSION_PATH = "/kaggle/input/mlp-term-3-2025-kaggle-assignment-3/sample_submission.csv"

# Model output path
MODEL_PATH = "model.joblib"
SUBMISSION_PATH = "submission.csv"

# Columns
TEXT_COL = "phrase"
TARGET_COL = "sentiment"
ID_COL = "id"

# Model parameters
RANDOM_STATE = 42

# Model hyperparameters
LOGISTIC_REGRESSION_MAX_ITER = 4000
LOGISTIC_REGRESSION_SOLVER = 'lbfgs'
LOGISTIC_REGRESSION_N_JOBS = -1
TFIDF_ANALYZER = 'char'
TFIDF_NGRAM_RANGE = (3, 5)
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.9
TFIDF_SUBLINEAR_TF = True
TFIDF_MAX_FEATURES = 200000
