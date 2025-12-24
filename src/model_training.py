from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import joblib
from src.config import (
    LOGISTIC_REGRESSION_MAX_ITER,
    LOGISTIC_REGRESSION_SOLVER,
    LOGISTIC_REGRESSION_N_JOBS,
    TFIDF_ANALYZER,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
    TFIDF_SUBLINEAR_TF,
    TFIDF_MAX_FEATURES,
)


def train_model(X_train, y_train, text_col):
    char_vectorizer = TfidfVectorizer(
        analyzer=TFIDF_ANALYZER,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        max_features=TFIDF_MAX_FEATURES,
    )

    model = Pipeline(
        steps=[
            ("tfidf", char_vectorizer),
            (
                "clf",
                LogisticRegression(
                    max_iter=LOGISTIC_REGRESSION_MAX_ITER,
                    solver=LOGISTIC_REGRESSION_SOLVER,
                    n_jobs=LOGISTIC_REGRESSION_N_JOBS,
                ),
            ),
        ]
    )

    model.fit(X_train[text_col], y_train)
    return model


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
