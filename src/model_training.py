from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import joblib

def train_model(X_train, y_train, text_col):
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        max_df=0.9,
        sublinear_tf=True,
        max_features=200000
    )

    model = Pipeline(steps=[
        ("tfidf", char_vectorizer),
        ("clf", LogisticRegression(
            max_iter=4000,
            solver="lbfgs",
            n_jobs=-1
        ))
    ])

    model.fit(X_train[text_col], y_train)
    return model

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
