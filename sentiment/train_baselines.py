import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sentiment.config import ARTIFACTS_DIR, DATA_DIR
from sentiment.evaluation import evaluate_predictions, save_results_table


def build_models() -> dict[str, Pipeline]:
    vectorizer = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    return {
        "TF-IDF Logistic Regression": Pipeline(
            [
                ("tfidf", vectorizer),
                ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ]
        ),
        "TF-IDF Linear SVM": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=100_000,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", LinearSVC(C=1.0, random_state=42)),
            ]
        ),
    }


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    models_dir = ARTIFACTS_DIR / "models"
    metrics_dir = ARTIFACTS_DIR / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics = []

    for name, model in build_models().items():
        model.fit(train["text"], train["label"])
        predictions = model.predict(test["text"])
        metrics.append(evaluate_predictions(name, test["label"], predictions, metrics_dir))
        dump(model, models_dir / f"{name.lower().replace(' ', '_')}.joblib")
        print(f"Finished held-out evaluation: {name}")

    print(save_results_table(metrics, metrics_dir).to_string(index=False))


if __name__ == "__main__":
    main()
