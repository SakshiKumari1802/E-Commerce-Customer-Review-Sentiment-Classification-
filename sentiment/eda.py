import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from sentiment.config import DATA_DIR, FIGURES_DIR, LABEL_NAMES


def run_eda() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    distribution = train["label"].map(LABEL_NAMES).value_counts().reindex(
        ["negative", "positive"]
    )
    figure, axis = plt.subplots(figsize=(6, 4))
    sns.barplot(x=distribution.index, y=distribution.values, hue=distribution.index,
                palette={"negative": "#d45b4c", "positive": "#278878"}, legend=False, ax=axis)
    axis.set(title="Training-set sentiment distribution", xlabel="Sentiment", ylabel="Reviews")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "sentiment_distribution.png", dpi=160)
    plt.close(figure)

    train = train.assign(word_count=train["text"].str.split().str.len())
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.histplot(
        data=train,
        x="word_count",
        hue=train["label"].map(LABEL_NAMES),
        bins=50,
        stat="density",
        common_norm=False,
        element="step",
        ax=axis,
    )
    axis.set(title="Review length by sentiment", xlabel="Words per review", ylabel="Density")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "review_lengths.png", dpi=160)
    plt.close(figure)

    for label, sentiment in LABEL_NAMES.items():
        vectorizer = CountVectorizer(stop_words="english", max_features=15)
        matrix = vectorizer.fit_transform(train.loc[train["label"] == label, "text"])
        counts = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        top_terms = pd.Series(counts, index=terms).sort_values().tail(15)
        figure, axis = plt.subplots(figsize=(7, 5))
        top_terms.plot.barh(ax=axis, color="#278878" if label else "#d45b4c")
        axis.set(title=f"Frequent terms in {sentiment} training reviews", xlabel="Occurrences")
        figure.tight_layout()
        figure.savefig(FIGURES_DIR / f"top_terms_{sentiment}.png", dpi=160)
        plt.close(figure)

    print(f"EDA figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
