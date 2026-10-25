import pandas as pd
import pyarrow as pa

from sentiment.preprocessing import clean_review

from sentiment.data import load_source_sample


def test_clean_review_removes_markup_urls_and_extra_whitespace():
    assert clean_review("  <b>Great</b>&amp; reliable https://shop.example/item  ") == (
        "Great & reliable"
    )


def test_clean_review_preserves_sentiment_words_and_punctuation():
    assert clean_review("I do NOT like this!") == "I do NOT like this!"


def test_source_sample_is_balanced_and_reproducible(monkeypatch):
    source = pd.DataFrame(
        {
            "sample_id": [f"row-{index}" for index in range(12)],
            "label": [0] * 8 + [1] * 4,
            "title": ["title"] * 12,
            "content": ["review"] * 12,
        }
    )

    monkeypatch.setattr(
        "sentiment.data.hf_hub_download", lambda **kwargs: "/unused/train.parquet"
    )

    class FakeReader:
        num_row_groups = 3

        class Metadata:
            @staticmethod
            def row_group(index):
                class Group:
                    num_rows = 4

                return Group()

        metadata = Metadata()

        @staticmethod
        def read_row_group(index, columns):
            return pa.Table.from_pandas(
                source.iloc[index * 4 : (index + 1) * 4].drop(columns="sample_id"),
                preserve_index=False,
            )

    monkeypatch.setattr("sentiment.data.parquet.ParquetFile", lambda path: FakeReader())
    first, rows_scanned, shard_count = load_source_sample(6, seed=42)
    second, _, _ = load_source_sample(6, seed=42)

    assert len(first) == 6
    assert first["label"].value_counts().to_dict() == {0: 3, 1: 3}
    assert first["sample_id"].tolist() == second["sample_id"].tolist()
    assert rows_scanned <= len(source)
    assert shard_count == 1
