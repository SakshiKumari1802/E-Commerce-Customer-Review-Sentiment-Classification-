import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

from sentiment.config import (
    DATA_DIR,
    DATASET_ID,
    DATASET_TEST_ROWS,
    DATASET_TRAIN_ROWS,
    DEFAULT_SAMPLE_SIZE,
    LABEL_NAMES,
    RANDOM_SEED,
)
from sentiment.preprocessing import clean_review


def prepare_data(
    max_samples: int | None = DEFAULT_SAMPLE_SIZE,
    output_dir: Path = DATA_DIR,
    seed: int = RANDOM_SEED,
) -> dict:
    """Download reviews, make one stratified split, and save it for every model."""
    if max_samples is not None and not 2 <= max_samples <= DATASET_TRAIN_ROWS:
        raise ValueError(f"max_samples must be between 2 and {DATASET_TRAIN_ROWS}, or None")

    frame, rows_scanned, shards_downloaded = load_source_sample(max_samples, seed)
    frame["text"] = (
        frame["title"].fillna("").astype(str)
        + ". "
        + frame["content"].fillna("").astype(str)
    ).map(clean_review)
    frame = frame.loc[frame["text"].str.len() > 0, ["sample_id", "text", "label"]]
    frame["label"] = frame["label"].astype("int8")
    train, held_out = train_test_split(
        frame,
        test_size=0.2,
        random_state=seed,
        stratify=frame["label"],
    )
    validation, test = train_test_split(
        held_out,
        test_size=0.5,
        random_state=seed,
        stratify=held_out["label"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "validation": validation, "test": test}
    for name, split in splits.items():
        split.to_csv(output_dir / f"{name}.csv", index=False)

    metadata = {
        "dataset_id": DATASET_ID,
        "source_train_rows": DATASET_TRAIN_ROWS,
        "source_test_rows": DATASET_TEST_ROWS,
        "published_total_rows": DATASET_TRAIN_ROWS + DATASET_TEST_ROWS,
        "rows_scanned_from_parquet": rows_scanned,
        "shards_downloaded": shards_downloaded,
        "used_rows": len(frame),
        "sample_limit": max_samples,
        "seed": seed,
        "label_names": LABEL_NAMES,
        "split_rows": {name: len(split) for name, split in splits.items()},
        "split_label_counts": {
            name: {
                LABEL_NAMES[int(label)]: int(count)
                for label, count in split["label"].value_counts().sort_index().items()
            }
            for name, split in splits.items()
        },
    }
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return metadata


def load_source_sample(max_samples: int | None, seed: int) -> tuple[pd.DataFrame, int, int]:
    """Read random Parquet row groups, stopping once both label quotas are met."""
    shard_count = DATASET_TRAIN_ROWS // 900_000
    shard_names = [
        f"amazon_polarity/train-{index:05d}-of-{shard_count:05d}.parquet"
        for index in range(shard_count)
    ]
    rng = np.random.default_rng(seed)
    collected = []
    rows_scanned = 0
    shards_downloaded = 0

    if max_samples is None:
        quotas = None
    else:
        base, remainder = divmod(max_samples, len(LABEL_NAMES))
        extra_label = sorted(LABEL_NAMES)[seed % len(LABEL_NAMES)]
        quotas = {
            label: base + int(label == extra_label and remainder > 0)
            for label in LABEL_NAMES
        }

    label_counts = {label: 0 for label in LABEL_NAMES}
    for shard_index, shard_name in enumerate(shard_names):
        local_path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=shard_name,
            repo_type="dataset",
        )
        reader = parquet.ParquetFile(local_path)
        shards_downloaded += 1
        row_groups = rng.permutation(reader.num_row_groups)

        for row_group_index in row_groups:
            table = reader.read_row_group(
                int(row_group_index), columns=["label", "title", "content"]
            )
            batch = table.to_pandas()
            rows_scanned += len(batch)
            first_row_id = sum(
                reader.metadata.row_group(index).num_rows
                for index in range(int(row_group_index))
            )
            batch["sample_id"] = [
                f"{shard_index}:{first_row_id + offset}" for offset in range(len(batch))
            ]
            collected.append(batch)
            for label, count in batch["label"].value_counts().items():
                label_counts[int(label)] += int(count)

            if quotas is not None and all(
                label_counts[label] >= quotas[label] for label in quotas
            ):
                break

        if quotas is not None and all(label_counts[label] >= quotas[label] for label in quotas):
            break

    if quotas is None:
        return pd.concat(collected, ignore_index=True), rows_scanned, shards_downloaded

    candidates = pd.concat(collected, ignore_index=True)
    if any(label_counts[label] < quotas[label] for label in quotas):
        raise ValueError("The downloaded training shards did not contain enough examples per label")
    selected = [
        candidates.loc[candidates["label"] == label].sample(
            n=quota,
            random_state=seed + label,
        )
        for label, quota in quotas.items()
    ]
    return pd.concat(selected, ignore_index=True), rows_scanned, shards_downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Stratified review sample from the 3.6M-row source train split; use 0 for all rows.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    prepare_data(
        max_samples=None if args.max_samples == 0 else args.max_samples,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
