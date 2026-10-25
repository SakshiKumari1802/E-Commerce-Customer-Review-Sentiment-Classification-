# E-Commerce Customer Review Sentiment Classification

An end-to-end sentiment classification project for real Amazon product reviews. It compares TF-IDF Logistic Regression and Linear SVM with a fine-tuned DistilBERT model, evaluates each on the same held-out test split, and provides an interactive Streamlit app.

## Dataset

This project uses [Amazon Review Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity), hosted on Hugging Face Datasets. The public dataset metadata reports **4,000,000 reviews total**: 3,600,000 training rows and 400,000 test rows, with binary labels `negative` and `positive`. The reported compressed download size is approximately 1.15 GB. Reviews have a title and body; they are joined into one text field before cleaning.

For manageable local training, preparation samples 10,000 rows from the source training partition by default. The Hugging Face repository has four train Parquet shards; preparation downloads one shard (about 260 MB) and reads seeded random row groups until it has enough examples for an exactly stratified sample. It then creates reproducible 80/10/10 train/validation/test splits. In the verified default run, 11,000 rows were scanned to select 10,000; the splits contain 8,000/1,000/1,000 rows, each exactly balanced between labels. Actual run metadata, including source rows scanned, shards downloaded, split sizes, and class counts, is written to `data/processed/dataset_metadata.json`. The source's original 400,000-row test partition is not used, so every model is compared on the exact same held-out test examples.

## Setup

Use Python 3.12 in Codespaces (the project's supported range is Python 3.10-3.12). The provided Codespace has Python 3.12 available, two CPU cores, and no detected NVIDIA GPU.

This Codespace's Python 3.12 does not include `ensurepip`, but `virtualenv` is already installed. Create the environment without sudo or network seeding:

```bash
cd /workspaces/E-Commerce-Customer-Review-Sentiment-Classification-
virtualenv --no-download --python=/usr/bin/python3.12 .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install CPU-only PyTorch in this CPU Codespace, then the project dependencies:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
```

For a CUDA-enabled machine, install the PyTorch build matching its CUDA runtime using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the remaining requirements. The DistilBERT script automatically uses CUDA when PyTorch detects it; otherwise it runs on CPU. CPU fine-tuning is supported but substantially slower. Reduce the dataset sample (for example, `--max-samples 2000`) before preparation for a smaller CPU experiment.

## Run In Stages

1. Download and prepare the shared dataset splits. This fetches the source dataset but does not train any models:

	```bash
	python -m sentiment.data --max-samples 10000
	```

	Use `--max-samples 2000` for a quicker CPU experiment, or `--max-samples 0` to use all 3.6 million source training rows. The full option downloads all four train shards and needs substantial disk space, memory, and training time.

2. Generate training-split exploratory plots:

	```bash
	python -m sentiment.eda
	```

3. Train both TF-IDF baselines and evaluate them on the shared test split:

	```bash
	python -m sentiment.train_baselines
	```

4. Fine-tune DistilBERT and evaluate it on the same test split:

	```bash
	python -m sentiment.train_distilbert --epochs 1 --batch-size 8
	```

	CPU usage can be reduced with `--batch-size 4`; GPU users may increase the batch size to suit available memory.

5. Start the review classifier:

	```bash
	streamlit run app.py
	```

6. Run the focused tests:

	```bash
	pytest
	```

## Project Layout

```text
app.py                         Streamlit review classifier
sentiment/config.py            Paths, labels, dataset, and seed
sentiment/data.py              Download, clean, sample, and split data
sentiment/preprocessing.py     Shared text cleaning
sentiment/eda.py               Training-split visualizations
sentiment/train_baselines.py   TF-IDF Logistic Regression and Linear SVM
sentiment/train_distilbert.py  Hugging Face fine-tuning workflow
sentiment/evaluation.py        Metrics, reports, and confusion matrices
tests/                         Focused unit tests
data/processed/                Prepared splits and dataset metadata (ignored)
artifacts/                     Models, metrics, and figures (ignored)
```

## Outputs

- `artifacts/models/`: serialized scikit-learn pipelines and fine-tuned DistilBERT model/tokenizer.
- `artifacts/metrics/`: per-model accuracy, weighted precision/recall/F1, full classification reports, confusion-matrix values, and `model_comparison.csv`.
- `artifacts/metrics/*_confusion_matrix.png`: held-out test confusion matrices.
- `artifacts/figures/`: sentiment counts, training review-length distribution, and frequent terms by sentiment.

All reported evaluation values are generated by running the training scripts; no expected scores are hard-coded or claimed in advance. The default Logistic Regression and DistilBERT model scores are not calibrated probabilities. Linear SVM does not expose probabilities in this configuration, so the app does not present its decision scores as confidence.