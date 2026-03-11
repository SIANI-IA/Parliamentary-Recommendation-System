# Parliamentary Speaker Recommendation System

Given a parliamentary document, this system learns from individual MP interventions to then recommend relevant documents to each Member of Parliament. The problem is framed as asymmetric multi-label classification: trained on individual interventions (one sample per speaker turn) and evaluated on full documents (all speakers at once).

---

## Project Structure

```
├── trainer.py                  # Main fine-tuning entry point (Transformers)
├── run_experiments.py          # CLI runner for IR/ML baselines
├── generate_benchmark.py       # Collects .pkl results → CSV + LaTeX table
├── trainers/                   # HuggingFace Trainer subclasses
│   ├── WeightedTrainer.py          # BCE with configurable pos_weight
│   ├── nnPUTrainer.py              # Non-negative PU loss
│   ├── StarSpaceTrainer.py         # Embedding-space loss
│   ├── GraphRegularizedTrainer.py  # BCE + co-occurrence graph regularization
│   └── trainer_utils.py            # Helpers: determinism, metrics, prior estimation
├── models/
│   ├── Recommender.py          # Base class (save_artifacts)
│   ├── IR/                     # Information retrieval baselines
│   │   ├── SparseTfidfRecommender.py
│   │   ├── SparseBM25Recommender.py
│   │   └── DenseEmbeddingRecommender.py
│   └── ML/                     # Machine learning baseline
│       ├── PULKMRecommender.py         # Full PU-Learning pipeline
│       ├── PULKMeans.py                # K-Means for reliable negative detection
│       ├── ParliamentaryVectorization.py
│       └── ParliamentaryClassifier.py
├── eval/
│   ├── Evaluator.py            # Multi-label + IR metrics (F1, MAP, nDCG, R-Prec, Recall@K)
│   └── IREvaluator.py          # Single-label ranking evaluator (MRR, nDCG, Recall@K)
├── dataset_builder.py          # Converts dataset to generative/LLM format ([MP_N] tokens)
├── extract_mp_intervertions_canada.py  # Parses raw Canadian parliament data
├── split_multilabel_dataset.py         # Stratified train/dev/test split
└── utils.py                    # Global constants (SEED=123, FOLDER_TO_SAVE_RESULTS)
```

---

## Datasets

| Path | Description | Tiny split |
|------|-------------|:---:|
| `dataset/canada-rec-split/` | Canadian parliament (English), original version | ✓ |
| `dataset/parcanDeb-rec-split/` | Canarian parliament (Spanish) | ✓ |

Each dataset directory contains `mp_mapping.json` with `id2label` and `label2id` dicts. Datasets are gitignored. To rebuild them:
1. `extract_mp_intervertions_canada.py` → produces `dataset/canada-rec/`
2. `split_multilabel_dataset.py` → produces `dataset/*-split/`

---

## Environment

```bash
conda activate mn5
# or without activating:
conda run -n mn5 python trainer.py ...
```

---

## IR/ML Baselines (`run_experiments.py`)

```bash
# TF-IDF (Canada)
python run_experiments.py --dataset dataset/canada-rec-split-updated --algorithm tfidf --lang english

# BM25 (Canarias)
python run_experiments.py --dataset dataset/parcanDeb-rec-split --algorithm bm25 --lang spanish

# PU-Learning with max_iter sweep
python run_experiments.py --dataset dataset/parcanDeb-rec-split \
  --algorithm pulkm --lang spanish --max_iter 5 10 30 60

# Dense: multiple models × both strategies × two token limits
python run_experiments.py --dataset dataset/canada-rec-split-updated \
  --algorithm dense \
  --model_name BAAI/bge-m3 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --batch_size 64 64 --token_limit 512 1024 --strategy both

# Large model with half-precision
python run_experiments.py --dataset dataset/canada-rec-split-updated \
  --algorithm dense --model_name Qwen/Qwen3-Embedding-4B --batch_size 8 \
  --token_limit 512 --strategy intervention --use_half_precision
```

### Key flags for `--algorithm dense`

| Argument | Description |
|----------|-------------|
| `--model_name M [M ...]` | One or more HuggingFace model IDs |
| `--batch_size B [B ...]` | Batch sizes zipped with `--model_name` (last value reused if fewer) |
| `--token_limit T [T ...]` | Outer loop: all model/strategy combinations run per value |
| `--strategy` | `centroid` (ir-p) \| `intervention` (ir-i) \| `both` |
| `--use_half_precision` | fp16 inference (CUDA only) |
| `--use_4bits` | 4-bit NF4 quantization (CUDA only) |

GPU memory is cleared with `torch.cuda.empty_cache()` between dense runs.

---

## Transformer Fine-tuning (`trainer.py`)

```bash
# Weighted BCE, full fine-tuning
python trainer.py \
  --data_path dataset/canada-rec-split-updated \
  --mapping_path dataset/canada-rec-split-updated/mp_mapping.json \
  --model_name jhu-clsp/mmBERT-small \
  --mode full --loss_type bce --epochs 3 --batch_size 8

# Quick smoke test (tiny split)
python trainer.py \
  --data_path dataset/canada-rec-split \
  --mapping_path dataset/canada-rec-split/mp_mapping.json \
  --model_name jhu-clsp/mmBERT-small \
  --tiny --epochs 1 --batch_size 4
```

### Key arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--mode` | `full` \| `lora` \| `qlora` | Fine-tuning strategy |
| `--loss_type` | `bce` \| `nnpuloss` \| `starspace` \| `graph_bce` | Loss function |
| `--intervention_strategy` | `split` \| `concat` | How interventions are segmented at train time |
| `--label_strategy` | `author` \| `all_participants` | Labels per training sample |
| `--pos_weight_type` | `linear` \| `sqrt` \| `log` | Positive class weighting strategy (BCE) |
| `--dapt_adapter_path` | path | LoRA adapter from DAPT to merge before fine-tuning |
| `--early_stopping_patience` | int | Early stopping patience (default: 3) |
| `--lora_r` / `--lora_alpha` | int | LoRA rank and alpha (defaults: 16 / 32) |

---

## Results and Benchmarking

Results are saved automatically to `results/<dataset>/<model_name>/` as `.pkl` files containing predictions, score matrices, optimal threshold, and metrics.

```bash
# Generate benchmark_results.csv and LaTeX table
python generate_benchmark.py
```

The classification threshold is **always optimized on dev** (maximizing Micro-F1) before reporting test metrics.

### Reported metrics

- **Classification**: Subset Accuracy, Micro/Macro F1, Micro/Macro Precision/Recall
- **IR**: MAP, nDCG, R-Precision, Recall@{1, 5, 10, 20}

---

## Asymmetric Train/Test Design

The central design decision of this system:

- **Train**: each individual MP intervention is a separate sample with a single label (or multi-label if `--label_strategy all_participants`).
- **Dev/Test**: each full document is one sample with a multi-label target (all MPs who spoke in it).

This simulates the real scenario: at inference time, the system receives a complete document and must predict who spoke in it.
