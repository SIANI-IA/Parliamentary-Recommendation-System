"""
run_experiments.py — CLI experiment runner for IR/ML baselines.

Examples:
    python run_experiments.py --dataset dataset/canada-rec-split-updated --algorithm tfidf --lang english
    python run_experiments.py --dataset dataset/parcanDeb-rec-split --algorithm pulkm --lang spanish --max_iter 5 10 30 60
    python run_experiments.py --dataset dataset/canada-rec-split-updated --algorithm dense \
        --model_name BAAI/bge-m3 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --batch_size 64 64 --token_limit 512 --strategy both
"""

import argparse
import sys


def print_metrics(metrics: dict, label: str):
    print(f"\n{'='*60}")
    print(f"  Results: {label}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
    print(f"{'='*60}\n")


def run_tfidf(dataset: str, lang: str):
    from models.IR.SparseTfidfRecommender import SparseTfidfRecommender
    label = f"TF-IDF | {dataset} | lang={lang}"
    print(f"\n>>> Running {label}")
    r = SparseTfidfRecommender(dataset, lang=lang)
    metrics = r.run_evaluation()
    print_metrics(metrics, label)
    return metrics


def run_bm25(dataset: str, lang: str):
    from models.IR.SparseBM25Recommender import SparseBM25Recommender
    label = f"BM25 | {dataset} | lang={lang}"
    print(f"\n>>> Running {label}")
    r = SparseBM25Recommender(dataset, lang=lang)
    metrics = r.run_evaluation()
    print_metrics(metrics, label)
    return metrics


def run_pulkm(dataset: str, lang: str, max_iters: list):
    from models.ML.PULKMRecommender import PULKMRecommender
    all_metrics = []
    for max_iter in max_iters:
        label = f"PULKM (max_iter={max_iter}) | {dataset} | lang={lang}"
        print(f"\n>>> Running {label}")
        r = PULKMRecommender(dataset, max_iter_pulk=max_iter, lang=lang)
        metrics = r.run_evaluation()
        print_metrics(metrics, label)
        all_metrics.append((max_iter, metrics))
    return all_metrics


def run_dense(
    dataset: str,
    model_names: list,
    batch_sizes: list,
    token_limits: list,
    strategy: str,
    use_half_precision: bool,
    use_4bits: bool,
):
    import torch
    from models.IR.DenseEmbeddingRecommender import DenseEmbeddingRecommender

    # Pad batch_sizes so it matches model_names length (reuse last value)
    if len(batch_sizes) < len(model_names):
        batch_sizes = batch_sizes + [batch_sizes[-1]] * (len(model_names) - len(batch_sizes))

    model_batch_pairs = list(zip(model_names, batch_sizes))
    all_metrics = []

    for token_limit in token_limits:
        for model_name, batch_size in model_batch_pairs:
            strategies = []
            if strategy in ("centroid", "both"):
                strategies.append(False)  # use_intervention_level=False → centroid
            if strategy in ("intervention", "both"):
                strategies.append(True)   # use_intervention_level=True → intervention

            for use_intervention_level in strategies:
                strat_name = "intervention" if use_intervention_level else "centroid"
                label = (
                    f"Dense ({model_name}, {strat_name}, token={token_limit}) "
                    f"| {dataset}"
                )
                print(f"\n>>> Running {label}")
                r = DenseEmbeddingRecommender(
                    dataset_path=dataset,
                    model_name=model_name,
                    batch_size=batch_size,
                    token_limit=token_limit,
                    use_intervention_level=use_intervention_level,
                    use_half_precision=use_half_precision,
                    use_4bits=use_4bits,
                )
                metrics = r.run_evaluation()
                print_metrics(metrics, label)
                all_metrics.append((model_name, strat_name, token_limit, metrics))

                # Free GPU memory between runs
                del r
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return all_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run IR/ML baseline experiments for Parliamentary Recommendation."
    )
    parser.add_argument("--dataset", required=True, help="Path to the HuggingFace dataset directory.")
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=["tfidf", "bm25", "pulkm", "dense"],
        help="Which algorithm to run.",
    )

    # Shared sparse options
    parser.add_argument(
        "--lang",
        default="spanish",
        choices=["english", "spanish"],
        help="Language for stopwords (tfidf/bm25/pulkm). Default: spanish.",
    )

    # PULKM options
    parser.add_argument(
        "--max_iter",
        type=int,
        nargs="+",
        default=[10],
        metavar="N",
        help="PULKM: max_iter values to sweep. Default: [10].",
    )

    # Dense options
    parser.add_argument(
        "--model_name",
        nargs="+",
        default=["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
        metavar="MODEL",
        help="Dense: one or more model names (HuggingFace hub IDs). Default: paraphrase-multilingual-MiniLM-L12-v2.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=[64],
        metavar="B",
        help="Dense: batch sizes, zipped with --model_name (last value reused if fewer). Default: [64].",
    )
    parser.add_argument(
        "--token_limit",
        type=int,
        nargs="+",
        default=[512],
        metavar="T",
        help="Dense: token limits; runs one experiment per value (outer loop). Default: [512].",
    )
    parser.add_argument(
        "--strategy",
        default="centroid",
        choices=["centroid", "intervention", "both"],
        help="Dense retrieval strategy. Default: centroid.",
    )
    parser.add_argument(
        "--use_half_precision",
        action="store_true",
        help="Dense: use fp16 (CUDA only).",
    )
    parser.add_argument(
        "--use_4bits",
        action="store_true",
        help="Dense: load model in 4-bit quantization (CUDA only).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    algo = args.algorithm

    if algo == "tfidf":
        run_tfidf(args.dataset, args.lang)

    elif algo == "bm25":
        run_bm25(args.dataset, args.lang)

    elif algo == "pulkm":
        run_pulkm(args.dataset, args.lang, args.max_iter)

    elif algo == "dense":
        run_dense(
            dataset=args.dataset,
            model_names=args.model_name,
            batch_sizes=args.batch_size,
            token_limits=args.token_limit,
            strategy=args.strategy,
            use_half_precision=args.use_half_precision,
            use_4bits=args.use_4bits,
        )

    else:
        print(f"Unknown algorithm: {algo}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
