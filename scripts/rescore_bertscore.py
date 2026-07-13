"""
Re-score BERTScore for all saved generations with a stronger backbone.

Addresses an R2 minor point: the original pipeline used bert-base-uncased
(weak, case-insensitive, no baseline rescaling), making absolute values hard
to interpret. This script recomputes BERTScore with roberta-large and
baseline rescaling (the bert-score package's recommended configuration for
English), over the same (reference docstring, generated docstring) pairs.

Resumable: one CSV per (model_dir, strategy) under results/rescored_bertscore/.

Usage:
    python scripts/rescore_bertscore.py
    python scripts/rescore_bertscore.py --model_type microsoft/deberta-xlarge-mnli
    python scripts/rescore_bertscore.py --limit 5   # smoke test
"""

import os
import sys
import glob
import argparse
import warnings

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

DEFAULT_DIRS = ["llama3_2_latest", "phi4_14b", "qwen3_8b"]


def find_result_files(results_root: str, model_dirs):
    for model_dir in model_dirs:
        pattern = os.path.join(results_root, model_dir, "comparison_*", "*_results.pkl")
        for path in sorted(glob.glob(pattern)):
            strategy = os.path.basename(path).replace("_results.pkl", "")
            yield model_dir, strategy, path


def main():
    parser = argparse.ArgumentParser(description="Recompute BERTScore with a stronger backbone")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS)
    parser.add_argument("--model_type", default="roberta-large")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from bert_score import BERTScorer

    out_dir = os.path.join(args.results_root, "rescored_bertscore")
    os.makedirs(out_dir, exist_ok=True)
    safe_backbone = args.model_type.replace("/", "_")

    scorer = BERTScorer(model_type=args.model_type, lang="en",
                        rescale_with_baseline=True, batch_size=args.batch_size)
    print(f"Scorer ready: {args.model_type} (rescale_with_baseline=True)")

    files = list(find_result_files(args.results_root, args.dirs))
    for model_dir, strategy, path in files:
        out_path = os.path.join(out_dir, f"{model_dir}__{strategy}__{safe_backbone}.csv")
        if os.path.exists(out_path):
            print(f"[skip] {model_dir}/{strategy} already done")
            continue

        df = pd.read_pickle(path)
        if args.limit:
            df = df.head(args.limit)

        refs = df["Comments"].fillna("").astype(str).tolist()
        cands = df["Generated_Docstring"].fillna("").astype(str).tolist()
        # Empty candidates crash some backbones; substitute a placeholder token
        cands = [c if c.strip() else "EMPTY" for c in cands]
        refs = [r if r.strip() else "EMPTY" for r in refs]

        P, R, F = scorer.score(cands, refs)
        result = pd.DataFrame({
            "model_dir": model_dir,
            "strategy": strategy,
            "sample_index": df["index"] if "index" in df.columns else df.index,
            "backbone": args.model_type,
            "bert_precision": P.numpy(),
            "bert_recall": R.numpy(),
            "bert_f1_rescaled": F.numpy(),
        })
        tmp = out_path + ".tmp"
        result.to_csv(tmp, index=False)
        os.replace(tmp, out_path)
        print(f"[done] {model_dir}/{strategy}: mean F1 = {result['bert_f1_rescaled'].mean():.4f}")

    parts = [pd.read_csv(p) for p in glob.glob(os.path.join(out_dir, "*__*.csv"))]
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(os.path.join(out_dir, "rescored_all.csv"), index=False)
        print(f"\nConsolidated {len(combined)} scores -> {out_dir}/rescored_all.csv")
        print(combined.groupby(["model_dir", "strategy"])["bert_f1_rescaled"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
