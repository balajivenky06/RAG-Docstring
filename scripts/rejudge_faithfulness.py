"""
Re-judge faithfulness of ALL saved generations against SOURCE CODE uniformly.

Addresses reviewer R2.2 (metric commensurability): the original pipeline judged
RAG outputs against retrieved context but No-RAG outputs against source code.
This script re-scores every saved docstring with the source code as the single
reference, optionally with multiple judge models (R1.4 second-judge check) and
an optional secondary context-grounding score for RAG strategies.

Resumable: one CSV per (model_dir, strategy, judge) under results/rejudged/;
completed files are skipped on restart.

Usage:
    python scripts/rejudge_faithfulness.py                       # 3 paper models, default judge
    python scripts/rejudge_faithfulness.py --judges deepseek-coder:6.7b qwen3:8b
    python scripts/rejudge_faithfulness.py --dirs qwen3_5_9b --limit 5
"""

import os
import sys
import time
import glob
import argparse
import warnings

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.evaluator import RAGEvaluator

warnings.filterwarnings('ignore')

DEFAULT_DIRS = ["llama3_2_latest", "phi4_14b", "qwen3_8b"]
DEFAULT_JUDGE = "deepseek-coder:6.7b"


def find_result_files(results_root: str, model_dirs):
    """Yield (model_dir, strategy_name, path) for every saved results pickle."""
    for model_dir in model_dirs:
        pattern = os.path.join(results_root, model_dir, "comparison_*", "*_results.pkl")
        for path in sorted(glob.glob(pattern)):
            strategy = os.path.basename(path).replace("_results.pkl", "")
            yield model_dir, strategy, path


def rejudge_file(evaluator, model_dir, strategy, path, judge, out_dir,
                 with_context_secondary=False, limit=None, draws=1):
    out_path = os.path.join(out_dir, f"{model_dir}__{strategy}__{judge.replace(':', '_').replace('.', '_')}.csv")
    if os.path.exists(out_path):
        print(f"  [skip] {model_dir}/{strategy} ({judge}) already done")
        return out_path

    df = pd.read_pickle(path)
    if limit:
        df = df.head(limit)

    rows = []
    start = time.time()
    for i, row in df.iterrows():
        code = row.get('Code_without_comments') or row.get('Full_code') or ''
        docstring = row.get('Generated_Docstring') or ''
        context = row.get('Retrieved_Context') or ''

        # LLM-judge scores have high draw-to-draw variance at default sampling
        # temperature; averaging k independent draws yields a reliable
        # per-sample score (single draws are only meaningful in aggregate).
        draw_scores = [evaluator.calculate_faithfulness_score(
            docstring, retrieved_context='', code=code, judge_model=judge)
            for _ in range(draws)]

        record = {
            'model_dir': model_dir,
            'strategy': strategy,
            'sample_index': i,
            'judge': judge,
            'faithfulness_code_ref': sum(draw_scores) / len(draw_scores),
            'n_draws': draws,
        }
        for d, s in enumerate(draw_scores, 1):
            record[f'draw_{d}'] = s
        if with_context_secondary and context:
            record['faithfulness_context_ref'] = evaluator.calculate_context_faithfulness_score(
                docstring, context, judge_model=judge)
        rows.append(record)

        done = len(rows)
        if done % 10 == 0 or done == len(df):
            rate = (time.time() - start) / done
            print(f"  {model_dir}/{strategy} ({judge}): {done}/{len(df)} "
                  f"({rate:.1f}s/sample, ~{rate * (len(df) - done):.0f}s left)")

    result = pd.DataFrame(rows)
    tmp_path = out_path + ".tmp"
    result.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    print(f"  [done] {out_path} mean={result['faithfulness_code_ref'].mean():.4f}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Re-judge faithfulness against source code uniformly")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS,
                        help="Model result directories under results/")
    parser.add_argument("--judges", nargs="+", default=[DEFAULT_JUDGE],
                        help="Ollama judge model tags")
    parser.add_argument("--with_context_secondary", action="store_true",
                        help="Also compute secondary context-referenced score for RAG rows")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Restrict to these strategy names (e.g. PlainLLM SimpleRAG)")
    parser.add_argument("--draws", type=int, default=1,
                        help="Independent judge draws per sample (mean reported); use >=3 for reliable per-sample scores")
    parser.add_argument("--limit", type=int, default=None, help="Samples per strategy (smoke test)")
    args = parser.parse_args()

    out_dir = os.path.join(args.results_root, "rejudged" if args.draws == 1 else f"rejudged_k{args.draws}")
    os.makedirs(out_dir, exist_ok=True)

    evaluator = RAGEvaluator()
    files = list(find_result_files(args.results_root, args.dirs))
    if args.strategies:
        files = [f for f in files if f[1] in set(args.strategies)]
    total = len(files) * len(args.judges)
    print(f"Re-judging {len(files)} result files x {len(args.judges)} judge(s) = {total} passes")

    for judge in args.judges:
        for model_dir, strategy, path in files:
            rejudge_file(evaluator, model_dir, strategy, path, judge, out_dir,
                         with_context_secondary=args.with_context_secondary, limit=args.limit,
                         draws=args.draws)

    # Consolidate
    parts = [pd.read_csv(p) for p in glob.glob(os.path.join(out_dir, "*__*.csv"))]
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined_path = os.path.join(out_dir, "rejudged_all.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nConsolidated {len(combined)} scores -> {combined_path}")
        summary = combined.groupby(['model_dir', 'strategy', 'judge'])['faithfulness_code_ref'].mean().round(4)
        print(summary.to_string())


if __name__ == "__main__":
    main()
