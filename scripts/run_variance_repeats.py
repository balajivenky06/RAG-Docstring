"""
Run-to-run variance estimation (R2 minor): repeated runs of representative
strategies at the paper's generation settings (temperature 0.5).

Reviewer 2 notes that single runs provide no variance estimate for stochastic
pipelines, and that the cost figures driving the "Reasoning Tax" argument need
stability evidence. This script repeats a subset of strategies k times and
reports per-run means for quality (deterministic metrics) and cost
(latency, API calls), plus across-run SDs.

Usage:
    python scripts/run_variance_repeats.py --model llama3.2:latest --runs 3
    python scripts/run_variance_repeats.py --model qwen3:8b --runs 3 --strategies PlainLLM GoTPlainLLM
"""

import os
import sys
import time
import argparse
import warnings

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

DEFAULT_STRATEGIES = ["PlainLLM", "SimpleRAG", "GoTPlainLLM"]


def main():
    parser = argparse.ArgumentParser(description="Repeated runs for run-to-run variance (R2 minor)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from rag_system import config, RAGEvaluator
    import rag_system as rs
    config.model.generator_model = args.model
    config.model.helper_model = "deepseek-coder:6.7b"

    safe_model = args.model.replace(":", "_").replace(".", "_")
    out_dir = os.path.join("results", safe_model, "variance")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_pickle("data/class_files_df.pkl")
    if args.limit:
        df = df.head(args.limit)

    evaluator = RAGEvaluator()
    summary = []

    for strategy in args.strategies:
        cls = getattr(rs, strategy)
        for run in range(1, args.runs + 1):
            run_pkl = os.path.join(out_dir, f"{strategy}_run{run}.pkl")
            if os.path.exists(run_pkl):
                print(f"[skip] {strategy} run {run} exists")
                continue

            print(f"--- {strategy} run {run}/{args.runs} ---")
            inst = cls(index_name=config.index_names.get('simple', 'rag-docstring'),
                       namespace=f"variance-{strategy.lower()}")
            rows = []
            for i, row in df.iterrows():
                code = row["Code_without_comments"]
                t0 = time.time()
                doc, cost = inst.generate_docstring(code)
                rows.append({
                    "index": i,
                    "Comments": row.get("Comments"),
                    "Code_without_comments": code,
                    "Generated_Docstring": doc,
                    "Retrieved_Context": (inst.retrieved_contexts[-1]
                                          if getattr(inst, "retrieved_contexts", None) else ""),
                    "RAG_Method": strategy,
                    "execution_time": time.time() - t0,
                    "api_calls": getattr(cost, "api_calls", None),
                })
                if len(rows) % 20 == 0:
                    print(f"  {len(rows)}/{len(df)}")
                    pd.DataFrame(rows).to_pickle(run_pkl + ".checkpoint")

            res = pd.DataFrame(rows)
            res.to_pickle(run_pkl)

            # deterministic quality metrics only (LLM-judge excluded: known unreliable per-sample)
            q = []
            for _, r in res.iterrows():
                q.append({
                    "bert": evaluator.calculate_bert_score(str(r["Comments"]), str(r["Generated_Docstring"])),
                    "rouge": evaluator.calculate_rouge_score(str(r["Comments"]), str(r["Generated_Docstring"])),
                    "param_cov": evaluator.calculate_parameter_coverage(str(r["Code_without_comments"]),
                                                                        str(r["Generated_Docstring"])),
                })
            qd = pd.DataFrame(q)
            summary.append({
                "strategy": strategy, "run": run,
                "bert_mean": qd["bert"].mean(), "rouge_mean": qd["rouge"].mean(),
                "param_cov_mean": qd["param_cov"].mean(),
                "latency_mean": res["execution_time"].mean(),
                "latency_median": res["execution_time"].median(),
                "api_calls_mean": res["api_calls"].mean(),
            })
            pd.DataFrame(summary).to_csv(os.path.join(out_dir, "variance_summary.csv"), index=False)
            print(pd.DataFrame(summary).round(4).tail(1).to_string(index=False))

    s = pd.DataFrame(summary)
    if len(s):
        print("\n=== ACROSS-RUN SD PER STRATEGY ===")
        print(s.groupby("strategy")[["bert_mean", "rouge_mean", "param_cov_mean",
                                     "latency_mean", "api_calls_mean"]].agg(['mean', 'std']).round(4).to_string())


if __name__ == "__main__":
    main()
