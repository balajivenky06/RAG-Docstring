"""
Mixed-effects re-analysis at the per-class unit of analysis.

Addresses reviewer R2.3: prior inferential tests compared per-strategy means
(n=3/9/13), discarding per-class observations and their dependence structure.
Here we fit linear mixed-effects models on the per-class data with class as a
random intercept and architecture family, reasoning mode, and model as fixed
effects, so identical classes evaluated under every condition are handled
correctly.

Inputs:  results/<model_dir>/comparison_<Strategy>/<Strategy>_evaluated.pkl
Optional: results/rejudged/rejudged_all.csv (code-referenced faithfulness from
          scripts/rejudge_faithfulness.py) — analyzed as 'faithfulness_code_ref'
          when present.

Usage:
    python scripts/mixed_effects_analysis.py
    python scripts/mixed_effects_analysis.py --metrics bert_score faithfulness_score
"""

import os
import sys
import glob
import argparse
import warnings

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

MODEL_DIRS = {
    "llama3_2_latest": "Llama 3.2 3B",
    "phi4_14b": "Phi-4 14B",
    "qwen3_8b": "Qwen3 8B",
}

# strategy name -> (architecture family, reasoning mode)
STRATEGY_FACTORS = {
    "PlainLLM": ("Plain", "Base"),
    "FewShotPlainLLM": ("FewShot", "Base"),
    "CoTPlainLLM": ("Plain", "CoT"),
    "ToTPlainLLM": ("Plain", "ToT"),
    "GoTPlainLLM": ("Plain", "GoT"),
    "SimpleRAG": ("RAG", "Base"),
    "CoTRAG": ("RAG", "CoT"),
    "ToTRAG": ("RAG", "ToT"),
    "GoTRAG": ("RAG", "GoT"),
    "SelfCorrectionRAG": ("IterCritique", "Base"),
    "CoTSelfCorrectionRAG": ("IterCritique", "CoT"),
    "ToTSelfCorrectionRAG": ("IterCritique", "ToT"),
    "GoTSelfCorrectionRAG": ("IterCritique", "GoT"),
}

DEFAULT_METRICS = ["bert_score", "faithfulness_score", "rouge_1_f1",
                   "parameter_coverage", "return_coverage", "exception_coverage"]


def load_long_dataframe(results_root: str) -> pd.DataFrame:
    frames = []
    for model_dir, model_label in MODEL_DIRS.items():
        pattern = os.path.join(results_root, model_dir, "comparison_*", "*_evaluated.pkl")
        for path in sorted(glob.glob(pattern)):
            strategy = os.path.basename(path).replace("_evaluated.pkl", "")
            if strategy not in STRATEGY_FACTORS:
                print(f"  [warn] unknown strategy {strategy}, skipping {path}")
                continue
            df = pd.read_pickle(path).copy()
            family, reasoning = STRATEGY_FACTORS[strategy]
            df["model"] = model_label
            df["model_dir"] = model_dir
            df["strategy"] = strategy
            df["family"] = family
            df["reasoning"] = reasoning
            df["class_id"] = df["index"] if "index" in df.columns else df.index
            frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(long_df)} per-class observations "
          f"({long_df['strategy'].nunique()} strategies x {long_df['model'].nunique()} models, "
          f"{long_df['class_id'].nunique()} classes)")
    return long_df


def merge_rejudged(long_df: pd.DataFrame, results_root: str) -> pd.DataFrame:
    path = os.path.join(results_root, "rejudged", "rejudged_all.csv")
    if not os.path.exists(path):
        print("  [info] no rejudged scores found; skipping faithfulness_code_ref")
        return long_df
    rj = pd.read_csv(path)
    primary_judge = rj["judge"].mode()[0]
    rj = rj[rj["judge"] == primary_judge]
    rj = rj.rename(columns={"sample_index": "class_id"})
    merged = long_df.merge(
        rj[["model_dir", "strategy", "class_id", "faithfulness_code_ref"]],
        on=["model_dir", "strategy", "class_id"], how="left")
    n = merged["faithfulness_code_ref"].notna().sum()
    print(f"  merged {n} code-referenced faithfulness scores (judge: {primary_judge})")
    return merged


def fit_metric(long_df: pd.DataFrame, metric: str, out_lines: list):
    df = long_df.dropna(subset=[metric]).copy()
    if df.empty or df[metric].nunique() < 2:
        out_lines.append(f"\n### {metric}: insufficient data, skipped\n")
        return

    header = f"\n{'='*70}\nMetric: {metric}  (N={len(df)} observations, {df['class_id'].nunique()} classes)\n{'='*70}"
    print(header)
    out_lines.append(header)

    # Random intercept per class; family/reasoning/model fixed effects.
    # Reference levels: Plain family, Base reasoning, Llama model.
    df["family"] = pd.Categorical(df["family"], ["Plain", "FewShot", "RAG", "IterCritique"])
    df["reasoning"] = pd.Categorical(df["reasoning"], ["Base", "CoT", "ToT", "GoT"])
    model_levels = ["Llama 3.2 3B", "Phi-4 14B", "Qwen3 8B"]
    df["model"] = pd.Categorical(df["model"], [m for m in model_levels if m in set(df["model"])])

    formula = f"{metric} ~ C(family) + C(reasoning) + C(model)"
    try:
        md = smf.mixedlm(formula, df, groups=df["class_id"])
        fit = md.fit(reml=True, method="lbfgs")
        summary = fit.summary().as_text()
        print(summary)
        out_lines.append(summary)

        # Approximate marginal/conditional variance decomposition
        var_re = float(fit.cov_re.iloc[0, 0])
        var_resid = float(fit.scale)
        icc = var_re / (var_re + var_resid)
        icc_line = f"Class random-intercept ICC: {icc:.3f} (var_class={var_re:.5f}, var_resid={var_resid:.5f})"
        print(icc_line)
        out_lines.append(icc_line)
    except Exception as e:
        msg = f"  [error] mixedlm failed for {metric}: {e}"
        print(msg)
        out_lines.append(msg)


def main():
    parser = argparse.ArgumentParser(description="Mixed-effects re-analysis (class as random effect)")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--metrics", nargs="+", default=None)
    args = parser.parse_args()

    long_df = load_long_dataframe(args.results_root)
    long_df = merge_rejudged(long_df, args.results_root)

    metrics = args.metrics or list(DEFAULT_METRICS)
    if "faithfulness_code_ref" in long_df.columns and long_df["faithfulness_code_ref"].notna().any():
        metrics = ["faithfulness_code_ref"] + [m for m in metrics if m != "faithfulness_code_ref"]

    out_lines = ["Mixed-effects re-analysis (R2.3): per-class unit of analysis",
                 f"Observations: {len(long_df)}"]
    for metric in metrics:
        fit_metric(long_df, metric, out_lines)

    out_dir = os.path.join(args.results_root, "mixed_effects")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mixed_effects_report.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"\nReport saved -> {out_path}")


if __name__ == "__main__":
    main()
