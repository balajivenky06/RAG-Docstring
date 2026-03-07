import os
import itertools
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation")


METHOD_TO_EVAL_FILE: Dict[str, str] = {
    "SimpleRAG": "simple_evaluation.pkl",
    "CodeAwareRAG": "code_aware_evaluation.pkl",
    "CorrectiveRAG": "corrective_evaluation.pkl",
    "SelfCorrectionRAG": "self_evaluation.pkl",
    "FusionRAG": "fusion_evaluation.pkl",
}


METRICS: List[str] = [
    "rouge_1_f1",
    "bleu_score",
    "bert_score",
    "flesch_reading_ease",
    "conciseness",
    "parameter_coverage",
    "return_coverage",
    "exception_coverage",
    "faithfulness_score",
    "pydocstyle_adherence",
]


def load_eval(method: str) -> pd.DataFrame:
    path = os.path.join(EVAL_DIR, METHOD_TO_EVAL_FILE[method])
    with open(path, "rb") as f:
        df = pickle.load(f)
    # ensure correct columns
    assert "index" in df.columns and "rag_method" in df.columns, f"Malformed eval for {method}"
    df = df.set_index("index").sort_index()
    # rename metric columns to include method for wide merges later if needed
    df["rag_method"] = method
    return df


def build_long_table(methods: List[str]) -> pd.DataFrame:
    frames = []
    for m in methods:
        frames.append(load_eval(m))
    long_df = pd.concat(frames, axis=0)
    # Reindex MultiIndex: (sample, method)
    long_df = long_df.reset_index().rename(columns={"index": "sample_id"})
    return long_df[["sample_id", "rag_method", *METRICS]].sort_values(["sample_id", "rag_method"]).reset_index(drop=True)


def compute_macro_averages(long_df: pd.DataFrame) -> pd.DataFrame:
    return (
        long_df.groupby("rag_method")[METRICS]
        .mean()
        .sort_values(by=["rouge_1_f1"], ascending=False)
        .reset_index()
    )


def compute_per_metric_deltas(macro_df: pd.DataFrame, baseline: str = "SimpleRAG") -> pd.DataFrame:
    df = macro_df.copy().set_index("rag_method")
    # deltas to best
    best = df[METRICS].max(axis=0)
    deltas_to_best = df[METRICS] - best
    deltas_to_best.columns = [f"delta_to_best__{c}" for c in deltas_to_best.columns]
    # deltas to baseline
    if baseline in df.index:
        base = df.loc[baseline, METRICS]
        deltas_to_baseline = df[METRICS].subtract(base, axis=1)
        deltas_to_baseline.columns = [f"delta_vs_{baseline}__{c}" for c in deltas_to_baseline.columns]
    else:
        deltas_to_baseline = pd.DataFrame(index=df.index)
    out = pd.concat([df, deltas_to_best, deltas_to_baseline], axis=1).reset_index()
    return out


def compute_ranks(long_df: pd.DataFrame) -> pd.DataFrame:
    # For each sample and metric, rank methods (higher is better)
    rows = []
    for metric in METRICS:
        for sample_id, g in long_df.groupby("sample_id"):
            vals = g[["rag_method", metric]].copy()
            # rank: 1 is best
            vals["rank"] = (-vals[metric]).rank(method="average").astype(float)
            for _, r in vals.iterrows():
                rows.append({"sample_id": sample_id, "metric": metric, "rag_method": r["rag_method"], "rank": r["rank"]})
    rank_df = pd.DataFrame(rows)
    avg_rank = rank_df.groupby(["rag_method", "metric"])['rank'].mean().reset_index()
    avg_rank_wide = avg_rank.pivot(index="rag_method", columns="metric", values="rank").reset_index()
    return rank_df, avg_rank_wide


def compute_win_rates(long_df: pd.DataFrame) -> pd.DataFrame:
    # Win = strictly highest value among methods for that sample & metric
    wins = []
    for metric in METRICS:
        for sample_id, g in long_df.groupby("sample_id"):
            max_val = g[metric].max()
            winners = g[g[metric] == max_val]["rag_method"].tolist()
            for m in long_df["rag_method"].unique():
                wins.append({
                    "sample_id": sample_id,
                    "metric": metric,
                    "rag_method": m,
                    "is_win": 1 if m in winners else 0,
                })
    win_df = pd.DataFrame(wins)
    win_rates = win_df.groupby(["rag_method", "metric"])['is_win'].mean().reset_index()
    win_rates_wide = win_rates.pivot(index="rag_method", columns="metric", values="is_win").reset_index()
    return win_df, win_rates_wide


def paired_tests(long_df: pd.DataFrame) -> pd.DataFrame:
    methods = sorted(long_df["rag_method"].unique())
    rows = []
    for m1, m2 in itertools.combinations(methods, 2):
        df1 = long_df[long_df["rag_method"] == m1].sort_values("sample_id")
        df2 = long_df[long_df["rag_method"] == m2].sort_values("sample_id")
        assert (df1["sample_id"].values == df2["sample_id"].values).all(), "Sample alignment mismatch"
        for metric in METRICS:
            x = df1[metric].values
            y = df2[metric].values
            # Filter out NaNs for fair pairwise
            mask = np.isfinite(x) & np.isfinite(y)
            x2, y2 = x[mask], y[mask]
            if len(x2) < 2:
                t_stat, t_p = np.nan, np.nan
                w_stat, w_p = np.nan, np.nan
            else:
                t_stat, t_p = stats.ttest_rel(x2, y2, nan_policy="omit")
                try:
                    w_stat, w_p = stats.wilcoxon(x2, y2, zero_method="wilcox", alternative="two-sided", method="auto")
                except Exception:
                    w_stat, w_p = np.nan, np.nan
            rows.append({
                "metric": metric,
                "method_a": m1,
                "method_b": m2,
                "ttest_stat": t_stat,
                "ttest_p": t_p,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "mean_a": np.nanmean(x2) if len(x2) else np.nan,
                "mean_b": np.nanmean(y2) if len(y2) else np.nan,
                "diff_mean": (np.nanmean(x2) - np.nanmean(y2)) if len(x2) else np.nan,
                "n_pairs": int(len(x2)),
            })
    return pd.DataFrame(rows)


def main():
    methods = list(METHOD_TO_EVAL_FILE.keys())
    long_df = build_long_table(methods)

    macro_df = compute_macro_averages(long_df)
    macro_with_deltas = compute_per_metric_deltas(macro_df, baseline="SimpleRAG")

    rank_df, avg_rank_wide = compute_ranks(long_df)
    win_df, win_rates_wide = compute_win_rates(long_df)

    tests_df = paired_tests(long_df)

    out_dir = os.path.join(RESULTS_DIR, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    # Save CSVs
    long_df.to_csv(os.path.join(out_dir, "per_sample_metrics_long.csv"), index=False)
    macro_with_deltas.to_csv(os.path.join(out_dir, "macro_averages_with_deltas.csv"), index=False)
    avg_rank_wide.to_csv(os.path.join(out_dir, "average_ranks.csv"), index=False)
    win_rates_wide.to_csv(os.path.join(out_dir, "win_rates.csv"), index=False)
    tests_df.to_csv(os.path.join(out_dir, "paired_tests.csv"), index=False)

    # Save Excel workbook
    xlsx_path = os.path.join(out_dir, "consolidated_comparison.xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        long_df.to_excel(writer, sheet_name="per_sample", index=False)
        macro_with_deltas.to_excel(writer, sheet_name="macro_averages", index=False)
        avg_rank_wide.to_excel(writer, sheet_name="average_ranks", index=False)
        win_rates_wide.to_excel(writer, sheet_name="win_rates", index=False)
        tests_df.to_excel(writer, sheet_name="paired_tests", index=False)

    print(f"Consolidated comparison written to: {out_dir}")


if __name__ == "__main__":
    main()


