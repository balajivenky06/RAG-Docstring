"""
Reconcile the N=51 human evaluation with the code-referenced re-judged scores.

Addresses reviewer R2.6: the paper's human-validation section reported strategy
orderings (SimpleRAG best, Plain worst) that contradicted the full-sample
Table 2 ordering. Hypothesis: the original full-sample faithfulness judged RAG
strategies against retrieved context (not code), while the human annotator
scored against code — the re-judged (code-referenced) scores should align with
human ratings if that explanation is correct.

Matches human-eval rows to re-judged rows by normalized generated-docstring
text (the human sheet lacks sample indices).

Usage:
    python scripts/reconcile_human_eval.py
"""

import os
import sys
import glob
import re
import warnings

import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

MODEL_LABEL_TO_DIR = {
    "Llama 3.2 8B": "llama3_2_latest",
    "Phi-4 14B": "phi4_14b",
    "Qwen3 8B": "qwen3_8b",
}

STRATEGY_LABEL_TO_NAME = {
    "Plain LLM (Base)": "PlainLLM",
    "Simple RAG (Base)": "SimpleRAG",
    "Iterative Critique RAG": "SelfCorrectionRAG",
    "Plain LLM + CoT": "CoTPlainLLM",
    "Plain LLM + GoT": "GoTPlainLLM",
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()[:300]


def main():
    sheet = pd.read_csv("evaluation/human_eval/human_eval_sheet.csv")
    sheet["human"] = pd.to_numeric(sheet["HUMAN_Faithfulness_Score (0-1)"], errors="coerce")
    sheet["norm_doc"] = sheet["Generated_Docstring"].apply(norm)

    rejudged_path = "results/rejudged/rejudged_all.csv"
    if not os.path.exists(rejudged_path):
        parts = [pd.read_csv(p) for p in glob.glob("results/rejudged/*__*.csv")]
        if not parts:
            sys.exit("No re-judged scores found yet — run scripts/rejudge_faithfulness.py first.")
        rejudged = pd.concat(parts, ignore_index=True)
    else:
        rejudged = pd.read_csv(rejudged_path)

    # attach generated docstrings to rejudged rows for text matching
    frames = []
    for (model_dir, strategy), grp in rejudged.groupby(["model_dir", "strategy"]):
        pkl = f"results/{model_dir}/comparison_{strategy}/{strategy}_results.pkl"
        if not os.path.exists(pkl):
            continue
        docs = pd.read_pickle(pkl)["Generated_Docstring"].fillna("").astype(str)
        grp = grp.copy()
        grp["norm_doc"] = grp["sample_index"].map(lambda i: norm(docs.iloc[i]) if i < len(docs) else "")
        frames.append(grp)
    rj = pd.concat(frames, ignore_index=True)

    merged = sheet.merge(
        rj[["model_dir", "strategy", "norm_doc", "faithfulness_code_ref"]],
        on="norm_doc", how="left").drop_duplicates(subset=["Sample_ID"])

    matched = merged["faithfulness_code_ref"].notna().sum()
    print(f"Matched {matched}/{len(sheet)} human-eval samples to re-judged scores")
    if matched < 30:
        print("WARNING: low match rate — inspect docstring normalization")

    m = merged.dropna(subset=["human", "faithfulness_code_ref"])
    pear = stats.pearsonr(m["human"], m["faithfulness_code_ref"])
    spear = stats.spearmanr(m["human"], m["faithfulness_code_ref"])
    mae = (m["human"] - m["faithfulness_code_ref"]).abs().mean()
    within = ((m["human"] - m["faithfulness_code_ref"]).abs() <= 0.2).mean() * 100

    print("\n=== Human vs CODE-REFERENCED judge (re-judged) ===")
    print(f"Pearson r = {pear[0]:.3f} (p={pear[1]:.2e})")
    print(f"Spearman rho = {spear[0]:.3f} (p={spear[1]:.2e})")
    print(f"MAE = {mae:.3f} | within +/-0.2: {within:.1f}%")

    print("\n=== Per-strategy means: human vs old judge vs new judge ===")
    tbl = m.groupby("Strategy").agg(
        n=("human", "size"), human=("human", "mean"),
        old_judge=("LLM_Judge_Faithfulness", "mean"),
        new_judge=("faithfulness_code_ref", "mean")).round(3)
    print(tbl.to_string())

    out = "results/rejudged/human_reconciliation.csv"
    m.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
