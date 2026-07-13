"""
Generate BLIND human re-annotation sheets from the original 51-sample eval sheet.

Addresses the judge-anchoring problem found during revision: the original sheet
displayed LLM_Judge_Faithfulness (and BERTScore/Token-Overlap) next to the
human score column, so the reported r=0.925 human-judge correlation is likely
anchored. These sheets contain ONLY the code, the docstring, and empty scoring
columns; row order is shuffled per annotator; the mapping back to
(model, strategy, sample) lives in a separate key file the annotators must not
open until scoring is complete.

Outputs (evaluation/human_eval/blind/):
    blind_sheet_annotator_A.xlsx / .csv
    blind_sheet_annotator_B.xlsx / .csv
    blind_key.csv   <-- DO NOT open while annotating

Usage:
    python scripts/generate_blind_human_eval_sheets.py
"""

import os
import pandas as pd

SRC = "evaluation/human_eval/human_eval_sheet.csv"
OUT_DIR = "evaluation/human_eval/blind"

INSTRUCTIONS = (
    "Score how faithful the docstring is to the source code, from 0.0 to 1.0. "
    "1.0 = every statement is supported by the code, all key parameters/returns/exceptions covered; "
    "0.5 = partially accurate or generic; 0.0 = hallucinated or contradicts the code. "
    "Flag Hallucination = Y if any claim is unsupported by the code. "
    "Do NOT consult the key file or any model scores until all rows are scored."
)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    src = pd.read_csv(SRC)

    blind_cols = pd.DataFrame({
        "Item_ID": range(1, len(src) + 1),
        "Code_Snippet": src["Code_Snippet"],
        "Generated_Docstring": src["Generated_Docstring"],
        "HUMAN_Faithfulness_Score (0-1)": "",
        "HUMAN_Hallucination_Noted (Y/N)": "",
        "HUMAN_Notes": "",
    })

    # key: blind Item_ID -> original identity (kept separate from annotator sheets)
    key = pd.DataFrame({
        "Item_ID": range(1, len(src) + 1),
        "Sample_ID": src["Sample_ID"],
        "Model": src["Model"],
        "Strategy": src["Strategy"],
    })
    key.to_csv(os.path.join(OUT_DIR, "blind_key.csv"), index=False)

    for annotator, seed in [("A", 7), ("B", 23)]:
        sheet = blind_cols.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        sheet.insert(0, "Row", range(1, len(sheet) + 1))
        csv_path = os.path.join(OUT_DIR, f"blind_sheet_annotator_{annotator}.csv")
        xlsx_path = os.path.join(OUT_DIR, f"blind_sheet_annotator_{annotator}.xlsx")
        sheet.to_csv(csv_path, index=False)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
            pd.DataFrame({"Instructions": [INSTRUCTIONS]}).to_excel(xw, sheet_name="README", index=False)
            sheet.to_excel(xw, sheet_name="Annotation", index=False)
        print(f"annotator {annotator}: {xlsx_path} ({len(sheet)} items, order seed={seed})")

    print(f"key file (do not open while annotating): {OUT_DIR}/blind_key.csv")


if __name__ == "__main__":
    main()
