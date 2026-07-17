"""
Build a 1,400-class benchmark corpus (data/class_files_df_1400.pkl) with the
same schema and transformations as the original 67-class benchmark
(data/class_files_df.pkl):

    Full_code             raw class source segment (docstring included)
    Comments              the class-level docstring, triple-quoted
    Code_without_comments AST-normalized class source with all docstrings removed
    Clean_classes         same, with class name -> dummy_class_1 and each
                          method declaration -> dummy_def_N

Sources (downloaded tarballs, see download_sources): Keras, scikit-learn,
the Flask/pallets ecosystem, and general utilities (NumPy, pandas).
Target composition mirrors the study's planned distribution:
TF/Keras 420 (30%), scikit-learn 280 (20%), Flask 70 (5%), other 630 (45%).

Filters: top-level classes, parseable, class docstring >= 200 chars,
20 <= raw lines <= 1,100, not in test paths, deduplicated, and excluding any
class already present in the original 67-class benchmark.

This corpus is for the scale-up REPLICATION runs; it does not alter the
67-class benchmark that the submitted results are computed on.

Usage:
    python scripts/build_1400_corpus.py             # download + build
    python scripts/build_1400_corpus.py --no-download
"""

import argparse
import ast
import copy
import hashlib
import io
import os
import random
import re
import sys
import tarfile
import urllib.request
import warnings

import pandas as pd

warnings.filterwarnings('ignore')

SOURCES = {
    # label -> (github tarball url, subdirs to scan)
    "keras":   ("https://codeload.github.com/keras-team/keras/tar.gz/refs/tags/v3.3.3", ["keras"]),
    "tfkeras": ("https://codeload.github.com/keras-team/tf-keras/tar.gz/refs/tags/v2.16.0", ["tf_keras"]),
    "sklearn": ("https://codeload.github.com/scikit-learn/scikit-learn/tar.gz/refs/tags/1.5.0", ["sklearn"]),
    "flask":   ("https://codeload.github.com/pallets/flask/tar.gz/refs/tags/3.0.3", ["src"]),
    "werkzeug":("https://codeload.github.com/pallets/werkzeug/tar.gz/refs/tags/3.0.3", ["src"]),
    "jinja":   ("https://codeload.github.com/pallets/jinja/tar.gz/refs/tags/3.1.4", ["src"]),
    "numpy":   ("https://codeload.github.com/numpy/numpy/tar.gz/refs/tags/v1.26.4", ["numpy"]),
    "pandas":  ("https://codeload.github.com/pandas-dev/pandas/tar.gz/refs/tags/v2.2.2", ["pandas"]),
    "scipy":   ("https://codeload.github.com/scipy/scipy/tar.gz/refs/tags/v1.13.1", ["scipy"]),
    "matplotlib": ("https://codeload.github.com/matplotlib/matplotlib/tar.gz/refs/tags/v3.9.0", ["lib"]),
    "networkx":("https://codeload.github.com/networkx/networkx/tar.gz/refs/tags/networkx-3.3", ["networkx"]),
    "sympy":   ("https://codeload.github.com/sympy/sympy/tar.gz/refs/tags/sympy-1.12.1", ["sympy"]),
}
GROUPS = {  # label -> composition group
    "keras": "TensorFlow/Keras", "tfkeras": "TensorFlow/Keras", "sklearn": "scikit-learn",
    "flask": "Flask", "werkzeug": "Flask", "jinja": "Flask",
    "numpy": "Other", "pandas": "Other", "scipy": "Other",
    "matplotlib": "Other", "networkx": "Other", "sympy": "Other",
}
TARGETS = {"TensorFlow/Keras": 420, "scikit-learn": 280, "Flask": 70, "Other": 630}
CACHE_DIR = "data/corpus_sources"
OUT_PKL = "data/class_files_df_1400.pkl"
SEED = 42


def download_sources():
    os.makedirs(CACHE_DIR, exist_ok=True)
    for label, (url, _) in SOURCES.items():
        dest = os.path.join(CACHE_DIR, f"{label}.tar.gz")
        if os.path.exists(dest):
            print(f"  [cached] {label}")
            continue
        print(f"  downloading {label} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "corpus-builder"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"  [ok] {label} ({os.path.getsize(dest)//1024} KiB)")


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node):
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)


def anonymize(tree):
    """Rename the top-level class to dummy_class_1 and each def to dummy_def_N
    (declaration names only, matching the original benchmark's convention)."""
    tree = copy.deepcopy(tree)
    tree.name = "dummy_class_1"
    counter = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            counter += 1
            node.name = f"dummy_def_{counter}"
    return tree


def extract_from_source(py_source, path):
    rows = []
    try:
        module = ast.parse(py_source)
    except SyntaxError:
        return rows
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        doc = ast.get_docstring(node, clean=False)
        if not doc or len(doc) < 200:
            continue
        seg = ast.get_source_segment(py_source, node)
        if seg is None:
            continue
        n_lines = seg.count("\n") + 1
        if not (20 <= n_lines <= 1100):
            continue
        try:
            stripped = _StripDocstrings().visit(copy.deepcopy(node))
            # match the original benchmark: source starts at the `class` line, and
            # decorators (e.g. @keras_export('real.Name')) must not leak the
            # anonymized class name into the blind inputs
            stripped.decorator_list = []
            ast.fix_missing_locations(stripped)
            code_wo = ast.unparse(stripped)
            clean = ast.unparse(anonymize(stripped))
        except Exception:
            continue
        rows.append({
            "Full_code": seg,
            "Comments": '"""' + doc + '"""',
            "Code_without_comments": code_wo,
            "Clean_classes": clean,
            "_class_name": node.name,
            "_path": path,
            "_lines": n_lines,
        })
    return rows


def harvest(label):
    url, subdirs = SOURCES[label]
    dest = os.path.join(CACHE_DIR, f"{label}.tar.gz")
    rows = []
    with tarfile.open(dest, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".py"):
                continue
            parts = member.name.split("/")
            if len(parts) < 2 or parts[1] not in [s.split("/")[0] for s in subdirs]:
                continue
            low = member.name.lower()
            if any(t in low for t in ("/tests/", "/test_", "conftest", "/benchmarks/", "/examples/", "/_vendor/")):
                continue
            try:
                src = tf.extractfile(member).read().decode("utf-8", errors="replace")
            except Exception:
                continue
            for r in extract_from_source(src, member.name):
                r["_library"] = label
                r["_group"] = GROUPS[label]
                rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    if not args.no_download:
        print("Downloading source tarballs...")
        download_sources()

    print("Harvesting classes...")
    all_rows = []
    for label in SOURCES:
        rows = harvest(label)
        print(f"  {label:9s} {len(rows):5d} candidate classes")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # dedupe (identical normalized code) and exclude the original 67 benchmark classes
    df["_hash"] = df["Code_without_comments"].map(lambda s: hashlib.sha1(" ".join(s.split()).encode()).hexdigest())
    df = df.drop_duplicates("_hash")
    orig = pd.read_pickle("data/class_files_df.pkl")
    orig_hashes = set(hashlib.sha1(" ".join(str(s).split()).encode()).hexdigest()
                      for s in orig["Code_without_comments"])
    orig_names = set()
    for s in orig["Full_code"].astype(str):
        m = re.match(r"class\s+(\w+)", s)
        if m:
            orig_names.add(m.group(1))
    before = len(df)
    df = df[~df["_hash"].isin(orig_hashes) & ~df["_class_name"].isin(orig_names)]
    print(f"deduped: {before} -> {len(df)} (removed duplicates and 67-benchmark overlaps)")

    # stratified sample to targets
    rng = random.Random(SEED)
    picked = []
    for group, n in TARGETS.items():
        pool = df[df["_group"] == group]
        if len(pool) < n:
            print(f"  [warn] {group}: only {len(pool)} available (target {n}) — taking all")
            picked.append(pool)
        else:
            idx = rng.sample(list(pool.index), n)
            picked.append(pool.loc[idx])
        print(f"  {group:18s} pool={len(pool):5d} -> sampled {min(n, len(pool))}")
    out = pd.concat(picked).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    meta = out[["_class_name", "_library", "_group", "_path", "_lines"]].copy()
    final = out[["Full_code", "Comments", "Code_without_comments", "Clean_classes"]].reset_index(drop=True)
    final.to_pickle(OUT_PKL)
    meta.to_csv(OUT_PKL.replace(".pkl", "_manifest.csv"), index=True)

    print(f"\nSaved {len(final)} classes -> {OUT_PKL}")
    print(meta["_group"].value_counts().to_string())
    print(f"lines/class: median {meta['_lines'].median():.0f}, max {meta['_lines'].max()}")


if __name__ == "__main__":
    main()
