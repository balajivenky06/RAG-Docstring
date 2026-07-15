"""
Retrieval-corpus ablation (R3.1): library-specific corpus vs generic style-guide corpus.

The paper's RAG knowledge base is 21 generic Python documentation sources
(PEP 257, style guides, tutorials). Reviewer 3 argues the "RAG does not help"
conclusion is unfair because the corpus contains no target-codebase context.
This script builds a LIBRARY-SPECIFIC corpus from the benchmark libraries'
guides and developer documentation (TensorFlow/Keras subclassing and base-layer
docs, scikit-learn developer/estimator documentation, Flask API docs) and runs
the stock SimpleRAG pipeline against it, changing ONLY the retrieval source.

Leakage control: only guide/convention/developer pages are indexed — never the
per-class API reference pages, which contain the benchmark classes' own
(ground-truth) docstrings.

Implementation: a stock SimpleRAG instance is created and its Pinecone index is
replaced with a local dense index exposing the same .query() interface, so the
enriched-query construction, context-rewrite call, and final generation prompt
are byte-identical to the paper's SimpleRAG condition.

Usage:
    python scripts/run_corpus_ablation.py --model llama3.2:latest
    python scripts/run_corpus_ablation.py --model phi4:14b --limit 10
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

LIBRARY_CORPUS_URLS = [
    # TensorFlow / Keras (67% of benchmark classes)
    "https://keras.io/guides/making_new_layers_and_models_via_subclassing/",
    "https://keras.io/api/layers/base_layer/",
    "https://keras.io/guides/writing_your_own_callbacks/",
    "https://keras.io/api/optimizers/base_optimizer/",
    "https://keras.io/guides/serialization_and_saving/",
    "https://www.tensorflow.org/guide/keras/custom_layers_and_models",
    # scikit-learn (31% of benchmark classes)
    "https://scikit-learn.org/stable/developers/develop.html",
    "https://scikit-learn.org/stable/glossary.html",
    "https://scikit-learn.org/stable/developers/contributing.html",
    "https://scikit-learn.org/stable/common_pitfalls.html",
    # Flask
    "https://flask.palletsprojects.com/en/stable/api/",
    "https://flask.palletsprojects.com/en/stable/quickstart/",
]

CORPUS_CACHE = "data/library_corpus_chunks.pkl"


def scrape_corpus():
    import requests
    from bs4 import BeautifulSoup

    docs = []
    for url in LIBRARY_CORPUS_URLS:
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (research corpus builder)"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
            if len(text) > 500:
                docs.append({"source": url, "text": text})
                print(f"  scraped {url} ({len(text)} chars)")
            else:
                print(f"  [warn] {url}: too little text, skipped")
        except Exception as e:
            print(f"  [warn] {url}: {e}")
    return docs


def chunk_documents(docs, chunk_words=750, overlap_words=150):
    """~1,000 tokens / 200 overlap, approximated in words (matches paper's chunking)."""
    chunks = []
    for d in docs:
        words = d["text"].split()
        step = chunk_words - overlap_words
        for start in range(0, max(1, len(words) - overlap_words), step):
            piece = " ".join(words[start:start + chunk_words])
            if len(piece) > 200:
                chunks.append({"source": d["source"], "text": piece})
    return chunks


class _Match:
    def __init__(self, text, source, score):
        self.metadata = {"text": text, "source": source}
        self.score = score


class _Result:
    def __init__(self, matches):
        self.matches = matches


class LocalDenseIndex:
    """Drop-in replacement for the Pinecone index used by SimpleRAG."""

    def __init__(self, chunks, embeddings, similarity_threshold=0.7):
        self.chunks = chunks
        self.embeddings = embeddings  # L2-normalized, (n, d)
        self.threshold = similarity_threshold

    def query(self, vector, top_k=3, include_metadata=True, namespace=None):
        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self.embeddings @ q
        order = np.argsort(sims)[::-1][:top_k]
        matches = [_Match(self.chunks[i]["text"], self.chunks[i]["source"], float(sims[i]))
                   for i in order if sims[i] >= self.threshold]
        return _Result(matches)


def build_index(embedding_model):
    if os.path.exists(CORPUS_CACHE):
        chunks = pd.read_pickle(CORPUS_CACHE)
        print(f"loaded cached corpus: {len(chunks)} chunks")
    else:
        print("Scraping library-specific corpus...")
        docs = scrape_corpus()
        chunks = chunk_documents(docs)
        pd.to_pickle(chunks, CORPUS_CACHE)
        print(f"corpus built: {len(docs)} pages -> {len(chunks)} chunks (cached)")

    texts = [c["text"] for c in chunks]
    emb = embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return LocalDenseIndex(chunks, np.asarray(emb, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description="Library-specific retrieval corpus ablation (R3.1)")
    parser.add_argument("--model", required=True, help="Generator model tag (e.g. llama3.2:latest)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from rag_system import SimpleRAG, RAGEvaluator, config
    config.model.generator_model = args.model
    config.model.helper_model = "deepseek-coder:6.7b"

    safe_model = args.model.replace(":", "_").replace(".", "_")
    out_dir = os.path.join("results", safe_model, "ablations", "corpus_library")
    os.makedirs(out_dir, exist_ok=True)
    out_pkl = os.path.join(out_dir, "SimpleRAG_librarycorpus_results.pkl")
    if os.path.exists(out_pkl):
        print(f"[skip] {out_pkl} already exists")
        return

    rag = SimpleRAG(index_name=config.index_names.get('simple', 'rag-docstring'),
                    namespace="corpus-ablation")
    rag.pinecone_index = build_index(rag.embedding_model)
    print("SimpleRAG retrieval backend swapped to library-specific local index")

    df = pd.read_pickle("data/class_files_df.pkl")
    if args.limit:
        df = df.head(args.limit)

    rows = []
    for i, row in df.iterrows():
        code = row["Code_without_comments"]
        t0 = time.time()
        doc, cost = rag.generate_docstring(code)
        ctx = rag.retrieved_contexts[-1] if rag.retrieved_contexts else ""
        rows.append({
            "index": i,
            "Full_code": row.get("Full_code"),
            "Comments": row.get("Comments"),
            "Code_without_comments": code,
            "Generated_Docstring": doc,
            "Retrieved_Context": ctx,
            "RAG_Method": "SimpleRAG_librarycorpus",
            "execution_time": time.time() - t0,
            "api_calls": cost.api_calls if hasattr(cost, "api_calls") else None,
        })
        if (len(rows)) % 10 == 0 or len(rows) == len(df):
            print(f"  {len(rows)}/{len(df)} generated")
            pd.DataFrame(rows).to_pickle(out_pkl + ".checkpoint")

    res = pd.DataFrame(rows)
    res.to_pickle(out_pkl)
    res.to_excel(out_pkl.replace(".pkl", ".xlsx"), index=False)

    print("Evaluating (deterministic metrics)...")
    evaluator = RAGEvaluator()
    ev = evaluator.evaluate_dataset(res)
    ev.to_pickle(os.path.join(out_dir, "SimpleRAG_librarycorpus_evaluated.pkl"))
    ev.to_excel(os.path.join(out_dir, "SimpleRAG_librarycorpus_evaluated.xlsx"), index=False)
    print(f"done -> {out_dir}")
    print(ev[["bert_score", "rouge_1_f1", "parameter_coverage", "faithfulness_score"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
