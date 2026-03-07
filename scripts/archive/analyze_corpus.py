import os
import sys
import pandas as pd
from collections import defaultdict
from bs4 import BeautifulSoup
import requests
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.config import config
from rag_system.base_rag import BaseRAG

def categorize_url(url):
    """Categorize URL into source types."""
    url_lower = url.lower()
    
    if "pep" in url_lower:
        return "Style Guide (PEP)"
    elif "styleguide" in url_lower or "coding-style" in url_lower:
        return "Style Guide"
    elif "stackoverflow" in url_lower:
        return "Community Q&A"
    elif "tutorial" in url_lower or "guide" in url_lower or "blog" in url_lower or "posts" in url_lower:
        return "Tutorial/Blog"
    elif "github" in url_lower and ".py" in url_lower:
        return "Code Repository"
    elif "docs" in url_lower or "readthedocs" in url_lower:
        return "API Reference"
    else:
        return "Other"

def analyze_corpus():
    print("Initializing Corpus Analysis...")
    
    # Initialize RAG to access Pinecone
    # We use 'simple' index as default
    from rag_system.simple_rag import SimpleRAG
    rag = SimpleRAG(index_name=config.index_names['simple'], namespace=config.index_namespaces['simple'])
    
    if not rag.pinecone_index:
        print("❌ Error: Could not connect to Pinecone index.")
        return
    
    print(f"Connected to index: {rag.index_name}")
    
    stats = rag.pinecone_index.describe_index_stats()
    print(f"Index Stats: {stats}")
    
    # We can't easily iterate all vectors in Pinecone without listing them (which is slow/expensive for large indexes).
    # However, we know the URLs we loaded from config.knowledge_base_urls.
    # We will analyze based on the Source URLs defined in the config, which represents the intended corpus.
    
    print("\n--- Corpus Composition Analysis (Based on Source URLs) ---")
    
    data = []
    
    for url in config.knowledge_base_urls:
        category = categorize_url(url)
        data.append({
            "Source URL": url,
            "Category": category
        })
        
    df = pd.DataFrame(data)
    
    # Aggregate stats
    summary = df['Category'].value_counts().reset_index()
    summary.columns = ['Source Type', 'Count']
    summary['Percentage'] = (summary['Count'] / len(df) * 100).round(1)
    
    print("\n=== Corpus Composition Table ===")
    print(summary.to_markdown(index=False))
    
    # Save to file
    output_file = "corpus_composition_table.md"
    with open(output_file, "w") as f:
        f.write("# Retrieval Corpus Composition\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## Detailed Sources\n\n")
        f.write(df.to_markdown(index=False))
        
    print(f"\nSaved composition table to {output_file}")
    
    # "Retrieval Evidence Audit" Simulation
    print("\n--- Retrieval Evidence Audit (Simulation) ---")
    # Simulate queries to see what gets retrieved
    test_queries = [
        "How to document a class?",
        "What is the numpy docstring format?",
        "python docstrings for exceptions",
        "example of a function docstring",
        "PEP 257 summary"
    ]
    
    audit_results = []
    
    for query in test_queries:
        print(f"Querying: '{query}'...")
        embedding = rag.embedding_model.encode(query).tolist()
        results = rag.pinecone_index.query(
            vector=embedding,
            top_k=3,
            include_metadata=True,
            namespace=rag.namespace
        )
        
        for match in results.matches:
            source = match.metadata.get('source', 'Unknown')
            category = categorize_url(source)
            score = match.score
            audit_results.append({
                "Query": query,
                "Retrieved Source": source,
                "Category": category,
                "Score": f"{score:.4f}"
            })
            
    audit_df = pd.DataFrame(audit_results)
    
    audit_file = "retrieval_evidence_audit.csv"
    audit_df.to_csv(audit_file, index=False)
    print(f"Saved retrieval audit to {audit_file}")
    
    # Print summary of retrieved categories
    retrieved_cats = audit_df['Category'].value_counts(normalize=True).mul(100).round(1)
    print("\nRetrieved Content Distribution (from Audit):")
    print(retrieved_cats.to_markdown())

if __name__ == "__main__":
    analyze_corpus()
