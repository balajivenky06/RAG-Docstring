import os
import sys
import pandas as pd
import time
import argparse
import warnings

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.simple_rag import SimpleRAG
from rag_system.evaluator import RAGEvaluator
from rag_system.config import config

warnings.filterwarnings('ignore')

def run_ablations(limit=None, model=None):
    print("Initializing Retrieval Ablations...")
    # Update config for model if provided
    if model:
        config.model.generator_model = model
        config.model.helper_model = model
        safe_model = model.replace(":", "_").replace(".", "_")
        output_dir = os.path.join("results", safe_model, "ablations")
    else:
        output_dir = "results/ablations"

    # Ablation settings
    k_values = [1, 3, 5, 10]
    chunk_sizes = [256, 512, 1024]
    
    # Load dataset (subset for speed if limit set)
    data_path = 'data/class_files_df.pkl'
    try:
        df = pd.read_pickle(data_path)
        if limit:
            df = df.head(limit)
        print(f"Loaded dataset: {len(df)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    evaluator = RAGEvaluator()
    results = []
    for chunk in chunk_sizes:
        for k in k_values:
            print(f"\n--- Running Ablation: Top-k = {k}, Chunk Size = {chunk} ---")
            
            original_top_k = config.rag_methods.common_config['top_k']
            original_chunk_size = config.retrieval.chunk_size
            
            config.rag_methods.common_config['top_k'] = k
            config.retrieval.chunk_size = chunk
            
            rag = SimpleRAG(index_name="rag-docstring", namespace=config.pinecone.namespace)
            rag.top_k = k 
            
            print(f"RAG initialized with top_k={rag.top_k}, chunk_size={chunk}")
            
            k_results = []
            
            for i, row in df.iterrows():
                user_code = row['Code_without_comments']
                
                start_time = time.time()
                docstring, cost = rag.generate_docstring(user_code)
                latency = time.time() - start_time
                
                contexts = rag.get_retrieved_contexts()
                context = contexts[-1] if contexts else ""
                retrieved_text = str(context)
                
                faithfulness = evaluator.calculate_faithfulness_score(docstring, retrieved_text, user_code)
                token_overlap = evaluator.calculate_token_overlap_faithfulness(docstring, retrieved_text)
                
                k_results.append({
                    "k": k,
                    "chunk_size": chunk,
                    "Sample_ID": i,
                    "Latency": latency,
                    "LLM_Faithfulness": faithfulness,
                    "Token_Faithfulness": token_overlap,
                    "Docstring_Length": len(docstring),
                    "Context_Length": len(retrieved_text)
                })
                
                if (i+1) % 5 == 0:
                    print(f"  Processed {i+1}/{len(df)} samples...")
                    
            config.rag_methods.common_config['top_k'] = original_top_k
            config.retrieval.chunk_size = original_chunk_size
            
            df_k = pd.DataFrame(k_results)
            print(f"  Result k={k}, chunk={chunk}: Faithfulness={df_k['LLM_Faithfulness'].mean():.3f}, Latency={df_k['Latency'].mean():.2f}s")
            results.extend(k_results)
            
    final_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    
    final_df.to_csv(os.path.join(output_dir, "retrieval_ablations_full.csv"), index=False)
    
    summary = final_df.groupby(['chunk_size', 'k']).agg({
        'LLM_Faithfulness': ['mean', 'std'],
        'Token_Faithfulness': ['mean', 'std'],
        'Latency': ['mean', 'std'],
        'Context_Length': 'mean'
    }).round(3)
    
    print("\n=== Ablation Summary ===")
    print(summary.to_string())
    
    with open(os.path.join(output_dir, "ablation_summary.md"), "w") as f:
        f.write(summary.to_markdown())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for quick test")
    parser.add_argument("--model", type=str, default=None, help="LLM to use (e.g., qwen2.5:8b, llama3.2:latest)")
    args = parser.parse_args()
    
    run_ablations(limit=args.limit, model=args.model)
