import os
import sys
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.simple_rag import SimpleRAG
from rag_system.plain_llm import PlainLLM
from rag_system.evaluator import RAGEvaluator
from rag_system.config import config

def generate_human_eval_sheet(num_samples=50):
    print("Initializing Human Evaluation Sheet Generation...")
    
    # Load dataset
    data_path = os.path.join(config.paths.data_dir, 'class_files_df.pkl') if hasattr(config.paths, 'data_dir') else 'data/class_files_df.pkl'
    try:
        df = pd.read_pickle(data_path)
        print(f"Loaded dataset with {len(df)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Check if we have enough samples
    if len(df) < num_samples:
        print(f"Warning: Dataset size ({len(df)}) is smaller than requested samples ({num_samples}). Using all samples.")
        num_samples = len(df)
        
    # Randomly sample, but let's take a stratified sample from existing results if possible, 
    # otherwise we generate fresh ones. 
    # To save time and API costs, let's try to load existing evaluated results first!
    
    results_dir = os.path.join(config.paths.results_dir, "comparison_SimpleRAG")
    eval_file = os.path.join(results_dir, "SimpleRAG_evaluated.pkl")
    
    eval_data = []
    
    if os.path.exists(eval_file):
        print(f"Found existing evaluated results at {eval_file}. Sampling {num_samples} rows from there...")
        results_df = pd.read_pickle(eval_file)
        if len(results_df) > num_samples:
            results_df = results_df.sample(n=num_samples, random_state=42).reset_index(drop=True)
            
        for i, row in results_df.iterrows():
            eval_data.append({
                "Sample_ID": row.get('index', i),
                "Method": "SimpleRAG",
                "Code_Snippet": str(row.get('code', ''))[:800] + "...",
                "Retrieved_Context": str(row.get('retrieved_context', ''))[:500] + "...",
                "Generated_Docstring": row.get('generated_docstring', ''),
                "LLM_Faithfulness_Score": row.get('faithfulness_score', ''),
                "Token_Overlap_Score": row.get('token_overlap_score', ''),
                "HUMAN_Faithfulness_Score (0.0 to 1.0)": "", 
                "HUMAN_Notes": ""
            })
            
    else:
        print("No existing evaluated results found. Generating fresh docstrings... This may take a while.")
        sample_df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
        rag = SimpleRAG(index_name=config.pinecone.namespace)
        evaluator = RAGEvaluator()
        
        for i, row in sample_df.iterrows():
            print(f"Processing {i+1}/{num_samples}...")
            user_code = row['Code_without_comments']
            docstring, _ = rag.generate_docstring(user_code)
            
            contexts = rag.get_retrieved_contexts()
            context = contexts[-1] if contexts else ""
            
            # Evaluate using LLM judge directly
            faith_score = evaluator.calculate_faithfulness_score(docstring, context, user_code)
            token_score = evaluator.calculate_token_overlap_faithfulness(docstring, context)
            
            eval_data.append({
                "Sample_ID": i + 1,
                "Method": "SimpleRAG",
                "Code_Snippet": user_code[:800] + "...",
                "Retrieved_Context": str(context)[:500] + "...",
                "Generated_Docstring": docstring,
                "LLM_Faithfulness_Score": faith_score,
                "Token_Overlap_Score": token_score,
                "HUMAN_Faithfulness_Score (0.0 to 1.0)": "", 
                "HUMAN_Notes": ""
            })
        
    # Create DataFrame
    eval_df = pd.DataFrame(eval_data)
    
    # Save to CSV and Excel
    output_dir = "results/human_eval"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "human_eval_sheet.csv")
    excel_path = os.path.join(output_dir, "human_eval_sheet.xlsx")
    
    eval_df.to_csv(csv_path, index=False)
    eval_df.to_excel(excel_path, index=False)
    
    print(f"\n✅ Generated human evaluation sheet with {len(eval_df)} samples.")
    print(f"Excel: {excel_path}")
    print("\nNext Steps for Evaluator:")
    print("1. Open the Excel file.")
    print("2. Read the Generated Docstring, comparing it against the Code and Context.")
    print("3. Enter your score from 0.0 to 1.0 in the 'HUMAN_Faithfulness_Score' column.")
    print("4. Save the file. We will use it later to generate the Human vs Judge Correlation chart.")

if __name__ == "__main__":
    generate_human_eval_sheet()
