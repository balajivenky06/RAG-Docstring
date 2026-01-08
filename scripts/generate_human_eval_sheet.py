import os
import sys
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.simple_rag import SimpleRAG
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
        
    # Randomly sample
    sample_df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    
    # Initialize RAG
    print("Initializing RAG system for generation...")
    rag = SimpleRAG(index_name=config.index_names['simple'])
    
    eval_data = []
    
    print(f"Generating docstrings for {num_samples} samples...")
    for i, row in sample_df.iterrows():
        print(f"Processing {i+1}/{num_samples}...")
        user_code = row['Code_without_comments']
        
        # Generate docstring
        docstring, _ = rag.generate_docstring(user_code)
        
        # Get retrieved context (last one)
        contexts = rag.get_retrieved_contexts()
        context = contexts[-1] if contexts else "No context retrieved"
        
        eval_data.append({
            "Sample_ID": i + 1,
            "Code_Snippet": user_code[:1000] + "\n..." if len(user_code) > 1000 else user_code,
            "Generated_Docstring": docstring,
            "Retrieved_Context_Snippet": str(context)[:500] + "..." if context else "None",
            "Faithfulness_Label (Supported/Unsupported)": "", # To be filled by human
            "Hallucination_Type (Parameter/Return/Logic/None)": "", # To be filled by human
            "Comments": ""
        })
        
    # Create DataFrame
    eval_df = pd.DataFrame(eval_data)
    
    # Save to CSV and Excel
    output_dir = "evaluation/human_eval"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "human_eval_sheet.csv")
    excel_path = os.path.join(output_dir, "human_eval_sheet.xlsx")
    
    eval_df.to_csv(csv_path, index=False)
    eval_df.to_excel(excel_path, index=False)
    
    print(f"\n✅ Generated human evaluation sheet with {len(eval_df)} samples.")
    print(f"CSV: {csv_path}")
    print(f"Excel: {excel_path}")
    print("\nInstructions for Evaluator:")
    print("1. Open the Excel file.")
    print("2. Read the Code and the Generated Docstring.")
    print("3. Check the Context if needed.")
    print("4. Mark 'Faithfulness_Label' as 'Supported' (fully accurate) or 'Unsupported' (contains hallucination).")
    print("5. If Unsupported, specify the type in 'Hallucination_Type'.")

if __name__ == "__main__":
    generate_human_eval_sheet()
