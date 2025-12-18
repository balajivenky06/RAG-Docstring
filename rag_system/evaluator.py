"""
Comprehensive evaluation framework for RAG-based docstring generation.
"""

import re
import subprocess
import tempfile
import os
import zlib
import sys
from typing import Dict, List, Optional, Tuple
import pandas as pd
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score
import nltk
import warnings
warnings.filterwarnings('ignore')

class RAGEvaluator:
    """Comprehensive evaluator for RAG-based docstring generation."""
    
    def __init__(self):
        self.rouge = Rouge()
        nltk.download('punkt', quiet=True)
    
    def calculate_rouge_score(self, reference: str, hypothesis: str) -> float:
        """Calculate ROUGE-1 F1 score."""
        try:
            scores = self.rouge.get_scores(hypothesis.lower(), reference.lower())
            return scores[0]['rouge-1']['f']
        except Exception as e:
            print(f"ROUGE calculation error: {e}")
            return 0.0
    
    def calculate_bleu_score(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score."""
        try:
            reference_tokens = [reference.lower().split()]
            hypothesis_tokens = hypothesis.lower().split()
            return sentence_bleu(reference_tokens, hypothesis_tokens, 
                               weights=(0.25, 0.25, 0.25, 0.25))
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            return 0.0
    
    def calculate_bert_score(self, reference: str, hypothesis: str) -> float:
        """Calculate BERT score."""
        try:
            _, _, bert_score_f1 = score([reference], [hypothesis], 
                                      lang='en', model_type='bert-base-uncased')
            return bert_score_f1.item()
        except Exception as e:
            print(f"BERT score calculation error: {e}")
            return 0.0
    
    def count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = re.sub(r'[^a-zA-Z]', '', word)
        vowels = 'aeiouy'
        syllables = 0
        last_was_vowel = False
        
        for char in word:
            if char.lower() in vowels:
                if not last_was_vowel:
                    syllables += 1
                last_was_vowel = True
            else:
                last_was_vowel = False
        
        # Adjust for words ending in 'e'
        if word.endswith(('e', 'es', 'ed')):
            syllables -= 1
        
        # Ensure at least 1 syllable
        if syllables == 0:
            syllables = 1
        
        return syllables
    
    def calculate_flesch_reading_ease(self, text: str) -> float:
        """Calculate Flesch Reading Ease score."""
        sentences = text.count('.') + text.count('!') + text.count('?') + 1
        words = len(re.findall(r'\b\w+\b', text))
        
        if words == 0:
            return 0.0
        
        syllables = sum(self.count_syllables(word) for word in text.split())
        
        # Flesch Reading Ease formula
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return score
    
    def calculate_conciseness(self, reference: str, hypothesis: str) -> float:
        """Calculate conciseness using compression ratio."""
        try:
            comp_ref = zlib.compress(reference.encode())
            comp_hyp = zlib.compress(hypothesis.encode())
            return sys.getsizeof(comp_hyp) / sys.getsizeof(comp_ref)
        except Exception as e:
            print(f"Conciseness calculation error: {e}")
            return 1.0
    
    def calculate_parameter_coverage(self, code: str, docstring: str) -> Optional[float]:
        """Calculate parameter coverage in docstring."""
        # Find function/method parameters
        match = re.search(r"def\s+\w+\s*\((.*?)\):", code)
        if not match:
            match = re.search(r"async\s+def\s+\w+\s*\((.*?)\):", code)
        
        if not match:
            return None
        
        params_str = match.group(1)
        if not params_str.strip():
            return 1.0
        
        # Extract parameter names
        potential_params = [p.strip().split('=')[0].split(':')[0].strip() 
                           for p in params_str.split(',')]
        actual_params = [p for p in potential_params 
                        if p and p not in ('self', 'cls') and not p.startswith('*')]
        
        if not actual_params:
            return 1.0
        
        # Check coverage
        covered_params = 0
        docstring_lower = docstring.lower()
        
        for param_name in actual_params:
            if re.search(r"\b" + re.escape(param_name.lower()) + r"\b", docstring_lower):
                covered_params += 1
            elif f"{param_name.lower()}:" in docstring_lower or \
                 f"parameter {param_name.lower()}" in docstring_lower:
                covered_params += 1
        
        return covered_params / len(actual_params)
    
    def calculate_return_coverage(self, code: str, docstring: str) -> float:
        """Calculate return value coverage."""
        # Check if code has return statements
        has_return = False
        for line in code.splitlines():
            stripped_line = line.strip()
            if (stripped_line.startswith("return ") and 
                not stripped_line.endswith("return None") and 
                len(stripped_line) > len("return ")):
                has_return = True
                break
        
        if not has_return:
            return 1.0
        
        # Check if docstring mentions return
        docstring_lower = docstring.lower()
        return_keywords = ["return", "returns", "yield", "yields"]
        
        if any(keyword in docstring_lower for keyword in return_keywords):
            return 1.0
        else:
            return 0.0
    
    def calculate_exception_coverage(self, code: str, docstring: str) -> Optional[float]:
        """Calculate exception coverage."""
        if not docstring.strip() or docstring.startswith(("# ERROR:", "# SKIPPED:")):
            return None
        
        # Find raised exceptions
        raised_exceptions = set(re.findall(r"raise\s+(\w+)", code))
        if not raised_exceptions:
            return 1.0
        
        docstring_lower = docstring.lower()
        mentions_raises_section = "raises:" in docstring_lower
        
        covered_exceptions = 0
        for exc_name in raised_exceptions:
            if re.search(r"\b" + re.escape(exc_name.lower()) + r"\b", docstring_lower):
                covered_exceptions += 1
        
        # If "Raises:" section exists, it's good
        if mentions_raises_section and raised_exceptions:
            return 1.0
        
        return covered_exceptions / len(raised_exceptions) if raised_exceptions else 1.0
    
    
    def calculate_faithfulness_score(self, generated_docstring: str, retrieved_context: str) -> float:
        """
        Calculate faithfulness using an LLM Judge.
        Scores 0.0 to 1.0 based on factual support using a detailed rubric.
        """
        if not retrieved_context or not generated_docstring:
            return 0.5 # Neutral if no context or docstring
            
        try:
            import ollama
            from .config import config  # Import config to get helper model
            
            prompt = f"""You are a strict technical judge evaluating Python docstrings.
            
            Context from Knowledge Base:
            {retrieved_context[:2000]}
            
            Generated Docstring:
            {generated_docstring}
            
            Evaluation Rubric:
            1. Hallucination Check: Does the docstring claim parameters/returns not present in the code or context?
            2. Contradiction Check: Does it contradict the provided context logic?
            3. Support Check: Is the description supported by the context or obvious code inference?
            
            Task:
            Assign a Faithfulness Score from 0.0 to 1.0.
            - 1.0: Fully supported by context/code, no hallucinations.
            - 0.5: Partially supported, some generic descriptions.
            - 0.0: Major hallucinations or contradictions.
            
            Return ONLY the numeric score in this format: Score: <number>
            """
            
            # Use configured helper model
            response = ollama.generate(model=config.model.helper_model, prompt=prompt)
            text = response.get('response', '')
            
            match = re.search(r"Score:\s*(\d+(\.\d+)?)", text)
            if match:
                return float(match.group(1))
            else:
                return 0.5
                
        except Exception as e:
            print(f"LLM Faithfulness calculation error: {e}")
            return 0.0
    
    def check_pydocstyle_adherence(self, code: str, docstring_content: str) -> float:
        """Check adherence to PEP 257 using pydocstyle.
        
        Focuses on docstring formatting issues, not missing method docstrings
        (since we only generate class docstrings, not method docstrings).
        """
        try:
            # Sanitize content
            safe_content = docstring_content.replace('\\', '\\\\')
            safe_content = safe_content.replace('"""', '\\"\\"\\"')
            safe_content = safe_content.replace("'''", "\\'\\'\\'")
            
            # Prepare docstring for insertion
            lines = safe_content.split('\n')
            if len(lines) == 1:
                indented_docstring_body = lines[0]
            else:
                indented_docstring_body = lines[0] + '\n' + '\n'.join(['    ' + line for line in lines[1:]])
            
            # Find class or function to insert docstring
            class_match = re.search(r"^(.*\bclass\s+\w+\s*\(?.*\)?:)", code, re.MULTILINE)
            func_match = re.search(r"^(.*\b(async\s+)?def\s+\w+\s*\(?.*\)?:)", code, re.MULTILINE)
            
            if class_match:
                code_prefix = code[:class_match.end()] + f'\n    """{indented_docstring_body}"""'
                code_suffix = code[class_match.end():]
                if not code_suffix.strip():
                    code_suffix = "\n    pass" + code_suffix
                code_for_check = code_prefix + code_suffix
            elif func_match:
                code_prefix = code[:func_match.end()] + f'\n    """{indented_docstring_body}"""'
                code_suffix = code[func_match.end():]
                if not code_suffix.strip():
                    code_suffix = "\n    pass" + code_suffix
                code_for_check = code_prefix + code_suffix
            else:
                code_for_check = f'"""{docstring_content}"""\n{code}'
            
            # Write to temporary file and run pydocstyle
            tmp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp_file:
                    tmp_file.write(code_for_check)
                    tmp_file_path = tmp_file.name
                
                command = ['pydocstyle', tmp_file_path]
                process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
                
                output = process.stdout.strip()
                if output:
                    all_errors = output.splitlines()
                    # Filter out irrelevant errors:
                    # - D100: Missing docstring in public module (not applicable for test files)
                    # - D102, D103: Missing docstring in public method/function (we only generate class docstrings)
                    # - D105, D106: Missing docstring in magic/nested class methods
                    # - D107: Missing docstring in __init__ (we don't generate these)
                    irrelevant_codes = ['D100', 'D102', 'D103', 'D105', 'D106', 'D107']
                    filtered_errors = []
                    for err in all_errors:
                        # Check if error is about missing method/function docstrings
                        is_irrelevant = any(err.endswith(f"{code}:") or f": {code}:" in err for code in irrelevant_codes)
                        if not is_irrelevant:
                            filtered_errors.append(err)
                    
                    filtered_errors_count = len(filtered_errors)
                else:
                    filtered_errors_count = 0
                
                # Improved scoring: use exponential decay instead of linear cutoff
                # This allows differentiation even with multiple errors
                # Score = 1/(1 + errors) so: 0 errors = 1.0, 1 error = 0.5, 2 errors = 0.33, etc.
                if filtered_errors_count == 0:
                    return 1.0
                else:
                    # Normalize by docstring complexity (longer docstrings may have more issues)
                    docstring_lines = len(docstring_content.split('\n'))
                    normalized_errors = filtered_errors_count / max(1, docstring_lines / 5)
                    return 1.0 / (1.0 + normalized_errors)
                
            finally:
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                    
        except Exception as e:
            print(f"Pydocstyle check error: {e}")
            return 0.0
    
    def evaluate_single_sample(self, code: str, ground_truth: str, generated_docstring: str, 
                              retrieved_context: str = "") -> Dict[str, float]:
        """Evaluate a single sample across all metrics."""
        return {
            'rouge_1_f1': self.calculate_rouge_score(ground_truth, generated_docstring),
            'bleu_score': self.calculate_bleu_score(ground_truth, generated_docstring),
            'bert_score': self.calculate_bert_score(ground_truth, generated_docstring),
            'flesch_reading_ease': self.calculate_flesch_reading_ease(generated_docstring),
            'conciseness': self.calculate_conciseness(ground_truth, generated_docstring),
            'parameter_coverage': self.calculate_parameter_coverage(code, generated_docstring),
            'return_coverage': self.calculate_return_coverage(code, generated_docstring),
            'exception_coverage': self.calculate_exception_coverage(code, generated_docstring),
            'faithfulness_score': self.calculate_faithfulness_score(generated_docstring, retrieved_context),
            'pydocstyle_adherence': self.check_pydocstyle_adherence(code, generated_docstring)
        }
    
    def evaluate_dataset(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate entire dataset and return results with metrics."""
        evaluation_results = []
        
        for idx, row in results_df.iterrows():
            # Handle different column name formats
            code = row.get('code', row.get('Code_without_comments', ''))
            ground_truth = row.get('ground_truth', row.get('Comments', ''))
            generated_docstring = row.get('generated_docstring', row.get('Generated_Docstring', ''))
            retrieved_context = row.get('retrieved_context', row.get('Retrieved_Context', ''))
            rag_method = row.get('rag_method', row.get('RAG_Method', ''))
            
            metrics = self.evaluate_single_sample(
                code=code,
                ground_truth=ground_truth,
                generated_docstring=generated_docstring,
                retrieved_context=retrieved_context
            )
            
            evaluation_results.append({
                'index': idx,
                'rag_method': rag_method,
                **metrics
            })
        
        return pd.DataFrame(evaluation_results)
    
    def generate_summary_report(self, results_df: pd.DataFrame, output_file: str) -> None:
        """Generate a comprehensive summary report of evaluation results."""
        try:
            # Calculate summary statistics
            summary_stats = results_df.groupby('rag_method').agg({
                'rouge_1_f1': ['mean', 'std', 'min', 'max'],
                'bleu_4': ['mean', 'std', 'min', 'max'],
                'bert_score': ['mean', 'std', 'min', 'max'],
                'parameter_coverage': ['mean', 'std', 'min', 'max'],
                'return_coverage': ['mean', 'std', 'min', 'max'],
                'exception_coverage': ['mean', 'std', 'min', 'max'],
                'python_style_adherence': ['mean', 'std', 'min', 'max'],
                'flesch_reading_ease': ['mean', 'std', 'min', 'max'],
                'conciseness': ['mean', 'std', 'min', 'max'],
                'faithfulness_score': ['mean', 'std', 'min', 'max']
            }).round(4)
            
            # Flatten column names
            summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
            summary_stats = summary_stats.reset_index()
            
            # Save summary report as Excel
            summary_excel_file = output_file.replace('.txt', '.xlsx')
            with pd.ExcelWriter(summary_excel_file, engine='openpyxl') as writer:
                summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                # Add detailed results sheet
                results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
                
                # Add method comparison sheet
                method_comparison = results_df.groupby('rag_method').mean().round(4)
                method_comparison.to_excel(writer, sheet_name='Method_Comparison', index=True)
            
            # Generate text summary report
            with open(output_file, 'w') as f:
                f.write("RAG Docstring Generation - Summary Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Dataset Overview:\n")
                f.write(f"- Total samples: {len(results_df)}\n")
                f.write(f"- RAG methods: {', '.join(results_df['rag_method'].unique())}\n\n")
                
                f.write("Performance Summary (Mean Scores):\n")
                f.write("-" * 30 + "\n")
                
                for method in results_df['rag_method'].unique():
                    method_data = results_df[results_df['rag_method'] == method]
                    f.write(f"\n{method}:\n")
                    f.write(f"  ROUGE-1 F1: {method_data['rouge_1_f1'].mean():.4f}\n")
                    f.write(f"  BLEU-4: {method_data['bleu_4'].mean():.4f}\n")
                    f.write(f"  BERT Score: {method_data['bert_score'].mean():.4f}\n")
                    f.write(f"  Parameter Coverage: {method_data['parameter_coverage'].mean():.4f}\n")
                    f.write(f"  Return Coverage: {method_data['return_coverage'].mean():.4f}\n")
                    f.write(f"  Exception Coverage: {method_data['exception_coverage'].mean():.4f}\n")
                    f.write(f"  Python Style Adherence: {method_data['python_style_adherence'].mean():.4f}\n")
                    f.write(f"  Flesch Reading Ease: {method_data['flesch_reading_ease'].mean():.4f}\n")
                    f.write(f"  Conciseness: {method_data['conciseness'].mean():.4f}\n")
                    f.write(f"  Faithfulness Score: {method_data['faithfulness_score'].mean():.4f}\n")
                
                f.write(f"\nDetailed results saved to: {summary_excel_file}\n")
            
            print(f"Summary report generated: {output_file}")
            print(f"Excel summary generated: {summary_excel_file}")
            
        except Exception as e:
            print(f"Error generating summary report: {e}")
