
import sys
import os
import logging

# Add parent directory to path so we can import rag_system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.plain_llm import PlainLLM

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_plain_llm():
    print("Initializing PlainLLM...")
    llm = PlainLLM()
    
    print(f"Generator Model: {llm.model_config.generator_model}")
    print(f"Helper Model: {llm.model_config.helper_model}")
    
    sample_code = """
    def calculate_fibonacci(n):
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
    """
    
    print("\nGenerating docstring for sample code...")
    docstring, metrics = llm.generate_docstring(sample_code)
    
    print("\n--- Generated Docstring ---")
    print(docstring)
    print("---------------------------")
    print("\n--- Cost Metrics ---")
    print(f"Execution Time: {metrics.execution_time:.3f}s")
    print(f"API Calls: {metrics.api_calls}")

if __name__ == "__main__":
    test_plain_llm()
