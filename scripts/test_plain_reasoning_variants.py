
import sys
import os
import logging
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.plain_llm import CoTPlainLLM, ToTPlainLLM, GoTPlainLLM

# Configure logging to show only INFO
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def run_test(model_class, name, samples):
    print(f"\n{'='*50}")
    print(f"Testing {name}")
    print(f"{'='*50}")
    
    try:
        llm = model_class()
        # Override Pinecone init just in case (though PlainLLM should handle it)
        llm._initialize_pinecone_index = lambda: None
        llm._load_data_into_pinecone = lambda: None
        
        for i, code in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            print(f"Code Length: {len(code)} chars")
            
            start = time.time()
            docstring, metrics = llm.generate_docstring(code)
            
            print(f"\n[Metrics]")
            print(f"API Calls: {metrics.api_calls}")
            print(f"Execution Time: {metrics.execution_time:.2f}s")
            print(f"Docstring Length: {len(docstring)} chars")
            
            if metrics.api_calls <= 1 and name in ["ToT", "GoT"]:
                print("⚠️  WARNING: API calls seems low for this reasoning method!")
            elif metrics.api_calls > 1:
                print("✅  API Call tracking seems to be working (Multi-turn verified)")
                
    except Exception as e:
        print(f"❌ Error testing {name}: {e}")

if __name__ == "__main__":
    
    sample1 = """
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    """

    sample2 = """
    class BinarySearchTree:
        def __init__(self):
            self.root = None
            
        def insert(self, val):
            if not self.root:
                self.root = Node(val)
            else:
                self._insert_recursive(self.root, val)
                
        def _insert_recursive(self, node, val):
            if val < node.val:
                if node.left:
                    self._insert_recursive(node.left, val)
                else:
                    node.left = Node(val)
            else:
                if node.right:
                    self._insert_recursive(node.right, val)
                else:
                    node.right = Node(val)
    """
    
    samples = [sample1, sample2]

    # Run tests
    run_test(CoTPlainLLM, "CoT (Chain of Thought)", samples)
    run_test(ToTPlainLLM, "ToT (Tree of Thoughts)", samples)
    run_test(GoTPlainLLM, "GoT (Graph of Thoughts)", samples)
