
import sys
import os
import logging
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.advanced_rag import ToTRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def test_tot_rag():
    print(f"\n{'='*50}")
    print(f"Testing ToTRAG (RAG + Tree of Thoughts)")
    print(f"{'='*50}")
    
    try:
        # Initialize ToTRAG (uses real Pinecone config)
        rag = ToTRAG(index_name="rag-docstring")
        
        # Disable strict pinecone check for this test if possible, or assume it works
        # We really just want to check the API counting logic in the reasoning loop
        
        sample_code = """
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
        """
        
        print(f"\nGenerating docstring...")
        start = time.time()
        
        # ToTRAG generation
        docstring, metrics = rag.generate_docstring(sample_code)
        
        print(f"\n[Metrics]")
        print(f"API Calls: {metrics.api_calls}")
        print(f"Execution Time: {metrics.execution_time:.2f}s")
        
        if metrics.api_calls > 5:
            print("✅  ToTRAG API Call tracking is working (Count > 5)")
        else:
            print(f"⚠️  WARNING: ToTRAG API calls low ({metrics.api_calls}). Check advanced_rag.py")

    except Exception as e:
        print(f"❌ Error testing ToTRAG: {e}")

if __name__ == "__main__":
    test_tot_rag()
