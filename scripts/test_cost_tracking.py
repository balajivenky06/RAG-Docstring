import sys
import os
import logging
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.advanced_rag import ToTRAG
from rag_system.config import config

# Setup mock logging
logging.basicConfig(level=logging.INFO)

def test_tot_cost_tracking():
    print("Testing ToTRAG cost tracking...")
    
    # Mock Ollama Client
    mock_ollama = MagicMock()
    # Setup mock response for generate/chat
    mock_ollama.generate.return_value = {'response': 'Mock response'}
    mock_ollama.chat.return_value = {'message': {'content': 'Mock docstring'}}
    
    # Initialize implementation
    try:
        rag = ToTRAG(index_name="test-index")
        
        # Inject mock client
        rag.ollama_client = mock_ollama
        
        # Force helper count to be high enough to simulate decomposition
        # ToT: 1 decomp + 3 tasks * (3 candidates + 1 eval) + 1 synthesis = 1 + 12 = 13 calls minimum?
        # Let's see actual implementation:
        # 1. Decompose -> 1 call
        # 2. For each task (approx 3 tasks):
        #    - Generate 3 candidates -> 3 calls
        #    - Evaluate 3 candidates -> 3 calls
        #    Total per task = 6 calls
        #    Total for 3 tasks = 18 calls
        # 3. Final synthesis -> 1 call
        # Total expected: 1 + 18 + 1 = 20 calls.
        
        user_code = "def hello(): pass"
        
        # Run generation
        docstring, cost = rag.generate_docstring(user_code)
        
        print(f"Reported API Calls: {cost.api_calls}")
        
        if cost.api_calls > 5:
            print("✅ SUCCESS: API calls > 5, indicating multi-step reasoning is tracked.")
        else:
            print(f"❌ FAILURE: API calls = {cost.api_calls}. Expected > 5.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        # Print codebase to debug if needed
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Disable actual Pinecone connection for test
    with patch('rag_system.base_rag.Pinecone'):
        with patch('rag_system.base_rag.SentenceTransformer'):
            test_tot_cost_tracking()
