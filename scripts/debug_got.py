
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.plain_llm import GoTPlainLLM

# Configure logging
logging.basicConfig(level=logging.INFO)

def debug_got():
    print("MRO of GoTPlainLLM:")
    for cls in GoTPlainLLM.mro():
        print(f" - {cls.__name__}")
        
    print("\nInitializing GoTPlainLLM...")
    llm = GoTPlainLLM()
    llm._initialize_pinecone_index = lambda: None
    llm._load_data_into_pinecone = lambda: None
    
    print(f"\nAttribute 'api_call_count' exists: {hasattr(llm, 'api_call_count')}")
    print(f"Initial Count: {llm.api_call_count}")
    
    # Mocking ollama_client to avoid real calls and just check logic
    class MockOllama:
        def generate(self, *args, **kwargs):
            return {'response': 'Mock Response'}
        def chat(self, *args, **kwargs):
            return {'message': {'content': 'Mock Chat'}}
            
    llm.ollama_client = MockOllama()
    
    # Run a dummy generation
    print("\nRunning generation...")
    try:
        docstring, metrics = llm.generate_docstring("def foo(): pass")
        print(f"Final Count from Metrics: {metrics.api_calls}")
        print(f"Final Count from Instance: {llm.api_call_count}")
        
        if metrics.api_calls > 0:
             print("SUCCESS: Increment happened.")
        else:
             print("FAILURE: Count is still 0.")
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_got()
