
import os
import sys
from pinecone import Pinecone

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_system.config import config

def list_indexes():
    api_key = config.pinecone.api_key
    if not api_key:
        print("Error: Pinecone API KEY not found in config.")
        return

    pc = Pinecone(api_key=api_key)
    indexes = pc.list_indexes()
    print("Existing Pinecone Indexes:")
    for idx in indexes:
        print(f"- {idx['name']}")

if __name__ == "__main__":
    list_indexes()
