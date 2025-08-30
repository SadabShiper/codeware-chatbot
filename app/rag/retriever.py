from typing import List, Dict, Any
from app.rag.faiss_store import FAISSStore
import os
import json
from pathlib import Path

class RAGRetriever:
    def __init__(self, persist_directory: str = "./faiss_db"):
        self.faiss_store = FAISSStore(persist_directory)
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """
        Initialize the knowledge base with JSON data if not already done
        """
        # Check if store is empty
        if len(self.faiss_store.documents) == 0:
            # Load JSON data
            json_path = Path(__file__).parent.parent / "data" / "codeware_bot_flow.json"
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Ingest data into FAISS store
            self.faiss_store.ingest_json_data(json_data)
            print("Knowledge base initialized with JSON data")
        else:
            print("Knowledge base already initialized")
    
    def retrieve_relevant_info(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant information from the knowledge base
        """
        return self.faiss_store.query(query, n_results=top_k)
    
    def format_context(self, retrieved_info: List[Dict[str, Any]]) -> str:
        """
        Format retrieved information into context for the LLM
        """
        if not retrieved_info:
            return "No relevant information found in knowledge base."
        
        context_parts = ["Relevant information from our knowledge base:"]
        
        for i, info in enumerate(retrieved_info):
            context_parts.append(f"\n{i+1}. {info['document']}")
            if 'metadata' in info and 'type' in info['metadata']:
                context_parts.append(f"   (Source: {info['metadata']['type']})")
        
        return "\n".join(context_parts)