import faiss
import numpy as np
import pickle
import os
import json
from typing import List, Dict, Any
from pathlib import Path
from app.rag.embeddings import EmbeddingGenerator

class FAISSStore:
    def __init__(self, persist_directory: str = "./faiss_db"):
        self.persist_directory = persist_directory
        self.embedding_generator = EmbeddingGenerator()
        self.index = None
        self.documents = []
        self.metadatas = []
        self._initialize_store()
    
    def _initialize_store(self):
        """
        Initialize the FAISS store, loading existing data if available
        """
        os.makedirs(self.persist_directory, exist_ok=True)
        
        index_path = os.path.join(self.persist_directory, "index.faiss")
        meta_path = os.path.join(self.persist_directory, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(meta_path):
            # Load existing index
            self.index = faiss.read_index(index_path)
            with open(meta_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadatas = data['metadatas']
            print(f"Loaded existing FAISS index with {len(self.documents)} documents")
        else:
            # Create new index
            embedding_size = 384  # Size of multilingual MiniLM embeddings
            self.index = faiss.IndexFlatL2(embedding_size)
            print("Created new FAISS index")
    
    def save(self):
        """
        Save the FAISS index and metadata to disk
        """
        index_path = os.path.join(self.persist_directory, "index.faiss")
        meta_path = os.path.join(self.persist_directory, "metadata.pkl")
        
        faiss.write_index(self.index, index_path)
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadatas': self.metadatas
            }, f)
        
        print(f"Saved FAISS index with {len(self.documents)} documents")
    
    def ingest_json_data(self, json_data: List[Dict[str, Any]]):
        """
        Ingest JSON data into the FAISS store
        """
        documents = []
        metadatas = []
        
        for item in json_data:
            # Extract text content from JSON for embedding
            text_content = self._extract_text_from_item(item)
            if text_content:
                documents.append(text_content)
                metadatas.append({
                    "type": "flow_item", 
                    "id": item.get("id", ""),
                    "original_data": item
                })
        
        # Generate embeddings
        if documents:
            embeddings = self.embedding_generator.generate_embeddings(documents)
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Add to index
            self.index.add(embeddings_array)
            self.documents.extend(documents)
            self.metadatas.extend(metadatas)
            
            # Save to disk
            self.save()
            
            print(f"Ingested {len(documents)} documents into FAISS store")
    
    def _extract_text_from_item(self, item: Dict[str, Any]) -> str:
        """
        Extract text content from JSON item for embedding
        """
        text_parts = []
        
        # Extract message content
        if "message" in item:
            text_parts.append(item["message"])
        
        # Extract options content
        if "options" in item and isinstance(item["options"], list):
            for option in item["options"]:
                if "label" in option:
                    text_parts.append(option["label"])
                if "value" in option:
                    text_parts.append(str(option["value"]))
        
        # Extract carousel content
        if "carousel" in item and isinstance(item["carousel"], list):
            for carousel_item in item["carousel"]:
                if "title" in carousel_item:
                    text_parts.append(carousel_item["title"])
                if "options" in carousel_item and isinstance(carousel_item["options"], list):
                    for option in carousel_item["options"]:
                        if "label" in option:
                            text_parts.append(option["label"])
        
        # Extract keywords
        if "keywords" in item and isinstance(item["keywords"], list):
            text_parts.extend(item["keywords"])
        
        return " ".join(text_parts) if text_parts else ""
    
    def query(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Query the FAISS store for similar content
        """
        if len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_array, n_results)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Ensure valid index
                results.append({
                    "id": self.metadatas[idx].get("id", f"doc_{idx}"),
                    "document": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "distance": float(distances[0][i])
                })
        
        return results