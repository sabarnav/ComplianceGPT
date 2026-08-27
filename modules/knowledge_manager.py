import chromadb
from sentence_transformers import SentenceTransformer
from modules.document_processor import DocumentProcessor
import os
import json

class KnowledgeManager:
    def __init__(self, collection_name="compliance_docs"):
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_processor = DocumentProcessor()
    
    def add_document(self, file_path: str) -> int:
        \"\"\"Add a document to the knowledge base\"\"\"
        try:
            print(f"   📄 Processing: {os.path.basename(file_path)}")
            
            # Process document
            processed = self.doc_processor.process_document(file_path)
            
            if not processed['chunks']:
                print(f"   ⚠️ No chunks extracted from {file_path}")
                return 0
            
            chunks = processed['chunks']
            texts = [chunk['text'] for chunk in chunks]
            
            # Filter empty texts
            valid_texts = []
            valid_chunks = []
            for i, text in enumerate(texts):
                if text and len(text.strip()) > 10:
                    valid_texts.append(text)
                    valid_chunks.append(chunks[i])
            
            if not valid_texts:
                print(f"   ⚠️ No valid text chunks from {file_path}")
                return 0
            
            # Generate embeddings
            print(f"   🔄 Generating embeddings for {len(valid_texts)} chunks...")
            embeddings = self.embedding_model.encode(valid_texts).tolist()
            
            # Prepare for ChromaDB
            file_hash = hashlib.md5(processed['file_name'].encode()).hexdigest()[:8]
            ids = [f"{file_hash}_{i}" for i in range(len(valid_chunks))]
            
            metadatas = [
                {
                    'source': processed['file_name'],
                    'chunk_id': chunk['chunk_id'],
                    'chunk_size': chunk['chunk_size']
                }
                for chunk in valid_chunks
            ]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=valid_texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"   ✅ Added {len(valid_chunks)} chunks")
            return len(valid_chunks)
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return 0
    
    def clear_all(self):
        \"\"\"Clear all documents\"\"\"
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("🗑️ Knowledge base cleared")
    
    def get_document_count(self) -> int:
        \"\"\"Get number of documents\"\"\"
        try:
            return self.collection.count()
        except:
            return 0
