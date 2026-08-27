import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import os
import pickle

class HybridRetriever:
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.chunk_texts = None
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def retrieve(self, query: str, query_processor, k: int = 5) -> Tuple[List[str], List[Dict]]:
        """Hybrid retrieval with multiple strategies"""
        
        # Process query
        processed_query = query_processor.process_query(query)
        
        # Get all chunks if not cached
        if self.chunk_texts is None:
            try:
                # Try to load from cache
                self._load_cache()
            except:
                # Load from collection
                all_chunks = self.collection.get()
                self.chunk_texts = all_chunks.get('documents', [])
                if self.chunk_texts and len(self.chunk_texts) > 1:
                    self._build_tfidf()
                    self._save_cache()
        
        # If no chunks, return empty
        if not self.chunk_texts:
            return [], []
        
        # 1. Semantic Search (Dense Retrieval)
        semantic_results = self._semantic_search(
            processed_query['cleaned'], 
            n_results=20
        )
        
        # 2. Keyword Search (Sparse Retrieval)
        keyword_results = self._keyword_search(
            processed_query['expanded'],
            n_results=20
        )
        
        # 3. Combine Results
        combined_results = self._combine_results(
            semantic_results, 
            keyword_results,
            alpha=0.7
        )
        
        # 4. Select top k
        top_results = combined_results[:k]
        
        # 5. Get full chunks and metadata
        chunks = []
        metadata = []
        for result in top_results:
            chunk_text = result['text']
            chunks.append(chunk_text)
            metadata.append({
                'score': result['score'],
                'source': result.get('source', 'unknown')
            })
        
        return chunks, metadata
    
    def _semantic_search(self, query: str, n_results: int = 20) -> List[Dict]:
        """Perform semantic search using embeddings"""
        try:
            query_embedding = self.embedding_model.encode(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            if results.get('documents') and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i] if results.get('distances') else [0]
                    formatted_results.append({
                        'text': doc,
                        'score': 1 - distance[0] if distance else 0,
                        'type': 'semantic',
                        'metadata': results['metadatas'][0][i] if results.get('metadatas') else {}
                    })
            
            return formatted_results
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def _keyword_search(self, query: str, n_results: int = 20) -> List[Dict]:
        """Perform keyword search using TF-IDF"""
        if not self.chunk_texts or self.tfidf_matrix is None:
            return []
        
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = np.argsort(similarities)[-n_results:][::-1]
            
            formatted_results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    formatted_results.append({
                        'text': self.chunk_texts[idx],
                        'score': similarities[idx],
                        'type': 'keyword',
                        'metadata': {'index': idx}
                    })
            return formatted_results
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []
    
    def _combine_results(self, semantic_results: List[Dict], 
                        keyword_results: List[Dict], 
                        alpha: float = 0.7) -> List[Dict]:
        """Combine and normalize scores from both methods"""
        combined_map = {}
        
        # Add semantic results
        for result in semantic_results:
            text = result['text']
            combined_map[text] = {
                'text': text,
                'score': 0,
                'semantic_score': result['score'],
                'keyword_score': 0,
                'metadata': result.get('metadata', {})
            }
        
        # Add keyword results
        for result in keyword_results:
            text = result['text']
            if text in combined_map:
                combined_map[text]['keyword_score'] = result['score']
            else:
                combined_map[text] = {
                    'text': text,
                    'score': 0,
                    'semantic_score': 0,
                    'keyword_score': result['score'],
                    'metadata': result.get('metadata', {})
                }
        
        # Normalize and combine
        semantic_scores = [v['semantic_score'] for v in combined_map.values() if v['semantic_score'] > 0]
        keyword_scores = [v['keyword_score'] for v in combined_map.values() if v['keyword_score'] > 0]
        
        max_semantic = max(semantic_scores) if semantic_scores else 1
        max_keyword = max(keyword_scores) if keyword_scores else 1
        
        for data in combined_map.values():
            norm_semantic = data['semantic_score'] / max_semantic if max_semantic > 0 else 0
            norm_keyword = data['keyword_score'] / max_keyword if max_keyword > 0 else 0
            data['score'] = (alpha * norm_semantic) + ((1 - alpha) * norm_keyword)
        
        return sorted(combined_map.values(), key=lambda x: x['score'], reverse=True)
    
    def _build_tfidf(self):
        """Build TF-IDF matrix for keyword search"""
        if self.chunk_texts and len(self.chunk_texts) > 0:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunk_texts)
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            cache_path = os.path.join(self.cache_dir, 'retriever_cache.pkl')
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'chunk_texts': self.chunk_texts,
                    'tfidf_matrix': self.tfidf_matrix,
                    'vectorizer': self.tfidf_vectorizer
                }, f)
        except:
            pass
    
    def _load_cache(self):
        """Load cache from disk"""
        cache_path = os.path.join(self.cache_dir, 'retriever_cache.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                self.chunk_texts = cache['chunk_texts']
                self.tfidf_matrix = cache['tfidf_matrix']
                self.tfidf_vectorizer = cache['vectorizer']