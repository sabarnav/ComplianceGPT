import time
from typing import Tuple, List, Dict
from modules.query_processor import QueryProcessor
from modules.hybrid_retriever import HybridRetriever
from modules.grok_llm_manager import GrokLLMManager
from modules.knowledge_manager import KnowledgeManager

class RAGEngine:
    def __init__(self):
        self.knowledge_manager = KnowledgeManager()
        self.query_processor = QueryProcessor()
        self.retriever = HybridRetriever(
            self.knowledge_manager.collection,
            self.knowledge_manager.embedding_model
        )
        self.llm_manager = GrokLLMManager()
        self.metrics = {
            'total_queries': 0,
            'avg_retrieval_time': 0,
            'avg_generation_time': 0
        }
    
    def ask(self, query: str, k: int = 5) -> Tuple[str, List[str]]:
        """Complete RAG pipeline"""
        start_time = time.time()
        
        # Check if knowledge base has documents
        if self.knowledge_manager.get_document_count() == 0:
            return "No documents found in knowledge base. Please upload compliance documents first.", []
        
        # 1. Query Processing
        processed_query = self.query_processor.process_query(query)
        
        # 2. Retrieval
        retrieval_start = time.time()
        chunks, chunk_metadata = self.retriever.retrieve(
            query, 
            self.query_processor, 
            k=k
        )
        retrieval_time = time.time() - retrieval_start
        
        # Check if we got any chunks
        if not chunks:
            return "I don't have relevant information in the knowledge base to answer this question.", []
        
        # 3. Build Prompt
        prompt = self._build_prompt(query, chunks, processed_query)
        
        # 4. Generate Response
        generation_start = time.time()
        answer = self.llm_manager.generate_response(prompt)
        generation_time = time.time() - generation_start
        
        # 5. Update metrics
        self.metrics['total_queries'] += 1
        self.metrics['avg_retrieval_time'] = (
            (self.metrics['avg_retrieval_time'] * (self.metrics['total_queries'] - 1) + retrieval_time) /
            self.metrics['total_queries']
        )
        self.metrics['avg_generation_time'] = (
            (self.metrics['avg_generation_time'] * (self.metrics['total_queries'] - 1) + generation_time) /
            self.metrics['total_queries']
        )
        
        # 6. Prepare sources
        sources = self._prepare_sources(chunk_metadata)
        
        return answer, sources
    
    def _build_prompt(self, query: str, chunks: List[str], processed_query: Dict) -> str:
        """Build enhanced prompt"""
        # Format context
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"""
[SOURCE {i+1} START]
{chunk}
[SOURCE {i+1} END]
""")
        
        context = "\n".join(context_parts)
        
        # Get query metadata
        question_type = processed_query.get('question_type', 'general')
        intent = processed_query.get('intent', 'general')
        key_phrases = ', '.join(processed_query.get('key_phrases', []))
        
        # Build the prompt
        prompt = f"""You are ComplianceGPT, a precise cybersecurity compliance assistant.

**CRITICAL INSTRUCTIONS:**
1. ONLY use information from the provided sources below
2. If the sources don't contain the answer, say "I don't have sufficient information in the knowledge base to answer this question."
3. DO NOT make up, guess, or hallucinate information
4. When citing information, reference the source number: [SOURCE X]
5. Be specific, concise, and professional
6. If the question is not compliance-related, politely redirect

**Question Type:** {question_type}
**Intent:** {intent}
**Key Terms:** {key_phrases}

**Sources:**
{context}

**User Question:** {query}

**Your Response:"""
        
        return prompt
    
    def _prepare_sources(self, chunk_metadata: List[Dict]) -> List[str]:
        """Prepare source citations"""
        sources = []
        for i, meta in enumerate(chunk_metadata):
            if meta and 'source' in meta:
                sources.append(f"Source {i+1}: {meta['source']}")
            else:
                sources.append(f"Source {i+1}: Document chunk")
        return sources
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        return self.metrics

# Singleton instance
_rag_engine = None

def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine

def ask_compliance_gpt(query: str) -> Tuple[str, List[str]]:
    """Legacy function for compatibility"""
    engine = get_rag_engine()
    return engine.ask(query)