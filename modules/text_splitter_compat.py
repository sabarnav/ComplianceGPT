"""
Compatibility layer for LangChain text splitters
Supports both old and new import paths
"""

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # Fallback implementation
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=500, chunk_overlap=100, separators=None, length_function=len):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
                self.length_function = length_function
            
            def split_text(self, text):
                chunks = []
                if len(text) <= self.chunk_size:
                    return [text]
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    chunk = text[start:end]
                    chunks.append(chunk)
                    start = end - self.chunk_overlap
                return chunks