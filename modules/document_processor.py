import os
import hashlib
import re
from typing import List, Dict, Any
import PyPDF2
import fitz

# Try importing langchain, fallback to custom
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
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

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        \"\"\"Process document and extract text\"\"\"
        
        # Extract text based on file type
        if file_path.lower().endswith('.pdf'):
            text = self._extract_pdf_text(file_path)
        else:
            text = self._extract_text_file(file_path)
        
        if not text or len(text.strip()) == 0:
            return {'chunks': [], 'metadata': {}, 'text': ''}
        
        # Clean text
        text = self._clean_text(text)
        
        # Calculate metadata
        metadata = self._calculate_metadata(file_path, text)
        
        # Create chunks
        chunks = self._create_chunks(text)
        
        # Create chunk metadata
        chunks_with_metadata = []
        for i, chunk in enumerate(chunks):
            chunks_with_metadata.append({
                'text': chunk,
                'chunk_id': i,
                'file_name': os.path.basename(file_path),
                'chunk_size': len(chunk)
            })
        
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'text': text,
            'metadata': metadata,
            'chunks': chunks_with_metadata
        }
    
    def _extract_pdf_text(self, file_path: str) -> str:
        \"\"\"Extract text from PDF\"\"\"
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except:
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
            except:
                pass
        return text
    
    def _extract_text_file(self, file_path: str) -> str:
        \"\"\"Extract text from TXT file\"\"\"
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except:
                continue
        return ""
    
    def _clean_text(self, text: str) -> str:
        \"\"\"Clean text\"\"\"
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\?]', '', text)
        return text.strip()
    
    def _calculate_metadata(self, file_path: str, text: str) -> Dict:
        \"\"\"Calculate metadata\"\"\"
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        return {
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'hash': file_hash[:8],
            'char_count': len(text),
            'word_count': len(text.split())
        }
    
    def _create_chunks(self, text: str) -> List[str]:
        \"\"\"Create text chunks\"\"\"
        if not text:
            return []
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        
        if not chunks and text:
            return [text]
        
        return chunks
