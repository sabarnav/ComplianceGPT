import re
import spacy
from typing import Dict, List
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

class QueryProcessor:
    def __init__(self):
        """Initialize Query Processor with NLP models"""
        # Load spaCy model with automatic download if missing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("📥 Downloading spaCy model 'en_core_web_sm'...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"⚠️  Error loading spaCy model: {e}")
            print("   Falling back to basic processing...")
            self.nlp = None
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Compliance-specific terminology
        self.compliance_terms = {
            'password': ['authentication', 'credentials', 'login', 'passphrase'],
            'security': ['protection', 'safeguard', 'defense', 'countermeasure'],
            'risk': ['threat', 'vulnerability', 'hazard', 'exposure'],
            'audit': ['review', 'assessment', 'evaluation', 'inspection'],
            'control': ['measure', 'safeguard', 'countermeasure', 'mitigation'],
            'policy': ['procedure', 'guideline', 'standard', 'regulation'],
            'compliance': ['regulation', 'standard', 'requirement', 'mandate'],
            'encryption': ['cipher', 'cryptography', 'encoding'],
            'firewall': ['network security', 'traffic control', 'packet filter'],
            'incident': ['breach', 'attack', 'compromise', 'security event']
        }
    
    def process_query(self, query: str) -> Dict[str, any]:
        """Comprehensive query processing pipeline"""
        # 1. Clean query
        cleaned_query = self._clean_text(query)
        
        # 2. Process with spaCy if available
        if self.nlp:
            doc = self.nlp(cleaned_query)
            key_phrases = self._extract_key_phrases(doc)
            entities = self._extract_entities(doc)
            keywords = self._extract_keywords(doc)
        else:
            # Fallback without spaCy
            key_phrases = []
            entities = []
            keywords = [word.lower() for word in cleaned_query.split() if len(word) > 3]
        
        # 3. Extract various features
        return {
            'original': query,
            'cleaned': cleaned_query,
            'expanded': self._expand_query(cleaned_query),
            'key_phrases': key_phrases,
            'entities': entities,
            'intent': self._classify_intent(cleaned_query),
            'keywords': keywords,
            'question_type': self._classify_question(cleaned_query),
            'sentiment': self._analyze_sentiment(cleaned_query)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\?]', ' ', text)
        return text.strip()
    
    def _extract_key_phrases(self, doc) -> List[str]:
        """Extract key phrases using noun chunks"""
        if not doc:
            return []
        phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:
                phrases.append(chunk.text.lower())
        return list(dict.fromkeys(phrases))[:5]
    
    def _extract_entities(self, doc) -> List[Dict]:
        """Extract named entities"""
        if not doc:
            return []
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_) if hasattr(spacy, 'explain') else ''
            })
        return entities
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms for better retrieval"""
        words = query.split()
        expanded = []
        
        for word in words:
            expanded.append(word)
            for term, synonyms in self.compliance_terms.items():
                if word.lower() == term or word.lower() in synonyms:
                    expanded.extend(synonyms[:2])
                    break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_expanded = []
        for word in expanded:
            if word not in seen:
                seen.add(word)
                unique_expanded.append(word)
        
        return ' '.join(unique_expanded)
    
    def _classify_intent(self, query: str) -> str:
        """Classify the intent of the question"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'how', 'when', 'why', 'who', 'where']):
            return 'question'
        elif any(word in query_lower for word in ['explain', 'describe', 'define']):
            return 'explanation'
        elif any(word in query_lower for word in ['list', 'provide', 'give']):
            return 'list'
        elif any(word in query_lower for word in ['yes', 'no', 'is', 'are', 'does']):
            return 'verification'
        else:
            return 'general'
    
    def _classify_question(self, query: str) -> str:
        """Classify the type of question"""
        query_lower = query.lower()
        
        if 'what' in query_lower:
            return 'definition'
        elif 'how' in query_lower:
            return 'process'
        elif 'why' in query_lower:
            return 'explanation'
        elif 'when' in query_lower:
            return 'timing'
        elif 'who' in query_lower:
            return 'actor'
        else:
            return 'general'
    
    def _extract_keywords(self, doc) -> List[str]:
        """Extract important keywords"""
        if not doc:
            return []
        keywords = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                if token.text.lower() not in self.stop_words and len(token.text) > 2:
                    keywords.append(token.lemma_.lower())
        return list(dict.fromkeys(keywords))
    
    def _analyze_sentiment(self, query: str) -> Dict:
        """Analyze sentiment of the query"""
        blob = TextBlob(query)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }