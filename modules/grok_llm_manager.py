import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GrokLLMManager:
    def __init__(
        self, 
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """Initialize Grok LLM Manager"""
        self.api_key = os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY not found in environment variables!")
        
        # Use environment variables or defaults
        self.model = model or os.environ.get("GROK_MODEL", "grok-4.1-fast-non-reasoning")
        self.temperature = temperature or float(os.environ.get("GROK_TEMPERATURE", 0.3))
        self.max_tokens = max_tokens or int(os.environ.get("GROK_MAX_TOKENS", 500))
        
        # Try to import xai_sdk
        try:
            from xai_sdk import Client
            self.client = Client(api_key=self.api_key)
            self.use_xai_sdk = True
        except ImportError:
            # Fallback to OpenAI SDK
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.x.ai/v1"
                )
                self.use_xai_sdk = False
            except ImportError:
                raise ImportError("Please install xai-sdk or openai: pip install xai-sdk")
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Grok API"""
        try:
            if self.use_xai_sdk:
                return self._generate_with_xai_sdk(prompt)
            else:
                return self._generate_with_openai(prompt)
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_with_xai_sdk(self, prompt: str) -> str:
        """Generate using xai-sdk"""
        from xai_sdk.chat import user, system
        
        # Create chat session
        chat = self.client.chat.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Add system instruction
        chat.append(system(
            "You are ComplianceGPT, a precise cybersecurity compliance assistant. "
            "ONLY use information from the provided context. "
            "If the context doesn't contain the answer, say 'I don't have sufficient information in the knowledge base.' "
            "DO NOT make up or hallucinate information. "
            "Be specific and concise in your answers."
        ))
        
        # Add user prompt
        chat.append(user(prompt))
        
        # Get response
        response = chat.sample()
        return self._clean_response(response.content)
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Generate using OpenAI-compatible interface"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are ComplianceGPT, a precise cybersecurity compliance assistant. ONLY use information from the provided context. If the context doesn't contain the answer, say you don't know. DO NOT make up information."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return self._clean_response(response.choices[0].message.content)
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        if not response:
            return "No response generated."
        
        # Remove extra whitespace
        response = ' '.join(response.split())
        
        # Ensure it's not too long
        if len(response) > 2000:
            response = response[:2000] + "..."
        
        return response.strip()