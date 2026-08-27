# ComplianceGPT – AI-Powered Cybersecurity Compliance Assistant

## 📌 Project Overview

ComplianceGPT is an AI-powered cybersecurity compliance assistant designed to simplify information retrieval from cybersecurity compliance documents.

The application allows users to upload PDF and TXT documents and ask questions using a conversational interface. Instead of manually searching through lengthy compliance documents, users can ask questions in natural language and receive answers based on the uploaded document knowledge base.

The system uses Retrieval-Augmented Generation (RAG), combining semantic search, keyword search, NLP-based query processing, ChromaDB, and Llama 3.3 70B Versatile through the Groq API.

---

## 🎯 Objectives

- Simplify access to cybersecurity compliance information.
- Allow users to query uploaded compliance documents using natural language.
- Extract and process PDF and TXT documents.
- Convert document chunks into embeddings.
- Store document embeddings in ChromaDB.
- Use semantic and keyword-based retrieval.
- Generate context-based answers using an LLM.
- Reduce unsupported or hallucinated responses.

---

## ✨ Key Features

- 📄 PDF and TXT document support
- 🔍 Semantic search
- 🔑 TF-IDF keyword search
- 🔄 Hybrid retrieval
- 🧠 NLP-based query processing
- 📦 ChromaDB vector database
- 🤖 Llama 3.3 70B Versatile
- ⚡ Groq API integration
- 💬 Conversational Streamlit interface
- 📚 Document-based question answering
- 🔐 Environment-variable based API configuration

---
Project Structure

ComplianceGPT/
├── app.py
├── ingest.py
├── modules/
├── data/
├── requirements.txt
└── PROJECT_DOCUMENTATION.md

## 🏗️ System Architecture

Document Upload
       ↓
Document Processing
       ↓
Text Extraction
       ↓
Text Cleaning
       ↓
Document Chunking
       ↓
Embedding Generation
       ↓
ChromaDB
       ↓
User Query
       ↓
Query Processing
       ↓
Hybrid Retrieval
   ↙          ↘
Semantic     Keyword
Search       Search
   ↘          ↙
   Relevant Chunks
          ↓
       RAG Engine
          ↓
    Groq API / Llama
          ↓
    Final Response
    
## Technology Stack

**Backend:** Python  
**Frontend:** Streamlit  
**Vector Database:** ChromaDB  
**Embeddings:** Sentence Transformers 
**NLP:** spaCy, NLTK, TextBlob  
**Retrieval:** Semantic Search + TF-IDF  
**LLM:** Llama 3.3 70B Versatile  
**LLM Provider:** Groq API
