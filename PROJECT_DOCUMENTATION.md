# ComplianceGPT

## Overview

ComplianceGPT is a local document-centric retrieval-augmented generation (RAG) system built for cybersecurity compliance assistance. It ingests text and PDF documents, stores them in a searchable ChromaDB knowledge base, and answers user questions with context sourced from the indexed documents.

The application includes a Streamlit-based UI (`app.py`) for chat, file upload, and knowledge base updates.

## Key Components

- `app.py`: Streamlit front-end for user chats, document upload, knowledge base refresh, and chat history management.
- `pipeline.py`: Main knowledge base builder that analyzes and processes supported documents, updates metadata, stores processed text, and indexes chunks in ChromaDB.
- `refresh_db.py`: Alternate refresh script that rebuilds the ChromaDB collection from the `data/` folder and supports `.txt` and `.pdf` files.
- `modules/`: Core processing modules.
- `data/`: Source documents to be ingested.
- `processed/`: Serialized processed text outputs.
- `metadata/`: Stored JSON metadata for processed documents.
- `chroma_db/`: Persistent ChromaDB database files.

## Modules

### `modules/document_analyzer.py`
- Reads a document file.
- Calculates metadata: file size, hash, character count, word count, and line count.
- Returns metadata in a structured dictionary.

### `modules/document_processor.py`
- Reads `.txt` documents.
- Splits text into fixed-size chunks (default 500 characters).
- Returns raw text and chunk list.

### `modules/metadata_manager.py`
- Saves metadata JSON files under `metadata/`.
- Detects whether a document was already processed by checking for existing metadata.

### `modules/processed_manager.py`
- Saves a plain text version of processed documents in `processed/`.

### `modules/knowledge_manager.py`
- Uses `SentenceTransformer("all-MiniLM-L6-v2")` for chunk embeddings.
- Adds chunk documents and embeddings to the `compliance_docs` ChromaDB collection.

### `modules/retriever.py`
- Encodes user queries to embeddings.
- Queries ChromaDB for the top relevant chunks.
- Returns matched chunks with source metadata.

### `modules/prompt_builder.py`
- Builds the LLM prompt from the user query and retrieved chunks.
- Includes source/ chunk labels and instructions to answer only from knowledge base context.

### `modules/llm_manager.py`
- Sends the final prompt to an Ollama model via subprocess.
- Cleans ANSI escape sequences from the output.
- Returns the generated answer.

### `modules/rag_engine.py`
- Orchestrates the retrieval-augmented generation workflow:
  1. Retrieve relevant chunks.
  2. Build the prompt.
  3. Generate the answer.
  4. Return answer and sources.

## Data Processing Workflow

### 1. Document ingestion
- Place supported documents in the `data/` folder.
- Supported file types:
  - `.txt` for `pipeline.py`.
  - `.txt` and `.pdf` for `refresh_db.py`.

### 2. Build or Refresh the knowledge base
- Use `python pipeline.py` to process text documents through the metadata/ChromaDB pipeline.
- Use `python refresh_db.py` to rebuild the ChromaDB collection from all supported files and clear existing embeddings.

### 3. Document analysis and storage
- Each document is analyzed for metadata.
- Text is chunked into 500-character segments.
- Processed text is saved into `processed/`.
- Metadata is saved into `metadata/`.
- Each chunk is embedded and stored in ChromaDB with metadata including source and chunk index.

## Query Workflow

### 1. User interaction
- Users access the Streamlit app in `app.py`.
- They can ask questions, upload new documents, update the knowledge base, and view recent chats.

### 2. Retrieval
- The user query is encoded into an embedding.
- ChromaDB returns the top `k` relevant chunks based on similarity.

### 3. Prompt construction
- The retrieved chunks and query are combined into a single prompt.
- The prompt instructs the model to answer only from the provided context.

### 4. Answer generation
- `ollama run mistral` is executed via subprocess.
- The cleaned answer is returned to the app.
- The app can store chat history in `chat_history/history.json`.

## File Upload Flow in Streamlit

- Upload a file through the sidebar.
- The file is saved to `data/`.
- Users can click `Update Knowledge Base` to run `pipeline.py` and index the newly added file.
- Available documents are listed in the sidebar.
- `Clear Chats` resets the session history and stored chat history file.

## Deployment / Usage

1. Install dependencies.
2. Place compliance documents in `data/`.
3. Run `python pipeline.py` or `python refresh_db.py`.
4. Start the UI with `streamlit run app.py`.
5. Ask questions in the app and use the knowledge base.

## Notes and Limitations

- `pipeline.py` currently only processes `.txt` files.
- `refresh_db.py` adds `.pdf` support via `pdf_utils.extract_text_from_pdf`.
- The prompt builder returns only the first chunk of retrieved context due to indentation/return placement in the current implementation.
- The LLM backend relies on `ollama` and the `mistral` model being available on the host.
- `requirements.txt` is empty, so install dependencies manually as needed (e.g. `sentence-transformers`, `chromadb`, `streamlit`, `pypdf`, `ollama` client if applicable).

## Directory Structure

- `app.py`
- `pipeline.py`
- `refresh_db.py`
- `ingest.py`
- `pdf_utils.py`
- `modules/`
  - `document_analyzer.py`
  - `document_processor.py`
  - `knowledge_manager.py`
  - `llm_manager.py`
  - `metadata_manager.py`
  - `processed_manager.py`
  - `prompt_builder.py`
  - `rag_engine.py`
  - `retriever.py`
- `data/`
- `processed/`
- `metadata/`
- `chroma_db/`
- `chat_history/history.json`

## Clean Up

- All test files (`test_analyzer.py`, `test_llm_manager.py`, `test_processor.py`, `test_prompt_builder.py`, `test_rag_engine.py`, `test_retriever.py`, `test_storage.py`) have been removed from the root directory.
