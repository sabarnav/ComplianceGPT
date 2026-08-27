import os
from modules.knowledge_manager import KnowledgeManager

def ingest_all_documents():
    print("=" * 50)
    print("📚 ComplianceGPT - Document Ingestion Tool")
    print("=" * 50)
    
    # Check data folder
    if not os.path.exists("data"):
        os.makedirs("data")
        print("❌ No data folder found. Created one.")
        print("📁 Please add documents to data/ folder")
        return
    
    # Get files
    files = []
    for file in os.listdir("data"):
        if file.endswith(('.txt', '.pdf')):
            files.append(os.path.join("data", file))
    
    if not files:
        print("❌ No documents found in data/ folder")
        print("📄 Supported formats: .txt, .pdf")
        return
    
    print(f"📄 Found {len(files)} documents")
    print()
    
    # Initialize knowledge manager
    km = KnowledgeManager()
    
    # Clear existing
    km.clear_all()
    print()
    
    # Process each file
    total_chunks = 0
    for file_path in files:
        chunks = km.add_document(file_path)
        total_chunks += chunks
    
    print()
    print("=" * 50)
    print(f"✅ Ingestion complete!")
    print(f"📊 Total chunks: {total_chunks}")
    print(f"📊 Documents in collection: {km.get_document_count()}")
    print("=" * 50)

if __name__ == "__main__":
    ingest_all_documents()
