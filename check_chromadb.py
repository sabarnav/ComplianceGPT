from modules.knowledge_manager import KnowledgeManager

km = KnowledgeManager()
all_chunks = km.collection.get()

print('Knowledge Base Status')
print('=' * 60)
print(f'Total chunks: {len(all_chunks["documents"])}')
print()

# Count chunks with 'password'
password_chunks = []
for i, doc in enumerate(all_chunks['documents']):
    if 'password' in doc.lower():
        password_chunks.append((i, doc))

print(f'Chunks containing "password": {len(password_chunks)}')
print()

if password_chunks:
    print('Password chunks:')
    print('-' * 60)
    for i, (idx, doc) in enumerate(password_chunks[:3]):
        print(f'Chunk {i+1}: {doc[:200]}...')
        print()
else:
    print('No password chunks found in ChromaDB!')

# Also check first 5 chunks overall
print('First 5 chunks in ChromaDB:')
print('-' * 60)
for i, doc in enumerate(all_chunks['documents'][:5]):
    print(f'Chunk {i+1}: {doc[:150]}...')
    print()
