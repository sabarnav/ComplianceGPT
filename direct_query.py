from modules.knowledge_manager import KnowledgeManager

km = KnowledgeManager()

# Query directly using ChromaDB
results = km.collection.query(
    query_texts=['password policy requirements'],
    n_results=5
)

print('🔍 Direct ChromaDB Query Test:')
print('=' * 60)

if results['documents'] and results['documents'][0]:
    print(f'✅ Found {len(results["documents"][0])} chunks!')
    print()
    for i, doc in enumerate(results['documents'][0]):
        print(f'Chunk {i+1}:')
        print(f'  {doc[:200]}...')
        distance = results['distances'][0][i] if results.get('distances') else 0
        print(f'  Distance: {distance:.4f}')
        print()
else:
    print('❌ No chunks found!')
