from semantic_chunker.core import SemanticChunker
from semantic_chunker.visualization import plot_attention_matrix, plot_semantic_graph, preview_clusters

# Example input — either raw strings or dicts with "text"
raw_chunks = [
    "Artificial intelligence is a growing field.",
    "Machine learning is a subset of AI.",
    "Photosynthesis occurs in plants.",
    "Deep learning uses neural networks.",
    "Plants convert sunlight into energy.",
]

# Wrap raw strings into {"text": ...} if needed
chunks = [{"text": c} if isinstance(c, str) else c for c in raw_chunks]

# Initialize the semantic chunker
chunker = SemanticChunker(max_tokens=100, cluster_threshold=0.4, similarity_threshold=0.4)

# Get debug info (includes similarity matrix, clusters, etc.)
debug_info = chunker.get_debug_info(chunks)

# Show original cluster preview
preview_clusters(debug_info["original_chunks"], debug_info["clusters"])

# Show visualizations
plot_attention_matrix(debug_info["similarity_matrix"], debug_info["clusters"], title="Similarity Matrix")
plot_semantic_graph(debug_info["original_chunks"], debug_info["semantic_pairs"], debug_info["clusters"])

# Print top semantic relationships
print("\n🔗 Top Semantic Relationships:")
for i, j, sim in debug_info["semantic_pairs"]:
    print(f"Chunk {i} ↔ Chunk {j} | Sim: {sim:.3f}")
    print(f"  - {debug_info['original_chunks'][i]['text']}")
    print(f"  - {debug_info['original_chunks'][j]['text']}")
    print()

# Print merged chunks
print("\n📦 Merged Chunks:")
for i, merged in enumerate(debug_info["merged_chunks"]):
    print(f"\nMerged Chunk {i + 1}")
    print(f"Text: {merged['text'][:100]}...")
    print(f"Metadata: {merged['metadata']}")
