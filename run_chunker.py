from semantic_chunker.core import ChunkAnalyzer
from semantic_chunker.visualization import plot_attention_matrix, plot_semantic_graph, preview_clusters

# Example input — either raw strings or dicts with "text"
raw_chunks = [
    "Artificial intelligence is a growing field.",
    "Machine learning is a subset of AI.",
    "Photosynthesis occurs in plants.",
    "Deep learning uses neural networks.",
    "Plants convert sunlight into energy.",
]

# Optional: wrap raw strings into {"text": ...} if needed
chunks = [{"text": c} if isinstance(c, str) else c for c in raw_chunks]

# Run the analysis
analyzer = ChunkAnalyzer(max_tokens=100)
results = analyzer.analyze_chunks(chunks, cluster_threshold=0.4, similarity_threshold=0.4)

# Show original cluster preview
preview_clusters(results["original_chunks"], results["clusters"])

# Optional: show visualizations
plot_attention_matrix(results["attention_matrix"], results["clusters"], title="Similarity Matrix")
plot_semantic_graph(results["original_chunks"], results["semantic_pairs"], results["clusters"])

# Print top semantic relationships
print("\n🔗 Top Semantic Relationships:")
for i, j, sim in results["semantic_pairs"]:
    print(f"Chunk {i} ↔ Chunk {j} | Sim: {sim:.3f}")
    print(f"  - {results['original_chunks'][i]['text']}")
    print(f"  - {results['original_chunks'][j]['text']}")
    print()

# Print merged chunks
print("\n📦 Merged Chunks:")
for i, merged in enumerate(results["merged_chunks"]):
    print(f"\nMerged Chunk {i + 1}")
    print(f"Text: {merged['text'][:100]}...")
    print(f"Metadata: {merged['metadata']}")
