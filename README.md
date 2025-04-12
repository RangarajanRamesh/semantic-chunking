# Semantic Chunker

Semantic Chunker is a lightweight Python package for semantically-aware chunking and clustering of text. It’s designed to support retrieval-augmented generation (RAG), LLM pipelines, and knowledge processing workflows by intelligently grouping related ideas.

Traditional chunking methods for LLM pipelines often rely on fixed-size windows or naive text boundaries, which can fragment meaning, split up related ideas, or fail to capture contextual coherence. This leads to inefficient retrieval, diluted embeddings, and suboptimal responses in RAG systems. Semantic Chunker addresses this by analyzing the meaning of each chunk — using sentence embeddings and clustering — to merge semantically similar chunks into more coherent units. The result is smarter preprocessing: fewer, denser, and more relevant chunks that enhance both the performance and interpretability of your downstream models.

---

## Features

- **Embedding-based chunk similarity** (via Sentence Transformers)
- **Token-aware merging** with real model tokenizers
- **Clustered chunk merging** for optimized RAG inputs
- **Preserves chunk metadata** through merging
- **Visual tools**: attention heatmaps, semantic graphs, cluster previews
- **Export options**: JSON, Markdown, CSV
- **CLI Interface** for scripting and automation
- Debug mode with embeddings, similarity matrix, semantic pairs

---

## Installation

```bash
pip install advanced-chunker
```

---

## Quick Start

```python
from semantic_chunker.core import SemanticChunker

chunks = [
    {"text": "Artificial intelligence is a growing field."},
    {"text": "Machine learning is a subset of AI."},
    {"text": "Photosynthesis occurs in plants."},
    {"text": "Deep learning uses neural networks."},
    {"text": "Plants convert sunlight into energy."},
]

chunker = SemanticChunker(max_tokens=512)
merged_chunks = chunker.chunk(chunks)

for i, merged in enumerate(merged_chunks):
    print(f"Chunk {i}:")
    print(merged["text"])
    print()

# Response

Merged Chunk 1
Text: Artificial intelligence is a growing field. Machine learning is a subset of AI. Deep learning uses n...
Metadata: [{'text': 'Artificial intelligence is a growing field.'}, {'text': 'Machine learning is a subset of AI.'}, {'text': 'Deep learning uses neural networks.'}]

Merged Chunk 2
Text: Photosynthesis occurs in plants. Plants convert sunlight into energy....
Metadata: [{'text': 'Photosynthesis occurs in plants.'}, {'text': 'Plants convert sunlight into energy.'}]

```
<img src="Images/similarity_matrix.png" alt="Similarity Matrix" width="400"/>

<img src="Images/relationships.png" alt="Relationships" width="400"/>

---

## Debugging & Visualization

```python
from semantic_chunker.visualization import plot_attention_matrix, plot_semantic_graph, preview_clusters

chunker = SemanticChunker(max_tokens=512)
debug = chunker.get_debug_info(chunks)

preview_clusters(debug["original_chunks"], debug["clusters"])
plot_attention_matrix(debug["similarity_matrix"], debug["clusters"])
plot_semantic_graph(debug["original_chunks"], debug["semantic_pairs"], debug["clusters"])
```

---

## CLI Usage

### Merge chunks semantically:
```bash
chunker chunk \
  --chunks path/to/chunks.json \
  --threshold 0.5 \
  --similarity-threshold 0.4 \
  --max-tokens 512 \
  --preview \
  --visualize \
  --export \
  --export-path output/merged \
  --export-format json
```

---

## Exports

Export clustered or merged chunks to:
- `.json`: for ML/data pipelines
- `.md`: for human-readable inspection
- `.csv`: for spreadsheets or BI tools

---

## Parameter Guide

| Argument               | Description |
|------------------------|-------------|
| `--chunks`             | Path to a JSON list of text chunks (each as a dict with a `text` field) |
| `--threshold`          | Agglomerative clustering distance threshold (controls granularity of clusters) |
| `--similarity-threshold` | Minimum cosine similarity required for two chunks to be considered semantically linked |
| `--max-tokens`         | Maximum number of tokens allowed per merged chunk |
| `--preview`            | Print out samples of clusters to the console |
| `--visualize`          | Show attention matrix heatmap plot of inter-chunk similarity |
| `--export`             | Enable file export of merged or clustered chunks |
| `--export-path`        | Output file path prefix (e.g. `output/merged`) |
| `--export-format`      | File format for export (`json`, `csv`, or `md`) |

---

## Glossary

| Term                 | Meaning |
|----------------------|---------|
| **Threshold**         | Controls how strictly the clustering groups similar chunks. Lower = more granular clusters. |
| **Similarity Threshold** | Sets the minimum cosine similarity to form a semantic pair between two chunks. |
| **Semantic Pairs**    | Pairs of chunks with high similarity; used for graph-based visualizations. |
| **Attention Matrix**  | The full NxN matrix of similarities between every chunk embedding. |
| **Merged Chunk**      | The final recombined text after semantic grouping. |
| **Cluster**           | A group of semantically similar chunks found by the algorithm. |

---

## Architecture

```text
Chunks → Embeddings → Cosine Similarity → Clustering → Merging
                               ↓
                         Semantic Pairs (Optional)
                               ↓
                     Visualization & Export (Optional)
```

---

## Integrations

### LangChain

Use `SemanticChunkerSplitter` to replace standard splitters:

```python
from langchain_core.documents import Document
from semantic_chunker.integrations.langchain import SemanticChunkerSplitter

splitter = SemanticChunkerSplitter(cluster_threshold=0.4, return_merged=True)

docs = [Document(page_content="Some long technical text here", metadata={"source": "report.pdf"})]
split = splitter.split_documents(docs)
```

---

## Testing

```bash
pytest tests/
```

---

## Contributing

Pull requests are welcome! Please open an issue first if you'd like to add a feature or fix a bug.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Sentence Transformers](https://www.sbert.net/)
- [LangChain](https://www.langchain.com/)
- [scikit-learn](https://scikit-learn.org/)
- Hugging Face ecosystem