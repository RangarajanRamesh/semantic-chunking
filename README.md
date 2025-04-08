# Semantic Chunker 🧠✂️

Semantic Chunker is a powerful, lightweight Python library for semantically-aware chunking and clustering of text. It’s designed to support retrieval-augmented generation (RAG), LLM pipelines, and knowledge processing workflows by intelligently grouping related ideas.

---

## 🔥 Features

- ✅ **Embedding-based chunk similarity** (via Sentence Transformers)
- ✅ **Token-aware merging** with real model tokenizers
- ✅ **Clustered chunk merging** for optimized RAG inputs
- ✅ **Preserves chunk metadata** through merging
- ✅ **Visual tools**: attention heatmaps, semantic graphs, cluster previews
- ✅ **Export options**: JSON, Markdown, CSV
- ✅ **CLI Interface** for scripting and automation
- 🧪 Debug mode with embeddings, similarity matrix, semantic pairs

---

## 🚀 Installation

```bash
pip install semantic-chunker
```

---

## 📦 Quick Start

```python
from semantic_chunker.refactor import SemanticChunker

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
```

---

## 🧠 Debugging & Visualization

```python
from semantic_chunker.visualization import plot_attention_matrix, plot_semantic_graph, preview_clusters

chunker = SemanticChunker(max_tokens=512)
debug = chunker.get_debug_info(chunks)

preview_clusters(debug["original_chunks"], debug["clusters"])
plot_attention_matrix(debug["similarity_matrix"], debug["clusters"])
plot_semantic_graph(debug["original_chunks"], debug["semantic_pairs"], debug["clusters"])
```

---

## 🛠 CLI Usage

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

## 📊 Exports

Export clustered or merged chunks to:
- `.json`: for ML/data pipelines
- `.md`: for human-readable inspection
- `.csv`: for spreadsheets or BI tools

---

## 📐 Architecture

```
Chunks → Embeddings → Cosine Similarity → Clustering → Merging
                                   ↓
                             Semantic Pairs (Optional)
                                   ↓
                             Visualization & Export
```

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first if you'd like to add a feature or fix a bug.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

- [Sentence Transformers](https://www.sbert.net/)
- [LangChain](https://www.langchain.com/)
- [scikit-learn](https://scikit-learn.org/)
- Hugging Face ecosystem ❤️