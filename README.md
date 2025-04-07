# PDF Chunk Attention Analyzer

This tool analyzes the self-attention and cross-attention patterns between chunks in a PDF document. It provides insights into how different sections of text relate to each other, which can be used to develop more intelligent chunking strategies for RAG (Retrieval Augmented Generation) systems.

## Overview

Traditional chunking methods split documents based on fixed token counts or simple delimiters, which often separates related content. This tool helps visualize the semantic relationships between different chunks by:

1. Extracting text from a PDF
2. Applying traditional fixed-size chunking
3. Generating embeddings for each chunk
4. Computing an attention (similarity) matrix between all chunks
5. Visualizing the results as heatmaps and relationship graphs

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- PyPDF2
- torch
- transformers
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage

### Usage

1. Edit the configuration variables at the top of `demo.py`:

```python
# Configuration (edit these variables directly)
pdf_path = "document.pdf"  # Path to your PDF file
chunk_size = 200           # Number of words per chunk
overlap = 20               # Number of overlapping words between chunks
output_dir = "./output"    # Directory to save outputs
```

2. Run the script:

```bash
python demo.py
```

Parameters you can modify in the script:
- `pdf_path`: Path to the PDF file
- `chunk_size`: Number of words per chunk (default: 200)
- `overlap`: Number of overlapping words between chunks (default: 20)
- `output_dir`: Directory to save outputs (default: ./output)

## Output Files

The tool generates several output files:

1. `attention_matrix.png`: Heatmap visualization of the attention matrix
2. `chunk_reference.png`: Reference image showing chunk contents
3. `enhanced_attention_matrix.png`: Enhanced visualization with highlighted relationships
4. `top_relationships.png`: Bar chart of top chunk relationships
5. `attention_matrix.csv`: Raw attention matrix data for further analysis
6. `chunks.txt`: Full text of each chunk for reference

## Interpreting the Results

### Attention Matrix

The attention matrix is a heatmap where:
- Each cell (i,j) represents the cosine similarity between chunk i and chunk j
- The diagonal (self-attention) shows how coherent each chunk is internally
- Off-diagonal elements (cross-attention) show relationships between different chunks
- Brighter colors indicate stronger relationships

### Top Relationships

The top relationships visualization shows pairs of chunks with the highest semantic similarity, which could potentially be merged in an improved chunking strategy.

## Next Steps

This tool provides the foundation for developing more advanced semantic chunking strategies:

1. **Iterative Clustering**: Use the attention matrix to iteratively merge related chunks
2. **Graph-Based Chunking**: Represent chunks as nodes in a graph and use community detection to identify natural groupings
3. **Adaptive Thresholds**: Develop heuristics to determine when chunks should be merged based on their similarity scores

## Example

Running the tool on a research paper might reveal that the "Methods" section is split across multiple chunks, but these chunks have high cross-attention scores. This suggests they should be kept together in a semantic chunking approach.