import PyPDF2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import re
import json
from collections import defaultdict

class HierarchicalChunkAnalyzer:
    """
    A hierarchical chunk analyzer that first creates small meaningful chunks
    and then analyzes relationships between them to form higher-level chunks.
    """
    
    def __init__(self):
        """Initialize the analyzer with TF-IDF for embeddings"""
        self.vectorizer = TfidfVectorizer(max_features=200)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            print("Try using a different PDF or check if it's text-based rather than scanned images.")
            return ""
    
    def create_atomic_chunks(self, text, chunk_size=50):
        """
        Create the smallest meaningful chunks based on natural boundaries.
        Prioritizes paragraph breaks, sentence boundaries, then falls back to word count.
        Adds metadata about chunk location and original paragraph.
        """
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        atomic_chunks = []
        chunk_metadata = []
        current_position = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            para_start_pos = current_position
            
            # If paragraph is very short, add it as-is
            if len(paragraph.split()) <= chunk_size:
                atomic_chunks.append(paragraph.strip())
                
                # Add metadata
                chunk_metadata.append({
                    'paragraph_idx': para_idx,
                    'start_position': para_start_pos,
                    'end_position': para_start_pos + len(paragraph),
                    'is_complete_paragraph': True
                })
                
                current_position += len(paragraph) + 2  # +2 for the paragraph break
                continue
            
            # Otherwise, split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            current_chunk = []
            current_size = 0
            chunk_start_pos = para_start_pos
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                
                # If single sentence exceeds chunk size, split by word count
                if sentence_words > chunk_size:
                    words = sentence.split()
                    
                    # Add any existing sentences to chunks first
                    if current_chunk:
                        chunk_text = " ".join(current_chunk).strip()
                        atomic_chunks.append(chunk_text)
                        
                        # Add metadata for this chunk
                        chunk_metadata.append({
                            'paragraph_idx': para_idx,
                            'start_position': chunk_start_pos,
                            'end_position': chunk_start_pos + len(chunk_text),
                            'is_complete_paragraph': False
                        })
                        
                        current_chunk = []
                        current_size = 0
                        chunk_start_pos += len(chunk_text) + 1  # +1 for space
                    
                    # Then chunk the long sentence
                    for i in range(0, len(words), chunk_size):
                        word_chunk = " ".join(words[i:i + chunk_size])
                        atomic_chunks.append(word_chunk.strip())
                        
                        # Add metadata for this word chunk
                        word_chunk_len = len(word_chunk)
                        chunk_metadata.append({
                            'paragraph_idx': para_idx,
                            'start_position': chunk_start_pos,
                            'end_position': chunk_start_pos + word_chunk_len,
                            'is_complete_paragraph': False,
                            'is_partial_sentence': True
                        })
                        
                        chunk_start_pos += word_chunk_len + 1  # +1 for space
                        
                # Otherwise add sentence to current chunk if it fits
                elif current_size + sentence_words <= chunk_size:
                    current_chunk.append(sentence)
                    current_size += sentence_words
                # Start a new chunk if it doesn't fit
                else:
                    chunk_text = " ".join(current_chunk).strip()
                    atomic_chunks.append(chunk_text)
                    
                    # Add metadata
                    chunk_metadata.append({
                        'paragraph_idx': para_idx,
                        'start_position': chunk_start_pos,
                        'end_position': chunk_start_pos + len(chunk_text),
                        'is_complete_paragraph': False
                    })
                    
                    current_chunk = [sentence]
                    current_size = sentence_words
                    chunk_start_pos += len(chunk_text) + 1  # +1 for space
            
            # Add the last chunk if there's anything left
            if current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                atomic_chunks.append(chunk_text)
                
                # Add metadata
                chunk_metadata.append({
                    'paragraph_idx': para_idx,
                    'start_position': chunk_start_pos,
                    'end_position': chunk_start_pos + len(chunk_text),
                    'is_complete_paragraph': False
                })
            
            current_position += len(paragraph) + 2  # +2 for the paragraph break
        
        # Filter out empty chunks and corresponding metadata
        valid_chunks = []
        valid_metadata = []
        for chunk, metadata in zip(atomic_chunks, chunk_metadata):
            if chunk.strip():
                valid_chunks.append(chunk)
                valid_metadata.append(metadata)
        
        return valid_chunks, valid_metadata
    
    def get_embeddings(self, chunks):
        """Generate embeddings for each chunk using TF-IDF."""
        if not chunks:
            return np.array([])
        
        # Fit and transform in one step
        tfidf_matrix = self.vectorizer.fit_transform(chunks)
        
        # Convert sparse matrix to dense array
        return tfidf_matrix.toarray()
    
    def compute_attention_matrix(self, embeddings):
        """Compute the attention (similarity) matrix between all chunks."""
        if embeddings.size == 0:
            return np.array([[]])
        
        return cosine_similarity(embeddings)
    
    def find_chunk_clusters(self, attention_matrix, threshold=0.5, max_dist=None):
        """
        Group chunks into clusters based on attention scores using a more sophisticated
        approach that finds strongly related chunks regardless of document location.
        
        Parameters:
        - attention_matrix: Similarity matrix between chunks
        - threshold: Minimum similarity to consider chunks related
        - max_dist: Maximum distance (in chunks) to allow in the same cluster (None for no limit)
        
        Returns a list of cluster assignments for each chunk.
        """
        if attention_matrix.size <= 1:
            return [0]
            
        n_chunks = attention_matrix.shape[0]
        clusters = [-1] * n_chunks  # -1 means unassigned
        current_cluster = 0
        
        # Step 1: Find the strongest connections in the matrix
        # Create a flattened view excluding the diagonal
        mask = np.ones_like(attention_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        
        # If max_dist is specified, also mask connections beyond that distance
        if max_dist is not None:
            for i in range(n_chunks):
                for j in range(n_chunks):
                    if abs(i - j) > max_dist:
                        mask[i, j] = False
        
        flat_similarities = attention_matrix[mask].flatten()
        
        # Get indices of connections sorted by strength (highest to lowest)
        sorted_indices = np.argsort(flat_similarities)[::-1]
        
        # Step 2: Build clusters from strongest connections first
        edges = []
        for flat_idx in sorted_indices:
            # Skip weak connections
            if flat_idx >= len(flat_similarities) or flat_similarities[flat_idx] < threshold:
                break
                
            # Convert flat index back to matrix coordinates 
            # This is tricky because our mask might have irregular shape now
            mask_indices = np.where(mask.flatten())[0]
            if flat_idx >= len(mask_indices):
                continue
                
            original_idx = mask_indices[flat_idx]
            i, j = np.unravel_index(original_idx, attention_matrix.shape)
                
            # Add this edge to our list
            edges.append((i, j, flat_similarities[flat_idx]))
            
        # Step 3: Process edges to form clusters
        # Union-find data structure for efficient clustering
        parent = list(range(n_chunks))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
            
        def union(x, y):
            parent[find(x)] = find(y)
        
        # Process edges in order of similarity
        for i, j, sim in edges:
            if sim >= threshold:
                union(i, j)
        
        # Assign cluster IDs based on connected components
        cluster_map = {}
        for i in range(n_chunks):
            root = find(i)
            if root not in cluster_map:
                cluster_map[root] = current_cluster
                current_cluster += 1
            clusters[i] = cluster_map[root]
        
        return clusters
        
    def find_semantic_clusters(self, chunks, embeddings=None, min_similarity=0.5, 
                              ignore_position=True, max_chunk_distance=None):
        """
        Find clusters of semantically related chunks across the document.
        
        Parameters:
        - chunks: List of text chunks
        - embeddings: Pre-computed embeddings (if None, will compute them)
        - min_similarity: Minimum similarity threshold for chunks to be related
        - ignore_position: If True, allows clustering chunks regardless of position
        - max_chunk_distance: Maximum distance (in chunks) to consider for relationships
                             (None means no limit - chunks from anywhere can be related)
        
        Returns:
        - cluster_assignments: List of cluster IDs for each chunk
        - semantically_related_pairs: List of (chunk_i, chunk_j, similarity) tuples 
        """
        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = self.get_embeddings(chunks)
            
        # Compute attention/similarity matrix
        attention_matrix = self.compute_attention_matrix(embeddings)
        
        # Find clusters
        cluster_assignments = self.find_chunk_clusters(
            attention_matrix, 
            threshold=min_similarity,
            max_dist=None if ignore_position else max_chunk_distance
        )
        
        # Find most semantically similar pairs (across the entire document if ignore_position=True)
        semantically_related_pairs = []
        
        # Create a mask for the pairs we want to consider
        mask = np.ones_like(attention_matrix, dtype=bool)
        np.fill_diagonal(mask, False)  # Exclude self-similarity
        
        if not ignore_position and max_chunk_distance is not None:
            # Only consider chunks within max_chunk_distance
            for i in range(len(chunks)):
                for j in range(len(chunks)):
                    if abs(i - j) > max_chunk_distance:
                        mask[i, j] = False
        
        # Get flattened indices sorted by similarity
        flat_indices = np.argsort(attention_matrix[mask].flatten())[::-1]
        
        # Get the top related pairs
        for flat_idx in flat_indices[:50]:  # Limit to top 50 connections
            # Convert flat index back to matrix coordinates
            mask_indices = np.where(mask.flatten())[0]
            if flat_idx >= len(mask_indices):
                continue
                
            original_idx = mask_indices[flat_idx]
            i, j = np.unravel_index(original_idx, attention_matrix.shape)
            
            similarity = attention_matrix[i, j]
            if similarity >= min_similarity:
                semantically_related_pairs.append((i, j, similarity))
        
        return cluster_assignments, semantically_related_pairs
    
    def create_hierarchical_chunks(self, atomic_chunks, clusters, chunk_metadata=None):
        """
        Create higher-level chunks by combining atomic chunks based on cluster assignments.
        Preserves metadata about the chunk origins if available.
        """
        if not atomic_chunks:
            return [], []
            
        # Group chunks by cluster
        cluster_map = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_map:
                cluster_map[cluster_id] = []
            cluster_map[cluster_id].append(i)  # Store indices instead of chunks
        
        # Create combined chunks
        hierarchical_chunks = []
        hierarchical_metadata = []
        
        for cluster_id, chunk_indices in sorted(cluster_map.items()):
            # Sort indices to help with metadata generation
            chunk_indices.sort()
            
            # Combine chunks
            combined_chunks = [atomic_chunks[i] for i in chunk_indices]
            combined_text = "\n\n".join(combined_chunks)
            hierarchical_chunks.append(combined_text)
            
            # Generate metadata for the hierarchical chunk
            if chunk_metadata:
                combined_meta = {
                    'cluster_id': cluster_id,
                    'chunk_count': len(chunk_indices),
                    'source_indices': chunk_indices,
                    'source_metadata': [chunk_metadata[i] for i in chunk_indices],
                    # Convert set to list for JSON serialization
                    'paragraph_indices': list(set(chunk_metadata[i]['paragraph_idx'] for i in chunk_indices))
                }
                
                # Are chunks contiguous in the original document?
                is_contiguous = (max(chunk_indices) - min(chunk_indices) + 1) == len(chunk_indices)
                combined_meta['is_contiguous'] = is_contiguous
                
                # Distance between furthest chunks
                if len(chunk_indices) > 1:
                    combined_meta['span_distance'] = max(chunk_indices) - min(chunk_indices)
                else:
                    combined_meta['span_distance'] = 0
                
                hierarchical_metadata.append(combined_meta)
            else:
                # Basic metadata if original chunk metadata isn't available
                hierarchical_metadata.append({
                    'cluster_id': cluster_id,
                    'chunk_count': len(chunk_indices),
                    'source_indices': chunk_indices
                })
        
        return hierarchical_chunks, hierarchical_metadata
    
    def visualize_attention_matrix(self, attention_matrix, chunks, clusters=None, 
                                  title="Chunk Attention Matrix", output_dir="./output"):
        """Visualize the attention matrix as a heatmap with cluster annotations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if attention_matrix.size <= 1:
            print("Not enough data to visualize attention matrix")
            return
        
        try:
            plt.figure(figsize=(14, 12))
            
            # Create the heatmap
            ax = sns.heatmap(attention_matrix, annot=False, cmap="viridis")
            
            # If clusters are provided, add cluster boundaries
            if clusters is not None:
                # Sort chunk indices by cluster
                sorted_indices = np.argsort(clusters)
                sorted_clusters = [clusters[i] for i in sorted_indices]
                
                # Reorder the attention matrix
                reordered_matrix = attention_matrix[sorted_indices][:, sorted_indices]
                
                # Clear the current plot and redraw with reordered data
                plt.clf()
                ax = sns.heatmap(reordered_matrix, annot=False, cmap="viridis")
                
                # Add cluster boundary lines
                cluster_boundaries = [0]
                for i in range(1, len(sorted_clusters)):
                    if sorted_clusters[i] != sorted_clusters[i-1]:
                        cluster_boundaries.append(i)
                cluster_boundaries.append(len(sorted_clusters))
                
                # Draw lines at cluster boundaries
                for boundary in cluster_boundaries:
                    plt.axhline(y=boundary, color='r', linestyle='-', linewidth=2)
                    plt.axvline(x=boundary, color='r', linestyle='-', linewidth=2)
                
                # Update labels to show cluster information
                labels = [f"C{sorted_clusters[i]}-{i}" for i in range(len(chunks))]
            else:
                labels = [f"Chunk {i+1}" for i in range(len(chunks))]
                
            # Use a reasonable number of ticks to avoid overcrowding
            max_ticks = 40
            stride = max(1, len(labels) // max_ticks)
            
            plt.xticks(np.arange(len(labels))[::stride] + 0.5, 
                      [labels[i] for i in range(0, len(labels), stride)], 
                      rotation=90)
            plt.yticks(np.arange(len(labels))[::stride] + 0.5, 
                      [labels[i] for i in range(0, len(labels), stride)], 
                      rotation=0)
            
            plt.title(title)
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(output_dir / "attention_matrix.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            # Create a reference plot showing chunks and their cluster assignments
            plt.figure(figsize=(14, len(chunks) * 0.4 + 2))
            
            # Calculate reasonable y spacing based on number of chunks
            y_spacing = min(0.02, 0.8 / max(1, len(chunks)))
            
            if clusters is not None:
                for i, chunk in enumerate(chunks):
                    # Truncate chunks for display
                    display_chunk = chunk[:50] + "..." if len(chunk) > 50 else chunk
                    plt.text(0, 1 - (i * y_spacing), 
                            f"Cluster {clusters[i]} - Chunk {i+1}: {display_chunk}")
            else:
                for i, chunk in enumerate(chunks):
                    display_chunk = chunk[:50] + "..." if len(chunk) > 50 else chunk
                    plt.text(0, 1 - (i * y_spacing), f"Chunk {i+1}: {display_chunk}")
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / "chunk_reference.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            print(f"Saved attention matrix visualization to '{output_dir}/attention_matrix.png'")
            print(f"Saved chunk reference to '{output_dir}/chunk_reference.png'")
        except Exception as e:
            print(f"Warning: Error visualizing attention matrix: {e}")
            print("Continuing with analysis without visualizations...")
    
    def analyze_pdf(self, pdf_path, atomic_size=50, cluster_threshold=0.5, 
                  ignore_position=True, output_dir="./output"):
        """
        Complete hierarchical pipeline to analyze a PDF:
        1. Extract text
        2. Create atomic chunks with metadata
        3. Generate embeddings for atomic chunks
        4. Compute attention between atomic chunks
        5. Cluster atomic chunks into hierarchical chunks
        6. Analyze and visualize both levels
        
        Parameters:
        - pdf_path: Path to the PDF file
        - atomic_size: Target size of atomic chunks (in words)
        - cluster_threshold: Similarity threshold for clustering chunks (0.0-1.0)
        - ignore_position: If True, chunks from anywhere in document can be clustered
        - output_dir: Directory to save outputs
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
            
        print(f"Extracted {len(text)} characters from PDF.")
        
        # Create atomic chunks with metadata
        atomic_chunks, chunk_metadata = self.create_atomic_chunks(text, atomic_size)
        print(f"Created {len(atomic_chunks)} atomic chunks with target size {atomic_size} words.")
        
        if not atomic_chunks:
            print("No chunks were created. Check if text extraction worked correctly.")
            return None
            
        # Generate embeddings for atomic chunks
        atomic_embeddings = self.get_embeddings(atomic_chunks)
        print(f"Generated atomic embeddings with shape {atomic_embeddings.shape}.")
        
        # Compute attention matrix for atomic chunks
        atomic_attention_matrix = self.compute_attention_matrix(atomic_embeddings)
        print(f"Computed atomic attention matrix with shape {atomic_attention_matrix.shape}.")
        
        # Find clusters and semantic relationships
        clusters, semantic_pairs = self.find_semantic_clusters(
            atomic_chunks,
            embeddings=atomic_embeddings,
            min_similarity=cluster_threshold,
            ignore_position=ignore_position
        )
        
        unique_clusters = len(set(clusters))
        print(f"Found {unique_clusters} semantic clusters from {len(atomic_chunks)} atomic chunks.")
        print(f"Found {len(semantic_pairs)} strong semantic relationships between chunks.")
        
        # Create hierarchical chunks with metadata
        hierarchical_chunks, hierarchical_metadata = self.create_hierarchical_chunks(
            atomic_chunks, clusters, chunk_metadata
        )
        print(f"Created {len(hierarchical_chunks)} hierarchical chunks.")
        
        # Generate embeddings for hierarchical chunks
        hierarchical_embeddings = self.get_embeddings(hierarchical_chunks)
        
        # Compute attention matrix for hierarchical chunks
        hierarchical_attention_matrix = self.compute_attention_matrix(hierarchical_embeddings)
        
        # Visualize results
        output_atomic_dir = Path(output_dir) / "atomic"
        output_hierarchical_dir = Path(output_dir) / "hierarchical"
        output_semantic_dir = Path(output_dir) / "semantic"
        
        output_atomic_dir.mkdir(exist_ok=True, parents=True)
        output_hierarchical_dir.mkdir(exist_ok=True, parents=True)
        output_semantic_dir.mkdir(exist_ok=True, parents=True)
        
        # Visualize atomic chunks with cluster annotations
        self.visualize_attention_matrix(
            atomic_attention_matrix, 
            atomic_chunks, 
            clusters, 
            title="Atomic Chunks Attention Matrix",
            output_dir=output_atomic_dir
        )
        
        # Visualize hierarchical chunks
        self.visualize_attention_matrix(
            hierarchical_attention_matrix, 
            hierarchical_chunks, 
            title="Hierarchical Chunks Attention Matrix",
            output_dir=output_hierarchical_dir
        )
        
        # Visualize semantic relationships
        self._visualize_semantic_relationships(
            atomic_chunks, semantic_pairs, clusters, output_semantic_dir
        )
        
        # Save all chunks to text files with metadata
        self._save_chunk_files(
            atomic_chunks, chunk_metadata, clusters, 
            hierarchical_chunks, hierarchical_metadata, 
            semantic_pairs, output_dir
        )
        
        # Create a visualization of chunk relationships
        self._visualize_chunk_relationships(
            atomic_chunks, hierarchical_chunks, 
            clusters, hierarchical_metadata, 
            hierarchical_attention_matrix, output_dir
        )
        
        print(f"Saved atomic chunks to {output_atomic_dir}/atomic_chunks.txt")
        print(f"Saved hierarchical chunks to {output_hierarchical_dir}/hierarchical_chunks.txt")
        print(f"Saved semantic analysis to {output_semantic_dir}")
        
        # Return results
        return {
            "atomic_chunks": atomic_chunks,
            "chunk_metadata": chunk_metadata,
            "atomic_embeddings": atomic_embeddings,
            "atomic_attention_matrix": atomic_attention_matrix,
            "clusters": clusters,
            "semantic_pairs": semantic_pairs,
            "hierarchical_chunks": hierarchical_chunks,
            "hierarchical_metadata": hierarchical_metadata,
            "hierarchical_embeddings": hierarchical_embeddings,
            "hierarchical_attention_matrix": hierarchical_attention_matrix
        }
    
    def _save_chunk_files(self, atomic_chunks, chunk_metadata, clusters,
                       hierarchical_chunks, hierarchical_metadata,
                       semantic_pairs, output_dir):
        """Save all chunk information to text files with metadata"""
        import json
        
        output_atomic_dir = Path(output_dir) / "atomic"
        output_hierarchical_dir = Path(output_dir) / "hierarchical"
        output_semantic_dir = Path(output_dir) / "semantic"
        
        # Ensure all directories exist
        output_atomic_dir.mkdir(exist_ok=True, parents=True)
        output_hierarchical_dir.mkdir(exist_ok=True, parents=True)
        output_semantic_dir.mkdir(exist_ok=True, parents=True)
        
        # Helper function to convert non-serializable objects
        def json_serialize(obj):
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Custom JSON encoder
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                return json_serialize(obj)
        
        # Save atomic chunks
        with open(f"{output_atomic_dir}/atomic_chunks.txt", "w", encoding="utf-8") as f:
            for i, chunk in enumerate(atomic_chunks):
                f.write(f"--- Atomic Chunk {i+1} (Cluster {clusters[i]}) ---\n")
                f.write(f"Metadata: {json.dumps(chunk_metadata[i], indent=2, cls=CustomJSONEncoder)}\n\n")
                f.write(chunk)
                f.write("\n\n")
        
        # Save hierarchical chunks
        with open(f"{output_hierarchical_dir}/hierarchical_chunks.txt", "w", encoding="utf-8") as f:
            for i, chunk in enumerate(hierarchical_chunks):
                f.write(f"--- Hierarchical Chunk {i+1} ---\n")
                f.write(f"Metadata: {json.dumps(hierarchical_metadata[i], indent=2, cls=CustomJSONEncoder)}\n\n")
                f.write(chunk)
                f.write("\n\n")
        
        # Save semantic relationships
        with open(f"{output_semantic_dir}/semantic_relationships.txt", "w", encoding="utf-8") as f:
            f.write(f"=== DOCUMENT SEMANTIC RELATIONSHIPS ===\n")
            f.write(f"Found {len(semantic_pairs)} strong semantic relationships\n\n")
            
            for i, (idx1, idx2, similarity) in enumerate(semantic_pairs):
                f.write(f"Relationship {i+1}: Chunks {idx1} and {idx2} (similarity: {similarity:.4f})\n")
                f.write(f"--- Chunk {idx1} (Cluster {clusters[idx1]}) ---\n")
                f.write(atomic_chunks[idx1][:200] + "...\n\n")
                f.write(f"--- Chunk {idx2} (Cluster {clusters[idx2]}) ---\n")
                f.write(atomic_chunks[idx2][:200] + "...\n\n")
                f.write("-" * 80 + "\n\n")
        
        # Save a combined JSON for potential use in other tools
        analysis_data = {
            "document_info": {
                "chunk_count": len(atomic_chunks),
                "cluster_count": len(set(clusters)),
                "hierarchical_chunk_count": len(hierarchical_chunks),
                "semantic_relationship_count": len(semantic_pairs)
            },
            "atomic_chunks": [
                {
                    "id": i,
                    "text": chunk,
                    "cluster": clusters[i],
                    "metadata": chunk_metadata[i]
                }
                for i, chunk in enumerate(atomic_chunks)
            ],
            "hierarchical_chunks": [
                {
                    "id": i,
                    "text": chunk,
                    "source_indices": metadata["source_indices"],
                    "metadata": metadata
                }
                for i, (chunk, metadata) in enumerate(zip(hierarchical_chunks, hierarchical_metadata))
            ],
            "semantic_relationships": [
                {
                    "chunk1_id": int(idx1),
                    "chunk2_id": int(idx2),
                    "similarity": float(similarity),
                    "same_cluster": clusters[idx1] == clusters[idx2]
                }
                for idx1, idx2, similarity in semantic_pairs
            ]
        }
        
        with open(f"{output_dir}/document_analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2, cls=CustomJSONEncoder)
        
    def _visualize_chunk_relationships(self, atomic_chunks, hierarchical_chunks, 
                                     clusters, hierarchical_metadata, 
                                     hierarchical_attention_matrix, output_dir):
        """
        Create a visualization showing the relationships between atomic chunks
        and how they're grouped into hierarchical chunks across the document.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            # Create a graph visualization of chunk relationships
            G = nx.Graph()
            
            # Add nodes for atomic chunks
            for i, chunk in enumerate(atomic_chunks):
                # Truncate chunk text for display
                short_text = chunk[:30] + "..." if len(chunk) > 30 else chunk
                G.add_node(f"A{i}", label=f"A{i}", type="atomic", 
                          cluster=clusters[i], text=short_text)
            
            # Add nodes for hierarchical chunks
            for i, chunk in enumerate(hierarchical_chunks):
                # Truncate chunk text for display
                short_text = chunk[:30] + "..." if len(chunk) > 30 else chunk
                G.add_node(f"H{i}", label=f"H{i}", type="hierarchical", 
                          text=short_text, size=hierarchical_metadata[i]['chunk_count'])
            
            # Add edges from hierarchical chunks to their atomic chunks
            for i, metadata in enumerate(hierarchical_metadata):
                for atomic_idx in metadata['source_indices']:
                    G.add_edge(f"H{i}", f"A{atomic_idx}")
            
            # Create a figure
            plt.figure(figsize=(16, 12))
            
            # Create positions - hierarchical chunks on top, atomic chunks below
            pos = {}
            
            # Position hierarchical chunks on top row
            h_nodes = [n for n in G.nodes() if n.startswith('H')]
            for i, node in enumerate(h_nodes):
                pos[node] = (i * (16.0 / max(1, len(h_nodes))), 1.0)
            
            # Position atomic chunks on bottom, grouped by original document order
            a_nodes = [n for n in G.nodes() if n.startswith('A')]
            for i, node in enumerate(a_nodes):
                a_idx = int(node[1:])
                pos[node] = (a_idx * (16.0 / max(1, len(a_nodes))), 0.0)
            
            # Draw the network
            hierarchical_nodes = [n for n in G.nodes() if n.startswith('H')]
            atomic_nodes = [n for n in G.nodes() if n.startswith('A')]
            
            # Draw atomic nodes colored by cluster
            unique_clusters = sorted(set(clusters))
            cluster_colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_clusters))))
            cluster_color_map = {cluster: color for cluster, color in zip(unique_clusters, cluster_colors)}
            
            atomic_colors = [cluster_color_map[clusters[int(n[1:])]] for n in atomic_nodes]
            
            nx.draw_networkx_nodes(G, pos, nodelist=atomic_nodes, 
                                  node_color=atomic_colors, node_size=200)
            
            # Draw hierarchical nodes with size based on number of contained chunks
            hierarchical_sizes = [G.nodes[n]['size'] * 100 + 300 for n in hierarchical_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=hierarchical_nodes, 
                                  node_color='lightblue', node_size=hierarchical_sizes,
                                  node_shape='s')
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            
            # Draw labels
            hierarchical_labels = {n: n for n in hierarchical_nodes}
            atomic_labels = {n: n for n in atomic_nodes}
            
            nx.draw_networkx_labels(G, pos, hierarchical_labels, font_size=12)
            nx.draw_networkx_labels(G, pos, atomic_labels, font_size=8)
            
            plt.axis('off')
            plt.title("Document Chunk Relationships")
            plt.savefig(output_dir / "chunk_relationships.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            # Create a more detailed visualization with chunk text
            plt.figure(figsize=(16, 24))
            
            # Determine space between text items based on number of chunks
            atomic_spacing = min(0.02, 0.8 / max(1, len(atomic_chunks)))
            hierarchical_spacing = min(0.03, 0.4 / max(1, len(hierarchical_chunks)))
            
            # Draw atomic chunks in document order
            for i, chunk in enumerate(atomic_chunks):
                # Truncate text for display
                short_text = chunk[:50] + "..." if len(chunk) > 50 else chunk
                cluster_id = clusters[i]
                
                y_pos = max(0.05, 0.95 - (i * atomic_spacing))
                plt.text(0.1, y_pos, 
                        f"A{i} [C{cluster_id}]: {short_text}", 
                        fontsize=8, 
                        bbox=dict(facecolor=cluster_color_map[cluster_id], alpha=0.2))
            
            # Draw hierarchical chunks below
            for i, (chunk, metadata) in enumerate(zip(hierarchical_chunks, hierarchical_metadata)):
                # Truncate text for display
                short_text = chunk.split("\n")[0][:100] + "..."
                source_indices = metadata['source_indices']
                
                y_pos = max(0.05, 0.5 - (i * hierarchical_spacing))
                
                plt.text(0.5, y_pos, 
                        f"H{i} (from A{min(source_indices)}-A{max(source_indices)}): {short_text}", 
                        fontsize=10, 
                        bbox=dict(facecolor='lightblue', alpha=0.3))
            
            plt.axis('off')
            plt.title("Document Chunks with Text Preview")
            plt.savefig(output_dir / "chunk_text_preview.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            print(f"Saved chunk relationship visualizations to {output_dir}")
            
            # Create a JSON file with all the data for interactive visualization
            visualization_data = {
                'atomic_chunks': [{'id': i, 'text': c[:100] + "..." if len(c) > 100 else c, 
                                  'cluster': int(clusters[i])} for i, c in enumerate(atomic_chunks)],
                'hierarchical_chunks': [{'id': i, 'text': c[:100] + "..." if len(c) > 100 else c, 
                                        'source_indices': [int(idx) for idx in m['source_indices']],
                                        'is_contiguous': bool(m.get('is_contiguous', False))} 
                                       for i, (c, m) in enumerate(zip(hierarchical_chunks, hierarchical_metadata))],
                'clusters': [int(c) for c in list(set(clusters))]
            }
            
            import json
            with open(output_dir / "visualization_data.json", "w", encoding="utf-8") as f:
                class CustomJSONEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, (np.integer, np.int64, np.int32)):
                            return int(obj)
                        if isinstance(obj, (np.floating, np.float64, np.float32)):
                            return float(obj)
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        if isinstance(obj, set):
                            return list(obj)
                        return super().default(obj)
                
                json.dump(visualization_data, f, indent=2, cls=CustomJSONEncoder)
            
            # Create a visualization of the hierarchical chunk network
            if hierarchical_attention_matrix.size > 1:
                self._visualize_hierarchical_network(hierarchical_chunks, hierarchical_metadata, 
                                                  hierarchical_attention_matrix, output_dir)
        except Exception as e:
            print(f"Warning: Error during visualization: {e}")
            print("Continuing with analysis without visualizations...")
            
    def _visualize_hierarchical_network(self, hierarchical_chunks, hierarchical_metadata, 
                                       attention_matrix, output_dir):
        """Create a visualization of relationships between hierarchical chunks"""
        if len(hierarchical_chunks) <= 1 or attention_matrix.size <= 1:
            return
            
        try:
            # Create a graph
            G = nx.Graph()
            
            # Add nodes for each hierarchical chunk
            for i, (chunk, metadata) in enumerate(zip(hierarchical_chunks, hierarchical_metadata)):
                short_text = chunk.split("\n")[0][:50] + "..."
                G.add_node(i, label=f"H{i}", text=short_text, 
                          size=metadata['chunk_count'],
                          contiguous=metadata.get('is_contiguous', False))
            
            # Add edges for chunk relationships above a threshold
            threshold = 0.3  # Minimum similarity to show an edge
            for i in range(len(hierarchical_chunks)):
                for j in range(i+1, len(hierarchical_chunks)):
                    similarity = attention_matrix[i, j]
                    if similarity >= threshold:
                        G.add_edge(i, j, weight=similarity)
            
            # Create visualization
            plt.figure(figsize=(14, 10))
            
            # Use a force-directed layout
            pos = nx.spring_layout(G, k=0.3, iterations=50)
            
            # Draw nodes with size based on number of atomic chunks
            node_sizes = [G.nodes[n]['size'] * 200 + 500 for n in G.nodes()]
            node_colors = ['lightgreen' if G.nodes[n]['contiguous'] else 'lightblue' for n in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
            
            # Draw edges with width based on similarity
            if G.edges():
                edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
                nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6)
            
            # Draw labels
            labels = {n: f"H{n}\n{G.nodes[n]['text']}" for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif')
            
            plt.axis('off')
            plt.title("Hierarchical Chunk Relationships")
            plt.savefig(output_dir / "hierarchical_network.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Warning: Error visualizing hierarchical network: {e}")
            print("Continuing with analysis...")
            
    def _visualize_semantic_relationships(self, chunks, semantic_pairs, clusters, output_dir):
        """Create visualizations of semantic relationships between chunks"""
        if not semantic_pairs:
            return
        
        try:
            # Create a graph of semantic relationships
            G = nx.Graph()
            
            # Add nodes for each chunk
            for i, chunk in enumerate(chunks):
                short_text = chunk[:50] + "..." if len(chunk) > 50 else chunk
                G.add_node(i, text=short_text, cluster=clusters[i])
            
            # Add edges for semantic relationships
            for idx1, idx2, similarity in semantic_pairs:
                G.add_edge(idx1, idx2, weight=similarity)
            
            # Visualize the graph
            plt.figure(figsize=(14, 12))
            
            # Use a spring layout with stronger repulsion
            pos = nx.spring_layout(G, k=0.3, iterations=50)
            
            # Color nodes by cluster
            cluster_values = sorted(list(set(clusters)))
            cluster_colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(cluster_values))))
            cluster_color_map = {cluster: color for cluster, color in zip(cluster_values, cluster_colors)}
            
            node_colors = [cluster_color_map[G.nodes[n]['cluster']] for n in G.nodes()]
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, alpha=0.8)
            
            # Draw edges with width based on semantic similarity
            if G.edges():
                edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
                nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6)
            
                # Highlight edges between chunks in different clusters
                different_cluster_edges = [(u, v) for u, v in G.edges() 
                                          if G.nodes[u]['cluster'] != G.nodes[v]['cluster']]
                if different_cluster_edges:
                    nx.draw_networkx_edges(G, pos, edgelist=different_cluster_edges,
                                         width=[G[u][v]['weight'] * 5 for u, v in different_cluster_edges],
                                         edge_color='red', alpha=0.7)
            
            # Draw labels for nodes
            labels = {n: f"{n}" for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=9)
            
            plt.axis('off')
            plt.title("Semantic Relationships Between Chunks")
            plt.savefig(output_dir / "semantic_network.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            # Create a more detailed visualization with top relationships
            plt.figure(figsize=(14, 24))
            
            # Sort relationships by similarity score
            sorted_pairs = sorted(semantic_pairs, key=lambda x: x[2], reverse=True)
            
            # Display top relationships with preview text
            for i, (idx1, idx2, similarity) in enumerate(sorted_pairs[:15]):  # Show top 15
                if i >= 15:  # Limit to prevent overcrowding
                    break
                    
                y_pos = max(0.05, 0.98 - (i * 0.06))
                plt.text(0.05, y_pos, 
                        f"Relationship {i+1}: Chunks {idx1}-{idx2} (sim={similarity:.4f})",
                        fontsize=12, fontweight='bold')
                
                # First chunk
                plt.text(0.05, y_pos - 0.02,
                        f"Chunk {idx1} (Cluster {clusters[idx1]}): {chunks[idx1][:100]}...",
                        fontsize=10)
                
                # Second chunk  
                plt.text(0.55, y_pos - 0.02,
                        f"Chunk {idx2} (Cluster {clusters[idx2]}): {chunks[idx2][:100]}...",
                        fontsize=10)
            
            plt.axis('off')
            plt.title("Top Semantic Relationships Between Chunks")
            plt.savefig(output_dir / "top_relationships.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Warning: Error visualizing semantic relationships: {e}")
            print("Continuing with analysis without visualizations...")