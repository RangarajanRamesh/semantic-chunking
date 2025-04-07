from chunkAnalyser import HierarchicalChunkAnalyzer
from pathlib import Path
import argparse

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Hierarchical PDF Chunk Analyzer")
    parser.add_argument("--pdf", default="test.pdf", help="Path to the PDF file")
    parser.add_argument("--atomic-size", type=int, default=50, help="Target size of atomic chunks (in words)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold for clustering (0.0-1.0)")
    parser.add_argument("--respect-position", action="store_true", help="Only cluster nearby chunks")
    parser.add_argument("--output-dir", default="./output", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Configuration from command-line arguments
    pdf_path = args.pdf
    atomic_size = args.atomic_size
    cluster_threshold = args.threshold
    ignore_position = not args.respect_position  # Invert the flag for clarity
    output_dir = args.output_dir
    
    print("=" * 60)
    print("SEMANTIC HIERARCHICAL CHUNK ANALYZER FOR RAG")
    print("=" * 60)
    print(f"Processing: {pdf_path}")
    print(f"Atomic chunk size: {atomic_size} words")
    print(f"Similarity threshold: {cluster_threshold}")
    print(f"Ignore document position: {ignore_position}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize the analyzer
    analyzer = HierarchicalChunkAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_pdf(
        pdf_path, 
        atomic_size=atomic_size, 
        cluster_threshold=cluster_threshold,
        ignore_position=ignore_position,
        output_dir=output_dir
    )
    
    if results:
        print("\nAnalysis complete!")
        print(f"All outputs saved to {output_dir}")
        
        # Print summary statistics
        atomic_chunks = results["atomic_chunks"]
        clusters = results["clusters"] 
        hierarchical_chunks = results["hierarchical_chunks"]
        semantic_pairs = results["semantic_pairs"]
        
        print("\n" + "=" * 40)
        print("ANALYSIS SUMMARY")
        print("=" * 40)
        
        # Cluster statistics
        cluster_counts = {}
        for cluster_id in clusters:
            if cluster_id not in cluster_counts:
                cluster_counts[cluster_id] = 0
            cluster_counts[cluster_id] += 1
        
        print(f"\nFound {len(set(clusters))} semantic clusters:")
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"  Cluster {cluster_id}: {count} atomic chunks")
        
        # Semantic relationship statistics
        if semantic_pairs:
            print(f"\nTop 5 semantic relationships across the document:")
            sorted_pairs = sorted(semantic_pairs, key=lambda x: x[2], reverse=True)[:5]
            
            for i, (idx1, idx2, similarity) in enumerate(sorted_pairs):
                print(f"  {i+1}. Chunks {idx1} and {idx2} (similarity: {similarity:.4f})")
                print(f"     Cluster {clusters[idx1]} → Cluster {clusters[idx2]}")
                print(f"     Chunk {idx1}: {atomic_chunks[idx1][:50]}...")
                print(f"     Chunk {idx2}: {atomic_chunks[idx2][:50]}...")
                print()
        
        # Cross-cluster connections
        cross_cluster_pairs = [(i, j, sim) for i, j, sim in semantic_pairs 
                              if clusters[i] != clusters[j]]
        
        if cross_cluster_pairs:
            print(f"\nFound {len(cross_cluster_pairs)} connections between different clusters")
            print(f"Top 3 cross-cluster connections:")
            
            sorted_cross = sorted(cross_cluster_pairs, key=lambda x: x[2], reverse=True)[:3]
            for i, (idx1, idx2, similarity) in enumerate(sorted_cross):
                print(f"  {i+1}. Cluster {clusters[idx1]} → Cluster {clusters[idx2]} (similarity: {similarity:.4f})")
        
        # Hierarchical chunk statistics
        print(f"\nCreated {len(hierarchical_chunks)} hierarchical chunks:")
        
        # Group hierarchical chunks by contiguity
        contiguous = 0
        non_contiguous = 0
        for metadata in results["hierarchical_metadata"]:
            if metadata.get("is_contiguous", False):
                contiguous += 1
            else:
                non_contiguous += 1
        
        print(f"  Contiguous chunks: {contiguous}")
        print(f"  Non-contiguous chunks: {non_contiguous}")
        
        print("\n" + "=" * 40)
        print("OUTPUT LOCATIONS")
        print("=" * 40)
        print(f"Atomic chunks: {output_dir}/atomic/atomic_chunks.txt")
        print(f"Hierarchical chunks: {output_dir}/hierarchical/hierarchical_chunks.txt") 
        print(f"Semantic relationships: {output_dir}/semantic/semantic_relationships.txt")
        print(f"Visualizations: {output_dir}/*.png")
        print(f"Complete analysis data: {output_dir}/document_analysis.json")
        
    else:
        print("\nAnalysis failed. Please check the PDF file.")

if __name__ == "__main__":
    main()

def main():
    # Configuration (edit these variables directly)
    pdf_path = "test.pdf"           # Path to your PDF file
    atomic_size = 50                # Target size of atomic chunks (in words)
    cluster_threshold = 0.5         # Similarity threshold for clustering chunks (0.0-1.0)
    output_dir = "./output"         # Directory to save outputs
    
    print("=" * 50)
    print("HIERARCHICAL PDF CHUNK ANALYZER")
    print("=" * 50)
    print(f"Processing: {pdf_path}")
    print(f"Atomic chunk size: {atomic_size} words")
    print(f"Cluster similarity threshold: {cluster_threshold}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize the analyzer
    analyzer = HierarchicalChunkAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_pdf(pdf_path, atomic_size, cluster_threshold, output_dir)
    
    if results:
        print("\nAnalysis complete!")
        print(f"All outputs saved to {output_dir}")
        
        # Print summary statistics
        atomic_chunks = results["atomic_chunks"]
        clusters = results["clusters"]
        hierarchical_chunks = results["hierarchical_chunks"]
        
        # Get some cluster statistics
        cluster_counts = {}
        for cluster_id in clusters:
            if cluster_id not in cluster_counts:
                cluster_counts[cluster_id] = 0
            cluster_counts[cluster_id] += 1
        
        print("\nCluster Statistics:")
        print("-" * 30)
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"Cluster {cluster_id}: {count} atomic chunks")
        
        # Find top connections between hierarchical chunks
        hier_matrix = results["hierarchical_attention_matrix"]
        if hier_matrix.size > 1:
            print("\nTop Hierarchical Chunk Relationships:")
            print("-" * 30)
            
            # Create mask to exclude self-connections
            mask = np.ones_like(hier_matrix, dtype=bool)
            np.fill_diagonal(mask, False)
            
            # Find top 3 connections
            flat_indices = np.argsort(hier_matrix[mask].ravel())[-3:]
            
            for idx, flat_idx in enumerate(reversed(flat_indices)):
                i, j = np.unravel_index(flat_idx, hier_matrix.shape)
                if i > j:  # Adjust for diagonal removal
                    i, j = j, i
                
                score = hier_matrix[i, j]
                print(f"Relationship {idx+1}: Hierarchical Chunks {i+1} and {j+1} (score: {score:.4f})")
                print(f"  Chunk {i+1}: {hierarchical_chunks[i][:50]}...")
                print(f"  Chunk {j+1}: {hierarchical_chunks[j][:50]}...")
                print()
        
        print("\nExample Atomic to Hierarchical Mapping:")
        print("-" * 30)
        
        # Show a few examples of how atomic chunks map to hierarchical chunks
        displayed_clusters = set()
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in displayed_clusters and len(displayed_clusters) < 3:
                displayed_clusters.add(cluster_id)
                hierarchical_idx = list(set(clusters)).index(cluster_id)
                
                print(f"Cluster {cluster_id} → Hierarchical Chunk {hierarchical_idx+1}:")
                print(f"  Atomic Chunk: {atomic_chunks[i][:100]}...")
                print(f"  Hierarchical Chunk: {hierarchical_chunks[hierarchical_idx][:100]}...")
                print()
    else:
        print("\nAnalysis failed. Please check the PDF file.")

if __name__ == "__main__":
    import numpy as np
    main()