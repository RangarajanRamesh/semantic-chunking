import argparse
from semantic_chunker.core import ChunkAnalyzer
from semantic_chunker.visualization import plot_attention_matrix, preview_clusters
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks', required=True, help='Path to JSON list of text chunks')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    with open(args.chunks, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    analyzer = ChunkAnalyzer()
    result = analyzer.analyze_chunks(chunks, cluster_threshold=args.threshold)

    print(f"Found {len(set(result['clusters']))} clusters.")
    preview_clusters(result['chunks'], result['clusters'])

    if args.visualize:
        plot_attention_matrix(result['attention_matrix'], result['clusters'], title='Chunk Similarity')

if __name__ == "__main__":
    main()
