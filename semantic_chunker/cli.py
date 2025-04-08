import argparse
import json
from semantic_chunker.core import SemanticChunker  # updated import
from semantic_chunker.visualization import plot_attention_matrix, preview_clusters
from semantic_chunker.export import export_chunks


def chunk_cmd(args):
    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunker = SemanticChunker(
        max_tokens=args.max_tokens,
        cluster_threshold=args.threshold,
        similarity_threshold=args.similarity_threshold
    )
    merged_chunks = chunker.chunk(chunks)
    print(f"Created {len(merged_chunks)} merged chunks.")

    if args.preview:
        debug = chunker.get_debug_info(chunks)
        preview_clusters(debug["original_chunks"], debug["clusters"])
        if args.visualize:
            plot_attention_matrix(debug["similarity_matrix"], debug["clusters"], title="Chunk Similarity")

    if args.export:
        export_chunks(merged_chunks, path=args.export_path, format=args.export_format)
        print(f"Exported merged chunks to {args.export_path}.{args.export_format}")


def main():
    parser = argparse.ArgumentParser(description="Semantic Chunker CLI")
    subparsers = parser.add_subparsers(dest="command")

    chunk_parser = subparsers.add_parser("chunk", help="Merge semantically similar chunks")
    chunk_parser.add_argument("--chunks", required=True, help="Path to input JSON list of chunks")
    chunk_parser.add_argument("--threshold", type=float, default=0.5, help="Cluster threshold")
    chunk_parser.add_argument("--similarity-threshold", type=float, default=0.4, help="Min similarity for clustering")
    chunk_parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per merged chunk")
    chunk_parser.add_argument("--preview", action="store_true", help="Show preview of clustered chunks")
    chunk_parser.add_argument("--visualize", action="store_true", help="Show similarity matrix plot")
    chunk_parser.add_argument("--export", action="store_true", help="Export merged chunks")
    chunk_parser.add_argument("--export-path", default="output/merged", help="Path for export file (no extension)")
    chunk_parser.add_argument("--export-format", default="json", choices=["json", "md", "csv"], help="Export file format")
    chunk_parser.set_defaults(func=chunk_cmd)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
