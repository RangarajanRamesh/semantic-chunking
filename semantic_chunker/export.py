# semantic_chunker/export.py
import json
import csv
from pathlib import Path
from typing import List, Dict


def export_to_json(chunks: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def export_to_markdown(chunks: List[Dict], path: str):
    lines = []
    for i, chunk in enumerate(chunks):
        lines.append(f"## Chunk {i+1}\n")
        lines.append(chunk["text"] + "\n")
        if "metadata" in chunk:
            lines.append("```json")
            lines.append(json.dumps(chunk["metadata"], indent=2, ensure_ascii=False))
            lines.append("```")
        lines.append("\n")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def export_to_csv(chunks: List[Dict], path: str):
    with open(path, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["chunk_id", "text", "metadata"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, chunk in enumerate(chunks):
            writer.writerow({
                "chunk_id": i,
                "text": chunk["text"],
                "metadata": json.dumps(chunk.get("metadata", {}), ensure_ascii=False)
            })


def export_chunks(chunks: List[Dict], path: str, format: str = "json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if not path.endswith(f".{format}"):
        path = f"{path}.{format}"

    if format == "json":
        export_to_json(chunks, path)
    elif format in {"md", "markdown"}:
        export_to_markdown(chunks, path)
    elif format == "csv":
        export_to_csv(chunks, path)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def export_all(results: Dict, base_path: str = "output/export", format: str = "json"):
    export_chunks(results.get("original_chunks", []), f"{base_path}_original", format)
    export_chunks(results.get("merged_chunks", []), f"{base_path}_merged", format)

    hier = results.get("hierarchical_clusters", {})
    for level, cluster_list in hier.items():
        level_chunks = []
        for cluster_id in sorted(set(cluster_list)):
            grouped = [results["original_chunks"][i] for i, cid in enumerate(cluster_list) if cid == cluster_id]
            level_chunks.append({
                "text": "\n\n".join([c["text"] for c in grouped]),
                "metadata": grouped
            })
        export_chunks(level_chunks, f"{base_path}_{level}", format)