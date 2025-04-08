# tests/test_export.py
import json
import csv
from pathlib import Path
from semantic_chunker.export import export_chunks, export_all


def sample_chunks():
    return [
        {"text": "Chunk A.", "metadata": {"source": "page_1"}},
        {"text": "Chunk B.", "metadata": {"source": "page_2"}},
    ]


def test_export_chunks_json(tmp_path):
    output_file = tmp_path / "chunks.json"
    export_chunks(sample_chunks(), str(output_file), format="json")
    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == 2


def test_export_chunks_md(tmp_path):
    output_file = tmp_path / "chunks.md"
    export_chunks(sample_chunks(), str(output_file), format="md")
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "## Chunk 1" in content
    assert "Chunk A." in content


def test_export_chunks_csv(tmp_path):
    output_file = tmp_path / "chunks.csv"
    export_chunks(sample_chunks(), str(output_file), format="csv")
    assert output_file.exists()
    with open(output_file, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["text"] == "Chunk A."


def test_export_all(tmp_path):
    results = {
        "original_chunks": sample_chunks(),
        "merged_chunks": sample_chunks(),
        "hierarchical_clusters": {
            "level_1": [0, 1],
            "level_2": [0, 0]
        }
    }
    base = tmp_path / "export"
    export_all(results, base_path=str(base), format="json")
    for suffix in ["_original.json", "_merged.json", "_level_1.json", "_level_2.json"]:
        assert (base.parent / (base.name + suffix)).exists()