from semantic_chunker.core import SemanticChunker

def test_semantic_chunker_success():
    chunks = [
        {"text": "Artificial intelligence is a growing field."},
        {"text": "Machine learning is a subset of AI."},
        {"text": "Photosynthesis occurs in plants."},
        {"text": "Deep learning uses neural networks."},
        {"text": "Plants convert sunlight into energy."},
    ]

    chunker = SemanticChunker(max_tokens=100)
    debug = chunker.get_debug_info(chunks)

    assert "embeddings" in debug
    assert "similarity_matrix" in debug
    assert "clusters" in debug
    assert "semantic_pairs" in debug
    assert "merged_chunks" in debug
    assert len(debug["clusters"]) == len(chunks)


def test_semantic_chunker_empty_input():
    chunker = SemanticChunker()
    debug = chunker.get_debug_info([])

    assert debug["original_chunks"] == []
    assert debug["embeddings"].size == 0
    assert debug["similarity_matrix"].shape == (0, 0)
    assert debug["merged_chunks"] == []
    assert debug["semantic_pairs"] == []


def test_semantic_chunker_partial_failure(monkeypatch):
    def broken_similarity(*args, **kwargs):
        raise ValueError("Sim matrix failed")

    chunker = SemanticChunker()
    monkeypatch.setattr(chunker, "compute_similarity", broken_similarity)

    chunks = [
        {"text": "A."},
        {"text": "B."}
    ]

    try:
        chunker.get_debug_info(chunks)
        assert False, "Expected failure not raised"
    except ValueError as e:
        assert "Sim matrix failed" in str(e)
