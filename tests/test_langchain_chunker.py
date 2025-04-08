from langchain_core.documents import Document
from semantic_chunker.integrations.langchain import SemanticChunkerSplitter

def test_langchain_chunker_split():
    documents = [
        Document(
            page_content="""
                Artificial intelligence is a growing field.
                Machine learning is a subset of AI.
                Photosynthesis occurs in plants.
                Deep learning uses neural networks.
                Plants convert sunlight into energy.
            """,
            metadata={"source": "test_doc"}
        )
    ]

    splitter = SemanticChunkerSplitter(cluster_threshold=0.4, return_merged=True)
    chunks = splitter.split_documents(documents)

    assert isinstance(chunks, list)
    assert all(isinstance(c, Document) for c in chunks)
    assert all("source_chunks" in c.metadata for c in chunks)
    assert any("Artificial intelligence" in c.page_content for c in chunks)

    print("✅ LangChain integration test passed with", len(chunks), "chunks.")


if __name__ == "__main__":
    test_langchain_chunker_split()
