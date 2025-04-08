from langchain_core.runnables import RunnableSerializable
from langchain_core.documents import Document
from pydantic import PrivateAttr
from semantic_chunker.core import SemanticChunker

class SemanticChunkerSplitter(RunnableSerializable):
    _return_merged: bool = PrivateAttr(default=True)
    _analyzer: SemanticChunker = PrivateAttr()

    def __init__(self, cluster_threshold=0.4, similarity_threshold=0.4, max_tokens=512, return_merged=True):
        super().__init__()
        self._return_merged = return_merged
        self._analyzer = SemanticChunker(
            cluster_threshold=cluster_threshold,
            similarity_threshold=similarity_threshold,
            max_tokens=max_tokens,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        texts = [{"text": doc.page_content, "metadata": doc.metadata} for doc in documents]
        chunks = self._analyzer.chunk(texts)

        return [
            Document(
                page_content=chunk["text"],
                metadata={"source_chunks": chunk.get("metadata", [])}

            )
            for chunk in chunks
        ]

    def invoke(self, input, config=None):
        return self.split_documents(input)
