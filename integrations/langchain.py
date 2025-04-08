# semantic_chunker/integrations/langchain_chunker.py
from typing import Any, List, Dict
from langchain_core.documents import Document
from langchain.text_splitter import TextSplitter
from semantic_chunker.core import ChunkAnalyzer


class SemanticChunkerSplitter(TextSplitter):
    def __init__(
        self,
        analyzer: ChunkAnalyzer = None,
        cluster_threshold: float = 0.5,
        similarity_threshold: float = 0.4,
        max_tokens: int = 512,
        return_merged: bool = True,
        **kwargs: Any
    ):
        super().__init__()
        self.cluster_threshold = cluster_threshold
        self.similarity_threshold = similarity_threshold
        self.return_merged = return_merged
        self.analyzer = analyzer or ChunkAnalyzer(max_tokens=max_tokens, **kwargs)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        texts = []
        for doc in documents:
            texts.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
            })

        results = self.analyzer.analyze_chunks(
            texts,
            cluster_threshold=self.cluster_threshold,
            similarity_threshold=self.similarity_threshold,
        )

        chunks = results["merged_chunks"] if self.return_merged else results["original_chunks"]

        return [
            Document(
                page_content=chunk["text"],
                metadata={"source_chunks": chunk.get("metadata", [])}
            )
            for chunk in chunks
        ]

    def split_text(self, text: str) -> List[str]:
        return [text]