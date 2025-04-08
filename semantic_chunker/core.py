import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from collections import defaultdict


class SemanticChunker:
    def __init__(
        self,
        model_name='all-MiniLM-L6-v2',
        max_tokens=512,
        cluster_threshold=0.5,
        similarity_threshold=0.4
    ):
        self.device = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        print(f"[Info] Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.max_tokens = max_tokens
        self.cluster_threshold = cluster_threshold
        self.similarity_threshold = similarity_threshold
        self.tokenizer = self.model.tokenizer if hasattr(self.model, "tokenizer") else None

    def get_embeddings(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return np.array([])
        texts = [chunk["text"] for chunk in chunks]
        return np.array(self.model.encode(texts, show_progress_bar=False))

    def compute_similarity(self, embeddings):
        if embeddings.size == 0:
            return np.zeros((0, 0))
        return cosine_similarity(embeddings)

    def cluster_chunks(self, similarity_matrix, threshold=0.5):
        n = similarity_matrix.shape[0]
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    union(i, j)

        clusters = [find(i) for i in range(n)]
        cluster_map = {cid: idx for idx, cid in enumerate(sorted(set(clusters)))}
        return [cluster_map[c] for c in clusters]

    def merge_chunks(self, chunks: List[Dict[str, Any]], clusters: List[int]) -> List[Dict[str, Any]]:
        if not chunks or not clusters:
            return []

        cluster_map = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            cluster_map[cluster_id].append(chunks[idx])

        merged_chunks = []
        for chunk_list in cluster_map.values():
            current_text = ""
            current_meta = []

            for chunk in chunk_list:
                next_text = (current_text + " " + chunk["text"]).strip()
                if self.tokenizer:
                    num_tokens = len(self.tokenizer.encode(next_text))
                else:
                    num_tokens = len(next_text.split())

                if num_tokens > self.max_tokens and current_text:
                    merged_chunks.append({
                        "text": current_text,
                        "metadata": current_meta
                    })
                    current_text = chunk["text"]
                    current_meta = [chunk]
                else:
                    current_text = next_text
                    current_meta.append(chunk)

            if current_text:
                merged_chunks.append({
                    "text": current_text,
                    "metadata": current_meta
                })

        return merged_chunks

    def find_top_semantic_pairs(self, similarity_matrix, min_similarity=0.4, top_k=50):
        pairs = []
        n = similarity_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity_matrix[i, j]
                if sim >= min_similarity:
                    pairs.append((i, j, sim))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_k]

    def chunk(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        embeddings = self.get_embeddings(chunks)
        similarity_matrix = self.compute_similarity(embeddings)
        clusters = self.cluster_chunks(similarity_matrix, threshold=self.cluster_threshold)
        return self.merge_chunks(chunks, clusters)

    def get_debug_info(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optional: for visualization or export/debug purposes."""
        embeddings = self.get_embeddings(chunks)
        similarity_matrix = self.compute_similarity(embeddings)
        clusters = self.cluster_chunks(similarity_matrix, threshold=self.cluster_threshold)
        merged_chunks = self.merge_chunks(chunks, clusters)
        semantic_pairs = self.find_top_semantic_pairs(similarity_matrix, min_similarity=self.similarity_threshold)

        return {
            "original_chunks": chunks,
            "embeddings": embeddings,
            "similarity_matrix": similarity_matrix,
            "clusters": clusters,
            "semantic_pairs": semantic_pairs,
            "merged_chunks": merged_chunks
        }
