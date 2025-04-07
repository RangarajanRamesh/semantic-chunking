# semantic_chunker/core.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class ChunkAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2', max_tokens=512):
        self.model = SentenceTransformer(model_name)
        self.max_tokens = max_tokens

    def get_embeddings(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return np.array([])
        texts = [chunk["text"] for chunk in chunks]
        return np.array(self.model.encode(texts, show_progress_bar=False))

    def compute_attention_matrix(self, embeddings):
        if embeddings.size == 0:
            return np.array([[]])
        return cosine_similarity(embeddings)

    def find_chunk_clusters(self, attention_matrix, threshold=0.5):
        n = attention_matrix.shape[0]
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for i in range(n):
            for j in range(i+1, n):
                if attention_matrix[i, j] >= threshold:
                    union(i, j)

        clusters = [find(i) for i in range(n)]
        cluster_map = {cid: idx for idx, cid in enumerate(sorted(set(clusters)))}
        return [cluster_map[c] for c in clusters]

    def find_top_semantic_pairs(self, attention_matrix, min_similarity=0.4, top_k=50):
        pairs = []
        n = attention_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                sim = attention_matrix[i, j]
                if sim >= min_similarity:
                    pairs.append((i, j, sim))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_k]

    def merge_clusters(self, chunks: List[Dict[str, Any]], clusters: List[int], tokenizer=None):
        from collections import defaultdict

        cluster_map = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            cluster_map[cluster_id].append(chunks[idx])

        merged_chunks = []
        for cluster_id, chunk_list in cluster_map.items():
            current_text = ""
            current_meta = []

            for chunk in chunk_list:
                next_text = (current_text + " " + chunk["text"]).strip()

                if tokenizer:
                    num_tokens = len(tokenizer.tokenize(next_text))
                else:
                    num_tokens = len(next_text.split())  # fallback: word count proxy

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

    def analyze_chunks(self, chunks: List[Dict[str, Any]], cluster_threshold=0.5, similarity_threshold=0.4):
        embeddings = self.get_embeddings(chunks)
        attention = self.compute_attention_matrix(embeddings)
        clusters = self.find_chunk_clusters(attention, threshold=cluster_threshold)
        semantic_pairs = self.find_top_semantic_pairs(attention, min_similarity=similarity_threshold)
        merged = self.merge_clusters(chunks, clusters)

        return {
            "original_chunks": chunks,
            "embeddings": embeddings,
            "attention_matrix": attention,
            "clusters": clusters,
            "semantic_pairs": semantic_pairs,
            "merged_chunks": merged,
        }