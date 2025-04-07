# semantic_chunker/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path

def plot_attention_matrix(attention_matrix, clusters=None, title="Attention Matrix", output_path=None):
    if attention_matrix.size <= 1:
        print("Not enough data to visualize attention matrix")
        return

    reordered_matrix = attention_matrix
    if clusters is not None:
        sorted_indices = np.argsort(clusters)
        reordered_matrix = attention_matrix[sorted_indices][:, sorted_indices]

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(reordered_matrix, cmap="viridis")

    if clusters is not None:
        sorted_clusters = [clusters[i] for i in sorted_indices]
        boundaries = [0]
        for i in range(1, len(sorted_clusters)):
            if sorted_clusters[i] != sorted_clusters[i - 1]:
                boundaries.append(i)
        boundaries.append(len(sorted_clusters))

        for b in boundaries:
            plt.axhline(b, color='red', linewidth=1)
            plt.axvline(b, color='red', linewidth=1)

    plt.title(title)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_semantic_graph(chunks, pairs, clusters, output_path=None):
    G = nx.Graph()

    for i, chunk in enumerate(chunks):
        G.add_node(i, label=f"{i}", cluster=clusters[i], text=chunk["text"][:30] + "...")

    for i, j, sim in pairs:
        G.add_edge(i, j, weight=sim)

    pos = nx.spring_layout(G, seed=42, k=0.3)
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(set(clusters))))
    color_map = {c: cluster_colors[i] for i, c in enumerate(sorted(set(clusters)))}
    node_colors = [color_map[clusters[n]] for n in G.nodes()]

    plt.figure(figsize=(14, 12))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, {n: n for n in G.nodes()}, font_size=9)
    plt.axis('off')
    plt.title("Semantic Relationships Between Chunks")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


def preview_clusters(chunks, clusters, output_path=None, max_per_cluster=3):
    cluster_map = {}
    for i, cluster_id in enumerate(clusters):
        cluster_map.setdefault(cluster_id, []).append(i)

    lines = []
    for cluster_id, indices in sorted(cluster_map.items()):
        lines.append(f"Cluster {cluster_id} ({len(indices)} chunks):")
        for idx in indices[:max_per_cluster]:
            text = chunks[idx].get("text", "")
            lines.append(f"  - Chunk {idx}: {text[:80]}...")
        lines.append("")

    output = "\n".join(lines)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output)