import json
import os
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# STAGE S5 (RQ1): Export the edge list and plot the developer
# compatibility graph.
# ============================================================
# Input:  outputs/compatibility_graph.json
# Output: outputs/network_edges.csv
#         outputs/network_plot_full.png       (all valid edges)
#         outputs/network_plot_top_subgraph.png (top-N by weight,
#                                                 for an illustrative
#                                                 figure in the paper)
# ============================================================

GRAPH_PATH = os.path.join("outputs", "compatibility_graph.json")
EDGES_CSV_PATH = os.path.join("outputs", "network_edges.csv")
FULL_PLOT_PATH = os.path.join("outputs", "network_plot_full.png")
SUBGRAPH_PLOT_PATH = os.path.join("outputs", "network_plot_top_subgraph.png")

TOP_N_EDGES_FOR_ILLUSTRATIVE_FIGURE = 20


def load_graph():
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.Graph()
    for node in data["nodes"]:
        G.add_node(
            node["id"],
            user_id=node["user_id"],
            type=node.get("type", "developer"),
            contractRoleName=node.get("contractRoleName", "unspecified"),
            hardskill=node.get("hardskill", []),
        )
    for edge in data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            weight=edge["weight"],
            source_user_id=edge["source_user_id"],
            target_user_id=edge["target_user_id"],
            N=edge.get("N", "?"),
        )
    return G


def export_edges_csv(G: nx.Graph):
    rows = [
        {
            "source": u,
            "source_user_id": d["source_user_id"],
            "target": v,
            "target_user_id": d["target_user_id"],
            "weight": d["weight"],
            "N": d["N"],
        }
        for u, v, d in G.edges(data=True)
    ]
    df = pd.DataFrame(rows).sort_values(by="weight", ascending=False)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(EDGES_CSV_PATH, index=False)
    print(f"OK: edge list saved to {EDGES_CSV_PATH}")
    return df


def plot_graph(G: nx.Graph, path: str, title: str, with_labels: bool = True):
    plt.figure(figsize=(12, 10))
    pos = nx.kamada_kawai_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=600)
    weights = [d["weight"] * 2 for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.8)

    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=9)
        edge_labels = {
            (u, v): f"w={d['weight']:.2f} | N={d.get('N', '?')}"
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title(title, fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"OK: figure saved to {path}")


def main():
    G = load_graph()
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    export_edges_csv(G)

    # Full graph (may be dense/unreadable if the number of edges is large --
    # useful for the supplementary material, not necessarily for the paper figure).
    plot_graph(
        G,
        FULL_PLOT_PATH,
        title="Developer Compatibility Graph (full)",
        with_labels=(G.number_of_edges() <= 60),
    )

    # Illustrative subgraph: top-N edges by weight, for a clean, readable
    # figure suitable for the paper (analogous to the "running example"
    # figure already in the manuscript).
    top_edges = sorted(G.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)
    top_edges = top_edges[:TOP_N_EDGES_FOR_ILLUSTRATIVE_FIGURE]
    sub_nodes = set()
    for u, v, _ in top_edges:
        sub_nodes.update([u, v])
    H = G.subgraph(sub_nodes).copy()
    # Keep only the top edges in the illustrative subgraph (a node may have
    # other, lower-weight edges inside the induced subgraph too).
    H_top = nx.Graph()
    H_top.add_nodes_from((n, G.nodes[n]) for n in sub_nodes)
    for u, v, d in top_edges:
        H_top.add_edge(u, v, **d)

    plot_graph(
        H_top,
        SUBGRAPH_PLOT_PATH,
        title=f"Developer Compatibility Graph (top {TOP_N_EDGES_FOR_ILLUSTRATIVE_FIGURE} edges by weight)",
        with_labels=True,
    )

    print(f"    Illustrative subgraph: {H_top.number_of_nodes()} nodes, "
          f"{H_top.number_of_edges()} edges")


if __name__ == "__main__":
    main()
