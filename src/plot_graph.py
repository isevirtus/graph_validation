import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# path input output
GRAPH_JSON = "../data/parsed_graph.json"
CSV_OUTPUT = "../data/network_edges_details_ordenado1.csv"
IMG_OUTPUT = "../data/network_plot.png"

# ───────────────────────────────────────────────────────────────
# 1. load data
# ───────────────────────────────────────────────────────────────
with open(GRAPH_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

nodes = data["nodes"]
edges = data["edges"]

# ───────────────────────────────────────────────────────────────
# 2. build graph
# ───────────────────────────────────────────────────────────────
G = nx.Graph()

for node in nodes:
    G.add_node(
        node["id"],
        user_id=node["user_id"],
        type=node.get("type", "developer"),
        contractRoleName=node.get("contractRoleName", "unspecified"),
        hardskill=node.get("hardskill", [])
    )

for edge in edges:
    G.add_edge(
        edge["source"],
        edge["target"],
        weight=edge["weight"],
        source_user_id=edge["source_user_id"],
        target_user_id=edge["target_user_id"],
        N=edge.get("N", "?")
    )

# ───────────────────────────────────────────────────────────────
# 3. Export edges
# ───────────────────────────────────────────────────────────────
rows = []
for u, v, d in G.edges(data=True):
    rows.append({
        "source": u,
        "source_user_id": d["source_user_id"],
        "target": v,
        "target_user_id": d["target_user_id"],
        "weight": d["weight"]
    })

df = pd.DataFrame(rows).sort_values(by="weight", ascending=False)
df.to_csv(CSV_OUTPUT, index=False)
print(f"✅ CSV saved → {CSV_OUTPUT}")

# ───────────────────────────────────────────────────────────────
# 4. Plot a graph
# ───────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)

# nodes
nx.draw_networkx_nodes(
    G, pos,
    nodelist=[n for n in G.nodes()],
    node_color='lightblue', node_size=600
)

# edges
weights = [d["weight"] * 1 for _, _, d in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, width=weights, alpha=0.8)

# nodes labes
nx.draw_networkx_labels(G, pos, font_size=6)


# edges labes (W=weight, N=number of projects)
edge_labels = {
    (u, v): f"W={d['weight']:.2f} | N={d.get('N', '?')}" 
    for u, v, d in G.edges(data=True)
}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)




nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title("Developer Compatibility Graph", fontsize=15)
plt.axis("off")
plt.tight_layout()
plt.savefig(IMG_OUTPUT, dpi=300)
plt.show()

print(f"✅ Graph image saved → {IMG_OUTPUT}")
