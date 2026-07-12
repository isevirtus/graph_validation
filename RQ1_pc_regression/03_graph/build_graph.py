import json
import os
import re

# ============================================================
# STAGE S5 (RQ1): Build the developer compatibility graph
# G = (V, E, w) from the PC pairs report and the developer base.
# ============================================================
# Inputs:
#   ../02_apply_to_pairs/outputs/pc_pairs_report.txt
#   ../00_data/developers_base.json
# Output:
#   outputs/compatibility_graph.json   ({"nodes": [...], "edges": [...]})
#
# Edge filter (paper, Section S5): a pair (i, j) is a valid edge
# only if PC > 0.3.
# ============================================================

REPORT_PATH = os.path.join("..", "02_apply_to_pairs", "outputs", "pc_pairs_report.txt")
DEVELOPERS_PATH = os.path.join("..", "00_data", "developers_base.json")
OUTPUT_PATH = os.path.join("outputs", "compatibility_graph.json")

PC_EDGE_THRESHOLD = 0.3

PAIR_PATTERN = re.compile(
    r"Developer\s*Pair\s*:.*?"
    r"ID\s*=\s*(\d+)\s*e\s*ID\s*=\s*(\d+).*?"
    r"N\s*=\s*(\d+).*?f\(N\)\s*=\s*[\d.,]+.*?"
    r"Final\s*PC\s*=\s*([\d.,]+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_edges(report_text: str):
    edges = []
    connected_ids = set()
    for m in PAIR_PATTERN.finditer(report_text):
        d1, d2, N, pc_str = m.groups()
        d1, d2, N = int(d1), int(d2), int(N)
        pc = float(pc_str.replace(",", "."))

        if pc <= PC_EDGE_THRESHOLD:
            continue

        edges.append({
            "source": f"Dev{d1}",
            "source_user_id": d1,
            "target": f"Dev{d2}",
            "target_user_id": d2,
            "weight": round(pc, 4),
            "N": N,
        })
        connected_ids.update([d1, d2])
    return edges, connected_ids


def build_nodes(developers_path: str, connected_ids: set):
    with open(developers_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    devs = data if isinstance(data, list) else data.get("developers", [])
    dev_by_id = {d["id"]: d for d in devs if "id" in d}

    def dedupe(items):
        seen, out = set(), []
        for x in items:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    nodes = []
    for dev_id in connected_ids:
        dev = dev_by_id.get(dev_id)
        if not dev:
            continue
        domain = dev.get("dominio") or []
        ecosystem = dev.get("ecossistema") or []
        language = dev.get("linguagens") or []
        hardskill = dedupe(list(domain) + list(ecosystem) + list(language))

        nodes.append({
            "id": f"Dev{dev_id}",
            "user_id": dev_id,
            "type": dev.get("type", "developer"),
            "contractRoleName": dev.get("contractRoleName", "unspecified"),
            "hardskill": hardskill,
        })
    return nodes


def main():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report_text = f.read()

    edges, connected_ids = parse_edges(report_text)
    print(f"[parse] pairs captured with PC > {PC_EDGE_THRESHOLD}: {len(edges)}")

    nodes = build_nodes(DEVELOPERS_PATH, connected_ids)
    print(f"[build] nodes with at least one valid edge: {len(nodes)}")

    os.makedirs("outputs", exist_ok=True)
    graph = {"nodes": nodes, "edges": edges}
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    print(f"OK: compatibility graph saved to {OUTPUT_PATH}")
    print(f"    Developers (nodes): {len(nodes)}")
    print(f"    Valid pairs (edges, PC > {PC_EDGE_THRESHOLD}): {len(edges)}")


if __name__ == "__main__":
    main()
