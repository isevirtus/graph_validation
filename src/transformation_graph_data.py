import re
import json

# --------- caminhos ---------
REPORT_PATH = "../data/report_pc.txt"          # saída do cálculo de PC
DATASET_PATH = "../data/dataset.json"     # mesmo JSON usado no cálculo de PC
OUTPUT_PATH = "../data/parsed_graph.json"         # arquivo final com nodes + edges

# --------- carregar relatório ---------
with open(REPORT_PATH, "r", encoding="utf-8") as f:
    report_text = f.read()

# --------- regex para pares ---------
pattern = re.compile(
    r"Developer Pair: .*?ID=(\d+)\).*?ID=(\d+)\).*?"
    r"N = (\d+) → f\(N\) = [0-9.]+\s+"
    r"🎯 Final PC = ([0-9.]+)",
    re.DOTALL
)


edges = []
dev_ids_in_edges = set()

for m in pattern.finditer(report_text):
    d1 = int(m.group(1))
    d2 = int(m.group(2))
    N = int(m.group(3))
    pc = float(m.group(4))

    edges.append({
        "source": f"Dev{d1}",
        "source_user_id": d1,
        "target": f"Dev{d2}",
        "target_user_id": d2,
        "weight": round(pc, 4),
        "N": N
    })
    dev_ids_in_edges.update([d1, d2])

# --------- construir nodes ---------
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dev_data = {d["id"]: d for d in json.load(f)["developers"]}

nodes = []
for dev_id in dev_ids_in_edges:                        # só quem aparece nos edges
    dev = dev_data[dev_id]
    nodes.append({
        "id": f"Dev{dev_id}",
        "user_id": dev_id,
        "type":  dev.get("type", "developer"),         # default “developer”
        "contractRoleName": dev.get("contractRoleName", "unspecified"),
        "hardskill": dev.get("dominio", []) +          # exemplo: usar campos do dataset
                     ([dev.get("ecossistema")] if dev.get("ecossistema") else []) +
                     ([dev.get("linguagens")] if dev.get("linguagens") else [])
    })

# --------- salvar ---------
graph = {"nodes": nodes, "edges": edges}
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(graph, f, indent=2)

print(f"✅ Graph (nodes + edges) salvo em {OUTPUT_PATH}")
