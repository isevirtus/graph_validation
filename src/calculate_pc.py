import json
import numpy as np
import itertools

# Load dataset
with open("../data/dataset.json", "r") as f:
    data = json.load(f)

developers = data['developers']

# Map each developer to their projects
dev_projects = {}
for dev in developers:
    dev_id = dev['id']
    dev_projects[dev_id] = {
        'name': dev['name'],
        'projects': {p['id']: {'osf': p['osf'], 'slf': p['slf']} for p in dev['projects']}
    }

# f(N) function
def f_N(N):
    if N <= 4:
        return N / 4
    else:
        return max(0.0, 1 - 0.1 * (N - 4))

# Formula parameters
alpha = 0.4674
beta = 0.5021
gamma = 0.2934
intercept = -0.1221

# Goal counts
target_counts = {4: 4, 5: 3, 6: 3}
current_counts = {4: 0, 5: 0, 6: 0}

seen = set()
report_lines = []

for d1, d2 in itertools.combinations(dev_projects.keys(), 2):
    pair = tuple(sorted((d1, d2)))
    if pair in seen:
        continue
    seen.add(pair)

    projects1 = dev_projects[d1]['projects']
    projects2 = dev_projects[d2]['projects']
    common = set(projects1.keys()) & set(projects2.keys())
    N = len(common)

    if N not in target_counts:
        continue

    if current_counts[N] >= target_counts[N]:
        continue

    osfs = [(projects1[pid]['osf'] + projects2[pid]['osf']) / 2 for pid in common]
    slfs = [(projects1[pid]['slf'] + projects2[pid]['slf']) / 2 for pid in common]
    osf_avg = np.mean(osfs)
    slf_avg = np.mean(slfs)
    fn = f_N(N)

    pc = alpha * osf_avg + beta * slf_avg + gamma * fn + intercept
    pc = max(0, min(1, pc))  # Clamp to [0, 1]

    output = (
        f"\n🔹 Developer Pair: {dev_projects[d1]['name']} (ID={d1}) and {dev_projects[d2]['name']} (ID={d2})\n"
        f"📂 Common Projects: {list(common)}\n"
        f"📊 Avg OSF: {osf_avg:.3f} | Avg SLF: {slf_avg:.3f}\n"
        f"🔢 N = {N} → f(N) = {fn:.3f}\n"
        f"🎯 Final PC = {pc:.3f}\n"
    )
    print(output)
    report_lines.append(output)
    current_counts[N] += 1

    # Stop when all target counts are reached
    if all(current_counts[n] >= target_counts[n] for n in target_counts):
        break

# Save to file
with open("../data/report_pc.txt", "w", encoding="utf-8") as f:
    f.writelines(report_lines)

print("✅ PC report saved to report_pc.txt")
