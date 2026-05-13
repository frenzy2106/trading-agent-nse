"""Quick inspector for the latest agent eval -- shows answers + judge rationales
for the cases where with_fc and without_fc diverged."""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

results_dir = Path("evals/results")
latest = max(results_dir.glob("agent-eval-*.json"), key=lambda p: p.stat().st_mtime)
print(f"Inspecting {latest.name}\n")
r = json.loads(latest.read_text(encoding="utf-8"))

swings = []
for entry in r["results"]:
    w = entry["conditions"]["with_fc"]["judge"]["score"]
    wo = entry["conditions"]["without_fc"]["judge"]["score"]
    if w is None or wo is None:
        continue
    if abs(w - wo) >= 0.25:
        swings.append((entry, w - wo))

if not swings:
    print("No swings >= 0.25 found.")
    sys.exit(0)

for entry, delta in swings:
    print("=" * 80)
    print(f"[{entry['entry']['id']}]  delta = {delta:+.2f}")
    print(f"QUERY:     {entry['entry']['query']}")
    print(f"REFERENCE: {entry['entry']['reference']}")
    print()
    for cond in ["with_fc", "without_fc"]:
        c = entry["conditions"][cond]
        print(f"--- {cond} | judge={c['judge']['score']} | t={c['elapsed_seconds']}s | {len(c['answer'])} chars ---")
        print(f"judge rationale: {c['judge']['rationale']}")
        print(c["answer"][:1500])
        print()
