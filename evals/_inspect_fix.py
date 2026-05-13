"""Quick inspector for the latest eval result -- prints answers for specific entry IDs."""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGETS = {"HAVELLS-segment-01"}
latest = max(Path("evals/results").glob("agent-eval-*.json"), key=lambda p: p.stat().st_mtime)
print(f"Latest: {latest.name}\n")
r = json.loads(latest.read_text(encoding="utf-8"))
for entry in r["results"]:
    if entry["entry"]["id"] not in TARGETS:
        continue
    print("=" * 80)
    print(f"[{entry['entry']['id']}]")
    print(f"QUERY:     {entry['entry']['query']}")
    print(f"REFERENCE: {entry['entry']['reference']}")
    print()
    for cond in ["with_fc", "without_fc"]:
        c = entry["conditions"][cond]
        print(f"--- {cond} | judge={c['judge']['score_mean']:.2f}±{c['judge']['score_std']:.2f} | {c['elapsed_seconds']}s | {len(c['answer'])} chars ---")
        print(f"rationales: {[run['rationale'] for run in c['judge']['runs']]}")
        print(c["answer"][:1500])
        print()
