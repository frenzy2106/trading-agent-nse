"""Quick coverage check: does each golden-eval entry's `must_contain_themes`
appear anywhere in the latest fact card for the same ticker?

This is NOT a precision eval -- it's a binary "are the facts even present" check
to tell us whether fact-card injection has the raw material to answer the
golden queries. If a theme is missing here, RAG would still need to find it;
if it's present, the agent gets it for free in the system prompt.
"""
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tools.fact_card import get_fact_card

eval_data = json.load(open("evals/golden_commentary.json", encoding="utf-8"))

print(f"{'ID':32s} {'coverage':>10s}   missing themes")
print("-" * 90)
for entry in eval_data:
    card = get_fact_card(entry["ticker"])
    if card is None:
        print(f"{entry['id']:32s} {'NO CARD':>10s}")
        continue
    text = card.model_dump_json().lower()
    themes = entry["must_contain_themes"]
    hits = [t for t in themes if t.lower() in text]
    misses = [t for t in themes if t.lower() not in text]
    coverage = f"{len(hits)}/{len(themes)}"
    miss_str = "" if not misses else "  miss: " + ", ".join(misses)
    print(f"{entry['id']:32s} {coverage:>10s}{miss_str}")
