# Macro Events Log

This file is automatically loaded into every analysis run, filtered to events within the last 365 days. Edit it to add new events as they happen, or to remove ones that are no longer relevant.

Format:
```
## YYYY-MM-DD — Short title
Affects: SECTORS, COMMA, SEPARATED   (e.g. ENERGY, IT, FINANCIAL, ALL)
Tags: free, comma, tags

A few sentences on what happened and why it matters for stock analysis.
```

Rules:
- Date in `YYYY-MM-DD` ISO format. Used to filter the log by the agent.
- Sections separated by an `## ` heading. Anything not under a dated `## ` heading is ignored by the loader.
- Keep each entry short — full text is injected into every run, so token cost matters.
- Delete entries when they're no longer load-bearing.

---

## 2024-04-13 — Iran-Israel direct conflict escalates
Affects: ENERGY, DEFENSE, ALL
Tags: geopolitics, oil
Iran launched ~300 drones/missiles at Israel — first direct state-on-state strike. Brent crude spiked to $90+. Sustained risk-off premium in EM equities. Continues to underpin oil-price expectations through 2025-26.

## 2024-06-04 — Indian general election results
Affects: ALL
Tags: politics, india
BJP won but with reduced majority (240 seats, down from 303). Coalition government with TDP/JDU. Markets initially fell ~6% intraday before recovering. Relevant context for any policy-sensitive sector (PSU banks, defense, infrastructure).

## 2025-09-22 — GST 2.0: rate rationalisation to two-slab structure (5% / 18%)
Affects: AUTO, FMCG, CONSUMER-DURABLES, PHARMA, REALTY, INSURANCE, TEXTILES
Tags: tax-policy, demand, india

GST Council's 56th meeting (chaired by FM Nirmala Sitharaman) collapsed slabs from 5/12/18/28% down to a simpler 5%/18% structure, with a 40% slab retained for luxury and sin goods (tobacco, pan masala, aerated drinks, high-end cars, yachts, private aircraft). Effective from 22-Sep-2025. Weighted-average GST drops from ~11.64% (FY24) toward single digits.

Demand-side tailwind running through FY26:
- **Auto + consumer durables**: rates cut on cars, ACs, TVs, dishwashers — improves affordability and volume.
- **FMCG**: many packaged items moved from 12% → 5%, supporting volume growth at the mass-market end.
- **Pharma + healthcare**: many medicines moved to nil / 0%; health and life insurance premiums cut from 18% to nil — direct margin lift.
- **Construction / realty**: lower input costs (cement, paints, fixtures) feed through to project margins and affordability.
- **Textiles**: rate cuts on apparel inputs.

When reading any name in these sectors, treat the YoY comps from Q3 FY26 onward as flattered by the cut. Base-effects start lapping in Q3 FY27 (~Sep 2026), narrowing the standout impact thereafter.

## 2026-01-15 — Claude Opus 4.7 launches
Affects: IT-services, TECH
Tags: ai, competition
Anthropic released next-generation Claude Opus 4.7. Increased competitive pressure on Indian IT services that rebrand or resell US AI capabilities. Long-term, may also accelerate productivity gains for IT services that integrate AI internally — net effect ambiguous.
