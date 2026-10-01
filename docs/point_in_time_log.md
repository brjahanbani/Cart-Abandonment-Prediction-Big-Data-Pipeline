# Step 3 — Point-in-time correctness of feature joins (log)

Goal (agreed 2026-09-29): a 4-page DEEM@SIGMOD short paper on point-in-time
correctness in feature pipelines, with cart abandonment as the case study,
layered on the existing prefix dataset (`1-Data/decision-time/prefix_dataset.parquet`).
Separate from the IJACSA paper; its numbers are not touched.

Code: `6-Point-In-Time/`. Probes in `6-Point-In-Time/probes/` are exploratory
and read-only (they print, they write nothing); every number below comes from
one of them.

## 2026-10-02 — The data: `item_properties`

`6-Point-In-Time/probes/profile_props.py` (≈90 s).

- 20,275,902 rows (`timestamp, itemid, property, value`), 417,053 items,
  1,104 properties. Only `categoryid` and `available` are readable; all other
  values are hashed tokens, numbers prefixed with `n` (e.g. `n15360.000`).
- **Weekly snapshots:** 18 distinct timestamps, every Sunday 03:00, 2015-05-10
  → 2015-09-13 (2015-06-21 missing). Same period as the events.
- **Change-log semantics:** a row is written when a value changes (static
  properties ≈1 row per item; property `790`, numeric and price-like, 4.3 rows
  per item and changes for 20% of items; `available` changes for 16%;
  `categoryid` for 5.6%).
- Cart sessions: 98.8% of first-cart items have property rows.

At each session's first add-to-cart (the k=0 decision point), comparing the
value **as of** the cart time with the **latest** value in the file:

| Property | as-of ≠ latest | no snapshot yet at cart time | as-of staleness (median / p90 days) |
|---|---|---|---|
| `available` | 36.4% | 8.9% | 4.9 / 78.7 |
| `790` (price-like) | 77.5% | 6.5% | 3.7 / 6.9 |
| `categoryid` | 6.0% | 16.2% | 46.8 / 104.7 |

The disagreement is almost independent of the outcome (`available`: 35.9%
abandoned vs 37.8% purchased; P(purchase | latest available = 0) = 0.284 vs
0.264 for = 1). The suspected "sold out after purchase" leak is weak.

## 2026-10-02 — Probe 1: item-property join modes

`6-Point-In-Time/probes/join_probe.py` (≈2 min). k=0 rows (43,745; 26.88%
positive). The cart item's `available`, log price (`790`) and missing-value
indicators are attached four ways; XGBoost (200 trees, depth 4) and LR,
5 seeds, visitor-grouped and temporal 80/20 splits. ROC-AUC (XGBoost mean):

| Join | item features only (vis / temp) | prefix + item (vis / temp) |
|---|---|---|
| none | — | 0.643 / 0.664 |
| as-of (`merge_asof` backward — correct) | 0.539 / 0.537 | 0.649 / 0.663 |
| nearest (`merge_asof` nearest — can look ≤3.5 days ahead) | 0.555 / 0.557 | 0.654 / 0.671 |
| latest (end-of-data table) | 0.547 / 0.539 | 0.648 / 0.663 |

Only **nearest** leaks measurably (+0.015–0.02 item-only; +0.005–0.008 with
prefix features). **Latest** ≈ as-of: the final snapshot is weeks after most
sessions. Leakage concentrates in the first snapshot after the outcome and
fades with distance. Too small, on its own, to carry a paper.

## 2026-10-02 — Probe 2: aggregate windows (feature-store style)

`6-Point-In-Time/probes/agg_probe.py` (≈75 s; re-run from another directory
reproduced every number). Two aggregates at the k=0 cart time: the cart item's
smoothed conversion rate (transactions / add-to-carts) and the visitor's
transaction count, each computed three ways. XGBoost ROC-AUC, 5 seeds:

| Aggregate window | aggregates only (vis / temp) | prefix + aggregates (vis / temp) |
|---|---|---|
| none | — | 0.645 / 0.664 |
| as-of: strictly before the cart time (correct) | 0.611 / 0.633 | **0.683 / 0.711** |
| whole dataset, scored session excluded | 0.630 / 0.658 | 0.679 / 0.726 |
| whole dataset, scored session **included** | **0.971 / 0.977** | **0.973 / 0.977** |

- **Self-inclusion is catastrophic** (0.66 → 0.98): the aggregate contains the
  scored session's own purchase — the same mechanism as the IJACSA paper's
  arithmetic proxy, now arising from a routine "lifetime purchases" or
  "item conversion rate" table.
- **Future window without self** is small (+0.02–0.025 aggregates only; mixed
  and noisy with prefix features, sd up to 0.022).
- **Correct as-of aggregates add real signal** (+0.05 temporal: 0.664 → 0.711).

## Emerging picture

Point-in-time violations differ by orders of magnitude:
self-inclusion ≫ future window ≈ nearest snapshot > latest snapshot ≈ 0.
This graded result — which violations matter, and why (distance between the
outcome event and the information the feature absorbs) — is a stronger DEEM
contribution than the item-property join alone.

Caveats so far: k=0 only; the first-cart item only; XGBoost under the temporal
split is deterministic (sd 0, one effective run); two aggregates only; no
significance tests yet; ties at identical millisecond timestamps make the
as-of cumulative counts order-dependent (rare).

## Open decision

Scope of the DEEM paper: item-property joins only (as planned), or the
graded set of point-in-time violations (item-property joins + aggregate
windows, possibly snapshot-time vs event-time joins) — see the 2026-10-02
report to the author.
