# M11 — The effort/latency curve is convex, and that decides where every future point gets spent

**Date:** 2026-07-25 · **Issue:** #158 (P1 agentic latency sweep) · **Status:** measured, decision-bearing
**Depends on:** M8 (LAFS = accuracy × latency) · M10 (search-flailing unifies accuracy and latency)

---

## 1. The measurement

Agent reasoning effort swept over the frozen 60-question dev manifest (32 web / 28 enterprise),
decoding pinned (temperature 0.6 / top_p 0.95 / top_k 20) per the M9 parity rule, everything else
held identical to the P4 configuration.

| arm | effort | accuracy | latency (mean) | LAFS |
|---|---|---|---|---|
| P4 (451q, banked) | xhigh | 66.10% | 196.9s | 0.0000 |
| L1 | high | 61.67% (37/60) | 126.3s | 0.0000 |
| L2 | medium | 58.33% (35/60) | 99.0s | 0.0000 |
| L3 | low | 56.67% (34/60) | 51.6s | 0.0000 |

Per-domain:

| arm | web | enterprise |
|---|---|---|
| L1 high | 68.8% @ 110.9s | 53.6% @ 144.0s |
| L2 medium | 65.6% @ 89.6s | 50.0% @ 109.7s |
| L3 low | 65.6% @ 47.0s | 46.4% @ 56.9s |

All four arms score **0.0000**. That is the least interesting thing about this table.

## 2. The curve is convex — the cheap latency is at the bottom

| step | accuracy cost | latency bought | points per 10s |
|---|---|---|---|
| xhigh → high | −4.43 | −70.6s | 0.63 |
| high → medium | −3.34 | −27.3s | 1.22 |
| **medium → low** | **−1.66** | **−47.4s** | **0.35** |

The last leg is the cheapest by a factor of ~3.5 against the leg above it. An earlier read of the
L1→L2 segment alone characterised the trade as "roughly 1:1"; that describes the top of the curve and
**does not generalise downward**. Notably, web accuracy is *identical* at medium and low (65.6% both)
while web latency nearly halves (89.6s → 47.0s) — on web, the last effort step is free.

## 3. Why this decides the operating point

Under LAFS, latency multiplies the value of accuracy. The same accuracy number is worth ~8× more at
low effort than at medium:

| accuracy | @ 99.0s (medium) | @ 51.6s (low) |
|---|---|---|
| 58.61% | 0.0002 | 0.0014 |
| 62.00% | 0.0576 | **0.4758** |
| 66.10% | 0.1271 | **1.0495** |
| 72.10% | 0.2288 | **1.8890** |

**Decision (recorded): low effort is the program's operating point.** Not as a fallback for when we
run out of accuracy, but as the place every future accuracy point should be banked, because each one
is worth roughly eight times more there. Effort is now a settled dial, not a lever we are still
searching over.

## 4. Distance to a non-zero score

At 51.6s the scoring window opens at **58.6%** accuracy. L3 sits at 56.67% — a gap of **1.94 points,
about 1.2 questions out of 60.** Value immediately above the threshold rises steeply: 60% → 0.196,
62% → 0.476, 66.1% → 1.049.

M7 (negative-evidence disclosure, #157) projects +6 points from the abstention mass. Landed at the
low-effort operating point that is ≈62.7% @ 51.6s ≈ **0.55** — and per M10 the same mechanism removes
searches rather than adding them, so latency should fall too, compounding the gain. The identical
mechanism landed at medium effort would be worth ≈0.07.

## 5. Instrument note (M9 discipline)

L3/web took exactly one reader-side HTTP 504; the harness records an empty response and scores it
wrong (`question_id edb69441`). This is a provider artifact, not a capability loss. L1 and L2 took
zero such errors.

- as-measured: **56.67%** (34/60)
- excluding the artifact: **57.63%** (34/59)

Both are below the 58.6% floor, so the artifact changes no decision here — recorded because at a
1.94-point margin a single question is material, and because L2 previously missed its floor by 0.3
points. **Rule reaffirmed: count provider-error rows before comparing any two arms.**

## 6. What this does and does not establish

**Establishes:** the shape of the effort/latency trade across the full dial; that low effort costs
almost nothing on web; that the operating point is low; that accuracy work is ~8× more valuable there;
that the remaining distance to a non-zero score is ~1.2 questions on the dev slice.

**Does not establish:** the honest uncontended latency (L1–L3 were measured with concurrent workers —
the L4 concurrency-1 probe is outstanding and can only move these numbers *down*, i.e. further in our
favour); that M7's projected +6 will materialise; that 60q dev-slice accuracy transfers to the full
451 (it has run ±3 points historically).

**Next:** M7 pilot (#157), executed at **low** effort, with `searches_per_question` instrumented as
the leading indicator per M10.
