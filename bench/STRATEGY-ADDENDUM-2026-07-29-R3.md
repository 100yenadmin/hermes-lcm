# Strategy addendum — R3 (2026-07-29, post-R2-ship; amends STRATEGY-2026-07-25.md)

*R2 shipped today (455/500 with its p-value, the latency claim, the published scaling curve — upstream
#436, all ten gates). This addendum records the R3 thesis, the F38 re-aim, and the owner-ratified
benchmark portfolio. The plan of record with full sequencing lives in the program state file; binding
gates stay in bench/specs/.*

## R3 thesis
**The metric-standard release: prove the thesis at scale, and define how the market must measure.**
The leaderboard is contested and non-comparable (different judges/readers, no reproducibility discipline,
no scaling curves anywhere); our moats are the three things R2 built — reproducibility discipline, the
only published latency+recall scaling curve, and answer-turn evidence-completeness measurement. R3's
headline is the metric-standard play backed by the extended scale curve (ANN → ms-class at 200k messages
with recall-parity evidence). Chasing vanity parity on a non-comparable leaderboard is explicitly not
the play.

## F38 re-aim (supersedes the "budget-fill on V1-small first" sequencing)
V1-small is saturated as a delivery-mechanism proving ground: realistic remaining headroom ≈1.6 pts
(hard cap 3.6), and the pre-registered Stage-2 gate was infeasible-as-registered (≈3% pass; adjudicated
on zero-spend evidence, experiment not run). Session expansion merged as dormant capability (#173,
flag-OFF byte-identical); its next gate is retrieval-completeness on the 389× instrument, pre-registered
before any run. F33's attribution narrowed: interventionally supported, salience confound documented —
public claims use the observational gradient.

## Benchmark portfolio (owner-ratified 2026-07-29 evening)
- **Primary tier move: LongMemEval V1-MEDIUM (`longmemeval_m`)** — 10× haystack (grep stops being free),
  frontier readers allowed, live leaderboard: the first official surface for the scale thesis. V2-medium
  only as a shared-store-cost secondary (weak fixed reader, empty leaderboard = visibility, not proof).
- **LoCoMo sidecar (R3.1, cred play):** competitors publish 83–94% there; adapter built under bench/tools
  discipline — pins, fail-close accounting, and a same-code noise floor BEFORE any published number.
  *(Amended 2026-07-30, owner decision: the CC BY-NC concern applies only to REPUBLISHING the dataset
  itself, not to publishing results measured on it — LoCoMo is the field's de-facto minimum and every
  memory vendor publishes on it. The publishing hold is LIFTED; the A/A′ noise floor executes first,
  unchanged.)*
- **Agentic eval: scoped, not committed.** Selection survey first (BEAM is vendor-harness/vendor-judge —
  non-comparable); adopt only an eval instrumentable to our standards. Decision at R3.1.
- **Our own scaling benchmark** (unchanged from Phase 3): publish with negative results included.
- **Sol production telemetry** (new lane): the dual-consumer principle's second consumer — benchmark
  scores argue; instrumented production evals prove. Measurement energy moves here as V1-small saturates.
- Standing rule: **one new instrument at a time, each trusted before the next** — a new benchmark is a
  new machine to debug, and we debug it before we believe it.

## Babysit lane (upstream #436)
Live with a declared stopping rule: mechanical batch 1 pushed (a53276c); three headline architecture
decisions in DECISIONS-R3-UPSTREAM-ARCH.md; six default-off/edge decisions tracked in fork #180; only
delivery-path/P1 findings get in-PR fixes from here.
