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

## ★ Owner amendment 2026-07-30 — the TWO-TIER testing doctrine (frontier-first)
The owner reframed benchmark purpose: **"whether it helps us find faults in our system... whether it
improves real agent, real-time, real-experience results"** — and explicitly de-weighted the
fixed-older-reader theory in favor of frontier models ("we're building for how people actually use
these systems with real frontier models"). Binding consequences:
- **Tier F (fault-finding):** frontier readers/judges, relaxed instrument bar, closed judges
  acceptable, speed over purity. Purpose = surface defects our own harness cannot see and measure
  real-experience improvement. LoCoMo lives here (it already found four instrument bugs + the
  config-class retrieval gap + the adversarial-attribution weakness — that IS the win). LoCoMo-Plus
  REOPENED under this tier (the closed-judge disqualifier applied a Tier-P bar to a Tier-F tool;
  adapter delta near-zero). Newer agentic benchmarks (BEAM-class and successors) re-surveyed under
  Tier-F criteria.
- **Tier P (published claims):** unchanged — full pins, disclosure standard (F46 §5's seven points),
  A/A′ noise floors, pre-registered gates. The metric-standard thesis lives at this tier and is
  STRENGTHENED by the split: we publish under discipline while testing at frontier speed.
- Official LongMemEval baselines keep their banked meaning; the FRONTIER-reader V1-medium move
  (already owner-ratified) becomes the flagship Tier-P surface going forward.

### Tier-F adoption decision (2026-07-30, survey: TIER-F-AGENTIC-SURVEY.md in the 07-30 deepdive artifacts)
Ranked by fault-finding value per integration effort: **(1) LongMemEval-V2** — the UCLA sequel to our own
primary instrument, agent-TRAJECTORY-based (dynamic state, workflow knowledge, environment gotchas; 451
curated questions confirmed unanswerable without working memory), Apache-2.0, adapter isomorphic to our
Provider, and THE benchmark the owner personally flagged. **ADOPTED as the next Tier-F build** (after the
LoCoMo config-fix replay resolves). **(2) AMA-Bench** (MIT, real tool-using agent harness, GPT-5.2 at only
72% = headroom) — queued second. **(3) MemoryAgentBench** (test-time learning + selective forgetting — the
only real-time-write instrument found) — watchlist. BEAM re-scored MEDIUM under Tier-F (judge unblocked,
but dialogue-only scale-stress; no agentic axis); PersonaMem MEDIUM-LOW; GAIA LOW (no memory axis).
τ²-Bench keeps R4 Tier-P; its memory-specific Tier-F use needs episode-chaining (noted, not funded).
