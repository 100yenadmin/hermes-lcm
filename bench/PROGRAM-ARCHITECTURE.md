# HERMES-LCM BENCHMARK PROGRAM — MASTER ARCHITECTURE v1 (frozen 2026-07-24 ~06:10 +07)

_Authored by the program architect (Fable) under owner grant of full authority (07-24 ~05:55: "You have full
capability and authority. You're the architect/designer/and release product manager"). This document + the
ROADMAP issues + OPUS-DRIVE-LOOP.md are the complete operating state: a continuation agent (Opus 4.8) must be
able to execute from these without access to this session's context. Strategy changes require either the owner
or a documented DECISION-RECORD addendum here — never silent drift._

## 0. Mission and the number ladder

Make agent memory better for all agents, proven publicly on LongMemEval-V2 (primary) and memorybench-V1
(secondary). The public ladder (leaderboard page, verified 07-24; leaderboard still EMPTY — first-mover slot open):

| Tier | Published points (small track) | Our position |
|---|---|---|
| Reader-only floor | 1.3% | — |
| Static (fixed reader) | RAG 42.8 @ ~0.2s · RAG+notes 51.0 @ 0.2s · AgentRunbook-R 58.6 @ 26.9s | **ours: 27.7 (125/451)** |
| Agentic (own agent) | Codex 69.9 @ 177.2s · AgentRunbook-C 74.9 @ 108.3s | not yet entered |

Leaderboard metric = **LAFS Gain: Pareto frontier over (accuracy, query-latency)**. Strategic consequence:
there are TWO winnable slots — a fast-compact static point AND a high-accuracy agentic point. We pursue both.

Targets (in order of expected attainment): static ≥42.8 → static ≥51.0 (beats every published static system)
→ agentic ≥69.9 (beats vanilla Codex with the SAME agent + our memory) → agentic ≥74.9. V1: 444 → ≥450.

## 1. The two load-bearing measurements (why this architecture)

**M1 — The official static run (125/451, integrity-gated, OFFICIAL-RESULTS.md):** the dominant loss class is
spurious-unknown — the official Qwen3.5-9B reader answers "unknown" on 95/323 (29%) of answerable questions
while holding full ~22k-token untruncated contexts (medians identical for answered vs unknown; zero truncation).
**The static score is bounded by context COMPACTNESS at the consumer, not recall volume.** Enterprise is worst
(64/95 spurious-unknowns; 20.9% vs web 33.8%).

**M2 — H5(b) adjacency no-go (H5B-SWEEP-REPORT.md):** pool-entry ceiling 4/30 vs gate ≥8/30, proven structural —
25/30 recall-miss targets sit on zero-pooled-state trajectories that same-source adjacency can never seed.
**The recall floor is reachability: states need direct semantic addressability (embedding backfill), not
neighborhood expansion.**

**M3 — H6-P0 protocol pin (07-24, file:line-verified):** the agentic tier ALSO answers through the same fixed
Qwen3.5-9B reader — the coding agent only CURATES evidence (memory_markdown + ≤20 trajectory spans) which the
reader consumes; scoring path identical to static. **Codex-69.9 vs our 27.7 is therefore entirely context-curation
quality into the same weak consumer.** The static/agentic ladder split is curation-by-pipeline vs curation-by-agent.

M1+M2+M3 converge: **reach the right states, then deliver them small — curation quality IS the benchmark.**
That is wave-3 (pipeline curation) and H6 (agent curation over our store); they share the same thesis.

**M4 — W3c enterprise classification (07-24, 20/20 sampled, #144):** REFINES M1 for the enterprise subset —
the needle was ABSENT from the delivered context in 20/20 sampled enterprise spurious-unknowns (the reader
abstained CORRECTLY). Mechanism: 3-4 generic hub trajectories (incident-list/search/creation flows) crowd out
the topic-specific trajectory in retrieval regardless of question topic (hub 96131e7b in 19/20 contexts).
Same magnet-family pathology as H3.1. CONSEQUENCE (recorded before any W3b build): W3b must lead with a
retrieval-diversity mechanism (hub-trajectory cap / per-source diversity quota) — repacking cannot surface
what retrieval never returned — and W3a state-level embeddings attack the same root (topic-specific states
become directly addressable instead of riding hub-trajectory coarse vectors). Compactness remains the live
thesis for the web subset (sample pending) and for the static→agentic gap (M3).

## 2. DECISION RECORD — the official protocol becomes the primary static instrument (NEW, 07-24)

The full 451-question official run cost **≈$1.1 and ~3.5h wall** (parallelizable to <1h in 4 batches). The Sol
internal protocol costs more and measures a frontier reader that masks the exact failure mode (bulk-context
tolerance) the official reader punishes. Therefore, effective immediately:
- **Static candidates iterate against the OFFICIAL protocol directly** (60q stratified slices for dev loops at
  ~$0.15, full 451 for gates), using the frozen batch machinery at
  `/Volumes/LEXAR/Codex/session-notes/2026-07-23/hermes-benchprog-h4/artifacts/phase3-openrouter/` (harness
  worktree: `/Volumes/LEXAR/hermes-work/wt-bench-h1-v2adapter`).
- Sol protocol is demoted to a dev-loop probe and for V1 (whose harness is Sol-native). Sol numbers remain
  non-official-labeled always.
- The 205/451 Sol baseline remains the tagged historical reference; the official 125/451 is the number of record.
Predeclared instrument rules carry over: snapshot-first, per-category integrity check on every full run,
both-sides rescoring if any scorer changes.

## 3. Lane architecture

### Lane S — static compactness (wave-3; epic issue W3)
Goal: official static 125 → ≥193 (42.8-parity) → ≥230 (51.0-parity). Mechanisms, in dependency order:
- **W3a — H5(a) state-level embedding backfill** (recall floor). Backfill Voyage embeddings for pool states so
  zero-pooled-state trajectories are directly reachable. Gate summary (the ISSUE #142 body is the sole complete
  source of truth): pool-entry ≥8/30 on the frozen H5 target set
  (`/Volumes/LEXAR/Codex/session-notes/2026-07-23/hermes-benchprog-h1/artifacts/h5-targets.json`); plus the
  delivered-recall, preservation/golden, and latency conditions per #142. Spend: size first (W3a-0, expect low
  single-digit $; hard cap $10 without owner ping; re-meter at 50% of the backfill before continuing).
- **W3b — compact delivery** (the spurious-unknown killer). Replace 22k bulk contexts with budgeted sharp
  contexts (target ≤4k tokens) via the selective/evidence-contract path (the R1 subsystem exists for exactly
  this). Development gate: 60q official slice, primary metric = spurious-unknown rate on answerable questions
  (currently 29%) + accuracy; promotion gate: paired full-451 official vs the 125 baseline, net ≥+15,
  category-integrity pass. NOTE: compact delivery also collapses query latency → LAFS position improves on both
  axes.
- **W3c — enterprise diagnosis**: read 20 enterprise spurious-unknown cases (artifacts have full contexts),
  classify (format? entity density? question style?), feed findings into W3b templates.
Sequencing: W3a and W3c parallel; W3b consumes both; one promotion gate at the end (one-primary law).

### Lane A — agentic (H6; spec SPEC-H6-AGENTIC-LANE.md; epic issue H6)
The controlled experiment: SAME coding agent, vanilla file memory vs hermes-lcm store. Phases P0 (protocol
read — dispatched 07-24 ~06:05) → P1 (vanilla-Codex 60q repro; validates env/cost/latency; measures per-q cost
for the full-run spend ask) → P2 (`hermes_lcm_agentic` memory module: insert = existing ingest; query = agent
workspace with store-backed search CLI) → P3 (paired 60q, gate: hermes ≥ vanilla +5pts) → P4 (full 451 +
integrity + LAFS). Model/effort/binary pinned per run manifest; published-point parity checked in P0/P1.

### Lane P — product-truth compaction-replay harness (H7; NEW — owner's concept, 07-24 ~05:55)
The owner's design intuition, architected: benchmarks above measure ingest-then-query; the PRODUCT reality is
a live agent whose context fills and must compact without losing operational memory. H7 harness:
1. Replay a long conversation/work stream into a live agent session (LongMemEval-V2 histories are the replay
   corpus; their questions are the probes).
2. At a context threshold (parameter; owner suggested ~230k tokens), trigger hermes-lcm compaction (ingest the
   evicted span into the store; keep the compact residue in-context).
3. After each compaction cycle, run the probe subset answerable from evicted material; compare three arms:
   (a) naive truncation, (b) summary-only compaction, (c) hermes-lcm compaction+store (agent may query the store).
4. Metrics: retention curve vs compaction count; probe accuracy per arm; residue token cost.
This measures "compactions provide memory relief AND keep data findable" — the product claim, directly. H7 is
sequenced AFTER H6-P3 (it reuses H6's agent-with-store machinery; building it first would duplicate work).
Design doc to be expanded in the H7 epic issue before any build.

### Lane V — memorybench V1 (cycle-2; spec SPEC-W2A-CYCLE2.md; epic issue W2A-C2)
Fixes 5 diagnosed classes; gate chain predeclared in the spec (paired ≥+8 AND ≥+3 vs cycle-1; redesigned
achievability-verified blind, losses ≤2 & net ≥0; ONE full ≥450, MISS 445-449 ⇒ bank+close). Priority: below
lanes S and A (V1 is secondary; 444 is already banked and respectable).

### Lane C — community/release (release-PM authority granted)
- R1 release (program-r1): PUBLISH (fork-local; owner asked for tagging/release push on 07-23).
- Wave-1 upstream PR: POST from `upstream-wave-1@0cb7f37` with the completed body (official number filled,
  CI story resolved). Shepherd per PR-shepherding standard; expect slow maintainer, Tosko4 reviews well.
- Leaderboard submission: **HOLD** (release-PM decision, standing): 27.7 as the first public number frames the
  project badly; submit when Lane S crosses ≥42.8 (RAG-parity) or Lane A produces a ≥69.9 entry. Revisit rule:
  any lane crossing its threshold triggers the submission packet (validator + integrity gate + GEX divergence
  rule; GEX cross-check continues meanwhile).
- #423 / #434 / harness PR #2: continue shepherding (respond-to-review autonomy per established pattern).

## 4. Standing discipline (unchanged, binding on all agents)
Snapshot-first before ANY subsequent run · manifests frozen+sha256 BEFORE launch, with achievability verified
at freeze (blind-R2 lesson: compute the control slice at freeze time) · predeclared gates, never relaxed
at one-short, revisions only as documented addenda BEFORE numbers · one-primary law per candidate · no re-ingest
of clonable state · file-keyed foreground polling, never process-exit or phantom background watches · durable
scheduling for anything outliving a session, and execute-inline when a cron misses · assert outward-action
success (ls-remote / re-read after push) · artifacts to LEXAR session-notes, never /tmp · Sol numbers labeled
non-official, always · routing ledger line per dispatch · pre-digest >200-line outputs.

## 5. Risk register (top 5, with mitigations)
1. **W3b compact contexts lose recall** (sharp but wrong) → dev gate tracks accuracy AND spurious-unknown
   jointly; paired full-451 promotion gate catches net harm.
2. **H6 P1 baseline repro misses published 69.9 badly** → P0 pins model/effort; if repro diverges >8pts,
   STOP the lane and file a protocol-parity issue rather than tuning toward a number (instrument-first rule).
3. **Agentic full-run cost blows up** → P1 measures per-question cost; P4 is explicitly owner-gated on that number.
4. **Continuation-agent drift** (Opus re-litigating settled decisions) → this doc's decision records are binding;
   OPUS-DRIVE-LOOP.md defines what Opus may decide alone vs park.
5. **Upstream PR stalls** (slow maintainer) → wave-1 is additive-only + fork-local value is already banked;
   stall costs nothing on the critical path; keep fork releases flowing.

## 6a. DECISION RECORD — benchmark portfolio & the frontier-consumer position (owner dialogue, 07-24 ~05:3x)

Owner input recorded: (i) authority now explicitly extends to release publishing, upstream PR waves, and
leaderboard submissions — all release-PM calls are the orchestrator's; (ii) the critical rule is
CHECK-THE-ANSWERS forensics — ">90% of the time it's bugs in the harness, not what the agent built"
(this program's own history: SpendGuard, comparator LaTeX, fixture date-bomb, hub-crowding all found by
reading answers); (iii) owner is nervous about LongMemEval-V2 (brand new, nobody has posted, hard to build
for) and asks whether AgentArena / MemoryArena / BEAM / AMA-Bench / PersonaMem / LongMemEval-v1 fit the
mission (agents as employees / chief-of-staff) better; (iv) the future-is-frontier thesis: tiny local readers
are not the agentic future (GLM-5.2-local ≈ the floor going forward; multimodal + hybrid memory beyond that),
while acknowledging the purist counter-theory (a memory system so good no-API-call retrieval alone scores).

**Decisions (standing until revised by a documented addendum):**
1. **LongMemEval-V2 stays the PRIMARY public target.** The empty leaderboard is a first-mover asset, our
   instrumentation there is now cheap (~$1/full run) and battle-tested, and the harness-risk the owner fears
   is exactly what our check-the-answers discipline converts into wins (M4). The three tiers map cleanly onto
   the three theories: static lane = the purist no-API-calls theory test · agentic lane (H6) = the
   frontier-consumer future · H7 = the product truth. The portfolio position is a HEDGE built into one bench.
2. **Portfolio spike (new issue):** agent-run survey of AgentArena, MemoryArena, BEAM, AMA-Bench, PersonaMem,
   LongMemEval-v1, and neighbors; scored on mission-fit (employee/chief-of-staff duties, agentic action
   continuity), maturity (harness quality, baselines, active leaderboard), effort-to-run, full-system
   comparability (does it exercise store+embeddings+tools end-to-end), and visibility. Output = a decision
   record proposing primary + up-to-2 secondary benches; orchestrator decides. X1 BEAM / X2 MemoryArena
   (#138) fold into this spike.
3. **LongMemEval-v1: do NOT invest in crushing it** (saturated, >95% public repos = low marginal signal).
   Instead MINE the winners: competitor-technique survey (new issue) extracting their retrieval/curation
   mechanics for W3b/H6-P2. Revisit only if the portfolio spike ranks a v1 number as cheap credibility.
4. **Frontier-local reader arm (deferred, noted):** GLM-5.2-local on GEX44 as an H6-class consumer arm once
   P1-P3 land — tests the owner's "minimum future standard" directly. Not scheduled yet.
5. **Submission authority:** now fully delegated; the #151 trigger rule stands as the orchestrator's own
   standing decision (submit on static ≥193 or agentic ≥69.9 — the first public number frames the project).

## 6. Pointers
GATE AUTHORITY NOTE: every gate summary in this document is an abbreviation — the GitHub ISSUE BODY carries the
complete, binding gate text; score against the issue verbatim.
Specs: SPEC-H6-AGENTIC-LANE.md · SPEC-W2A-CYCLE2.md · SPEC-H5b (superseded, archived) · H7 epic (#149).
Evidence: hermes-benchprog-h4/artifacts/OFFICIAL-RESULTS.md (+OFFICIAL-FULL-RAW) · H5B-SWEEP-REPORT.md ·
W2A-* artifacts · check-in #2 (#107 comment). Ops: bench/RUNBOOK.md (fork) · ~/.claude/runbooks/
hermes-benchmark-ops.md · OPUS-DRIVE-LOOP.md (continuation operating system). Tracker: #107 map + the
wave-3/H6/H7 milestones+issues posted 07-24.
