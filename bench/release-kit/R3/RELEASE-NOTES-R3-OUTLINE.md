# R3 release notes — OUTLINE (assemble after Track B lands)

Headline: **the metric-standard release** — we publish how agent memory must be measured, with
our own numbers (good and bad) as the reference implementation.

1. The disclosure standard (DISCLOSURE-STANDARD.md) — the proposed practice.
2. The scoreboard — disclosure-first, append-only, fail-closed generator; live rows incl. V1
   455/500 + latency, V2 re-baseline 123/451 (F47 gate PASS, blind-adjudicated), the F44 scale
   curve, LoCoMo 47% with full decomposition (F46) → superseded by the declared-config re-run (F48, pending).
3. The honesty narrative: F45 (a pre-registered gate KILLED our own feature), F42→F44 (six gate
   executions, five root causes, confirmation-run rule), F47 (blind adjudication + the EINTR
   append-only continuation).
4. The two-tier doctrine: Tier-F frontier fault-finding vs Tier-P published claims — why both,
   and the "stale-world measurement" argument (market moved; undated configs are unanchored).
5. LoCoMo arc as the worked example: 47% → instrument bugs + corrupted gold + config-class gap +
   one real product weakness → every fix landed & disclosed → declared-config re-run result.
6. Upstream record: #436 (R2 train, review record, maintainer engagement), harness reports #6/#7,
   memorybench instrument PRs #3/#5, adapters #4/#6.
PENDING INPUTS: F48 (Track B scored read) · V1-medium baseline (Track D) · owner branding for the
public scoreboard surface.
