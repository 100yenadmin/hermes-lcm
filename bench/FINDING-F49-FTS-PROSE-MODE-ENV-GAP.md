# FINDING F49 — LoCoMo FTS inertness solved: prose-mode flag never set (env-wiring gap)

Date: 2026-08-02. Status: mechanism CONFIRMED (zero-spend; reproduced in-main). Follow-on
measurement (replay effect estimate) in flight. Supersedes the open question in F48 §3-CORRECTION
finding 1 ("why does the prose-mode arm lose every fused slot").

## 1. The finding
The declared-config LoCoMo run (F48, 54.6%) executed with **FTS effectively dark: 3/49,650
delivered hits** — because `LCM_FTS_PROSE_MODE` (the opt-in flag gating #183's prose-mode fix,
`config.py:388`, default False) **was never set in the run environment**. The prose-mode CODE was
present in the measured build (product 9d181aa); the FEATURE was off. Every FTS query ran in the
old conjunctive-AND form, which near-never matches multi-word conversational questions.

**"Code shipped ≠ feature on."** The pins captured the git sha faithfully; nothing in the
registration process compared the run env against the product's feature-flag inventory.

## 2. Evidence (verified in-main, not agent-trusted)
- `run-locomo-aa.sh:15-27` (the actual runtime env source): 8 exports, no `LCM_FTS_PROSE_MODE`.
  `data/pins-locomo.yaml` env block likewise omits it. `config.py:388` `_EnvFieldSpec("fts_prose_mode",
  "LCM_FTS_PROSE_MODE", bool)` → False when unset; `store.py:1147` gates disjunctive routing on it;
  `tools.py:2513-2544` only pass it downstream when set.
- One-question reproduction on a retained store (conv-26 q55, copied read-only): as-run conjunctive
  query → **0 rows**; the product's own classifier `should_use_fts_prose_mode(query)` → True; the
  disjunctive form it would have built → **52 rows including the literal gold-evidence turn**.
- Not a one-off query shape: 29/30 uniform-random locomo10 questions classify prose-mode-eligible.
- Alternative mechanisms REFUTED: index fully populated (419/419 rows 1:1, H1); fusion favors the
  FTS arm in every pull round — there was simply nothing to pull (H3; matches F46 §2's 0/41 raw
  top-200 measurement); delivered-hits accounting is a faithful literal count (H4).

## 3. Implications
1. **Headroom, honestly labeled:** the banked 54.6% was earned with one retrieval arm off. An
   FTS-prose-ON config is a NEW declared config — own registration, pre-declared bars, its own
   A/A′ noise floor. No silent retune, no retroactive edits to the F48 row.
2. **`HERMES_MB_FUSION=quota:fts=1,chunk=2` was tuned against a near-empty FTS arm.** The ratio
   must be re-derived once FTS returns real candidates (part of the new config's registration).
3. The F46 §2 "0/41 gold turns in raw FTS top-200" measurement is now explained, not mysterious.

## 4. Pre-registration requirements for the FTS-ON declared config (before any paid run)
1. **Replay effect estimate (zero-spend, in flight):** replay the F46 41-miss list + a 100-question
   random sample through both query forms on retained stores; bank gold-recovery delta + candidate
   volumes. Proceed to a paid run only if the replay shows material recovery.
2. **Precision guard:** confirm classifier routing leaves compact-keyword/quoted queries unchanged.
3. **Fusion-ratio re-derivation** from replay candidate volumes, fixed before registration.

## 5. New standing rule (program-wide, effective immediately)
**Feature-flag inventory diff at registration:** a declared config's pins must include the diff of
the run env against the product's full env-flag inventory (`_EnvFieldSpec` table or equivalent) —
every flag either explicitly set or explicitly listed as default-with-value. A shipped,
default-off feature is OFF, and the registration must show it was known to be off. (Family:
wire-as-you-go / gate-every-caller; the pins pinned the code, nobody pinned the switches.)

## 6. Artifacts
- Decomposition + reproduction: session-notes 2026-08-02 `fts-inertness/` (H1–H4 verdicts,
  copied sample store); replay (pending): `fts-prose-replay/`.
- Code refs: wt-locomo-product `config.py:388,688`, `store.py:1147`, `tools.py:2387,2513-2544,
  4702-4737`, `search_query.py:158-200`; bridge `hermes_lcm_bridge.py:134-198`.
