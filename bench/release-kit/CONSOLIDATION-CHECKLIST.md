# R2 consolidation checklist — the 95% bar, operationalized (owner granted publish authority 2026-07-29)

**Authority:** owner pre-authorized upstream push + maintainer tagging + PR closeouts once every gate below is
green and the architect's confidence is ≥95%. Fork-first review cycles, then upstream via PR #436.

## The consolidated base (supersedes F32 §2's "ship e99f342 exactly" — intent preserved, see F32 note)
`bench/w3b-on-wave1` @ e99f342 **+ PR #169** (the two Phase 1A ceilings) **+ #434 cherry-pick** (query-path
spend-guard configurability — verified ABSENT from wave-1) **+ #164(a)** (store_id on summary hits — prevents
the measured 1.6% fail-close loss). Released artifact == validated artifact: validation is Phase 1B + the
sanity slice below, run ON the consolidated base.

## Gates (ALL must be green before upstream push)
1. ☐ PR #169 cross-model review verdict processed (sol·max, running); mandatory fixes applied + re-reviewed.
2. ☐ #434 cherry-pick + #164(a) fix landed on the consolidation branch; targeted tests green.
3. ☐ **Phase 1B**: F31 instrument re-run on the consolidated base — A3 recall does NOT collapse across the
   ladder (shape, not level, per #167 acceptance); A2 returns non-empty at 8k+ with p50 within 2× of A3
   (#168 acceptance); latency curve still sub-linear and ≤ file-scan at top rung. → F34 verdict doc.
4. ☐ **V1 sanity slice**: paired ~100q (the F26 slice, known baseline 44/100) on the consolidated base under
   F32 pins — flips within the measured noise reference (≤18 discordant, |net| ≤ 6). Full 500 re-run ONLY if
   the slice moves beyond that.
5. ☐ **Fork mono-PR review cycles**: consolidated branch → fork PR → Codex + CodeRabbit + evaOS review bot;
   iterate until a CLEAN ROUND (no new HIGH+ findings), minimum 2 rounds, cap 5 (owner suggested 4–5;
   diminishing-returns stop allowed after 2 clean).
6. ☐ Release docs final: RELEASE-NOTES-R2 (candour lead, locked), README section, #436 body updated,
   evidence links resolve on the fork.
7. ☐ #436 branch (`upstream-wave-1`) updated to the consolidated content (merge, no force-push if possible);
   CI green including 3.13.
8. ☐ Closeouts: #434 closed w/ prepared comment (only AFTER its fix is verifiably in #436); #423 already
   closed (verified 2026-07-29); fork-side #169 + consolidation PR merged.
9. ☐ Architect's 95% statement written into the release notes PR comment: what was validated, what wasn't,
   known issues (#165 harness-side; anything open).
10. ☐ Upstream: push #436 update, comment tagging the maintainer with the review guide, monitor per
    pr-shepherding standard (never push-and-abandon).

## Explicitly NOT gated on
Stage-2 session expansion (parallel capability lane; ships in a later train regardless of outcome);
upstream #436 merge itself (maintainer's timeline); the LongMemEval-V2 upstream reports (#6/#7 — informational).
