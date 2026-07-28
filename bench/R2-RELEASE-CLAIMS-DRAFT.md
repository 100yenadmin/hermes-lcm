# R2 release — claims section DRAFT (owner-gated; do not publish without sign-off)

**Status:** DRAFT 2026-07-29. Task #15's requirement is baked in: R2 carries the M12-corrected claim.
**Pending inputs before publish:** (1) the full-500 wave-1 V1 number (running); (2) Phase 1A's scaling curve
(running); (3) the fresh-eyes audit (running); (4) owner's framing choice in §3 below.

---

## 1. What R2 claims — every sentence measured, nothing else

**Established (ship as the headline):**
> Hermes-LCM makes an agent on LongMemEval-V1 **22% faster end-to-end** (−56.3 s/question, p<0.0001, faster on
> 48/60 paired questions) at **equal accuracy**, replacing the agent's file-scan exploration with indexed
> retrieval.

**Explicitly corrected from R1 (state it; do not bury it):**
> R1 implied an accuracy advantage. Paired re-measurement (three arms, McNemar) found **no measurable accuracy
> effect** — the accuracy claim is withdrawn. The speed claim replaced it because that is what the data supports.

**The attribution statement (the honest differentiator — see §3 for framing choice):**
> On V1 our retrieval is **saturated**: at least one gold session retrieved on 500/500 questions, every gold
> session on 486/500 (97.2%). 52 of the model's 56 failures occurred with the complete gold evidence already in
> the prompt. The remaining V1 gap belongs overwhelmingly to the reader model's synthesis, not to memory. **No
> memory system wins those questions by retrieving or presenting better** — claims of large V1 margins imply
> reader-side spend (stronger model, more passes), which buyers should price accordingly.

**Token cost:** flat (+3.2%, CI spans zero). Say "no token overhead", never "saves tokens".
**Compression:** NOT claimed — the coding agent compresses; vanilla does it too (H2). R1's silence stays silent.

## 2. What R2 must NOT say

- Any accuracy delta on V1 or V2-small without a fresh paired McNemar attached.
- Any "scales better" language **unless Phase 1A's censored curve supports it** — pending. If A-arms hold recall
  flat at 389× where file-scan latency grows ~linearly (B: 4.4 ms → 411 ms p50), the scaling sentence writes
  itself and cites the curve. If file-scan wins both axes at 19,829, R2 says that too (protocol §2 row 4).
- Any Phase 1A recall LEVEL as a production number — fastembed 384-dim ≠ production Voyage 1024-dim; shapes
  transfer, levels do not (protocol §3c).
- "1/6 preference" or any enriched-slice rate (F27 §0). Preference is 25/30.
- LAFS: every banked configuration scores 0.0000 against the reference frontier. R2 does not mention LAFS as a
  win; if asked, that is the number and the reason the scaling regime is the roadmap.

## 3. OWNER CHOICE — the one sentence that sets R2's posture (F27 §Q1)

Both are true; they read very differently. Pick one as the lead:

- **(a) Confidence lead:** "Our memory layer is at its measurable ceiling on this benchmark — 100% gold-session
  recall — and it gets there 22% faster than file scanning." *Strong, but invites "so you can't improve?"*
- **(b) Candour lead:** "We measured exactly where a memory layer stops mattering on LongMemEval-V1 — and where
  it starts: scale. Below ~500 sessions an agent can grep; past that, indexes are the only thing that holds."
  *Positions Phase 1A/scaling as the product story; slightly ahead of the data until the curve lands.*

Recommendation: **(b) if Phase 1A's curve supports it, else (a).** Decide after the curve, not before.

## 4. Release mechanics (unchanged from the R2 checklist)

Base = the winning provider from the pending full-500 comparison, pinned by commit. Include the #164(a)
`store_id` fix if landed; the harness fail-close (#165) is documented as an instrument caveat, not patched
silently. Findings docs F20–F28 ship in `bench/` as the evidence trail, including the three same-day
self-corrections — the correction record is part of the credibility claim.
