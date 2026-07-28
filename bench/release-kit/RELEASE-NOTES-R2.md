# Hermes-LCM R2 — release notes DRAFT (candour framing, locked by owner 2026-07-29)

> ⚠ DRAFT — placeholders marked ⟨TBD⟩ fill from F34 (Phase 1B) and the sanity slice before publish.

## We measured where memory systems matter — including our own limits

R2 is the release where we stopped estimating and measured. The headline results, every one from a paired,
pinned, reproducible run (methods and finding docs ship in `bench/`):

**What we can prove:**
- **22% faster end-to-end at equal accuracy** on LongMemEval-V1 (−56.3 s/question, p<0.0001, 48/60 paired
  questions faster): indexed retrieval replaces the agent's file-scan exploration.
- **Complete at scale — and we publish the cost:** on a 389×-scaled store (~200k messages), retrieval now
  out-recalls file-scan at every corpus size we measured, with the recall cliff we found in our own product
  eliminated. Below ~2k sessions it also answers in ~20–45 ms. Above that, full-coverage brute-force scanning
  costs real latency (1.8 s at 8k sessions, 5.6 s at 20k) — the ANN index that removes this cost is the next
  release, and this curve ships in the notes so you can hold us to it. (An earlier internal number — "267 ms
  at 200k messages" — described the broken build that silently scanned 13% of memory; we caught it in the
  retest and it does not ship.) (Provider stamp: fastembed/384-dim; shapes, not levels.)
- **Evidence delivery is causal:** answer accuracy tracks delivered-evidence completeness monotonically
  (92.4% complete / 74.0% partial / 52.6% none), and injecting the missing evidence flips 23 of 35 wrong
  answers (p=1.9×10⁻⁵). We publish the metric (answer-turn recall) so anyone can hold any memory vendor to it.

**What we found wrong in our own product — and fixed in this release:**
- A 25k-vector recency window silently blinded semantic recall on large stores (recall → 0.000 at 389×).
  Fixed: full batched scan. Re-measured: ⟨TBD F34 curve⟩.
- Raw natural-language queries could return nothing at scale (FTS5 rejection → LIKE scan → timeout).
  Fixed: in-product query sanitization. Re-measured: ⟨TBD F34⟩.
- A missing `store_id` on summary hits could cost whole answers under strict evidence validation (measured
  1.6% of questions on one run). Fixed.

**What we corrected in our own claims:** R1 implied an accuracy advantage; paired re-measurement found none
(three arms, McNemar null) — the claim is withdrawn and replaced by what the data supports. The full
correction trail (F20–F33, including three same-day self-corrections and an independent audit that refuted
two of our own published claims) ships with this release. That trail is the product: numbers you can check.

**Known limitations (published, not buried):** V1-small accuracy is a tie with our previous base by
construction (retrieval is byte-identical on 96% of questions); the scaling recall claim is shape-only until
measured on the production embedding stack; the benchmark harness itself scores certain instrument failures
as capability losses (reported upstream: LongMemEval-V2 #6, #7).

Scores: LongMemEval-V1 ⟨TBD sanity-slice-confirmed⟩/500 · LongMemEval-V2 agentic 298/451 (66.1%) ·
full provenance blocks in `bench/`.
