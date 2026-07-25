# M18 — The `small` tier has NO retrieval problem: 2 distinct candidate sets for 451 questions

**Date:** 2026-07-25 · **Status:** decisive — explains the entire attribution result
**Supersedes the explanation in:** VISION-AND-ATTRIBUTION §2 (right conclusion, understated cause)

---

## 1. The measurement

| tier | questions | union of candidates | candidates/q | **DISTINCT candidate sets** | selectivity |
|---|---|---|---|---|---|
| **small** | 451 | **200** | 100 | **2** | 100/100 per domain = **1.00** |
| **medium** | 451 | **1,473** | 387–500 | **433** | 500/1473 = **0.34** |

**In the `small` tier every web question receives the identical 100 trajectories, and every enterprise
question receives the identical 100.** Two candidate sets serve all 451 questions. Mean pairwise overlap in
medium is 0.48 — genuinely different pools per question.

## 2. What this settles

**There is no retrieval task in the `small` tier.** Not a small one — none. The candidate set is fixed,
identical across questions, and equals the entire domain pool. Nothing has to be *found*.

This is the complete explanation for the attribution result (vanilla Codex 63.3% vs ours, McNemar null):
**vanilla Codex is handed a folder of 100 fixed files and greps it, and on this tier that IS the whole task.**
Earlier I framed this as "the corpus is small so grep is competitive." Too weak. The correct statement is
that the tier we have spent the entire program on **does not exercise retrieval at all**, so a memory
system's core function is unmeasured by construction.

It also retires any residual worry that our architecture underperformed. It was never tested.

## 3. `medium` is different in KIND, not degree
433 distinct candidate sets, 34% selectivity, 0.48 mean overlap. **Medium is the first tier where a
per-question retrieval decision exists.** That is why the retrieval-heavy published systems degrade there
(M17: RAG −5.1, AgentRunbook-C −4.8 and 31s slower) while the agentic ones barely move — the ones that
depend on retrieval quality start actually needing it.

Combined with M17, the strategic picture is settled: **every number this program has banked comes from a tier
with no retrieval problem.** 125/451, 298/451, and the V1 444/500 are all small-variant results.

## 4. Practical blocker for the medium run (must be planned, not assumed)
The current store holds **exactly 100 trajectory sources per domain** (`lcm_trajectory_sources`: web 100,
enterprise 100 = 200 total) — precisely the small tier's union. **Medium needs 1,473 distinct trajectories,
a 7.4x ingest**, roughly 900MB of source text (corpus mean ~640KB/trajectory).

So `--tier medium` is **not a config flip**; it requires a real ingest job first. Plan it explicitly:
build a medium store, verify `lcm_trajectory_sources` reaches the required union per domain, and re-run the
corpus-coverage audit (M5) against the medium haystack before trusting any accuracy number from it.

## 5. Why this was missed for the whole program
The store had exactly the trajectories the small tier needed, every run succeeded, and every number looked
plausible. Nothing failed. **A benchmark that quietly omits the hard half of the task produces perfectly
clean, perfectly misleading results** — and the tell was available from one query
(`count DISTINCT candidate sets`) that nobody ran until the owner asked whether larger variants existed.

**Standing check, added to discipline:** before treating any benchmark as a measure of retrieval, compute the
number of distinct candidate sets and the per-question selectivity. If selectivity is ~1.0, the benchmark
measures reading, not retrieval, and must not be used to justify a retrieval claim in either direction.
