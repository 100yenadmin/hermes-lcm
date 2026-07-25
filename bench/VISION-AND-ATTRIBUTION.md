# VISION & ATTRIBUTION — what the frontier model does, what WE do, and where our value is measurable

**Date:** 2026-07-25 · **Status:** program vision (architect, under owner directive)
**Owner directive:** *"Our users care about the frontier... understand what percentage of performance comes
from the frontier model itself and what you're actually improving. That gives you a baseline to build on."*

---

## 1. The attribution, measured and honest

| capability | who actually delivers it | our marginal contribution |
|---|---|---|
| corpus → pack compression (~7.6M tok → 17k, ~444x) | **the coding agent, either one** | **none** — vanilla Codex is handed a `trajectories/` dir and greps it to the same ~20k pack |
| accuracy | agent + fixed reader | **none measurable** — McNemar null across 3 paired arms (p=0.727/1.000/0.774) |
| context tokens delivered | either | **none meaningful** — +3.2%, 95% CI spans zero (sign test p=0.027 on direction only) |
| **retrieval latency** | **our indexed store vs its file scan** | **−56.3s/question (−22%), 48/60 questions, p<0.0001** |

**One real, measured advantage: speed.** Everything else attributed to us so far was either the agent's work
or noise. This is the baseline the owner asked for, and it is deliberately unflattering.

## 2. Why the advantage is so thin HERE — and it is structural, not a failure

**The benchmark corpus is 1,870 trajectories total, and each question is handed 100 pre-selected candidates.**

That is the whole explanation. At 100 candidate files, brute-force exploration is *competitive* — `grep` over
100 files is fast, so an index has almost nothing to beat. The benchmark **pre-solves the candidate-selection
problem**, which is precisely the problem a memory system exists to solve, and then measures only what
remains: reading a small pre-filtered set.

**So LME-V2 structurally cannot showcase our architecture.** Not because the benchmark is bad at what it
measures, but because what it measures is downstream of our value. Vanilla Codex scoring 63.3% without us is
not a threat to the product — **it is evidence that this task does not need a memory system**, and we should
say so plainly rather than pretend a 2-point wobble is a moat.

## 3. What that implies — the scaling hypothesis (our actual thesis, now testable)

File scanning is **O(n)** in corpus size. Indexed retrieval is sub-linear. So:

> **The latency advantage we measure should GROW with corpus size.** 22% at 100 candidates is the floor, not
> the ceiling — it is our advantage leaking through at a scale engineered to suppress it.

That is a falsifiable, product-relevant claim, and it is the owner's lossless-raw + read-time-intelligence
thesis stated as an experiment: *keep everything, and still answer fast.* The value proposition is not
"+2 accuracy points"; it is **"your memory can grow without your queries getting slower."**

**This is what we should be building and publishing.** It also explains why the corpus-coverage ceiling (M5:
1/451 gold absent) was reassuring but not decisive — storage completeness is necessary and we have it; the
differentiator is retrieval cost at scale.

## 4. The dual-objective consequence (with M16 §2b)

| objective | consumer | can it show our value? | action |
|---|---|---|---|
| LME-V2 leaderboard | fixed Qwen3.5-9B, both tiers (`EXPECTED_READER_MODEL_SUBSTRING = "qwen3.5-9b"`) — **no frontier track exists** | Weakly — pre-filtered small corpus | compete for the LAFS latency axis; do not expect a moat |
| **Product / Sol (frontier)** | what customers actually run | **Yes — but only if the corpus is realistic** | **our own benchmark, scaled corpus, published on our repo** |
| LME-V1 | more permissive | Yes (444/500 banked) | publishable now |

Frontier results cannot be submitted to LME-V2 (weak-reader-only by construction) but **can** be published on
our own repo — which is what users read anyway.

## 5. What to build next (priority, revised by this analysis)

1. **SCALING BENCHMARK (new, top priority).** Same questions, but sweep the candidate corpus: 100 → 1,000 →
   10,000+ trajectories per query, no oracle pre-filter. Measure accuracy, latency, and tokens for
   (a) our indexed store, (b) vanilla agent + file scan, (c) naive RAG. **The hypothesis is that (b)
   degrades linearly and we do not.** This is the experiment that either proves the product or kills the
   thesis — and it is the only one that can, because LME-V2 cannot.
2. **Frontier-consumer measurement** (H6-P5, #156) — read the same packs with a frontier reader. Establishes
   the product-truth number and how much of the score is the model vs us.
3. **M7b** (running) — the leaderboard path. Keep it, but it is no longer the strategic centre.
4. **Full-scale confirmation of the latency claim** (vanilla@451) — hardens our one real result.

## 6. The discipline this vision inherits
Today's audit cycle retired four claims we believed (M12–M15) and its root cause was one underdesigned
instrument (M16). The compression number in §1 was *one step from becoming a fifth* — it was the owner who
asked "didn't vanilla score well?", which is exactly the question that caught it. **Attribution claims get
the same paired, adversarial treatment as accuracy claims: name who does the work, and measure the
counterfactual where we are absent.**
