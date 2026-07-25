# F26 — wave-1 does NOT improve LongMemEval-V1. A placebo arm made the result readable.

**Date:** 2026-07-26 · **Issue:** #163 · **Verdict:** NULL — wave-1's effect is inside the noise floor.
**Design credit:** the executing agent added a **placebo arm** (#423's identical code, re-run) that I did not
specify. Without it this experiment would have produced a false positive.

---

## 1. Three arms, same 100-question slice, same Voyage stores, same `gpt-5.6-sol` answerer/judge

| arm | correct |
|---|---|
| **A** — banked #423 verdicts | 44/100 |
| **A′** — #423 code RE-RUN (placebo) | **50/100** |
| **B** — wave-1 code | **51/99** (1 answer failure, paired on 99) |

| paired comparison | fixed (b) | broken (c) | net | McNemar p |
|---|---|---|---|---|
| **B vs A** (wave-1 vs banked) | 9 | 2 | **+7** | 0.065 |
| **A′ vs A** (identical code vs banked) | 8 | 2 | **+6** | 0.109 |

**The placebo reproduced almost the entire effect.** wave-1 beats a re-run of the same code by **+1 question**.

## 2. The overlap is the clincher

| | count | ids |
|---|---|---|
| fixes shared by wave-1 AND placebo | **5** | `7405e8b1`, `80ec1f4f_abs`, `ba358f49`, `gpt4_31ff4165`, `gpt4_ab202e7f` |
| fixes unique to wave-1 | 4 | `a2f3aa27`, `c4ea545c`, `dad224aa`, `gpt4_59149c78` |
| breaks unique to wave-1 | 1 | `gpt4_fe651585` |

**5 of wave-1's 9 "fixes" are questions that flip on a plain re-run.** Effect beyond resampling:
**+4 fixed, −1 broken = net +3**, against a measured ±6 noise floor (F24). **Null.**

## 3. Per-category — the multi-session "gain" is entirely resampling

| category | n | wave-1 net | **placebo net** |
|---|---|---|---|
| multi-session | 34 | +5 | **+5** ← identical |
| knowledge-update | 13 | +3 | +1 |
| temporal-reasoning | 24 | −1 | 0 |
| single-session-user | 7 | 0 | 0 |
| single-session-assistant | 15 | 0 | 0 |
| **single-session-preference** | 6 | **0** | **0** |

The multi-session improvement — the most tempting result in the table — is **exactly matched by the placebo**.

**★ And preference application scores 1/6 in ALL THREE ARMS.** Baseline 1, re-run 1, wave-1 1. This is the
category where OMEGA is **30/30** and we are 25/30 (F23). **Neither of our codebases moves it at all.** That is a
stable, reproducible, unaddressed weakness — the strongest confirmation yet that it is a *missing mechanism*, not
variance.

## 4. This confirms F25 by an independent route
F25 measured that both arms retrieve **identical content on 97/100** questions and that V1 gold-session recall is
**99.7%** for both. A null score result is the necessary consequence: **wave-1's changes are not engaged on the
V1 path.** Two independent measurements — retrieval-identity and a placebo-controlled score — agree.

## 5. ★ CORRECTION to F24's noise figure
F24 reported "±6 questions per 100" and implied that scales. The full-run numbers in this analysis show
otherwise: the banked run scored **444/500** and the re-run scored **442/500** — a difference of only **2
questions on 500**. So:
- on a **representative** 500-question run, variance is ≈**2 questions**;
- on this **hard-enriched** 100-question slice, variance is ≈**6 questions**.

The enriched slice amplifies variance because it is loaded with boundary questions that flip easily. **Both
figures are correct for their own instrument; do not transfer either.** F24's rule (always run a control repeat)
stands and is now doubly justified.

## 6. Decisions
- **Do not switch the V1 base to wave-1.** It does not improve V1.
- **Do not close #423.** Its 6 answer-layer commits remain the only thing measured to matter on V1, and wave-1
  does not contain them.
- **wave-1 remains the better V2 base** (298/451) and the better *local checkout* for trajectory capability. The
  two lanes are genuinely complementary, exactly as the disjoint schemas (F-series/M-series) implied.
- **The V1 integration is still worth doing** — port #423's 6 commits onto wave-1 — but for *unification*, not
  because wave-1 is expected to lift V1.
- **All V1 effort goes to the answer layer**, and specifically to preference application, which is 1/6 and
  immovable across both codebases.

## 7. Process note
Fifth instance in two days of a subagent's unrequested addition changing the outcome: the placebo arm here, the
repeat baseline (F24), `retrieval-diff.json` (F25), the asymmetric-adjustment catch and the false-premise catch
(M15 §7/§8). **Standing dispatch item, now proven five times over: require a placebo/control-repeat arm and an
"is the mechanism even engaged?" check in every A/B.** My dispatch asked for a two-arm paired comparison; a
two-arm comparison here would have reported +7 at p=0.065 and been wrong.
