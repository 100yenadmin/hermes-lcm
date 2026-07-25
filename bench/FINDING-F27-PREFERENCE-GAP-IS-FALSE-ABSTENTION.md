# F27 — The V1 preference gap is FALSE ABSTENTION, and the "1/6" figure was a selection artifact

**Date:** 2026-07-25 · **Instrument:** banked 444 report.json (`v1l1-full500-primary-2026-07-23T18-26-13-780Z`)
**Spend:** zero — pure re-analysis of an existing artifact.
**Artifacts:** `/Volumes/LEXAR/Codex/session-notes/2026-07-25/hermes-preference-gap/artifacts/`
(`pref_diag.py`, `slice_check.py`, `retrieval_check.py`, `abstain_mass.py`, `gold_map2.py`,
**`recall_by_date.py` ← the correct recall method**, `shift_test.py`/`recall_v3.py` ← the wrong ones, kept as
counter-examples; outputs `pref_rows.json`, `failure_split.json`, `recall_by_date.json`)

---

## 0. Correction first — I compared our worst-selected 6 questions against OMEGA's full 30

F26 §5 reported "preference application scores **1/6** in all three arms" and set it beside OMEGA's 30/30 as
"the sharpest open question". The dispatch packet for the full-500 wave-1 run repeated it. **The comparison was
invalid**, and the reason is structural rather than arithmetical.

The 100-question cross-test slice is **failure-enriched by construction**. Measured directly:

| instrument | on the 100q slice | on all 500 |
|---|---|---|
| banked 444 | **44/100 = 44.0%** | 444/500 = 88.8% |

The slice's 6 preference questions are **exactly the 5 known banked failures plus 1 pass**
(`09d032c9`, `0edc2aef`, `195a1a1b`, `35a27287`, `38146c39`, and `afdc33df` passing).

So **1/6 is what the banked run itself scores on those 6.** The cross-test arms scoring 1/6 means they
*reproduced* the banked outcome, not that they regressed. F26's "neither codebase moves it" survives as a
statement about **our 5 hardest preference questions**; it does not license a category rate. The true banked
preference number is **25/30 = 83.3%** (which F23's table had right — the error was in F26's framing and in the
sentence I put in the dispatch, not in F23).

**Rule this reinforces:** a rate computed on an enriched slice is not a rate. Enriched slices buy *paired power*
on a target class; they cannot produce a category accuracy, and they must never be set beside a competitor's
full-set number. Report enriched results as **flip counts against the same questions' own baseline**, never as
percentages.

## 1. What the preference gap actually is

Five questions separate us from OMEGA's 30/30. Their failure modes, read off the artifact:

| qid | failure mode | answer emitted |
|---|---|---|
| `0edc2aef` | abstention | "I don't know" |
| `09d032c9` | abstention | "I don't know." |
| `38146c39` | abstention | "I don't know." |
| `35a27287` | abstention | "I don't know." |
| `195a1a1b` | wrong content | recommended watching TV / an indie film to a user who excludes screens in the evening |

**4 of the 5 are false abstentions.** And in each of the four, the preference evidence is textually present in the
logged retrieved hits — `turbinado` (2 occurrences), `power bank` (3), `hot tub`/`balcony`/`rooftop` (9/7/1),
`Spanish`/`French` (6/5). For `0edc2aef` and `35a27287` the evidence sits at hit ranks **0–6**, i.e. the top of
the list. The system had the preference in front of it and declined to answer.

The fifth (`195a1a1b`) is the same defect in a different coat: a *negative* preference ("not phone, not TV,
before 9:30pm") was retrieved and then violated in the recommendation.

**Reading:** this is not a retrieval capability gap and it is not "we lack a preference mechanism". It is the
answer layer treating a preference question as a lookup question. "Suggest a hotel in Miami" contains no
retrievable hotel, so an evidence-seeking reader concludes the memory does not contain the answer and abstains —
when the task is to *condition a recommendation on retrieved preferences*. Preference questions have no literal
target in memory by design.

## 2. Whole-run failure decomposition (the honest map of the 56)

| bucket | n | share of failures |
|---|---|---|
| abstentions ("I don't know") | **14** | 25% |
| wrong / incomplete answers | **42** | 75% |

By category:

| category | abstentions | wrong answers | banked score |
|---|---|---|---|
| multi-session | 1 | **25** | 107/133 = 80.5% |
| knowledge-update | 0 | 7 | 71/78 = 91.0% |
| temporal-reasoning | 4 | 6 | 123/133 = 92.5% |
| single-session-assistant | 4 | 1 | 51/56 = 91.1% |
| single-session-preference | 4 | 1 | 25/30 = 83.3% |
| single-session-user | 1 | 2 | 67/70 = 95.7% |

Two things fall out that change where V1 effort should go:

**(a) The biggest single bucket is multi-session wrong answers — 25 questions, 5.0 points of the 11.2 available.**
Not abstention; not preference. Multi-session is also our worst category (80.5%). Its failures are almost
entirely *committed wrong answers* (25 of 26), which is the signature of cross-session synthesis failure rather
than refusal.

**(b) False abstention is worth ≤14 questions (2.8 points)**, and it is spread across four categories rather than
concentrated in preference. Preference is the *cleanest* instance (4 of 5 failures, evidence at rank 0), which
makes it the right place to develop the fix — but the prize is the 14, not the 5.

For calibration: the reader abstains **correctly** on 20 questions where gold is genuinely unanswerable. So the
capability to abstain is present and roughly half-miscalibrated (20 right, 14 wrong). This is a threshold
problem, not a missing behaviour.

## 3. The lever this leaves us — and the line it must not cross

We may not touch the reader/answer prompt (§2b standing rule). What we *do* own is **how memory is presented**:
`answerPresentationMode: evidence_cards_v1` renders our hits into the prompt. A preference hit currently arrives
indistinguishable from a candidate factual evidence hit.

**Mechanism hypothesis (untested, for the #150 lane):** label retrieved preference/constraint content *as* a
standing preference rather than as candidate evidence for the literal question, so the reader applies it instead
of searching it for an answer. This is presentation-layer, on our side of the line, and it addresses both failure
shapes at once — the 4 false abstentions and the 1 violated negative preference.

Retrieval is now excluded as the cause by measurement (§4), which makes the answer layer the only remaining
locus for these 5.

**Do not** pre-register this as a win. It is one hypothesis about 5 questions, and per §2b the gate has to be
declared before the run: paired, on the enriched preference set plus a non-preference control set to catch
collateral damage, with the floor condition that no currently-passing preference question breaks.

## 4. F25 independently CONFIRMED — and the mapping trap that nearly made me refute it

**Result: retrieval is saturated. ANY-gold = 94/94 = 100.0%, ALL-golds = 89/94 = 94.68%** on the cross-test
questions, measured against the banked 444's own hits. F25 reported all-gold 98/100 with mean 0.9967 on the same
slice; the two agree within the 94-vs-100 question difference. **F25 stands, by an independent method.**

The conditional accuracy is the load-bearing number:

| condition | n | accuracy | abstentions |
|---|---|---|---|
| **every gold session retrieved** | 89 | **41/89 = 46.1%** | 12 |
| partial gold coverage | 5 | 1/5 = 20.0% | 1 |
| no gold retrieved | 0 | — | — |

**48 of the 52 failures on this slice had the complete gold evidence in hand.** (These are enriched-slice
questions, so 46.1% is not a category rate — but the *decomposition* is valid: retrieval is not the binding
constraint on the questions we lose.) All five preference failures had their gold session retrieved (each has
exactly 1 gold; each retrieved it).

**⚠ The trap — positional session mapping is invalid.** I first measured any-gold at **66.2%** against an
accuracy of 88.8%, which is incoherent (you cannot answer more often than you retrieve the evidence). That
incoherence is what exposed the method error, and the cause is worth recording:

- Gold sessions are opaque ids (`answer_772472c8`); the harness names sessions `<qid>-session-N`. The obvious
  join — index into `haystack_session_ids` — **is wrong**, because **86 of 94 stores contain FEWER sessions than
  the question's haystack**. Sessions are dropped during ingest, so every index after the first drop is shifted.
  Raw order gave 66.2%, date-sorted 60.4%, shift −1 54.6% — all garbage, and the "best" of them looked plausible
  enough to report.
- The correct join is the **`<store>.dates.json` sidecar** (`<qid>-session-N` → ISO timestamp) against the
  dataset's `haystack_dates`, which is parallel to `haystack_session_ids`. Order-independent, and it resolves
  100% of golds (0 questions with a gold absent from the store — the dropped sessions are all non-gold).
- A red herring en route: three spot-checks showed evidence text in session N while gold mapped to N+1, looking
  like an off-by-one. It is not — those are LongMemEval's **designed topical distractor sessions** adjacent to
  the gold. Text matching cannot identify a gold session in this dataset; that is the dataset's whole point.

Also true, and still worth knowing: `searchResults` is exactly 25 hits for 500/500 questions — a fixed top-k. It
did not obstruct this measurement, but it is a snapshot of one search, so **do not** treat it as the complete
evidence an agentic provider finally saw.

**Process note:** F25's 99.67% was reported without its producer script being persisted — only
`retrieval-diff.json` survived, so the number could not be re-derived and had to be re-measured from scratch.
Every gate-relevant statistic must ship with the script that produced it, in the artifacts dir. Added to §6e.

## 5. What this does and does not change

**Confirms:** F25 — V1 retrieval is saturated (100% any-gold, 94.7% all-golds, re-measured independently).
**Changes:** the V1 answer-layer project (#150) re-aims from "preference application, 1/6" to **multi-session
synthesis (25 questions) as the primary and false abstention (14) as the secondary**, with preference as the
development vehicle for the abstention fix. #154's competitor question sharpens from "how does OMEGA apply
preferences" to "how does OMEGA get 30/30 on preference *and* what does it do on multi-session".

**Does not change:** any banked number. 444/500 stands, the per-category table stands, F23's 22-question
decomposition stands. This is a re-reading of an existing artifact, and the corrected item is my own framing of
a slice statistic — no measurement was wrong.
