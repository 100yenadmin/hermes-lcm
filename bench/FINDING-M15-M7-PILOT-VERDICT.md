# M15 — M7 pilot: NO-GO on the frozen gate, with the failure mechanism precisely located

**Date:** 2026-07-25 · **Issue:** #157 · **Gate:** SPEC-M7-NEGATIVE-EVIDENCE.md §5 (frozen pre-run)
**Control:** L3 (paired, same 60q manifest, M9 parity verified: effort=low both sides, `require_evidence_gate`
the only intended diff, questions + haystack byte-identical)

---

## 1. VERDICT: NO-GO

| axis | L3 | M7 | delta | bar | result |
|---|---|---|---|---|---|
| **PRIMARY** abstention subset | 5/17 = 29.4% | **9/17 = 52.9%** | **+23.5 pts (+4q)** | strictly up | **met** |
| **FLOOR** answerable subset | 29/43 = 67.4% | 24/43 = 55.8% | −11.6 pts (−5q) | not down >2.0 pts | **BREACHED** |
| FLOOR, artifact-adjusted | 29/43 = 67.4% | 26/43 = 60.5% | **−7.0 pts** | not down >2.0 pts | **BREACHED by 5.0** |
| SECONDARY overall | 56.67% | 55.00% | −1.67 pts | reported only | net negative |
| latency (official) | 51.6s | 54.0s | +2.4s | flat-or-down | slightly worse |
| LAFS | 0.0000 | 0.0000 | — | — | — |

Neither subset move is individually significant (abstention p=0.219, answerable p=0.180) — but **the floor
was written as an absolute-magnitude bar, not a significance bar**, and −7.0 breaches it. Per the frozen
gate and the never-relax rule: **NO-GO.** This is the exact outcome spec §5 predeclared as
"primary up, floor breached."

**Mechanically the change worked perfectly:** gate-field emission **60/60 (100%)**.

## 2. Instrument note — provider errors changed the verdict's MAGNITUDE

M7 took **2** reader-side provider errors (`31f146ba`, `4329b535`) vs L3's 1. **Both were answerable
questions that L3 answered correctly**, so both landed inside the floor breach as false losses. Counting
them (M11 §5) moved the breach from −11.6 to **−7.0 points**. The verdict is unchanged; the reported
magnitude was wrong by 4.6 points until adjusted. **The rule earned its place twice in one day.**

## 3. The failure mechanism is NOT the one spec §4 predicted

§4 predicted we would import the static lane's spurious-unknown disease — the reader abstaining when it
should answer. **That is not what happened.** Of the 7 raw answerable losses:

- **all 7 carried status `directly_supported`** — not an absence status;
- **the reader went UNKNOWN on 0 of 7** (and on 0 of 7 in L3 either).

The reader remained willing to answer and simply answered *worse*. So the harm is **degradation of pack
quality on questions the mechanism was not even targeting** — not over-abstention.

**The targeting itself is good.** Cross-tab of `evidence_status` × gold class × correctness:

| status | on abstention | on answerable |
|---|---|---|
| `near_match_only` | **6/8 = 75%** | 1/4 = 25% |
| `contradicts_premise` | **2/2 = 100%** | 1/2 = 50% |
| `directly_supported` | 1/7 = 14% | 22/37 = 59.5% |

Absence statuses score **8/10 on the abstention class**, and **4 of the 5 abstention gains came via
`near_match_only`**. The channel works where it fires. Residual M7 failure is concentrated in
`directly_supported`-on-abstention (1/7 = 14%) — the agent still fails to notice a false premise 7 times.

**Why pack quality degraded:** two plausible contributors, both consistent with the data — (a) the
`## Evidence Assessment` section is prepended FIRST, occupying the position a weak 9B reader weights most
heavily (M1/M4), displacing real evidence; (b) the contract's mandatory targeted absence search consumes
agent effort and the ≤20-state span budget. Latency rising +2.4s supports (b).

## 4. The motivated next variant — and an honest bound on it

**Variant M7b: render `## Evidence Assessment` ONLY on absence statuses** (`near_match_only`,
`contradicts_premise`, `insufficient`), never on `directly_supported`. That leaves the 44
`directly_supported` packs unperturbed by rendering while keeping the signal exactly where it earns 8/10.

**Counterfactual upper bound** (directly_supported → L3 outcome, absence-status → M7 outcome):

| | value |
|---|---|
| overall | **63.33%** (vs L3 56.67%, M7 55.00%) |
| abstention | 52.9% (keeps M7's full gain) |
| answerable | 67.4% (keeps L3's level) |
| **LAFS @51.6s** | **0.6623** |
| distance to window A floor | **+4.7 pts above** |

**This is an UPPER BOUND, not an estimate, and the reason is important.** The counterfactual assumes
`directly_supported` packs would be unchanged. They would not be: the contract change makes the agent run
absence searches **before** any status is known, so its curation shifts everywhere. Measured on those 37
questions: spans +0.19, context +779 tokens (+3.9%), span counts differing on **11 of 37**. The
perturbation is real but modest (26/37 identical), so the rendering is plausibly the dominant term — but
**the true M7b result lies somewhere in [56.67%, 63.33%]** and must be measured, not assumed.

If M7b is run, the gate must be frozen with the same two-sided structure, and 60q remains a **screen**.

## 5. What this establishes for the program
- **A real, working read-time absence channel exists** and lifts the abstention class (+4q, 8/10 where it
  fires) using the harness's own dormant vocabulary. That is a genuine result about read-time absence
  signalling and is publishable as such — including the negative half.
- **The binding constraint is now pack-quality collateral, not the absence signal.** That is a materially
  more tractable problem than "the reader over-asserts", which is where M7 started.
- M14 established M7 was the only live path to a non-zero score. It failed as specified. **M7b is a
  diagnosed, targeted follow-on — not another knob** — but if it also fails, the honest conclusion is that
  the program needs a genuinely new mechanism, and that should be stated rather than absorbed by variants.
