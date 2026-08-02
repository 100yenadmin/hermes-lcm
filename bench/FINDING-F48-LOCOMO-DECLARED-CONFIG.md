# F48 — LoCoMo declared-config A/A′: 54.6% (+7.6 vs old config); noise floor banked; two category-level faults surfaced (2026-08-02)

## 1. The pair (both arms integrity-clean)
Config of record: RUN-SHEET-LOCOMO-DECLARED-CONFIG.md — product 9d181aa, harness 19d9c0b+f55eba3
(+2c36f98 run-prep), fusion quota fts:chunk=1:2, chunk threshold 10, sol answerer / sol-low judge
(narrowed rubric), fastembed bge-small, pins signed pre+post both arms, storefreeze verified.
- **Arm A: 1,085/1,986 = 54.63%** · A′: 1,070/1,986 = 53.88% (fresh stores, same config).
- **Noise floor: 69/1,986 discordant (3.47%)** — 42 A-only + 27 A′-only; aggregate spread 0.76 pts.
  (V1 reference pair: 3.6%. Agreement HEALTHY → arm A is the scored read per the run sheet.)
- Fail-closes: 0 in both arms. Drops: 0. Ops incidents during the pair (3, all recovered with data
  intact, none scoring-relevant): two search-start bridge-init starvations under shared-disk I/O and
  one lingering-process wedge — mechanisms + workarounds tracked on #191; RUN-LOG has the timeline.

## 2. Aggregate vs the old config — honest attribution
47.0% → 54.6% (+7.6). NOT a pure retrieval delta: the harness fixes (PR #3) changed the judge
rubric and adversarial gold BETWEEN runs (disclosed below); the F46 decomposition anticipated
~+3 instrument + config-class retrieval recovery. Corrupted-gold exposure UNCHANGED (~95% ceiling,
99 documented-bad rows still scored as-is).

## 3. Per-category — the fault-finding yield
| category | old config | declared | Δ | reading |
|---|---|---|---|---|
| multi-hop | 43.8 | **64.8** | +21.0 | fusion-quota + threshold recoveries (the F46 buried/excluded turns were multi-hop-heavy) |
| temporal | 29.1 | **41.7** | +12.6 | same class + date metadata via caption/threshold-recovered turns |
| world-knowledge | 52.9 | **69.7** | +16.8 | same class |
| single-hop | 49.2 | **37.2** | **−12.0** | ★ NEW FAULT: suspected quota side effect — capping FTS slots 1:2 likely demotes exact-match wins on rows where FTS-first ranking was correct. The fusion sim (25 rows, 4 controls) was NOT category-balanced — a gate-proxy-calibration echo. DECOMPOSITION NEXT (zero-spend: per-question results + delivered hits on disk). |
| adversarial | 45.5 | 32.7 | −12.8 | measured-honest drop: old run judged against a buggy gold ("undefined"→trap fields); declared config uses canonical abstention gold + a rubric that only credits verifiable premise-rejection. The B3 product weakness (speaker misattribution with the fact retrieved) remains unaddressed by design — that is the roadmapped fix. |

## 4. Seven-point disclosure (for the scoreboard row)
1 Fix-state: product 9d181aa; harness feat/locomo-hermes-prep @ f55eba3 (+run-prep 2c36f98).
2 Judge: gpt-5.6-sol @ low; full prompts pinned (defaults.ts sha 7662f6…); NARROWED abstention
  rubric (premise-rejection only) — stricter than the stock LoCoMo judge.
3 Dataset defects: 99 documented corrupted-gold rows all ran; ceiling ≈95%; unchanged vs F46 §6.
4 Retrieval: fastembed bge-small; HERMES_MB_FUSION=quota:fts=1,chunk=2; chunk threshold 10;
  answer-ready 2,400 chars; every knob's measured justification in the run sheet.
5 Per-category: §3 table; per-question rows in run artifacts.
6 Variance: A/A′ pair, 69/1,986 discordant (3.47%), aggregate spread 0.76 pts.
7 Fail-close: 0/1,986 both arms; union-drop 0.

## 5. Dispositions
- Scoreboard: new row supersedes locomo10-1986-arm-a-2026-07-30 (the 47% row STAYS visible).
- NEXT (zero-spend): single-hop regression decomposition — per-row diff of delivered hits old-vs-new
  config on the single-hop flips; validates or refutes the quota-side-effect hypothesis and, if
  confirmed, feeds a category-aware fusion refinement (own registration, not a silent retune).
- B3 (adversarial premise-check product work) unchanged as the roadmapped product fix.
- Voyage-embedding variant (owner steer) = a FUTURE declared config with its own A/A′.
