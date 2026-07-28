## Benchmarks (R2) — measured, not marketed

Every number below comes from a paired, pinned, reproducible run; methods, finding docs (F20–F34), and full
provenance ship in [`bench/`](bench/). The short version of our philosophy: we publish the metric definitions,
our correction history, and the bugs we found in our own product — so you can check us.

| claim | number | evidence |
|---|---|---|
| End-to-end speedup vs agent file-scan (LongMemEval-V1) | **−56.3 s/question (22%), p<0.0001** | M12 |
| Retrieval recall at ~200k messages (389× store) | **beats file-scan at every rung; cliff eliminated** | F34 |
| Retrieval latency (full-coverage scan) | **20–45 ms ≤2k sessions · 5.6 s at 20k — ANN next train** | F34 |
| LongMemEval-V1 | **⟨TBD⟩/500** | F32 + slice |
| LongMemEval-V2 (agentic, fixed 9B reader) | **298/451** | H6-P4 |
| Evidence-delivery → accuracy causality | **23/35 flips, p=1.9×10⁻⁵** | F33 |

**Known limitations we publish on purpose:** V1-small accuracy ties our previous base (retrieval is
byte-identical on 96% of questions — that benchmark regime is delivery-bound, not retrieval-bound); scaling
recall is shape-validated on a non-production embedder; two harness defects we found are reported upstream
(LongMemEval-V2 #6, #7) because they affect every vendor's numbers, not just ours.
