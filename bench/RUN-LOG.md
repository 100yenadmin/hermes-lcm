# RUN LOG — append-only registry of every benchmark run (owner directive 2026-08-02)
Format: `id | date(UTC) | benchmark/tier | config (one line) | artifacts | status → result`
Older lineage (R1→R2) is indexed by finding docs; this log is authoritative from R3 onward.

- v2-paired-control | 2026-07-30 | LME-V2 static 451q | main@e5acbbf, official harness 6bfd58a, #166 both-arms | session-notes/2026-07-30/hermes-v2-paired/artifacts/control | DONE → 120/451 raw (F47)
- v2-paired-treatment(+resume) | 2026-07-30/31 | LME-V2 static 451q | #190@a3a31dd, same instrument; EINTR fail-close + append-only continuation (AMENDMENT-2) | …/treatment + …/treatment-resume | DONE → 123/451 raw; gate PASS (F47) → merged 9d181aa
- locomo-aa-paid-3-armA | 2026-07-30 | LoCoMo 1,986q | R2-era 543e9ea, pre-fix harness, fastembed/top-25/RRF | session-notes/2026-07-30/hermes-locomo-aa/artifacts/paid-aa-3/a | DONE → 47.0% (F46 decomposition); A′ cancelled (documented)
- locomo-aa-20260730T202131Z-a | 2026-07-31→08-02 | LoCoMo 1,986q DECLARED CONFIG | product 9d181aa, harness 19d9c0b+f55eba3+2c36f98, quota 1:2, threshold 10, sol/sol-low, fastembed | session-notes/2026-07-31/hermes-locomo-declared/artifacts/paid-aa-20260730T202131Z/a | DONE → 1085/1986 (54.6%) UNADJUDICATED pending A′ agreement; 0 fail-closes; 2 ops incidents (zombie-contention hang → resume; bun-never-exits → kill-and-continue, tracked #191)
- locomo-aa-20260730T202131Z-a-prime | 2026-08-02 | same declared config, fresh stores | same pins | …/a-prime | RUNNING (ingest)
- lme-m-smoke-1 | 2026-08-02 | LME-V1 MEDIUM 25q SMOKE (timing pilot) | PR #198 instrument (prepare+prepared-dir), VOYAGE embeddings (owner steer — frontier stack), label m provenance | session-notes/2026-08-02/hermes-lme-m-smoke/artifacts | LAUNCHING
