# OPUS-DRIVE-LOOP — continuation-agent operating system (hermes-lcm benchmark program)

_For the Opus 4.8 (or any successor) agent taking over when Fable credits exhaust. Read order on takeover:
(1) this file, (2) PROGRAM-ARCHITECTURE.md (same dir — binding strategy + decision records), (3) the plan file
`~/.claude/plans/i-don-t-see-anything-cached-axolotl.md` STATE DELTA, (4) open issues in the wave-3/H6/H7
milestones on 100yenadmin/hermes-lcm. Do NOT re-derive strategy — it is settled; your job is execution._

## The loop (each cycle)

1. **Pick**: highest-priority OPEN issue whose dependencies are done (priority: Lane S > Lane A > Lane C
   shepherding > Lane V > Lane P). Issues carry `lane:*` and `blocked-by` markers.
2. **Execute** per the issue's spec section. Route work per the routing table below — you are the brain;
   delegate typing where the table says to.
3. **Gate**: score results against the issue's PREDECLARED gate exactly as written. Pass ⇒ next step in issue.
   Fail ⇒ execute the issue's predeclared MISS branch. NEVER relax, reinterpret, or re-run-until-pass.
   If a gate proves structurally unpassable (verify with data, like blind-R2's control-100/100), document the
   analysis in the issue, mark it `needs-owner`, and move to the next issue.
4. **Bank**: snapshot-first (cp -R raw outputs to the session-notes artifacts dir BEFORE any further run);
   ledger line to ~/.claude/routing-ledger.jsonl; comment the result on the issue; tag score-bearing milestones
   (`bench-*` tags, pattern in git history).
5. **Advance**: close the issue with a result comment, or update its state. Return to step 1 without stopping.

## What you may decide alone vs must park

**Alone:** dispatch routing; retry/rerun of INFRA failures (hung CI runner, transport errors); executing
predeclared MISS branches; fork-local merges after your own review of green CI; shepherding replies on
upstream PRs (repro-first, own real bugs plainly — see #423 history); fork releases for milestones the owner
already blessed in kind; spend within existing categories and caps (OpenRouter $15 total cap stands, ~$1.1
used; Voyage W3a cap $10; codex/Sol lane = existing account).
**Park as `needs-owner` (comment on #107 + the issue, continue other work):** gate REVISIONS after numbers
exist; NEW spend categories or cap raises (H6-P4 full agentic run is explicitly this); leaderboard submission
(standing HOLD until a threshold crossing — architecture §3 Lane C); strategy/lane changes; anything
irreversible outside our fork.

## Routing table (Fable-era table adapted for an Opus brain)

| Work | Lane |
|---|---|
| Strategy/gates/scoring/promotions | YOU (Opus), inline — never delegated |
| Complex repo-native implementation | `coder` agent (Opus) or codex dispatch per `codex-dispatch` skill (well-spec'd + non-urgent → codex luna·xhigh/sol·high) |
| Mechanical volume, launches, digests | `fast-worker` (Sonnet) |
| Cross-model review | Codex-authored code → you review; your/Claude-authored load-bearing code → codex sol·max review. Never self-review load-bearing code |
| Official-protocol runs | the frozen batch machinery in hermes-benchprog-h4/artifacts/phase3-openrouter/ (60q slices for dev, 4×60+enterprise for full); keys in keychain (OPENROUTER_API_KEY, VOYAGE_API_KEY) — never print |

## Discipline (binding; violations have all bitten this program — see memory files)

- Snapshot-first; manifests frozen + sha256 + ACHIEVABILITY CHECKED before launch (compute the control slice
  at freeze — the blind-R2 lesson).
- One-primary law: one full primary run per candidate; no reruns of completed runs; never point a relaunch at
  a dir that held a completed run.
- File-keyed FOREGROUND polling (report.json), never process-exit; dispatch prompts must mandate foreground
  polling and the artifacts dir explicitly.
- Durable crons for anything outliving the session; if a cron misses (fires only while idle), execute its
  prompt inline — both aggregation crons tonight were executed inline.
- Assert outward success: ls-remote after pushes; re-read after posts; no `2>/dev/null` on outward actions.
- Keepalive sleep-ticks during active waiting phases (session-scoped sentinels; `touch /tmp/worldos_last_tick.
  $CLAUDE_CODE_SESSION_ID; sleep 250; echo tick` as last call; self-extinguishing); dense turns — process every
  landed wake fully and do parallel plan work before ending a turn.
- Artifacts → /Volumes/LEXAR/Codex/session-notes/YYYY-MM-DD/<slug>/artifacts/, never /tmp. Scratchpad =
  implementation-notes.html in the packet dir; timestamped entries at every decision.
- GitHub API budget: 5k/hr shared — don't poll gh in tight loops (burned once tonight); use ScheduleWakeup or
  sleep-tick micro-checks.

## Current in-flight state at handoff (07-24 ~06:15)

- H6 P0 recon agent running (deep-reasoner; report → P1 issue).
- GEX44 fp8 cross-check running (independent; non-blocking; fold into submission packet when relevant).
- Fork main CI: rerun in progress after a hung 3.13 runner was cancelled (5/6 legs green pre-cancel; expect green).
- #423 awaiting maintainer; #434 awaiting review; harness PR #2 open.
- W2a parked (cycle-2 spec ready, owner gave full-go authority — treat cycle-2 as AUTHORIZED, priority per lane
  order); H5(a) = W3a AUTHORIZED under the $10 Voyage cap.
- R1 release publish + wave-1 PR post: being executed by Fable in the handoff window — VERIFY completed
  (release live, PR exists) before assuming; if not done, they are yours (authorized).

## Escalation

Real blockers (missing credential, hard external dependency, owner-taste fork) → comment `needs-owner` on the
issue + #107, continue other lanes. If ALL lanes block: write a status comment on #107 with the precise asks,
then stop cleanly. Never idle-stop with lanes open; never fabricate progress; report failures verbatim.
