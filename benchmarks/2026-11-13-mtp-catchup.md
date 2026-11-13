# 2026-11-13: MTP catch-up correction

## Changes and supported scope

This update carries two already-qualified Rust changes: skip unused head Q/attention/feed-forward/vocabulary prediction work when the whole prompt fits one 64-token chunk, and add the transfer read-before-overwrite dependency for head history. Multi-chunk head catch-up retains the complete original schedule; broader abbreviated schedules failed exact head-state checks and were not shipped.

The regression checks complete active head-KV words and subsequent drafts/state against an independently serialized full catch-up oracle at prompt lengths 1, 2, 63, 64, 65 and 129. Arithmetic, kernels, weights and numerical tolerances are unchanged. The scope is Ternary-Bonsai-2-27B PQ2_0 with its compatible MTP head on the AMD FirePro W9100, not general model/device qualification.

## Whole-request measurements

Three complete baseline/candidate/baseline cycles used discarded warmups, the same model/device/settings and a policy selected from original C10 measurements **before candidate timing**. No timing cycle was discarded or retried. Each baseline case could use its fastest qualified existing mode; the plain conversation actually reused 80 prompt tokens. MTP won the frozen conversation narrowly, while plain mode won the long-prompt/short-reply case.

The primary metric is the sum of per-case warm complete-request medians, including both real conversation turns and client history construction. It decreased from **32.619503 s to 32.476448 s**, or **0.4386% less time** (speed ratio 1.004405). This is a small finite-workload observation, not a statistical-significance claim, a universal speedup or a substantial new decode-throughput improvement.

| Case | Original C10 seconds | Candidate seconds | Less time |
|---|---:|---:|---:|
| capital | 3.561955 | 3.553308 | 0.2428% |
| code | 3.596490 | 3.589526 | 0.1936% |
| chat | 4.137083 | 4.115380 | 0.5246% |
| long | 2.647843 | 2.643182 | 0.1760% |
| chat-explain | 4.535727 | 4.483693 | 1.1472% |
| chat-list | 4.424679 | 4.402540 | 0.5004% |
| two-turn conversation | 9.715725 | 9.688818 | 0.2769% |

Medians use six original-baseline observations and three candidate observations per case, not rounded table sums. The largest baseline bracket drift was 1.0178%, below the precommitted 5% limit. None of the primary medians regressed. Short-prompt prefill improved modestly; decode medians were largely unchanged.

Cold process-to-health medians were **38.952488 s original C10** (six starts) and **39.227746 s candidate** (three starts). Cold startup did not improve and is excluded from warm-request gains.

## API and cache policy

MTP is greedy-only, request-selectable and default-off. It is not automatically the fastest mode for every request. Ordinary prefix caching is one-slot exact consumed-history extension with opt-out/reset on divergence. MTP clears/bypasses that cache because head/checkpoint/pending-row state is not represented by the ordinary token-history cache. Combined MTP/prefix reuse is deliberately unsupported; removing the selector exclusion would not establish correct reuse.

## Qualification and provenance

The accepted implementation completed **101 Rust tests**, **88 complete input/output comparisons**, and **141 exact complete mixed requests over 602.97 seconds**. The full canonical target accounting includes the entire lifetime target rerun after an administrative harness deadline interrupted it; partial execution was not counted as a pass. A separate private deployment guard passed 18 tests, not included in public Cargo coverage. Full-token arrays include terminal EOS; actual GPU dispatch and allocation/lifetime behavior were checked.

All staged GPU phases were exclusively monitored: observed peak 72 C, minimum free VRAM 1,312,870,400 bytes and maximum sample gap 1.602051 s, within the unchanged below-80 C / at-least-1-GiB / 30-second limits. No new GPU fault was recorded. This is finite qualification, not an adversarial audit or service SLA.

These are **retained hardware-qualification results for the exact implementation**, not GPU benchmarks rerun for this public checkout. The two changed Rust files match the qualified source byte-for-byte; remaining exported production files retain their baseline identities. `SOURCE_MANIFEST.json` records their hashes and the qualified executable. Public source builds are not assumed binary-identical. See [TESTING.md](../TESTING.md) for fresh CPU/build checks and hardware prerequisites.

Historical native-reference responses remain independently pinned, complete-token evidence; the historical native executable identity is unknown and no fresh native replay is claimed. Private raw request/monitor logs, machine-specific deployment controls/configuration, credentials and full model/head files are not published. Historical C10-versus-Prism figures in `qualified-results.json` remain separate from this incremental update.
