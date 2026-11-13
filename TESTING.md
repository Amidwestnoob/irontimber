# Public-source verification

## 2026-11-13 MTP catch-up update

`src/vk/mod.rs` and `src/vk/catchup_tests.rs` are byte-identical to the accepted W9100 implementation. All other exported production source and shader bytes retain their baseline identities. `SOURCE_MANIFEST.json` records the current source and the stored qualified executable; a fresh host rebuild is not assumed binary-identical.

Fresh public-source verification with Rust 1.98.1: release binaries built, all test targets compiled, and **19 model-free CPU tests passed**, zero failures or ignored tests. Gitleaks and a private-path/key-header byte scan passed with no findings. Two pre-existing compiler warnings were retained without unrelated source changes. These commands were used:

```sh
cargo build --locked --release --bins -j 2
cargo test --locked --release --no-run -j 2
cargo test --locked --release -j 2 --test gguf_malformed --test api_review_r011 --test combined_review --test prefill_lut
```

The new `state_only_catchup_preserves_all_head_kv_and_drafts` test is compiled by `--no-run`. Execution requires the actual W9100, both model files and exclusive supervised GPU access. It checks prompt lengths 1, 2, 63, 64, 65 and 129, comparing complete active head state and subsequent drafts against an independently serialized full schedule. Its passing hardware run is retained qualification, not a new model-free CPU result.

The corresponding qualified implementation passed 101 Rust tests and 88 complete input/output comparisons, plus 141 exact mixed requests over 602.97 seconds. A separate private deployment guard passed 18 tests; that guard/configuration/logging is not shipped here and those 18 are **not** public Cargo tests. See the [scoped qualification summary](benchmarks/2026-11-13-mtp-catchup.md). No new GPU qualification is implied by public build/CPU checks.

## Original C10 public export

The following checks describe the original export. Its production source and shader hashes match the [baseline source manifest](https://github.com/Amidwestnoob/irontimber/blob/5b3f0974c05b594fd294071ebf8dcb63639881b4/SOURCE_MANIFEST.json); no inference implementation was renamed or edited for branding.

Fresh public export checks:

- `cargo build --locked --release --bins -j 4`: passed with Rust 1.98.1. Two pre-existing compiler warnings were retained rather than modifying the qualified source.
- `cargo test --locked --release --test gguf_malformed --test api_review_r011 --test combined_review --test prefill_lut`: **19 passed**, zero failures or ignored tests. These are model-free CPU checks.
- `tests/review_regressions.rs`: **8 passed**, zero failures or ignored tests, using the real model tokenizer on the W9100 host. Its scripted engine is a mock; these tests do not prove GPU numerical quality. The first invocation without `BONSAI_MODEL` failed its prerequisite and was rerun with the actual model, without source changes.
- Gitleaks 8.30.1 directory scan: no secret findings. A separate byte scan found no private machine paths, private LAN addresses or private-key headers in the public export.
- Existing source whitespace warnings were preserved to keep the qualified snapshot byte-identical.

The first offline build attempt lacked a cached Cargo dependency; the normal locked build obtained the pinned dependencies and passed. No model weights are published.

The **100-test / full-token-history / full-context / soak / quality** numbers in the README are retained C10 W9100 qualification, not newly rerun GPU tests of a different build. Full historical private orchestration and machine logs are deliberately not included. The public rebuild is not claimed byte-identical to the stored qualified binary.

For model-dependent regression tests, provide `BONSAI_MODEL=/path/to/the/model.gguf`. Additional tests may also require `BONSAI_MTP_HEAD`, a separately built Prism/ggml oracle, or the actual W9100 device; inspect test prerequisites before interpreting a full suite result.
