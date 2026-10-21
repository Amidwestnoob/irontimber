# Historical initial public-snapshot verification

At public base 5b3f0974c05b594fd294071ebf8dcb63639881b4, the exported C10 production source and shader hashes matched `SOURCE_MANIFEST.json`; no inference implementation was renamed or edited for branding. The NVIDIA repair changes selected host paths and adds separate shader modules. That historical manifest is retained for provenance, not presented as a current-tree manifest. Current repair checks and their hardware applicability are described in NVIDIA-REPAIR.md and the accompanying job receipts.

Fresh public export checks:

- `cargo build --locked --release --bins -j 4`: passed with Rust 1.98.1. Two pre-existing compiler warnings were retained rather than modifying the qualified source.
- `cargo test --locked --release --test gguf_malformed --test api_review_r011 --test combined_review --test prefill_lut`: **19 passed**, zero failures or ignored tests. These are model-free CPU checks.
- `tests/review_regressions.rs`: **8 passed**, zero failures or ignored tests, using the real model tokenizer on the W9100 host. Its scripted engine is a mock; these tests do not prove GPU numerical quality. The first invocation without `BONSAI_MODEL` failed its prerequisite and was rerun with the actual model, without source changes.
- Gitleaks 8.30.1 directory scan: no secret findings. A separate byte scan found no private machine paths, private LAN addresses or private-key headers in the public export.
- Existing source whitespace warnings were preserved to keep the qualified snapshot byte-identical.

The first offline build attempt lacked a cached Cargo dependency; the normal locked build obtained the pinned dependencies and passed. No model weights are published.

The **100-test / full-token-history / full-context / soak / quality** numbers in the README are retained C10 W9100 qualification, not newly rerun GPU tests of a different build. Full historical private orchestration and machine logs are deliberately not included. The public rebuild is not claimed byte-identical to the stored qualified binary.

For model-dependent regression tests, provide `BONSAI_MODEL=/path/to/the/model.gguf`. Additional tests may also require `BONSAI_MTP_HEAD`, a separately built Prism/ggml oracle, or the actual W9100 device; inspect test prerequisites before interpreting a full suite result.
