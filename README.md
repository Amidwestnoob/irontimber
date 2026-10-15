# IronTimber

**Local inference, forged in Rust.**

An experimental Rust/ash/Vulkan inference engine for **Ternary-Bonsai-2-27B**, qualified on the **AMD FirePro W9100 (16 GB)**. IronTimber gives older hardware useful local inference without requiring a CUDA runtime.

This initial public release exports the verified **C10** engine. The repository has a new name; the Cargo package and executable remain **`bonsai`** to preserve the tested implementation and command compatibility.

## What works

- Existing PQ2_0 GGUF loading, tokenizer, chat template, and all 64 trunk layers.
- Rust host with `ash` for Vulkan and `rust-gpu` kernels compiled to SPIR-V. GLSL reference kernels are included for comparison; this is not a claim that every supporting tool is Rust.
- Greedy generation, a persistent text/chat HTTP API, streaming, and request-selectable multi-token prediction (MTP).
- Qualified context capacity **8192**, using **f16 KV**, with MTP head loaded once.
- Same-process state isolation, cancellation recovery, terminal rollback and reload/lifetime checks in the qualified evidence.

This is **one-model, one-card qualification**, not a general-purpose replacement for llama.cpp. Separate Nvidia development is not included or claimed to be qualified here.

## Measured on the W9100

Against the pinned Prism Vulkan baseline, using the same model and greedy requests:

| Measurement | Prism | IronTimber plain | IronTimber MTP |
|---|---:|---:|---:|
| Warm decode across six measured cases | 7.8–7.9 tok/s | about 14.6 tok/s | 16.3–19.0 tok/s |

Warm prefill improved approximately **40% / 26% / 23% / 20%** for short / 2k / 4k / 7k prompts, with three native/candidate/native comparison cycles per size. A 50% improvement was a stretch goal, not an achieved result.

The six-workload sum of warm whole-request medians was about **2.05× faster with MTP** (47.34 s versus 23.06 s). This excludes startup and is not a promise for every workload or long prompt.

Qualification retained **100 canonical tests**, **54 matching measured token histories**, **142 exact requests in a ten-minute soak**, and **16/16 inspected quality cases**. Full matching token histories, not matching prefixes, were required. Scope and per-case numbers are in [`benchmarks/qualified-results.json`](benchmarks/qualified-results.json); these are retained qualification results, not a new public-snapshot GPU benchmark.

## Build

Requirements: Rust/Cargo, a Vulkan loader, and a working GPU Vulkan driver. The qualified host toolchain was **Rust 1.98.1** on Linux. No full model or MTP-head files are included; small model-derived numerical test vectors retain the model license.

```sh
cargo build --locked --release
```

Prebuilt SPIR-V modules are checked in. `glslangValidator`, when installed, rebuilds GLSL reference shaders; otherwise the build uses their checked-in equivalents. Rebuilding the Rust kernels separately requires the toolchain pinned in `tools/spvbuild/rust-toolchain.toml`:

```sh
cd tools/spvbuild
cargo run --locked --release
```

Rebuilding shaders or using another driver/compiler creates a new build that needs its own qualification. The qualified executable SHA-256 is recorded in `SOURCE_MANIFEST.json`; a fresh rebuild is not assumed byte-identical.

## Run

Obtain the existing model and compatible MTP head separately from the publisher. Set these variables to your own files:

```sh
export MODEL=/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf
export MTP_HEAD=/path/to/head.gguf
. config/w9100-c10.env

./target/release/bonsai serve   --model "$MODEL" --backend vulkan --device HAWAII --layers 64   --ctx 8192 --kv f16 --prompt-cache off   --host 127.0.0.1:18082   --mtp --mtp-head "$MTP_HEAD" --mtp-default off
```

The configuration file is part of the qualified runtime setup; do not omit it and assume unqualified defaults reproduce the reported performance. Set aside the W9100 for this engine rather than loading another GPU model alongside it. Qualified runs preserved at least 1 GiB free VRAM.

```sh
curl --fail http://127.0.0.1:18082/health

curl --fail http://127.0.0.1:18082/completion   -H 'Content-Type: application/json'   -d '{"prompt":"The capital of France is","n_predict":64,"temperature":0,"mtp":true}'

curl --fail --no-buffer http://127.0.0.1:18082/v1/chat/completions   -H 'Content-Type: application/json'   -d '{"messages":[{"role":"user","content":"Name three primary colors."}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64,"temperature":0,"mtp":true,"stream":true}'
```

Set `mtp:false` for ordinary decoding. Omitted MTP is off in this launch configuration. MTP requires greedy-compatible sampling settings; explicit sampled MTP is rejected. Prompt plus subsequent decode positions must fit the context; oversized explicit requests are rejected rather than silently truncated.

## Testing and safety

`cargo test --locked --release` contains CPU and hardware/model-dependent tests. Read individual test prerequisites: absence of model environment variables can cause model-dependent tests to skip work, and the W9100 state tests require the actual qualified device. A passing model-free test command is **not** full GPU qualification.

- The HTTP API is serial and unauthenticated. Keep it on **loopback**; do not expose it directly to the internet. Health requests can queue behind generation.
- This is not an adversarial security audit or a production SLA.
- `BONSAI_REVIEW_TRACE=1` enables diagnostic traces that can contain prompt/output token IDs. It is not enabled by the public example; avoid recording sensitive inputs.
- No vision support, 262k context claim, or Nvidia performance parity is implied.
- Model weights keep their own license. This project is independent of Prism ML; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

MIT for this code, with upstream notices preserved. Full models are not redistributed; small numerical reference fixtures retain their model license. See [`LICENSE`](LICENSE).
