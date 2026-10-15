# Attribution and third-party notices

IronTimber is an independent experimental runtime, not an official Prism ML product.

- **Prism ML** created the Bonsai model and its ternary formats. The reference implementation is [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), derived from llama.cpp/ggml. The upstream MIT notice is retained in `licenses/llama.cpp-MIT.txt` and the root license.
- Development originated with Drew Schuyler's Forge/Grok Rust port, followed by Astra review, correctness and lifetime repairs, MTP qualification, and prefill optimization. IronTimber is the public name for the qualified C10 snapshot, not a newly written engine.
- The runtime uses `ash`, `rust-gpu`/`spirv-builder`, and the libraries pinned in the Cargo lockfiles. Their respective licenses continue to apply. Dependencies are obtained through Cargo; their source is not vendored here.
- Full model weights and the MTP head are **not distributed** in this repository. Small tensor/dequantization test vectors under `tests/data/` are model-derived reference fixtures and retain the model's Apache 2.0 license, included in `licenses/Bonsai-model-LICENSE.txt`. Obtain them from the model publisher and comply with their licenses. The [Bonsai GGUF model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) identifies its license as Apache 2.0. The code license is not a license grant for model weights.
- C/C++ sources in `tools/oracle/` are independent reference-test helpers, not part of the inference runtime. They require the separate upstream reference build.
