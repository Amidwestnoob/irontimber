# Restricted NVIDIA repair candidate

This local candidate starts from public IronTimber C10 commit
5b3f0974c05b594fd294071ebf8dcb63639881b4, selectively integrating code by the
contributors to Amidwestnoob/contribution-workshop at
b931d06b916d773601629e5fb6022ef1c386a299. Original MIT and model/upstream notices
remain authoritative. No publication or deployment is implied.

## Scope and exactness

The NVIDIA specialization is limited to matvec_r16x2, matvec_r16x4,
attn_s_h2, attn_comb2 and gdn_p6. Public f32-activation prefill and the public
MTP-head matrix arithmetic are retained. The workshop's int8-activation CM9
prefill and f16-rounded Q8-head GEMM are not normal inference paths in this
candidate: their changed arithmetic is unnecessary for this repair. No new
performance superiority claim is made. This conservative selection trades the
workshop's tensor-core prefill speed for preservation of strict reference
numerics. The four known historical Prism differences are not a requirement
for new Prism equivalence; full histories are compared to the preserved Rust
backend, with the same model, head, quantization and dispatch configuration.

The five separate NVIDIA shader modules retain the rust-gpu builder's declared
capabilities. Device creation queries all requested extension features and
32-lane compute-shuffle support, plus cooperative-matrix shapes conservatively.
Unsupported devices use the public backend. BONSAI_NV=0 explicitly selects
that preserved reference. AMD never requests NVIDIA features or loads these
modules. No VulkanMemoryModelDeviceScope or cooperative-matrix2 feature is
requested. Only the five selected extra pipelines are constructed.

The persistent-grid gbar benchmark and its device-side spin barrier were
removed, not timed out. No gbar-bench command is integrated. No fused FFN or
fz1/fz2/fz3 pipeline is loaded. Historical unused kernel source is retained for
attribution/rebuild context, not emitted by the selected builder. Fz3 failed
strict intermediate-state parity (5.6e-4 versus 1e-4); matching final tokens
never turn that failure into a pass. BONSAI_DIAG, BONSAI_FUSE,
BONSAI_FAST_DECODE, BONSAI_DMV, BONSAI_CM, BONSAI_C9 and BONSAI_CM2 are rejected
by presence before acquiring a Vulkan instance, including on AMD. Empty/zero
values do not silently enable experiments. Public supported W9100 settings and
config/w9100-c10.env are unchanged.

## Ownership and builds

The public Arc device/instance/buffer owners and partial-construction cleanup
are retained, including shader-module cleanup and descriptor/command-pool
ownership. Process exit is not the lifecycle acceptance test.

Default builds use Vulkan; `cargo test --locked --release --no-default-features
--no-run` also compiles every test target without ash. GPU CLI branches and
GPU-only tests are feature-guarded. Build artifacts must be outside the checkout.

NVIDIA shader rebuilding uses tools/spvbuild-nv's pinned nightly and locked
rust-gpu dependencies. Set BONSAI_SPV_OUT to an isolated output directory,
then run `cargo run --locked --release` there. Only the five selected modules
are emitted. Validate each with `spirv-val --target-env vulkan1.1` before
incorporating it. Unrelated public GLSL/Rust shader binaries are preserved,
not silently regenerated. A structural SPIR-V header check is not validation.

## Current qualification and historical evidence

The target scope is the existing Ternary-Bonsai-2-27B PQ2_0 trunk (all 64
layers), compatible existing MTP head, f16 KV, context up to 8192 on the
RTX 3090 Ti. Other models, GPUs, drivers and context sizes are not thereby
qualified. Consult the accompanying job REPORT.md, TESTS.md, final-source.json,
test-receipts.json and hardware.json for actual outcomes and exact executable,
compiler, shader and source identities. A build or this document alone does
not establish qualification. SOURCE_MANIFEST.json, benchmarks and the README's
W9100 numbers are explicitly retained historical C10 records, not current
NVIDIA runs. No fresh W9100 GPU pass is claimed without a new hardware receipt.

examples/nvidia_qualification.rs implements desktop-only kernel/f64 oracles,
strict active-state/rollback/cache/streaming checks, 8192 capacity and
pre-dispatch overflow recovery, complete plain/MTP histories against the
public-path Rust reference, three same-process full-engine reload cycles per
mode after warm-up, and a >=600-second mixed-request soak. It requires actual
fixtures and exclusive ownership; it does not turn missing fixtures into passes.
The memory return bound is 128 MiB: allows asynchronous driver retirement and
small desktop allocation noise, while rejecting retained model/arena buffers
measured in GiB. Each run must keep at least 1 GiB free VRAM.

Keep the serial unauthenticated API on loopback. MTP remains request-selectable;
the public serve example defaults MTP off, not globally on. Do not use historical
performance to imply current NVIDIA superiority over Prism. No CUDA runtime is
needed, but rust-gpu/SPIR-V tooling and supporting dependencies are not a claim
that every tool in the stack is Rust.
