//! Builds ../../kernels-nv (rust-gpu, NVIDIA/Ampere cooperative-matrix kernels) into one SPIR-V module per entry point,
//! copied to ../../shaders/spv-rust-nv/<entry>.spv. Separate from tools/spvbuild so the W9100 modules never declare the
//! CooperativeMatrixKHR / Float16 capabilities. Run on the box: `cd tools/spvbuild-nv && cargo run --release`.
use spirv_builder::{Capability, SpirvBuilder, SpirvMetadata};
fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let res = SpirvBuilder::new(root.join("kernels-nv"), "spirv-unknown-vulkan1.1")
        .multimodule(true)
        .spirv_metadata(SpirvMetadata::None)
        .capability(Capability::CooperativeMatrixKHR)
        .capability(Capability::Float16)
        .capability(Capability::Int8)
        .capability(Capability::GroupNonUniformShuffle)
        .capability(Capability::DotProduct)
        .capability(Capability::DotProductInput4x8BitPacked)

        .extension("SPV_KHR_integer_dot_product")
        .extension("SPV_KHR_cooperative_matrix")
        .build()
        .expect("rust-gpu build failed");
    // Never rewrite reviewed source implicitly. Copy only the selected kernels
    // to an explicit isolated output, then validate/hash before incorporation.
    let out = std::path::PathBuf::from(std::env::var_os("BONSAI_SPV_OUT").expect("BONSAI_SPV_OUT required"));
    std::fs::create_dir_all(&out).unwrap();
    for (entry, path) in res.module.unwrap_multi() {
        if !matches!(entry.as_str(), "matvec_r16x2" | "matvec_r16x4" | "attn_s_h2" | "attn_comb2" | "gdn_p6") { continue; }
        std::fs::copy(path, out.join(format!("{entry}.spv"))).unwrap();
        println!("{entry} <- {}", path.display());
    }
}
