//! Builds ../../kernels (rust-gpu) into one SPIR-V module per entry point and copies them to
//! ../../shaders/spv-rust/<entry>.spv. Run on the box only: `cd tools/spvbuild && cargo run --release`.
use spirv_builder::{SpirvBuilder, SpirvMetadata};
fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let res = SpirvBuilder::new(root.join("kernels"), "spirv-unknown-vulkan1.1")
        .multimodule(true)
        .spirv_metadata(SpirvMetadata::None)
        .build()
        .expect("rust-gpu build failed");
    let out = root.join("shaders/spv-rust");
    std::fs::create_dir_all(&out).unwrap();
    for (entry, path) in res.module.unwrap_multi() {
        std::fs::copy(path, out.join(format!("{entry}.spv"))).unwrap();
        println!("{entry} <- {}", path.display());
    }
}
