//! rust-gpu kernels vs GLSL kernels on the same Vulkan device (lavapipe on the box), tol 1e-4.
#[test]
#[cfg(feature = "vulkan")]
fn rust_gpu_kernels_match_glsl() {
    assert!(!bonsai::vk::list_devices().expect("Vulkan discovery").is_empty(), "actual device required");
    let (report, ok) = bonsai::vk::kparity::run(None, 1e-4).expect("kernel parity run");
    println!("{report}");
    assert!(ok, "{report}");
}
#[test]
#[cfg(feature="vulkan")]
fn r012_f16_split_k_independent_oracle() { bonsai::vk::r012_parity::run().unwrap(); }
