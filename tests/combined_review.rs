//! CPU-only combined geometry boundary; no device acquisition.
#[test]
fn accepted_head_geometry_and_plain_prefill_batch_cpu() {
    for n in [64,128,256,512] {
        assert_eq!(bonsai::vk::effective_prefill_batch(n,false),n);
        assert_eq!(bonsai::vk::effective_prefill_batch(n,true),64);
    }
    assert!(std::panic::catch_unwind(||bonsai::vk::effective_prefill_batch(0,true)).is_err());
    assert!(std::panic::catch_unwind(||bonsai::vk::effective_prefill_batch(8192,false)).is_err());
}
