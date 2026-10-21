//! Selected NVIDIA integration from contribution-workshop b931d06.
//! No persistent grids, device-scope barriers, output-changing diagnostics,
//! fused fz1/fz2/fz3, or unqualified knob-selected pipelines are loaded.
use ash::vk;

pub(super) const SHADERS: &[(&str, &[u8])] = &[

    ("matvec_r16x2", include_bytes!("../../shaders/spv-rust-nv/matvec_r16x2.spv")),
    ("matvec_r16x4", include_bytes!("../../shaders/spv-rust-nv/matvec_r16x4.spv")),
    ("attn_s_h2", include_bytes!("../../shaders/spv-rust-nv/attn_s_h2.spv")),
    ("attn_comb2", include_bytes!("../../shaders/spv-rust-nv/attn_comb2.spv")),
    ("gdn_p6", include_bytes!("../../shaders/spv-rust-nv/gdn_p6.spv")),
];
pub(super) const EXTENSIONS: [&std::ffi::CStr; 4] = [
    c"VK_KHR_cooperative_matrix", c"VK_KHR_shader_float16_int8",
    c"VK_KHR_vulkan_memory_model", c"VK_KHR_shader_integer_dot_product",
];

#[derive(Clone, Copy, Debug, Default)]
pub struct Capabilities {
    pub vendor: u32, pub subgroup: u32, pub extensions: bool, pub compute_shuffle: bool,
    pub cooperative: bool, pub float16: bool, pub int8: bool,
    pub memory_model: bool, pub integer_dot: bool,
    pub f16_shape: bool, pub int8_shape: bool,
}
impl Capabilities {
    pub fn supported(self) -> bool {
        self.vendor == 0x10de && self.subgroup == 32 && self.extensions && self.compute_shuffle
            && self.cooperative && self.float16 && self.int8 && self.memory_model
            && self.integer_dot && self.f16_shape && self.int8_shape
    }
}

/// All requested optional features are queried. DeviceScope and CM2 are not
/// required, requested or used. Unsupported devices retain the public backend.
pub(super) unsafe fn supported(entry: &ash::Entry, inst: &ash::Instance, pd: vk::PhysicalDevice) -> bool {
    let props = inst.get_physical_device_properties(pd);
    if props.vendor_id != 0x10de || std::env::var("BONSAI_NV").as_deref() == Ok("0") { return false; }
    let Ok(exts) = inst.enumerate_device_extension_properties(pd) else { return false; };
    if !EXTENSIONS.iter().all(|n| exts.iter().any(|e| e.extension_name_as_c_str().ok() == Some(*n))) { return false; }
    let mut sg = vk::PhysicalDeviceSubgroupProperties::default();
    inst.get_physical_device_properties2(pd, &mut vk::PhysicalDeviceProperties2::default().push_next(&mut sg));
    let mut cm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut f = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut mm = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
    let mut dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
    inst.get_physical_device_features2(pd, &mut vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut cm).push_next(&mut f).push_next(&mut mm).push_next(&mut dot));
    let ci = ash::khr::cooperative_matrix::Instance::new(entry, inst);
    let Ok(shapes) = ci.get_physical_device_cooperative_matrix_properties(pd) else { return false; };
    Capabilities { vendor: props.vendor_id, subgroup: sg.subgroup_size, extensions: true,
        compute_shuffle: sg.supported_stages.contains(vk::ShaderStageFlags::COMPUTE) && sg.supported_operations.contains(vk::SubgroupFeatureFlags::SHUFFLE),
        cooperative: cm.cooperative_matrix != 0, float16: f.shader_float16 != 0, int8: f.shader_int8 != 0,
        memory_model: mm.vulkan_memory_model != 0, integer_dot: dot.shader_integer_dot_product != 0,
        f16_shape: shapes.iter().any(|c| c.m_size == 16 && c.n_size == 16 && c.k_size == 16
            && c.a_type == vk::ComponentTypeKHR::FLOAT16 && c.b_type == vk::ComponentTypeKHR::FLOAT16
            && c.c_type == vk::ComponentTypeKHR::FLOAT32 && c.result_type == vk::ComponentTypeKHR::FLOAT32 && c.scope == vk::ScopeKHR::SUBGROUP),
        int8_shape: shapes.iter().any(|c| c.m_size == 16 && c.n_size == 16 && c.k_size == 32
            && c.a_type == vk::ComponentTypeKHR::SINT8 && c.b_type == vk::ComponentTypeKHR::SINT8
            && c.c_type == vk::ComponentTypeKHR::SINT32 && c.result_type == vk::ComponentTypeKHR::SINT32 && c.scope == vk::ScopeKHR::SUBGROUP),
    }.supported()
}

pub fn validate_experiment(name: &str, value: Option<&str>) -> Result<(), String> {
    if value.is_some() { Err(format!("{name} is not supported in production; remove the experimental setting")) } else { Ok(()) }
}
pub(super) fn validate_environment() -> Result<(), String> {
    for name in ["BONSAI_DIAG", "BONSAI_FUSE", "BONSAI_FAST_DECODE", "BONSAI_DMV", "BONSAI_CM", "BONSAI_C9", "BONSAI_CM2"] {
        // Reject even empty/non-Unicode values: presence must not silently change semantics.
        validate_experiment(name, std::env::var_os(name).as_ref().map(|_| "present"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_requested_feature_is_required() {
        let good = Capabilities {vendor:0x10de,subgroup:32,extensions:true,compute_shuffle:true,cooperative:true,float16:true,int8:true,memory_model:true,integer_dot:true,f16_shape:true,int8_shape:true};
        assert!(good.supported());
        for c in [Capabilities{vendor:0x1002,..good},Capabilities{subgroup:64,..good},
            Capabilities{extensions:false,..good},Capabilities{compute_shuffle:false,..good},Capabilities{cooperative:false,..good},
            Capabilities{float16:false,..good},Capabilities{int8:false,..good},
            Capabilities{memory_model:false,..good},Capabilities{integer_dot:false,..good},
            Capabilities{f16_shape:false,..good},Capabilities{int8_shape:false,..good}] { assert!(!c.supported(), "{c:?}"); }
    }
    #[test]
    fn diagnostics_and_fusions_fail_closed() {
        for name in ["BONSAI_DIAG", "BONSAI_FUSE", "BONSAI_FAST_DECODE"] {
            assert!(validate_experiment(name,None).is_ok());
            for value in ["", "0", "1", "fz3", "nobar", "nodispatch", "tiny", "bogus"] { assert!(validate_experiment(name,Some(value)).is_err()); }
        }
    }
}
