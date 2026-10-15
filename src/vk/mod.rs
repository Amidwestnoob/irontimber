//! Vulkan (ash) backend. Design rules (lessons from the LFM2.5 host):
//!  * all weights resident on the device, uploaded once, repacked to an aligned 2-bit layout;
//!  * one persistent activation/state arena — no per-op allocations, no per-op descriptor writes;
//!  * descriptor sets built once at load (one per ternary tensor);
//!  * one command buffer + one submit + one fence wait per eval() chunk (a whole forward pass),
//!    with only compute->compute memory barriers between dispatches.
use crate::cpu;
use crate::gguf::{GgmlType, Gguf};
use crate::model::WRef;
use crate::quant::Packed2;
use ash::vk;
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Arc;
mod lifetime;
use lifetime::{DeviceOwner, InstanceOwner, ShaderModule};

const SHADERS: &[(&str, &[u8])] = &[
    ("matvec_t2", include_bytes!(concat!(env!("OUT_DIR"), "/matvec_t2.spv"))),
    ("matvec_t2_1", include_bytes!(concat!(env!("OUT_DIR"), "/matvec_t2_1.spv"))),
    ("matvec_f32", include_bytes!(concat!(env!("OUT_DIR"), "/matvec_f32.spv"))),
    ("rmsnorm", include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm.spv"))),
    ("hadamard", include_bytes!(concat!(env!("OUT_DIR"), "/hadamard.spv"))),
    ("ewise", include_bytes!(concat!(env!("OUT_DIR"), "/ewise.spv"))),
    ("conv", include_bytes!(concat!(env!("OUT_DIR"), "/conv.spv"))),
    ("matmul_t2", include_bytes!(concat!(env!("OUT_DIR"), "/matmul_t2.spv"))),
    ("matmul_t2s", include_bytes!(concat!(env!("OUT_DIR"), "/matmul_t2s.spv"))),
    ("gdn_p", include_bytes!(concat!(env!("OUT_DIR"), "/gdn_p.spv"))),
    ("gdn", include_bytes!(concat!(env!("OUT_DIR"), "/gdn.spv"))),
    ("argmax", include_bytes!(concat!(env!("OUT_DIR"), "/argmax.spv"))),
    ("rope", include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv"))),
    ("attn", include_bytes!(concat!(env!("OUT_DIR"), "/attn.spv"))),
    ("matvec_q8", include_bytes!(concat!(env!("OUT_DIR"), "/matvec_q8.spv"))),
];
/// rust-gpu builds of the same kernels (kernels/ crate -> tools/spvbuild -> shaders/spv-rust), entry point = kernel name.
const SHADERS_RUST: &[(&str, &[u8])] = &[
    ("matvec_t2", include_bytes!("../../shaders/spv-rust/matvec_t2.spv")),
    ("matvec_t2_1", include_bytes!("../../shaders/spv-rust/matvec_t2_1.spv")),
    ("matvec_c8", include_bytes!("../../shaders/spv-rust/matvec_c8.spv")),
    ("attn_s", include_bytes!("../../shaders/spv-rust/attn_s.spv")),
    ("attn_comb", include_bytes!("../../shaders/spv-rust/attn_comb.spv")),
    ("attn_s_h", include_bytes!("../../shaders/spv-rust/attn_s_h.spv")),
    ("kv_pack", include_bytes!("../../shaders/spv-rust/kv_pack.spv")),
    ("matvec_c8x2", include_bytes!("../../shaders/spv-rust/matvec_c8x2.spv")),
    ("matvec_c8x4", include_bytes!("../../shaders/spv-rust/matvec_c8x4.spv")),
    ("matvec_f32", include_bytes!("../../shaders/spv-rust/matvec_f32.spv")),
    ("rmsnorm", include_bytes!("../../shaders/spv-rust/rmsnorm.spv")),
    ("hadamard", include_bytes!("../../shaders/spv-rust/hadamard.spv")),
    ("ewise", include_bytes!("../../shaders/spv-rust/ewise.spv")),
    ("conv", include_bytes!("../../shaders/spv-rust/conv.spv")),
    ("matmul_t2", include_bytes!("../../shaders/spv-rust/matmul_t2.spv")),
    ("matmul_pf64", include_bytes!("../../shaders/spv-rust/matmul_pf64.spv")),
    ("matmul_pf128", include_bytes!("../../shaders/spv-rust/matmul_pf128.spv")),
    ("matmul_pf128x32", include_bytes!("../../shaders/spv-rust/matmul_pf128x32.spv")),
    ("matmul_pf64x32", include_bytes!("../../shaders/spv-rust/matmul_pf64x32.spv")),
    ("matmul_pf64k64", include_bytes!("../../shaders/spv-rust/matmul_pf64k64.spv")),
    ("matmul_pf512", include_bytes!("../../shaders/spv-rust/matmul_pf512.spv")),
    ("matmul_pf_h16", include_bytes!("../../shaders/spv-rust/matmul_pf_h16.spv")),
    ("matmul_r64", include_bytes!("../../shaders/spv-rust/matmul_r64.spv")),
    ("matmul_r128", include_bytes!("../../shaders/spv-rust/matmul_r128.spv")),
    ("matmul_ru8", include_bytes!("../../shaders/spv-rust/matmul_ru8.spv")),
    ("matmul_rb64", include_bytes!("../../shaders/spv-rust/matmul_rb64.spv")),
    ("matmul_rb128", include_bytes!("../../shaders/spv-rust/matmul_rb128.spv")),
    ("matmul_col_dt", include_bytes!("../../shaders/spv-rust/matmul_col_dt.spv")),
    ("matmul_col_dr", include_bytes!("../../shaders/spv-rust/matmul_col_dr.spv")),
    ("matmul_col_dr512", include_bytes!("../../shaders/spv-rust/matmul_col_dr512.spv")),
    ("matmul_col_dr16", include_bytes!("../../shaders/spv-rust/matmul_col_dr16.spv")),
    ("matmul_col_tail", include_bytes!("../../shaders/spv-rust/matmul_col_tail.spv")),
    ("matmul_col_sq", include_bytes!("../../shaders/spv-rust/matmul_col_sq.spv")),
    ("matmul_col_sq4", include_bytes!("../../shaders/spv-rust/matmul_col_sq4.spv")),
    ("matmul_col_sw4", include_bytes!("../../shaders/spv-rust/matmul_col_sw4.spv")),
    ("matmul_col_sw16", include_bytes!("../../shaders/spv-rust/matmul_col_sw16.spv")),
    ("matmul_col", include_bytes!("../../shaders/spv-rust/matmul_col.spv")),
    ("matmul_col_k16", include_bytes!("../../shaders/spv-rust/matmul_col_k16.spv")),
    ("matmul_col_pipe", include_bytes!("../../shaders/spv-rust/matmul_col_pipe.spv")),
    ("matmul_colsoa", include_bytes!("../../shaders/spv-rust/matmul_colsoa.spv")),
    ("matmul_colwave", include_bytes!("../../shaders/spv-rust/matmul_colwave.spv")),
    ("matmul_lut2", include_bytes!("../../shaders/spv-rust/matmul_lut2.spv")),
    ("matmul_lutv", include_bytes!("../../shaders/spv-rust/matmul_lutv.spv")),
    ("matmul_trit4", include_bytes!("../../shaders/spv-rust/matmul_trit4.spv")),
    ("matmul_trits", include_bytes!("../../shaders/spv-rust/matmul_trits.spv")),
    ("matmul_col16", include_bytes!("../../shaders/spv-rust/matmul_col16.spv")),
    ("matmul_col8", include_bytes!("../../shaders/spv-rust/matmul_col8.spv")),
    ("prefill_expand", include_bytes!("../../shaders/spv-rust/prefill_expand.spv")),
    ("matmul_dense64", include_bytes!("../../shaders/spv-rust/matmul_dense64.spv")),
    ("matmul_col128u4", include_bytes!("../../shaders/spv-rust/matmul_col128u4.spv")),
    ("matmul_col128u8", include_bytes!("../../shaders/spv-rust/matmul_col128u8.spv")),
    ("matmul_col128u32", include_bytes!("../../shaders/spv-rust/matmul_col128u32.spv")),
    ("matmul_col_w512", include_bytes!("../../shaders/spv-rust/matmul_col_w512.spv")),
    ("matmul_col_r512", include_bytes!("../../shaders/spv-rust/matmul_col_r512.spv")),
    ("matmul_col_1024", include_bytes!("../../shaders/spv-rust/matmul_col_1024.spv")),
    ("matmul_col_r128t32", include_bytes!("../../shaders/spv-rust/matmul_col_r128t32.spv")),
    ("matmul_col_r32t64", include_bytes!("../../shaders/spv-rust/matmul_col_r32t64.spv")),
    ("matmul_col_r32t32", include_bytes!("../../shaders/spv-rust/matmul_col_r32t32.spv")),
    ("matmul_col_u4", include_bytes!("../../shaders/spv-rust/matmul_col_u4.spv")),
    ("matmul_col_u8", include_bytes!("../../shaders/spv-rust/matmul_col_u8.spv")),
    ("matmul_col_u16", include_bytes!("../../shaders/spv-rust/matmul_col_u16.spv")),
    ("matmul_col_k64", include_bytes!("../../shaders/spv-rust/matmul_col_k64.spv")),
    ("matmul_col_fma8", include_bytes!("../../shaders/spv-rust/matmul_col_fma8.spv")),
    ("matmul_col_fma32", include_bytes!("../../shaders/spv-rust/matmul_col_fma32.spv")),
    ("attn_prefill_h4", include_bytes!("../../shaders/spv-rust/attn_prefill_h4.spv")),
    ("attn_gqa8", include_bytes!("../../shaders/spv-rust/attn_gqa8.spv")),
    ("attn_gqa6", include_bytes!("../../shaders/spv-rust/attn_gqa6.spv")),
    ("attn_gqa3", include_bytes!("../../shaders/spv-rust/attn_gqa3.spv")),
    ("attn_gqa2", include_bytes!("../../shaders/spv-rust/attn_gqa2.spv")),
    ("attn_prefill_h4_256", include_bytes!("../../shaders/spv-rust/attn_prefill_h4_256.spv")),
    ("attn_prefill_h2_256", include_bytes!("../../shaders/spv-rust/attn_prefill_h2_256.spv")),
    ("matmul_t2s", include_bytes!("../../shaders/spv-rust/matmul_t2s.spv")),
    ("gdn_p", include_bytes!("../../shaders/spv-rust/gdn_p.spv")),
    ("gdn", include_bytes!("../../shaders/spv-rust/gdn.spv")),
    ("argmax", include_bytes!("../../shaders/spv-rust/argmax.spv")),
    ("rope", include_bytes!("../../shaders/spv-rust/rope.spv")),
    ("attn", include_bytes!("../../shaders/spv-rust/attn.spv")),
    ("matvec_q8", include_bytes!("../../shaders/spv-rust/matvec_q8.spv")),
];
pub mod kparity;
pub mod prefill_pack;
pub mod r012_parity;
const NONE: u32 = u32::MAX;
/// Kernel source: "rust" (default, rust-gpu SPIR-V) or "glsl" (env BONSAI_KERNELS).
pub fn rust_kernels() -> bool { std::env::var("BONSAI_KERNELS").map(|v| v != "glsl").unwrap_or(true) }
/// Token count at/above which the tiled matmul is used (env BONSAI_TILE_MIN; set very large to disable).
fn tile_min() -> u32 { std::env::var("BONSAI_TILE_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(22) }
pub const MAXT: usize = 512; // hard bound; allocation follows explicit/default64 batch
pub fn validate_prefill_batch(value:Option<&str>)->Result<usize,String>{
    match value {None|Some("64")=>Ok(64),Some("128")=>Ok(128),Some("256")=>Ok(256),Some("512")=>Ok(512),_=>Err("unsupported prefill batch; use 64/128/256/512".into())}
}
/// Avoid a complete extra 64-column tile for a tiny final tail. No token is removed.
/// The tail executes through the existing short-prompt path.
pub fn prefill_chunks(len:usize,batch:usize)->Vec<std::ops::Range<usize>> {
    assert!(matches!(batch,64|128|256|512));let mut ranges=vec![];let mut start=0;
    while start<len {let mut n=(len-start).min(batch);let tail=n%64;if n>64 && tail>0 && tail<22 {n-=tail;}ranges.push(start..start+n);start+=n;}
    ranges
}

/// Split-K's minimum one-partial ABI is 260 words/query/head, even at tiny ctx.
/// Legacy/default64 allocation is unchanged; small contexts retain the 260-word ABI.
pub fn attention_scratch_words(ctx:usize,heads:usize)->usize { 64*heads*ctx.max(260) }
/// Combined package: accepted MTP head dispatches/catch-up were screened at64.
/// Its recurrent checkpoint + head weights leave insufficient1GiB reserve for a
/// second full column layout AND a512-token head arena. Keep full8192/f16 and
/// bounded accepted head geometry, without changing ordinary no-head prefill.
pub fn effective_prefill_batch(requested:usize,head:bool)->usize {
    assert!(matches!(requested,64|128|256|512));
    if head {64}else{requested}
}
/// Preserve r012's chunk choice for nt<=64 even when arena allocation is larger.
pub fn attention_dispatch_room(ctx:usize,heads:usize,nt:usize)->usize {
    assert!(nt>0 && nt<=MAXT);nt.div_ceil(64)*64*heads*ctx.max(260)
}

/// Ternary -d/0/d/2d must all be losslessly representable as normal f16 or zero.
pub fn lossless_weight_lds_scale(d:f32)->bool {
    d.is_finite() && (d==0.0 || (d.abs()>=0.00006103515625 && d.abs()<=32752.0))
        && half::f16::from_f32(d).to_f32().to_bits()==d.to_bits()
}

/// Peel tiny padded token tails for all column-prefill matrices, not prompt names.
/// Full tiles retain<=256-token dispatches; tiny tails use the existing c8 path.
pub fn column_prefill_segments(nt:u32,peel:bool)->Vec<(u32,u32)>{
    assert!(nt<=MAXT as u32);
    let mut v=Vec::new();
    for start in (0..nt).step_by(256){let n=(nt-start).min(256);let rem=n%64;
        if peel && n>64 && rem>0 && rem<8 {v.push((start,n-rem));v.push((start+n-rem,rem));}else{v.push((start,n));}
    }v
}

pub fn prefill_column_kernel(m:u32,k:u32,nt:u32,value:&str)->(&'static str,u32,u32){
    match value {
        "shape" if nt>=256 && (m,k)==(5120,17408)=>("matmul_col_sw16",64,64),

        _=>("matmul_col",64,64)
    }
}
pub fn validate_prefill_matmul(value:Option<&str>)->Result<(),String>{
    if matches!(value,None|Some("base"|"shape")) {Ok(())} else {Err("unsupported prefill matmul policy".into())}
}

/// CPU-only validation; reject unsupported configuration before Vulkan acquisition.
pub fn validate_prefill_config(ctx:usize, rust:bool, kv:Option<&str>, tile:Option<&str>) -> Result<(),String> {
    if ctx == 0 || ctx > (u32::MAX as usize)/(MAXT*24*4) { return Err("invalid/overflowing context capacity geometry".into()); }
    if !matches!(kv,None|Some("f16")|Some("f32")) { return Err("unsupported KV dtype".into()); }
    if !matches!(tile,None|Some("old")|Some("pf64")|Some("pf128")|Some("pf128x32")|Some("pf64x32")|Some("pf64k64")|Some("pf512")|Some("h16")|Some("r64")|Some("r128")|Some("ru8")|Some("rb64")|Some("rb128")|Some("col")|Some("col16")|Some("col8")|Some("col128u4")|Some("col128u8")|Some("dense")) { return Err("unsupported prefill tile".into()); }
    if !rust && (kv==Some("f16") || !matches!(tile,None|Some("old"))) { return Err("f16 KV/prefill tile needs rust-gpu kernels".into()); }
    Ok(())
}

pub fn validate_prefill_attention(rust:bool, value:&str) -> Result<(),String> {
    if !matches!(value,"old"|"quad"|"quad256"|"pair256"|"gqa6"|"gqa3"|"gqa2") || (!rust && value!="old") { return Err("unsupported prefill attention configuration".into()); }
    Ok(())
}
pub fn select_prefill_attention(nt:u32,cn:u32,value:&str)->(&'static str,u32){
    if nt<8 || value=="old" {return ("attn_s_h",1);}
    match (value,cn<=256) {("quad256",true)=>("attn_prefill_h4_256",4),("pair256",true)=>("attn_prefill_h2_256",2),_=>("attn_prefill_h4",4)}
}

/// GQA specialization is only for the measured 24/4/256 geometry and long prefill.
/// Decode and MTP verification (nt<=2), small contexts and other configs fall back.
pub fn select_prefill_gqa(nt:u32,cn:u32,pos0:u32,nh:u32,nkv:u32,hd:u32,value:&str)->(&'static str,u32,u32){
    if nt>=8 && cn==256 && pos0>=1024 && (nh,nkv,hd)==(24,4,256) {
        match value {"gqa6"=>return ("attn_gqa6",1,6),"gqa3"=>return ("attn_gqa3",1,3),"gqa2"=>return ("attn_gqa2",1,2),_=>()}
    }
    let (pipe,q)=select_prefill_attention(nt,cn,if matches!(value,"gqa6"|"gqa3"|"gqa2") {"quad256"}else{value});(pipe,q,1)
}

/// Validate production specializations before constructing a Vulkan instance.
pub fn validate_geometry(c: &crate::model::Config, ctx_len: usize, kv16: bool, rust: bool) -> Result<(), String> {
    if ctx_len == 0 || ctx_len > 8192 { return Err("supported context capacity is 1..=8192".into()); }
    if kv16 && !rust { return Err("f16 KV requires Rust kernels".into()); }
    if c.head_dim != 256 || c.n_head == 0 || c.n_head_kv == 0 || c.n_head % c.n_head_kv != 0
        || c.n_embd == 0 || c.n_embd % 1024 != 0 || c.n_ff == 0 || c.n_ff % 1024 != 0
        || c.ssm_state != 128 || c.ssm_groups == 0 || c.ssm_vheads == 0
        || c.ssm_vheads % c.ssm_groups != 0 || c.ssm_inner != c.ssm_vheads * c.ssm_state
        || c.ssm_inner % 1024 != 0 || !(2..=4).contains(&c.conv_k)
        || c.full_attn_interval == 0 || c.n_layer == 0 || c.n_layer > 64
        || c.had_block != 1024 || c.n_rot == 0 || c.n_rot > c.head_dim || c.n_rot % 2 != 0 {
        return Err("unsupported Vulkan kernel geometry".into());
    }
    Ok(())
}

/// Packed KV readback decodes half words, not two halves reinterpreted as an f32.
pub fn decode_kv_readback(bytes: &[u8], f16: bool) -> Vec<f32> {
    if f16 { bytes.chunks_exact(2).map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()).collect() }
    else { bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect() }
}

pub struct Buf { pub buf: vk::Buffer, pub mem: vk::DeviceMemory, pub size: u64, pub ptr: *mut u8, owner: Arc<DeviceOwner> }

pub struct Ctx {
    owner: Arc<DeviceOwner>, pub instance: ash::Instance, pub pdev: vk::PhysicalDevice, pub device: ash::Device,
    pub queue: vk::Queue, pub mem_props: vk::PhysicalDeviceMemoryProperties, pub props: vk::PhysicalDeviceProperties,
    pub cpool: vk::CommandPool, pub cb: vk::CommandBuffer, pub fence: vk::Fence,
    /// extra command buffers so one forward pass can be split into several queue submissions
    pub seg: Vec<vk::CommandBuffer>, pub seg_used: std::cell::Cell<usize>, pub cur: std::cell::Cell<vk::CommandBuffer>,
    pub dsl: vk::DescriptorSetLayout, pub layout: vk::PipelineLayout, pub dpool: vk::DescriptorPool,
    pub pipes: HashMap<&'static str, vk::Pipeline>, pub staging: Buf,
    // Review trace counts recorded dispatches only after their fence completed.
    pub(crate) rust: bool, trace: bool, pending_dispatches: std::cell::Cell<u64>, nobar: std::cell::Cell<bool>,
    pending_q8: std::cell::Cell<u64>, pub completed_dispatches: std::cell::Cell<u64>, pub completed_q8: std::cell::Cell<u64>,
    /// BONSAI_PROFILE=1: GPU timestamps after every dispatch, aggregated per kernel
    pub prof: Option<(vk::QueryPool, std::cell::RefCell<Vec<&'static str>>, std::cell::RefCell<HashMap<&'static str, (f64, u64)>>)>,
}

pub fn list_devices() -> Result<Vec<String>, String> {
    unsafe {
        let entry = ash::Entry::load().map_err(|e| e.to_string())?;
        let inst = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1)), None).map_err(|e| e.to_string())?;
        let owner = InstanceOwner { instance: inst, _entry: Some(entry) };
        let inst = &owner.instance;
        let v = inst.enumerate_physical_devices().map_err(|e| e.to_string())?.iter().map(|&p| inst.get_physical_device_properties(p).device_name_as_c_str().unwrap().to_string_lossy().into_owned()).collect();
        Ok(v)
    }
}

impl Ctx {
    /// `want`: substring of the device name (env BONSAI_VK_DEVICE overrides); default prefers a discrete GPU.
    pub fn new(want: Option<&str>, max_sets: u32) -> Result<Self, String> { Self::new_with(want, max_sets, rust_kernels()) }
    /// Same, choosing the kernel set explicitly (rust-gpu SPIR-V when `rust`, else GLSL).
    pub fn new_with(want: Option<&str>, max_sets: u32, rust: bool) -> Result<Self, String> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("vulkan loader: {e}"))?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1).application_name(c"bonsai");
            let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None).map_err(|e| e.to_string())?;
            let instance_owner = Arc::new(InstanceOwner { instance: instance.clone(), _entry: Some(entry) });
            let devs = instance.enumerate_physical_devices().map_err(|e| e.to_string())?;
            let want = std::env::var("BONSAI_VK_DEVICE").ok().or(want.map(String::from));
            let name = |p: vk::PhysicalDevice| instance.get_physical_device_properties(p).device_name_as_c_str().unwrap().to_string_lossy().to_lowercase();
            let pdev = match &want {
                Some(w) => *devs.iter().find(|&&p| name(p).contains(&w.to_lowercase())).ok_or_else(|| format!("no Vulkan device matching '{w}'"))?,
                None => *devs.iter().find(|&&p| instance.get_physical_device_properties(p).device_type == vk::PhysicalDeviceType::DISCRETE_GPU).or(devs.first()).ok_or("no Vulkan devices")?,
            };
            let props = instance.get_physical_device_properties(pdev);
            eprintln!("[vk] device: {} (maxStorageBufferRange {} MiB)", CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy(), props.limits.max_storage_buffer_range >> 20);
            let qfi = instance.get_physical_device_queue_family_properties(pdev).iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)).ok_or("no compute queue")? as u32;
            let prio = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qfi).queue_priorities(&prio)];
            let device = instance.create_device(pdev, &vk::DeviceCreateInfo::default().queue_create_infos(&qci), None).map_err(|e| e.to_string())?;
            let owner = Arc::new(DeviceOwner { device: device.clone(), _instance: instance_owner });
            let queue = device.get_device_queue(qfi, 0);
            let mem_props = instance.get_physical_device_memory_properties(pdev);
            // Own every handle immediately, so any later ? drops exactly the
            // successfully constructed prefix (including staging/map failures).
            let mut c = Self::empty(owner, pdev, queue, mem_props, props, rust);
            c.cpool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(qfi).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).map_err(|e| e.to_string())?;
            c.cb = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(c.cpool).command_buffer_count(1).level(vk::CommandBufferLevel::PRIMARY)).map_err(|e| e.to_string())?[0];
            c.cur.set(c.cb);
            c.seg = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(c.cpool).command_buffer_count(96).level(vk::CommandBufferLevel::PRIMARY)).map_err(|e| e.to_string())?;
            c.fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?;
            let binds: Vec<_> = (0..4).map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
            c.dsl = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds), None).map_err(|e| e.to_string())?;
            let pcr = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(128)];
            let dsls = [c.dsl];
            c.layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls).push_constant_ranges(&pcr), None).map_err(|e| e.to_string())?;
            let ps = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(4 * max_sets)];
            c.dpool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(max_sets).pool_sizes(&ps), None).map_err(|e| e.to_string())?;
            eprintln!("[vk] kernels: {}", if rust { "rust-gpu (shaders/spv-rust)" } else { "GLSL (shaders/*.comp)" });
            for (n, spv) in if rust { SHADERS_RUST } else { SHADERS } {
                // Dev/A-B hook: BONSAI_SPV_DIR=<dir> overrides individual rust kernels with <dir>/<name>.spv.
                let ov = if rust { std::env::var("BONSAI_SPV_DIR").ok().and_then(|d| std::fs::read(format!("{d}/{n}.spv")).ok()) } else { None };
                let code = ash::util::read_spv(&mut std::io::Cursor::new(ov.as_deref().unwrap_or(spv))).map_err(|e| e.to_string())?;
                let m = ShaderModule(&device, device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| e.to_string())?);
                let rows: u32 = std::env::var("BONSAI_MV_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
                let entries = [vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 }];
                let data = rows.to_le_bytes();
                let spec = vk::SpecializationInfo::default().map_entries(&entries).data(&data);
                let entry_name = std::ffi::CString::new(*n).unwrap();
                let mut st = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(m.1).name(if rust { &entry_name } else { c"main" });
                if *n == "matvec_t2_1" { st = st.specialization_info(&spec); }
                let p = match device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(st).layout(c.layout)], None) {
                    Ok(p) => p[0],
                    Err((partial, error)) => {
                        for p in partial { if p != vk::Pipeline::null() { device.destroy_pipeline(p, None); } }
                        return Err(format!("pipeline {n}: {error:?}"));
                    }
                };
                c.pipes.insert(*n, p);
            }
            if std::env::var("BONSAI_PROFILE").is_ok() {
                let qp = c.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(8192), None).map_err(|e| e.to_string())?;
                c.prof = Some((qp, Default::default(), Default::default()));
            }
            c.staging = c.buffer(64 << 20, true)?;
            Ok(c)
        }
    }

    pub fn buffer(&self, size: u64, host: bool) -> Result<Buf, String> {
        unsafe {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
            let mut b = Buf::empty(self.owner.clone());
            b.size = size.max(16);
            b.buf = self.device.create_buffer(&vk::BufferCreateInfo::default().size(b.size).usage(usage), None).map_err(|e| e.to_string())?;
            let req = self.device.get_buffer_memory_requirements(b.buf);
            let want = if host { vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT } else { vk::MemoryPropertyFlags::DEVICE_LOCAL };
            let mt = (0..self.mem_props.memory_type_count).find(|&i| req.memory_type_bits & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(want)).ok_or("no memory type")?;
            b.mem = self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt), None).map_err(|e| format!("allocate {} MiB: {e}", size >> 20))?;
            self.device.bind_buffer_memory(b.buf, b.mem, 0).map_err(|e| e.to_string())?;
            if host { b.ptr = self.device.map_memory(b.mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).map_err(|e| e.to_string())? as *mut u8; }
            Ok(b)
        }
    }

    fn begin(&self) { self.pending_q8.set(0); self.pending_dispatches.set(0); self.seg_used.set(0); self.cur.set(self.cb); unsafe { self.device.reset_command_buffer(self.cb, vk::CommandBufferResetFlags::empty()).unwrap(); self.device.begin_command_buffer(self.cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
        if let Some((qp, names, _)) = &self.prof { self.device.cmd_reset_query_pool(self.cb, *qp, 0, 8192); self.device.cmd_write_timestamp(self.cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, *qp, 0); names.borrow_mut().clear(); } } }
    /// fold the timestamps of the last submit into the per-kernel totals
    pub fn collect_profile(&self) {
        let Some((qp, names, tot)) = &self.prof else { return };
        let n = names.borrow().len(); if n == 0 { return; }
        let mut ts = vec![0u64; n + 1];
        unsafe { self.device.get_query_pool_results(*qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT).unwrap(); }
        let period = self.props.limits.timestamp_period as f64;
        let mut t = tot.borrow_mut();
        for (i, name) in names.borrow().iter().enumerate() { let e = t.entry(name).or_insert((0.0, 0)); e.0 += (ts[i + 1].wrapping_sub(ts[i])) as f64 * period * 1e-6; e.1 += 1; }
    }
    pub fn print_profile(&self) {
        let Some((_, _, tot)) = &self.prof else { return };
        let t = tot.borrow(); let all: f64 = t.values().map(|v| v.0).sum();
        let mut v: Vec<_> = t.iter().collect(); v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
        eprintln!("[profile] GPU time by kernel (ms total, calls, %):");
        for (k, (ms, n)) in v { eprintln!("  {k:12} {ms:10.2} ms {n:7} calls {:5.1}%", 100.0 * ms / all); }
    }
    /// End the current command buffer and queue it (no wait); keep recording into a fresh one.
    /// Keeps each GPU job short (RADV/amdgpu resets long-running jobs) while still waiting only once.
    pub fn split(&self) {
        let i = self.seg_used.get();
        if i >= self.seg.len() { return; }
        unsafe {
            let cur = self.cur.get();
            self.device.end_command_buffer(cur).unwrap();
            let cbs = [cur];
            self.device.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], vk::Fence::null()).unwrap();
            let n = self.seg[i];
            self.device.reset_command_buffer(n, vk::CommandBufferResetFlags::empty()).unwrap();
            self.device.begin_command_buffer(n, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            self.cur.set(n); self.seg_used.set(i + 1);
        }
        self.barrier();
    }
    fn submit_wait(&self) {
        unsafe {
            let cur = self.cur.get();
            self.device.end_command_buffer(cur).unwrap();
            let cbs = [cur];
            self.device.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], self.fence).unwrap();
            self.device.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
            self.completed_dispatches.set(self.completed_dispatches.get()+self.pending_dispatches.get());
            self.completed_q8.set(self.completed_q8.get()+self.pending_q8.get());
            if self.trace && self.pending_dispatches.get() > 0 {
                eprintln!("[review] completed rust={} device=0x{:04x} dispatches={} matvec_q8={}", self.rust, self.props.device_id, self.pending_dispatches.get(), self.pending_q8.get());
            }
            self.device.reset_fences(&[self.fence]).unwrap();
        }
    }
    fn barrier(&self) {
        let mb = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ)];
        unsafe { self.device.cmd_pipeline_barrier(self.cur.get(), vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &mb, &[], &[]); }
    }
    /// upload bytes into a device buffer at byte offset, through the staging buffer
    pub fn upload(&self, dst: &Buf, off: u64, data: &[u8]) {
        for (i, ch) in data.chunks(self.staging.size as usize).enumerate() {
            unsafe { std::ptr::copy_nonoverlapping(ch.as_ptr(), self.staging.ptr, ch.len()); }
            self.begin();
            unsafe { self.device.cmd_copy_buffer(self.cur.get(), self.staging.buf, dst.buf, &[vk::BufferCopy { src_offset: 0, dst_offset: off + (i * self.staging.size as usize) as u64, size: ch.len() as u64 }]); }
            self.submit_wait();
        }
    }
    pub fn download(&self, src: &Buf, off: u64, out: &mut [u8]) {
        let n = self.staging.size as usize;
        for (i, ch) in out.chunks_mut(n).enumerate() {
            self.begin();
            unsafe { self.device.cmd_copy_buffer(self.cur.get(), src.buf, self.staging.buf, &[vk::BufferCopy { src_offset: off + (i * n) as u64, dst_offset: 0, size: ch.len() as u64 }]); }
            self.submit_wait();
            unsafe { std::ptr::copy_nonoverlapping(self.staging.ptr, ch.as_mut_ptr(), ch.len()); }
        }
    }
    pub fn set(&self, b: [(&Buf, u64, u64); 4]) -> vk::DescriptorSet {
        unsafe {
            let dsls = [self.dsl];
            let s = self.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.dpool).set_layouts(&dsls)).expect("descriptor sets")[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = b.iter().map(|(bf, o, r)| [vk::DescriptorBufferInfo { buffer: bf.buf, offset: *o, range: *r }]).collect();
            let w: Vec<_> = infos.iter().enumerate().map(|(i, inf)| vk::WriteDescriptorSet::default().dst_set(s).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(inf)).collect();
            self.device.update_descriptor_sets(&w, &[]);
            s
        }
    }
    fn dispatch(&self, pipe: &str, set: vk::DescriptorSet, pc: &[u32], g: (u32, u32, u32)) {
        unsafe {
            self.device.cmd_bind_pipeline(self.cur.get(), vk::PipelineBindPoint::COMPUTE, self.pipes[pipe]);
            self.device.cmd_bind_descriptor_sets(self.cur.get(), vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[set], &[]);
            let mut p = [0u32; 32]; p[..pc.len()].copy_from_slice(pc);
            self.device.cmd_push_constants(self.cur.get(), self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck_u32(&p));
            self.device.cmd_dispatch(self.cur.get(), g.0, g.1, g.2);
            self.pending_dispatches.set(self.pending_dispatches.get() + 1);
            if pipe == "matvec_q8" { self.pending_q8.set(self.pending_q8.get()+1); }
            if let Some((qp, names, _)) = &self.prof {
                let mut n = names.borrow_mut();
                if n.len() + 1 < 8192 { n.push(self.pipes.keys().find(|k| self.pipes[*k] == self.pipes[pipe]).copied().unwrap()); self.device.cmd_write_timestamp(self.cur.get(), vk::PipelineStageFlags::BOTTOM_OF_PIPE, *qp, n.len() as u32); }
            }
        }
        if !self.nobar.get() { self.barrier(); }
    }
    /// r012: record mutually independent dispatches (disjoint outputs, none reads another's output) with ONE
    /// trailing barrier instead of one per dispatch. BONSAI_NOGROUP=1 restores per-dispatch barriers.
    pub(crate) fn group(&self, f: impl FnOnce()) {
        if std::env::var_os("BONSAI_NOGROUP").is_some() { f(); return; }
        let was = self.nobar.replace(true); f(); self.nobar.set(was); if !was { self.barrier(); }
    }
}
fn bytemuck_u32(p: &[u32]) -> &[u8] { unsafe { std::slice::from_raw_parts(p.as_ptr() as *const u8, p.len() * 4) } }
impl Ctx {
    fn expand_prefill(&self,set:vk::DescriptorSet,m:u32,k:u32,off:u32){
        let n=m*(k/16);self.group(||{for start in (0..n).step_by(65536) {let count=(n-start).min(65536);self.dispatch("prefill_expand",set,&[m,k,off,start,count],(count.div_ceil(256),1,1));}});
    }
}
fn fb(x: f32) -> u32 { x.to_bits() }

/// Device-resident ternary matrix: one buffer, words then scales (256-aligned).
pub struct DevMat { pub rows: u32, pub cols: u32, pub buf: Buf, pub set: vk::DescriptorSet, prefill: Option<(Buf,vk::DescriptorSet)> }

/// Offsets (in f32 elements) of everything in the arena.
#[derive(Default, Debug, Clone)]
struct Lay { x: u32, xn: u32, hx: u32, big1: u32, big2: u32, conv: u32, beta: u32, alpha: u32, o: u32, scr: u32, logits: u32, dense:u32, dense_words:usize,
    conv_st: Vec<u32>, ssm: Vec<u32>, kc: Vec<u32>, vc: Vec<u32>, total: u64,
    // MTP (only allocated when a head is attached)
    hrow: u32, hprev: u32, mh: u32, me: u32, mcat: u32, mx: u32, mkc: u32, mvc: u32, kvs: u32, logits2: u32, amx: u32, ck_conv: Vec<u32>, ck_ssm: Vec<u32> }

/// GPU-resident MTP head (int8-group matrices + norm params).
struct GMtp { eh: DevMat, q: DevMat, k: DevMat, v: DevMat, o: DevMat, gate: DevMat, up: DevMat, down: DevMat,
    enorm: u32, hnorm: u32, attn_norm: u32, post_norm: u32, q_norm: u32, k_norm: u32, head_norm: u32 }

/// Offsets into Params.
#[derive(Default, Clone)]
struct PLayer { attn_norm: u32, post_norm: u32, conv_w: u32, dt: u32, a: u32, beta_w: u32, alpha_w: u32, ssm_norm: u32, q_norm: u32, k_norm: u32 }

enum DLayer { Gdn { qkv: DevMat, gate: DevMat, out: DevMat }, Attn { q: DevMat, k: DevMat, v: DevMat, o: DevMat } }
struct DL { kind: DLayer, gate: DevMat, up: DevMat, down: DevMat }

pub struct VkEngine<'a> {
    pub ctx: Ctx, pub cpu: cpu::Model<'a>, arena: Buf, params: Buf, lay: Lay, pl: Vec<PLayer>, out_norm: u32,
    signs: HashMap<usize, u32>, layers: Vec<DL>, output: DevMat, set0: vk::DescriptorSet, pub pos: usize, ctx_len: usize,
    pub timing: bool,
    /// r012: f16 (f16x2-packed) KV cache for the trunk attention layers (BONSAI_KV=f16 / --kv f16). Default f32.
    pub kv16: bool,
    ubatch: usize,
    mtp: Option<(GMtp, crate::mtp::MtpHead<'a>)>, mpos: usize, hp: (u32, usize),
}

impl<'a> VkEngine<'a> {
    pub fn new(g: &'a Gguf, n_layers: Option<usize>, ctx_len: usize, device: Option<&str>) -> Result<Self, String> { Self::new_with_mtp(g, n_layers, ctx_len, device, None) }
    /// `mtp_from`: GGUF holding the NextN head (the model itself, or a head-only sidecar).
    pub fn new_with_mtp(g: &'a Gguf, n_layers: Option<usize>, ctx_len: usize, device: Option<&str>, mtp_from: Option<&'a Gguf>) -> Result<Self, String> {
        let t0 = std::time::Instant::now();
        let requested_batch=validate_prefill_batch(std::env::var("BONSAI_PREFILL_BATCH").ok().as_deref())?;
        validate_prefill_matmul(std::env::var("BONSAI_PREFILL_MATMUL").ok().as_deref())?;
        let ubatch=effective_prefill_batch(requested_batch,mtp_from.is_some());
        validate_prefill_config(ctx_len, rust_kernels(), std::env::var("BONSAI_KV").ok().as_deref(), std::env::var("BONSAI_PREFILL_TILE").ok().as_deref())?;
        let attn=std::env::var("BONSAI_PREFILL_ATTN").unwrap_or_else(|_|"old".into());
        validate_prefill_attention(rust_kernels(),&attn)?;
        if let Some(hg) = mtp_from { crate::mtp::validate_compatibility(hg, g)?; }
        let geometry = crate::model::Config::from_gguf(g)?;
        let kv16 = std::env::var("BONSAI_KV").map_or(false, |v| v == "f16");
        validate_geometry(&geometry, ctx_len, kv16, rust_kernels())?;
        let m = cpu::Model::load(g, n_layers)?;
        let mut head = match mtp_from { Some(hg) => Some(crate::mtp::MtpHead::load(hg, &m.cfg)?.ok_or("no MTP head in the given GGUF")?), None => None };
        let column_prefill=std::env::var("BONSAI_PREFILL_TILE").map_or(false,|v|matches!(v.as_str(),"col"|"col16"|"col8"|"col128u4"|"col128u8"));
        // Plan exact extra allocation before Vulkan. Only the prefill copy is permuted;
        // original matrices and every nt=1/nt=2/MTP decode load remain untouched.
        let mut matrices=vec![&m.output];
        for l in &m.layers {
            matrices.extend([&l.ffn_gate,&l.ffn_up,&l.ffn_down]);
            match &l.kind {cpu::Kind::Gdn{qkv,gate,out,..}=>matrices.extend([qkv,gate,out]),cpu::Kind::Attn{q,k,v,o,..}=>matrices.extend([q,k,v,o])};
        }
        let matrix_bytes=|w:&&WRef,pad:bool|->u64 {let r=if pad {w.rows.div_ceil(4)*4}else{w.rows};let wb=(r*w.cols/16*4) as u64;wb.div_ceil(256)*256+(r*w.cols/128*4) as u64};
        let original_bytes:u64=matrices.iter().map(|w|matrix_bytes(w,false)).sum();
        let column_bytes:u64=if column_prefill {matrices.iter().skip(1).map(|w|matrix_bytes(w,true)).sum()}else{0};
        let dense_words=if std::env::var("BONSAI_PREFILL_TILE").map_or(false,|v|v=="dense") {matrices.iter().skip(1).map(|w|w.rows*w.cols).max().unwrap_or(0)}else{0};
        let head_bytes:u64=head.as_ref().map_or(0,|h|[&h.eh,&h.q,&h.k,&h.v,&h.o,&h.gate,&h.up,&h.down].iter().map(|w|(w.words.len() as u64*4).div_ceil(256)*256+w.scales.len() as u64*4).sum());
        let c = m.cfg.clone();
        let ctx = Ctx::new(device, 1024)?;
        // ---- params
        let mut p: Vec<f32> = Vec::new();
        let mut push = |v: &[f32]| -> u32 { while p.len() % 64 != 0 { p.push(0.0); } let o = p.len() as u32; p.extend_from_slice(v); o };
        let mut pl = vec![]; 
        for l in &m.layers {
            let mut q = PLayer { attn_norm: push(&l.attn_norm), post_norm: push(&l.post_norm), ..Default::default() };
            match &l.kind {
                cpu::Kind::Gdn { conv, dt, a, beta, alpha, norm, .. } => { q.conv_w = push(conv); q.dt = push(dt); q.a = push(a); q.beta_w = push(&beta.f32_vec()); q.alpha_w = push(&alpha.f32_vec()); q.ssm_norm = push(norm); }
                cpu::Kind::Attn { q_norm, k_norm, .. } => { q.q_norm = push(q_norm); q.k_norm = push(k_norm); }
            }
            pl.push(q);
        }
        let out_norm = push(&m.output_norm);
        let hn = head.as_ref().map(|h| [push(&h.enorm), push(&h.hnorm), push(&h.attn_norm), push(&h.post_norm), push(&h.q_norm), push(&h.k_norm), push(&h.head_norm)]);
        let mut signs = HashMap::new();
        if let Some(h) = &m.had { assert_eq!(h.block, 1024, "GPU hadamard kernel is specialised for block 1024"); for (w, s) in &h.signs { signs.insert(*w, push(s)); } }
        let params = ctx.buffer((p.len() * 4) as u64, false)?;
        ctx.upload(&params, 0, unsafe { std::slice::from_raw_parts(p.as_ptr() as *const u8, p.len() * 4) });
        // ---- arena
        let e = c.n_embd; let t = ubatch;
        let wide = [e, c.ssm_inner, c.n_ff, c.n_head * c.head_dim].into_iter().max().unwrap();
        let big = [c.conv_dim(), c.n_ff, 2 * c.n_head * c.head_dim].into_iter().max().unwrap();
        let mut off: u64 = 0;
        let mut al = |n: usize| -> u32 { let o = off; off += (n as u64 + 63) / 64 * 64; o as u32 };
        let mut lay = Lay { x: al(t * e), xn: al(t * e), hx: al(t * wide), big1: al(t * big), big2: al(t * big), conv: al(t * c.conv_dim()),
            beta: al(t * c.ssm_vheads), alpha: al(t * c.ssm_vheads), o: al(t * wide), scr: al(t*c.n_head*ctx_len.max(260)), logits: al(c.n_vocab + 64), ..Default::default() };
        if dense_words>0 {lay.dense=al(dense_words);lay.dense_words=dense_words;}
        let kvd = c.n_head_kv * c.head_dim;
        let kvn = if kv16 { ctx_len * kvd / 2 } else { ctx_len * kvd };
        if kv16 { lay.kvs = al(2 * t * kvd); }
        for il in 0..c.n_layer {
            if c.is_recurrent(il) { lay.conv_st.push(al((c.conv_k - 1) * c.conv_dim())); lay.ssm.push(al(c.ssm_vheads * c.ssm_state * c.ssm_state)); lay.kc.push(0); lay.vc.push(0); }
            else { lay.conv_st.push(0); lay.ssm.push(0); lay.kc.push(al(kvn)); lay.vc.push(al(kvn)); }
        }
        if head.is_some() {
            let kvd = kvd as usize;
            lay.hrow = al(t * e); lay.hprev = al(e); lay.mh = al(t * e); lay.me = al(t * e); lay.mcat = al(t * 2 * e); lay.mx = al(t * e);
            lay.mkc = al(ctx_len * kvd); lay.mvc = al(ctx_len * kvd); lay.logits2 = al(2 * (c.n_vocab + 64)); lay.amx = al(64);
            for il in 0..c.n_layer {
                if c.is_recurrent(il) { lay.ck_conv.push(al((c.conv_k - 1) * c.conv_dim())); lay.ck_ssm.push(al(c.ssm_vheads * c.ssm_state * c.ssm_state)); }
                else { lay.ck_conv.push(0); lay.ck_ssm.push(0); }
            }
        }
        lay.total = off * 4;
        if lay.total > ctx.props.limits.max_storage_buffer_range as u64 { return Err(format!("arena {} MiB exceeds maxStorageBufferRange; lower --ctx", lay.total >> 20)); }
        let arena = ctx.buffer(lay.total, false)?;
        if column_prefill || dense_words>0 {
            if ctx.props.device_id!=0x67a0 {return Err("column-prefill capacity guard requires W9100".into());}
            let ds:Vec<_>=std::fs::read_dir("/sys/class/drm").map_err(|e|e.to_string())?.filter_map(Result::ok).filter(|x|x.file_name().to_string_lossy().starts_with("card")).map(|x|x.path().join("device")).filter(|p|std::fs::read_to_string(p.join("device")).map_or(false,|x|x.trim()=="0x67a0")).collect();
            if ds.len()!=1 {return Err("column-prefill W9100 capacity identity missing/ambiguous".into());}
            let read=|n:&str|->Result<u64,String>{std::fs::read_to_string(ds[0].join(n)).map_err(|e|e.to_string())?.trim().parse::<u64>().map_err(|e|e.to_string())};
            let(total,used)=(read("mem_info_vram_total")?,read("mem_info_vram_used")?);
            let allocation=original_bytes.checked_add(column_bytes).and_then(|x|x.checked_add(head_bytes)).ok_or("column-prefill size overflow")?;
            if !prefill_pack::allocation_fits(total,used,allocation) {return Err(format!("column-prefill VRAM guard: total={total} used={used} additional={allocation}; need 1GiB reserve"));}
            eprintln!("[prefill-layout] {}",serde_json::json!({"layout":"column-quad","original_bytes":original_bytes,"additional_prefill_bytes":column_bytes,"head_bytes":head_bytes,"vram_total":total,"vram_used_before_weights":used,"reserve_bytes":1u64<<30}));
        }
        // ---- weights
        let wbytes = std::cell::Cell::new(0u64);
        let mut mat = |w: &WRef| -> Result<DevMat, String> {
            assert!(matches!(w.ty, GgmlType::PQ2_0 | GgmlType::PTQ1_0), "GPU path expects ternary weights");
            let pk = Packed2::from_ggml(w.ty, w.data, w.rows, w.cols);
            if std::env::var("BONSAI_PREFILL_TILE").map_or(false,|v|v=="h16") && !pk.scales.iter().all(|&d|lossless_weight_lds_scale(d)) { return Err(format!("lossless h16 LDS unsupported scale in {}x{}",w.rows,w.cols)); }
            let wb = (pk.words.len() * 4) as u64; let so = (wb + 255) / 256 * 256; let sb = (pk.scales.len() * 4) as u64;
            let buf = ctx.buffer(so + sb, false)?;
            ctx.upload(&buf, 0, unsafe { std::slice::from_raw_parts(pk.words.as_ptr() as *const u8, wb as usize) });
            ctx.upload(&buf, so, unsafe { std::slice::from_raw_parts(pk.scales.as_ptr() as *const u8, sb as usize) });
            wbytes.set(wbytes.get() + so + sb);
            let set = ctx.set([(&arena, 0, vk::WHOLE_SIZE), (&params, 0, vk::WHOLE_SIZE), (&buf, 0, wb), (&buf, so, sb)]);
            let prefill=if column_prefill && w.rows!=c.n_vocab {
                let(cw,cd)=prefill_pack::columns(w.rows,w.cols,&pk.words,&pk.scales)?;
                let wb=(cw.len()*4) as u64;let so=wb.div_ceil(256)*256;let sb=(cd.len()*4) as u64;
                let b=ctx.buffer(so+sb,false)?;
                ctx.upload(&b,0,unsafe{std::slice::from_raw_parts(cw.as_ptr() as *const u8,wb as usize)});
                ctx.upload(&b,so,unsafe{std::slice::from_raw_parts(cd.as_ptr() as *const u8,sb as usize)});
                let s=ctx.set([(&arena,0,vk::WHOLE_SIZE),(&params,0,vk::WHOLE_SIZE),(&b,0,wb),(&b,so,sb)]);
                wbytes.set(wbytes.get()+so+sb);Some((b,s))
            }else{None};
            Ok(DevMat { rows: w.rows as u32, cols: w.cols as u32, buf, set, prefill })
        };
        let mut layers = vec![];
        for (il, l) in m.layers.iter().enumerate() {
            let kind = match &l.kind {
                cpu::Kind::Gdn { qkv, gate, out, .. } => DLayer::Gdn { qkv: mat(qkv)?, gate: mat(gate)?, out: mat(out)? },
                cpu::Kind::Attn { q, k, v, o, .. } => DLayer::Attn { q: mat(q)?, k: mat(k)?, v: mat(v)?, o: mat(o)? },
            };
            layers.push(DL { kind, gate: mat(&l.ffn_gate)?, up: mat(&l.ffn_up)?, down: mat(&l.ffn_down)? });
            if il % 8 == 7 { eprintln!("[vk] uploaded {} layers ({:.1} GiB)", il + 1, wbytes.get() as f64 / (1u64 << 30) as f64); }
        }
        let output = mat(&m.output)?;
        let qmat = |w: &mut crate::quant::Q8G16| -> Result<DevMat, String> {
            let wb = (w.words.len() * 4) as u64; let so = (wb + 255) / 256 * 256; let sb = (w.scales.len() * 4) as u64;
            let buf = ctx.buffer(so + sb, false)?;
            ctx.upload(&buf, 0, unsafe { std::slice::from_raw_parts(w.words.as_ptr() as *const u8, wb as usize) });
            ctx.upload(&buf, so, unsafe { std::slice::from_raw_parts(w.scales.as_ptr() as *const u8, sb as usize) });
            wbytes.set(wbytes.get() + so + sb);
            let set = ctx.set([(&arena, 0, vk::WHOLE_SIZE), (&params, 0, vk::WHOLE_SIZE), (&buf, 0, wb), (&buf, so, sb)]);
            let d = DevMat { rows: w.rows as u32, cols: w.cols as u32, buf, set, prefill:None };
            w.words = Vec::new(); w.scales = Vec::new(); // host copy no longer needed
            Ok(d)
        };
        let gm = match (head.as_mut(), hn) {
            (Some(h), Some(n)) => Some(GMtp { eh: qmat(&mut h.eh)?, q: qmat(&mut h.q)?, k: qmat(&mut h.k)?, v: qmat(&mut h.v)?, o: qmat(&mut h.o)?, gate: qmat(&mut h.gate)?, up: qmat(&mut h.up)?, down: qmat(&mut h.down)?,
                enorm: n[0], hnorm: n[1], attn_norm: n[2], post_norm: n[3], q_norm: n[4], k_norm: n[5], head_norm: n[6] }),
            _ => None,
        };
        if gm.is_some() { eprintln!("[vk] MTP head attached (int8-group matrices, block {})", head.as_ref().unwrap().il); }
        let set0 = output.set;
        eprintln!("[vk] weights {:.2} GiB, arena {:.2} GiB, ctx {ctx_len}, load {:.1}s", wbytes.get() as f64 / (1u64 << 30) as f64, lay.total as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64());
        let mut s = VkEngine { ctx, cpu: m, arena, params, lay, pl, out_norm, signs, layers, output, set0, pos: 0, ctx_len, timing: std::env::var("BONSAI_TIMING").is_ok(), kv16, ubatch,
            mtp: gm.zip(head), mpos: 1, hp: (0, 0) };
        s.zero_state();
        if std::env::var_os("BONSAI_REVIEW_TRACE").is_some() {
            eprintln!("[api-review] {}", serde_json::json!({"event":"engine_constructed", "pid":std::process::id(), "engine":s.review_identity(), "weights_bytes":wbytes.get(), "arena_bytes":s.lay.total}));
        }
        Ok(s)
    }

    pub fn review_identity(&self) -> serde_json::Value {
        serde_json::json!({"device":format!("0x{:04x}",self.ctx.props.device_id), "rust":self.ctx.rust,
            "layers":self.cpu.cfg.n_layer, "arena":format!("{:?}",self.arena.buf), "arena_bytes":self.lay.total,"prefill_batch":self.ubatch,
            "head_loaded":self.has_mtp(), "context_capacity":self.ctx_len, "kv_dtype":if self.kv16 {"f16"} else {"f32"}, "completed_dispatches":self.ctx.completed_dispatches.get(), "completed_q8":self.ctx.completed_q8.get(), "position":self.pos})
    }

    fn zero_state(&mut self) {
        let start = std::time::Instant::now();
        self.ctx.begin();
        unsafe { self.ctx.device.cmd_fill_buffer(self.ctx.cur.get(), self.arena.buf, 0, vk::WHOLE_SIZE, 0); }
        self.ctx.submit_wait();
        self.pos = 0; self.mpos = 1; self.hp = (0, 0);
        if std::env::var_os("BONSAI_REVIEW_TRACE").is_some() { eprintln!("[api-review] {}", serde_json::json!({"event":"reset", "pid":std::process::id(), "seconds":start.elapsed().as_secs_f64()})); }
    }

    fn mv(&self, w: &DevMat, ntok: u32, xo: u32, xs: u32, yo: u32, ys: u32, acc: bool) {
        assert!(xo % 4 == 0 && (ntok == 1 || xs % 4 == 0), "activation offsets must be vec4 aligned");
        if ntok >= tile_min() {
            let tile=std::env::var("BONSAI_PREFILL_TILE").unwrap_or_else(|_| "old".into());
            if tile=="dense" && self.ctx.rust && w.rows%4==0 && w.rows as usize*w.cols as usize<=self.lay.dense_words {
                // Shared expansion scratch cannot participate in independent-mv grouping.
                self.ctx.barrier();self.ctx.expand_prefill(w.set,w.rows,w.cols,self.lay.dense);self.ctx.barrier();
                self.ctx.dispatch("matmul_dense64",w.set,&[w.rows,w.cols,ntok,xo,xs,yo,ys,acc as u32,self.lay.dense],(w.rows.div_ceil(64),ntok.div_ceil(64),1));
                self.ctx.barrier();return;
            }
            if matches!(tile.as_str(),"col"|"col16"|"col8"|"col128u4"|"col128u8") {
                let(pipe,tokens)=match tile.as_str(){"col16"=>("matmul_col16",64),"col8"=>("matmul_col8",64),"col128u4"=>("matmul_col128u4",128),"col128u8"=>("matmul_col128u8",128),_=>("matmul_col",64)};
                let (_,set)=w.prefill.as_ref().expect("column-prefill matrix must be planned at load");
                // Retain screened <=256-token dispatch bounds even with larger ubatches.
                let peel=std::env::var("BONSAI_PREFILL_PEEL").map_or(false,|v|v=="1");
                let policy=std::env::var("BONSAI_PREFILL_MATMUL").unwrap_or_else(|_|"base".into());
                self.ctx.group(||{for (start,n) in column_prefill_segments(ntok,peel) {
                    if peel && n<8 && ntok>64 {self.mv(w,n,xo+start*xs,xs,yo+start*ys,ys,acc);}
                    else {let(p,rows,toks)=if tile=="col" {prefill_column_kernel(w.rows,w.cols,n,&policy)}else{(pipe,64,tokens)};
                        self.ctx.dispatch(p,*set,&[w.rows,w.cols,n,xo+start*xs,xs,yo+start*ys,ys,acc as u32],(w.rows.div_ceil(rows),n.div_ceil(toks),1));}
                }});return;
            }
            let (pipe,rows,toks)=match tile.as_str() { "rb64" if self.ctx.rust => ("matmul_rb64",64,64), "rb128" if self.ctx.rust => ("matmul_rb128",128,128), "r64" if self.ctx.rust => ("matmul_r64",64,64), "r128" if self.ctx.rust => ("matmul_r128",128,128), "ru8" if self.ctx.rust => ("matmul_ru8",64,64), "pf64" if self.ctx.rust => ("matmul_pf64",64,64), "pf128" if self.ctx.rust => ("matmul_pf128",128,64), "pf128x32" if self.ctx.rust => ("matmul_pf128x32",128,32), "pf64x32" if self.ctx.rust => ("matmul_pf64x32",64,32), "pf64k64" if self.ctx.rust => ("matmul_pf64k64",64,64), "pf512" if self.ctx.rust => ("matmul_pf512",64,64), "h16" if self.ctx.rust => ("matmul_pf_h16",64,64), _ => ("matmul_t2",64,64) };
            self.ctx.dispatch(pipe, w.set, &[w.rows, w.cols, ntok, xo, xs, yo, ys, acc as u32], (w.rows.div_ceil(rows), ntok.div_ceil(toks), 1));
            return;
        }
        // r012: 2..tile_min tokens (MTP verify, short prompts): 2 tokens share every weight load.
        if ntok >= 2 && self.ctx.rust && std::env::var("BONSAI_MV").map_or(true, |v| v != "t2_1") {
            let groups = w.rows.div_ceil(8); let gx = groups.min(65535); let gy = groups.div_ceil(gx);
            self.ctx.dispatch("matvec_c8x2", w.set, &[w.rows, w.cols, ntok, xo, xs, yo, ys, acc as u32], (gx, gy, ntok.div_ceil(2)));
            return;
        }
        let small = std::env::var("BONSAI_SMALL").unwrap_or_else(|_| "mv1".into());
        if ntok >= 2 && small == "mv1" {
            self.ctx.dispatch("matvec_t2_1", w.set, &[w.rows, w.cols, ntok, xo, xs, yo, ys, acc as u32], (ntok, w.rows.div_ceil(8), 1));
            return;
        }
        if ntok >= 2 && small == "t2s" && std::env::var("BONSAI_NO_SMALL").is_err() {
            self.ctx.dispatch("matmul_t2s", w.set, &[w.rows, w.cols, ntok, xo, xs, yo, ys, acc as u32], (w.rows.div_ceil(128), ntok.div_ceil(8), 1));
            return;
        }
        // r012: coalesced 8-row decode matvec (~1.8x the bandwidth of matvec_t2_1 on the W9100); BONSAI_MV=t2_1 restores the old path.
        if ntok == 1 && self.ctx.rust && std::env::var("BONSAI_MV").map_or(true, |v| v != "t2_1") {
            let groups = w.rows.div_ceil(8); let gx = groups.min(65535); let gy = groups.div_ceil(gx);
            self.ctx.dispatch("matvec_c8", w.set, &[w.rows, w.cols, 1, xo, xs, yo, ys, acc as u32], (gx, gy, 1));
            return;
        }
        let pipe = if ntok == 1 && std::env::var("BONSAI_NO_T1").is_err() { "matvec_t2_1" } else { "matvec_t2" };
        let rpg = if pipe == "matvec_t2_1" && rust_kernels() { 8 } else if pipe == "matvec_t2_1" { std::env::var("BONSAI_MV_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(8) } else { 4 };
        let groups = w.rows.div_ceil(rpg); let gx = groups.min(65535); let gy = groups.div_ceil(gx);
        self.ctx.dispatch(pipe, w.set, &[w.rows, w.cols, ntok, xo, xs, yo, ys, acc as u32], (gx, gy, if pipe == "matvec_t2" { ntok.div_ceil(8) } else { 1 }));
    }
    /// Causal GQA attention + sigmoid gate into o. r012: split-K partials + combine (head_dim 256, rust
    /// kernels); BONSAI_ATTN=old keeps the single-workgroup-per-(head, token) kernel.
    fn attention(&self, qo: u32, qts: u32, kco: u32, vco: u32, kvd: u32, pos0: u32, hd: u32, nh: u32, nkv: u32, oo: u32, nt: u32) {
        self.attention_kv(qo, qts, kco, vco, kvd, pos0, hd, nh, nkv, oo, nt, false)
    }
    fn attention_kv(&self, qo: u32, qts: u32, kco: u32, vco: u32, kvd: u32, pos0: u32, hd: u32, nh: u32, nkv: u32, oo: u32, nt: u32, f16: bool) {
        let scale = fb(1.0 / (hd as f32).sqrt()); let l = &self.lay;
        if f16 {
            assert!(hd == 256 && qo % 4 == 0 && kco % 4 == 0 && vco % 4 == 0 && l.scr % 4 == 0, "f16 KV layout");
            let room = attention_dispatch_room(self.ctx_len,nh as usize,nt as usize) as u32; let total = pos0 + nt;
            let mut c = 64u32; while c < 1024 && nt * nh * total.div_ceil(c) * 260 > room { c *= 2; }
            let smax = total.div_ceil(c); assert!(nt * nh * smax * 260 <= room, "attention scratch");
            let mode=std::env::var("BONSAI_PREFILL_ATTN").unwrap_or_else(|_|"old".into());
            let(pipe,queries,heads)=select_prefill_gqa(nt,c,pos0,nh,nkv,hd,&mode);
            if heads>1 {
                // Never enlarge the screened64-query GQA geometry for batch512.
                self.ctx.group(||{for start in (0..nt).step_by(64) {let n=(nt-start).min(64);
                    self.ctx.dispatch(pipe,self.set0,&[qo+start*qts,qts,kco,vco,kvd,pos0+start,hd,nh,nkv,l.scr+start*nh*smax*260,c,smax,scale,n],(nh/heads,n,smax));
                }});
            } else {
                self.ctx.dispatch(pipe, self.set0, &[qo, qts, kco, vco, kvd, pos0, hd, nh, nkv, l.scr, c, smax, scale, nt], (nh, nt.div_ceil(queries), smax));
            }
            self.ctx.dispatch("attn_comb", self.set0, &[qo, qts, pos0, nh, l.scr, c, smax, oo, nh * hd], (nh, nt, 1));
            return;
        }
        if hd == 256 && self.ctx.rust && std::env::var("BONSAI_ATTN").map_or(true, |v| v != "old") && qo % 4 == 0 && kco % 4 == 0 && vco % 4 == 0 && l.scr % 4 == 0 {
            let room = attention_dispatch_room(self.ctx_len,nh as usize,nt as usize) as u32; let total = pos0 + nt;
            let mut c = 64u32; while c < 1024 && nt * nh * total.div_ceil(c) * 260 > room { c *= 2; }
            let smax = total.div_ceil(c);
            if nt * nh * smax * 260 <= room {
                self.ctx.dispatch("attn_s", self.set0, &[qo, qts, kco, vco, kvd, pos0, hd, nh, nkv, l.scr, c, smax, scale], (nh, nt, smax));
                self.ctx.dispatch("attn_comb", self.set0, &[qo, qts, pos0, nh, l.scr, c, smax, oo, nh * hd], (nh, nt, 1));
                return;
            }
        }
        self.ctx.dispatch("attn", self.set0, &[qo, qts, kco, vco, kvd, pos0, hd, nh, nkv, oo, nh * hd, l.scr, self.ctx_len as u32, scale], (nh, nt, 1));
    }
    fn mvq(&self, w: &DevMat, ntok: u32, xo: u32, xs: u32, yo: u32, ys: u32, acc: bool) {
        assert!(xo % 4 == 0 && (ntok == 1 || xs % 4 == 0) && w.rows % 4 == 0 && w.cols % 16 == 0);
        self.ctx.dispatch("matvec_q8", w.set, &[w.rows, w.cols, ntok, xo, xs, yo, ys, acc as u32], (ntok, w.rows / 4, 1));
    }
    fn copy(&self, src: u32, dst: u32, n: u32) {
        unsafe { self.ctx.device.cmd_copy_buffer(self.ctx.cur.get(), self.arena.buf, self.arena.buf, &[vk::BufferCopy { src_offset: src as u64 * 4, dst_offset: dst as u64 * 4, size: n as u64 * 4 }]); }
    }
    fn had(&self, n: usize, ntok: u32, io: u32, its: u32, oo: u32, ots: u32, perm: Option<(u32, u32, u32)>, inverse: bool) {
        let so = self.signs.get(&n).copied().unwrap_or(NONE);
        let (hd, nk, rep) = perm.unwrap_or((0, 0, 0));
        self.ctx.dispatch("hadamard", self.set0, &[n as u32, io, its, oo, ots, so, inverse as u32, hd, nk, rep], ((n / 1024) as u32, ntok, 1));
    }
    #[allow(clippy::too_many_arguments)]
    fn norm(&self, n: usize, rows: u32, ntok: u32, io: u32, irs: u32, its: u32, oo: u32, ors: u32, ots: u32, w: u32, mode: u32, z: (u32, u32, u32)) {
        self.ctx.dispatch("rmsnorm", self.set0, &[n as u32, io, irs, its, oo, ors, ots, w, mode, z.0, z.1, z.2, fb(self.cpu.cfg.eps)], (rows, ntok, 1));
    }

    /// Record and run one chunk of <= MAXT tokens; one submit, one wait.
    fn run_chunk(&mut self, toks: &[u32], logits: bool, argmax_only: bool) -> Option<Vec<f32>> { self.run_chunk_v(toks, logits, argmax_only, false) }
    /// `verify` (MTP, 2 tokens): recurrent kernels run per token exactly as in single-token decode, the recurrent
    /// state after token 0 is checkpointed, and both tokens' argmaxes are returned.
    fn run_chunk_v(&mut self, toks: &[u32], logits: bool, argmax_only: bool, verify: bool) -> Option<Vec<f32>> {
        let c = self.cpu.cfg.clone(); let l = self.lay.clone(); let nt = toks.len() as u32; let e = c.n_embd as u32;
        assert!(self.pos + toks.len() <= self.ctx_len, "context full");
        // embeddings (latent rows + inverse Hadamard) on the CPU: 5120 floats/token
        let mut emb = Vec::with_capacity(toks.len() * c.n_embd);
        for &t in toks { emb.extend(self.cpu.embed(t)); }
        unsafe { std::ptr::copy_nonoverlapping(emb.as_ptr() as *const u8, self.ctx.staging.ptr, emb.len() * 4); }
        let cx = &self.ctx;
        cx.begin();
        unsafe { cx.device.cmd_copy_buffer(cx.cur.get(), cx.staging.buf, self.arena.buf, &[vk::BufferCopy { src_offset: 0, dst_offset: l.x as u64 * 4, size: (emb.len() * 4) as u64 }]); }
        cx.barrier();
        let pos0 = self.pos as u32;
        let (hd, nh, nkv) = (c.head_dim as u32, c.n_head as u32, c.n_head_kv as u32); let kvd = hd * nkv;
        let wide = 0; let _ = wide;
        for (il, dl) in self.layers.iter().enumerate() {
            let p = &self.pl[il];
            self.norm(c.n_embd, 1, nt, l.x, 0, e, l.xn, 0, e, p.attn_norm, 0, (0, 0, 0));
            match &dl.kind {
                DLayer::Gdn { qkv, gate, out } => {
                    let cd = c.conv_dim() as u32; let inner = c.ssm_inner as u32; let nv = c.ssm_vheads as u32; let sd = c.ssm_state as u32;
                    self.had(c.n_embd, nt, l.xn, e, l.hx, e, None, false);
                    cx.group(|| {
                        self.mv(qkv, nt, l.hx, e, l.big1, cd, false);
                        self.mv(gate, nt, l.hx, e, l.big2, inner, false);
                        cx.dispatch("matvec_f32", self.set0, &[nv, e, nt, l.xn, e, l.beta, nv, p.beta_w], (nv, nt, 1));
                        cx.dispatch("matvec_f32", self.set0, &[nv, e, nt, l.xn, e, l.alpha, nv, p.alpha_w], (nv, nt, 1));
                    });
                    if verify {
                        let csz = (c.conv_k as u32 - 1) * cd; let ssz = nv * sd * sd;
                        for t in 0..nt {
                            cx.dispatch("conv", self.set0, &[cd, 1, l.big1 + t * cd, cd, l.conv + t * cd, cd, p.conv_w, l.conv_st[il], c.conv_k as u32], (cd.div_ceil(256), 1, 1));
                            if t == 0 { self.copy(l.conv_st[il], l.ck_conv[il], csz); cx.barrier(); }
                        }
                        self.norm(sd as usize, 2 * c.ssm_groups as u32, nt, l.conv, sd, cd, l.conv, sd, cd, NONE, 2, (0, 0, 0));
                        for t in 0..nt {
                            let gpc = [1, l.conv + t * cd, cd, l.beta + t * nv, l.alpha + t * nv, nv, p.dt, p.a, l.ssm[il], l.o + t * inner, inner, c.ssm_groups as u32, sd, c.qk_dim() as u32];
                            cx.dispatch("gdn", self.set0, &gpc, (nv, 1, 1));
                            if t == 0 { self.copy(l.ssm[il], l.ck_ssm[il], ssz); cx.barrier(); }
                        }
                    } else {
                    cx.dispatch("conv", self.set0, &[cd, nt, l.big1, cd, l.conv, cd, p.conv_w, l.conv_st[il], c.conv_k as u32], (cd.div_ceil(256), 1, 1));
                    self.norm(sd as usize, 2 * c.ssm_groups as u32, nt, l.conv, sd, cd, l.conv, sd, cd, NONE, 2, (0, 0, 0));
                    let gpc = [nt, l.conv, cd, l.beta, l.alpha, nv, p.dt, p.a, l.ssm[il], l.o, inner, c.ssm_groups as u32, sd, c.qk_dim() as u32];
                    if nt > 1 && sd == 128 && std::env::var("BONSAI_OLD_GDN").is_err() { cx.dispatch("gdn_p", self.set0, &gpc, (nv, 2, 1)); } else { cx.dispatch("gdn", self.set0, &gpc, (nv, 1, 1)); }
                    }
                    self.norm(sd as usize, nv, nt, l.o, sd, inner, l.o, sd, inner, p.ssm_norm, 1, (l.big2, sd, inner));
                    let perm = if c.gdn_v_grouped { Some((sd, c.ssm_groups as u32, nv / c.ssm_groups as u32)) } else { None };
                    self.had(c.ssm_inner, nt, l.o, inner, l.hx, inner, perm, false);
                    self.mv(out, nt, l.hx, inner, l.x, e, true);
                }
                DLayer::Attn { q, k, v, o } => {
                    let qf = 2 * nh * hd;
                    let (kc, vc) = if self.kv16 { (l.kvs, l.kvs + self.ubatch as u32 * kvd) } else { (l.kc[il] + pos0 * kvd, l.vc[il] + pos0 * kvd) };
                    self.had(c.n_embd, nt, l.xn, e, l.hx, e, None, false);
                    cx.group(|| {
                        self.mv(q, nt, l.hx, e, l.big1, qf, false);
                        self.mv(k, nt, l.hx, e, kc, kvd, false);
                        self.mv(v, nt, l.hx, e, vc, kvd, false);
                    });
                    cx.group(|| {
                        self.norm(c.head_dim, nh, nt, l.big1, 2 * hd, qf, l.big1, 2 * hd, qf, p.q_norm, 0, (0, 0, 0));
                        self.norm(c.head_dim, nkv, nt, kc, hd, kvd, kc, hd, kvd, p.k_norm, 0, (0, 0, 0));
                    });
                    let ts = (c.rope_base).powf(-2.0 / c.n_rot as f32); let s = c.rope_sections;
                    let half = c.n_rot as u32 / 2;
                    cx.group(|| {
                        cx.dispatch("rope", self.set0, &[l.big1, 2 * hd, qf, nh, nt, pos0, c.n_rot as u32, fb(ts), s[0] as u32, s[1] as u32, s[2] as u32, s[3] as u32], ((nh * half).div_ceil(64), nt, 1));
                        cx.dispatch("rope", self.set0, &[kc, hd, kvd, nkv, nt, pos0, c.n_rot as u32, fb(ts), s[0] as u32, s[1] as u32, s[2] as u32, s[3] as u32], ((nkv * half).div_ceil(64), nt, 1));
                    });
                    if self.kv16 {
                        let n = nt * kvd; let g = (n / 2).div_ceil(64);
                        cx.group(|| {
                            cx.dispatch("kv_pack", self.set0, &[kc, l.kc[il] + pos0 * kvd / 2, n], (g, 1, 1));
                            cx.dispatch("kv_pack", self.set0, &[vc, l.vc[il] + pos0 * kvd / 2, n], (g, 1, 1));
                        });
                    }
                    self.attention_kv(l.big1, qf, l.kc[il], l.vc[il], kvd, pos0, hd, nh, nkv, l.o, nt, self.kv16);
                    self.had(c.n_head * c.head_dim, nt, l.o, nh * hd, l.hx, nh * hd, None, false);
                    self.mv(o, nt, l.hx, nh * hd, l.x, e, true);
                }
            }
            let ff = c.n_ff as u32;
            self.norm(c.n_embd, 1, nt, l.x, 0, e, l.xn, 0, e, p.post_norm, 0, (0, 0, 0));
            self.had(c.n_embd, nt, l.xn, e, l.hx, e, None, false);
            cx.group(|| {
                self.mv(&dl.gate, nt, l.hx, e, l.big1, ff, false);
                self.mv(&dl.up, nt, l.hx, e, l.big2, ff, false);
            });
            cx.dispatch("ewise", self.set0, &[1, ff, nt, l.big1, ff, l.big2, ff, l.big1, ff], (ff.div_ceil(256), nt, 1));
            self.had(c.n_ff, nt, l.big1, ff, l.hx, ff, None, false);
            self.mv(&dl.down, nt, l.hx, ff, l.x, e, true);
            // keep each queue submission short for multi-token chunks (amdgpu job timeout)
            let every = if nt >= 8 { 1 } else if nt > 1 { 4 } else { 0 };
            if every > 0 && (il + 1) % every == 0 && il + 1 < self.layers.len() { cx.split(); }
        }
        if self.mtp.is_some() {
            // h_nextn rows for the MTP head: output-normed hidden state of every token of this pass
            self.norm(c.n_embd, 1, nt, l.x, 0, e, l.hrow, 0, e, self.out_norm, 0, (0, 0, 0));
        }
        if verify {
            let nvp = c.n_vocab as u32 + 64;
            self.had(c.n_embd, nt, l.hrow, e, l.hx, e, None, false);
            self.mv(&self.output, nt, l.hx, e, l.logits2, nvp, false);
            cx.group(|| { for t in 0..nt { cx.dispatch("argmax", self.set0, &[l.logits2 + t * nvp, c.n_vocab as u32, l.amx + t], (1, 1, 1)); } });
            unsafe { cx.device.cmd_copy_buffer(cx.cur.get(), self.arena.buf, cx.staging.buf, &[vk::BufferCopy { src_offset: l.amx as u64 * 4, dst_offset: 0, size: nt as u64 * 4 }]); }
        } else if logits {
            let last = l.x + (nt - 1) * e;
            self.norm(c.n_embd, 1, 1, last, 0, e, l.xn, 0, e, self.out_norm, 0, (0, 0, 0));
            self.had(c.n_embd, 1, l.xn, e, l.hx, e, None, false);
            self.mv(&self.output, 1, l.hx, e, l.logits, 0, false);
            if argmax_only {
                let slot = l.logits + c.n_vocab as u32 + 4; // scratch slot just past the logits (arena allocs are 64-padded)
                cx.dispatch("argmax", self.set0, &[l.logits, c.n_vocab as u32, slot], (1, 1, 1));
                unsafe { cx.device.cmd_copy_buffer(cx.cur.get(), self.arena.buf, cx.staging.buf, &[vk::BufferCopy { src_offset: slot as u64 * 4, dst_offset: 0, size: 4 }]); }
            } else {
                unsafe { cx.device.cmd_copy_buffer(cx.cur.get(), self.arena.buf, cx.staging.buf, &[vk::BufferCopy { src_offset: l.logits as u64 * 4, dst_offset: 0, size: (c.n_vocab * 4) as u64 }]); }
            }
        }
        let t = std::time::Instant::now();
        cx.submit_wait();
        cx.collect_profile();
        if self.timing { eprintln!("[vk] chunk {nt} tok: gpu+submit {:.1} ms", t.elapsed().as_secs_f64() * 1e3); }
        self.pos += toks.len();
        if verify { let mut u = vec![0u32; nt as usize]; unsafe { std::ptr::copy_nonoverlapping(cx.staging.ptr as *const u32, u.as_mut_ptr(), nt as usize); } return Some(u.into_iter().map(f32::from_bits).collect()); }
        if !logits { return None; }
        if argmax_only { let mut u = 0u32; unsafe { std::ptr::copy_nonoverlapping(cx.staging.ptr as *const u32, &mut u, 1); } return Some(vec![f32::from_bits(u)]); }
        let mut v = vec![0f32; c.n_vocab];
        unsafe { std::ptr::copy_nonoverlapping(cx.staging.ptr as *const f32, v.as_mut_ptr(), c.n_vocab); }
        Some(v)
    }

    pub fn eval_tokens(&mut self, toks: &[u32], want: bool) -> Option<Vec<f32>> {
        let n = toks.len(); let mut r = None;
        for range in prefill_chunks(n,self.ubatch) {let last=range.end==n;r=self.run_chunk(&toks[range],want && last,false);}
        r
    }
    /// greedy step with GPU argmax: reads back one u32
    pub fn eval_argmax(&mut self, toks: &[u32]) -> u32 {
        let n = toks.len(); let mut r = None;
        for range in prefill_chunks(n,self.ubatch) {let last=range.end==n;r=self.run_chunk(&toks[range],last,true);}
        r.unwrap()[0].to_bits()
    }
    /// One MTP head pass over `toks` at positions pos0.. paired with h rows at arena offset `hoff`; returns the
    /// argmax of the last row (the draft).
    fn mtp_pass(&mut self, toks: &[u32], pos0: u32, hoff: u32) -> u32 {
        let c = self.cpu.cfg.clone(); let l = self.lay.clone(); let nt = toks.len() as u32; let e = c.n_embd as u32;
        assert!(toks.len() <= MAXT && pos0 >= 1 && (pos0 as usize) + toks.len() <= self.ctx_len);
        let (gm, hd) = self.mtp.as_ref().unwrap();
        let mut emb = Vec::with_capacity(toks.len() * c.n_embd);
        for &t in toks { emb.extend(hd.embed(&self.cpu, t)); }
        let cx = &self.ctx;
        unsafe { std::ptr::copy_nonoverlapping(emb.as_ptr() as *const u8, cx.staging.ptr, emb.len() * 4); }
        cx.begin();
        unsafe { cx.device.cmd_copy_buffer(cx.cur.get(), cx.staging.buf, self.arena.buf, &[vk::BufferCopy { src_offset: 0, dst_offset: l.me as u64 * 4, size: (emb.len() * 4) as u64 }]); }
        cx.barrier();
        let (hdm, nh, nkv) = (c.head_dim as u32, c.n_head as u32, c.n_head_kv as u32); let kvd = hdm * nkv; let qf = 2 * nh * hdm; let ff = c.n_ff as u32;
        self.norm(c.n_embd, 1, nt, l.me, 0, e, l.mcat, 0, 2 * e, gm.enorm, 0, (0, 0, 0));
        self.norm(c.n_embd, 1, nt, hoff, 0, e, l.mcat + e, 0, 2 * e, gm.hnorm, 0, (0, 0, 0));
        self.mvq(&gm.eh, nt, l.mcat, 2 * e, l.mx, e, false);
        self.norm(c.n_embd, 1, nt, l.mx, 0, e, l.xn, 0, e, gm.attn_norm, 0, (0, 0, 0));
        let slot0 = pos0 - 1; let kc = l.mkc + slot0 * kvd; let vc = l.mvc + slot0 * kvd;
        self.mvq(&gm.q, nt, l.xn, e, l.big1, qf, false);
        self.mvq(&gm.k, nt, l.xn, e, kc, kvd, false);
        self.mvq(&gm.v, nt, l.xn, e, vc, kvd, false);
        self.norm(c.head_dim, nh, nt, l.big1, 2 * hdm, qf, l.big1, 2 * hdm, qf, gm.q_norm, 0, (0, 0, 0));
        self.norm(c.head_dim, nkv, nt, kc, hdm, kvd, kc, hdm, kvd, gm.k_norm, 0, (0, 0, 0));
        let ts = (c.rope_base).powf(-2.0 / c.n_rot as f32); let s = c.rope_sections; let half = c.n_rot as u32 / 2;
        cx.dispatch("rope", self.set0, &[l.big1, 2 * hdm, qf, nh, nt, pos0, c.n_rot as u32, fb(ts), s[0] as u32, s[1] as u32, s[2] as u32, s[3] as u32], ((nh * half).div_ceil(64), nt, 1));
        cx.dispatch("rope", self.set0, &[kc, hdm, kvd, nkv, nt, pos0, c.n_rot as u32, fb(ts), s[0] as u32, s[1] as u32, s[2] as u32, s[3] as u32], ((nkv * half).div_ceil(64), nt, 1));
        self.attention(l.big1, qf, l.mkc, l.mvc, kvd, slot0, hdm, nh, nkv, l.o, nt);
        self.mvq(&gm.o, nt, l.o, nh * hdm, l.mx, e, true);
        self.norm(c.n_embd, 1, nt, l.mx, 0, e, l.xn, 0, e, gm.post_norm, 0, (0, 0, 0));
        self.mvq(&gm.gate, nt, l.xn, e, l.big1, ff, false);
        self.mvq(&gm.up, nt, l.xn, e, l.big2, ff, false);
        cx.dispatch("ewise", self.set0, &[1, ff, nt, l.big1, ff, l.big2, ff, l.big1, ff], (ff.div_ceil(256), nt, 1));
        self.mvq(&gm.down, nt, l.big1, ff, l.mx, e, true);
        self.norm(c.n_embd, 1, 1, l.mx + (nt - 1) * e, 0, e, l.xn, 0, e, gm.head_norm, 0, (0, 0, 0));
        self.had(c.n_embd, 1, l.xn, e, l.hx, e, None, false);
        self.mv(&self.output, 1, l.hx, e, l.logits, 0, false);
        let slot = l.logits + c.n_vocab as u32 + 4;
        cx.dispatch("argmax", self.set0, &[l.logits, c.n_vocab as u32, slot], (1, 1, 1));
        unsafe { cx.device.cmd_copy_buffer(cx.cur.get(), self.arena.buf, cx.staging.buf, &[vk::BufferCopy { src_offset: slot as u64 * 4, dst_offset: 0, size: 4 }]); }
        cx.submit_wait();
        cx.collect_profile();
        let mut u = 0u32; unsafe { std::ptr::copy_nonoverlapping(cx.staging.ptr as *const u32, &mut u, 1); }
        u
    }
    /// Independent regression readback of ACTIVE recurrent and KV state only. Rejected KV
    /// suffix slots are logically invalidated by pos; subsequent writes overwrite them.
    pub fn review_state(&self, head: bool) -> Vec<f32> {
        let c=&self.cpu.cfg; let l=&self.lay; let mut out=vec![];
        let mut add=|off:u32,n:usize,packed:bool| { let mut b=vec![0u8;n*if packed {2} else {4}]; if n>0 { self.ctx.download(&self.arena,off as u64*4,&mut b); } out.extend(decode_kv_readback(&b,packed)); };
        let kvd=c.n_head_kv*c.head_dim;
        if head { add(l.mkc,self.mpos.saturating_sub(1)*kvd,false); add(l.mvc,self.mpos.saturating_sub(1)*kvd,false); }
        else { for il in 0..c.n_layer { if c.is_recurrent(il) {add(l.conv_st[il],(c.conv_k-1)*c.conv_dim(),false);add(l.ssm[il],c.ssm_vheads*c.ssm_state*c.ssm_state,false);} else { add(l.kc[il],self.pos*kvd,self.kv16);add(l.vc[il],self.pos*kvd,self.kv16);} } }
        out
    }
    pub fn has_mtp(&self) -> bool { self.mtp.is_some() }
    pub fn reset(&mut self) { self.zero_state(); }
    /// Read back a slice of the arena (debug / tests)
    pub fn read_x(&self) -> Vec<f32> { let mut b = vec![0u8; self.cpu.cfg.n_embd * 4]; self.ctx.download(&self.arena, self.lay.x as u64 * 4, &mut b); b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
}

impl crate::mtp::MtpBackend for VkEngine<'_> {
    fn mtp_single(&mut self, x: u32) -> u32 { self.eval_argmax(&[x]) }
    fn mtp_prefill(&mut self, prompt: &[u32]) -> u32 {
        crate::mtp::MtpBackend::reset_all(self);
        let e = self.cpu.cfg.n_embd as u32; let l = self.lay.clone(); let n = prompt.len();
        let mut next = 0;
        for range in prefill_chunks(n,self.ubatch) {
            let s=range.start;let last=range.end==n;let ch=&prompt[range];
            let r = self.run_chunk(ch, last, true);
            if last { next = r.unwrap()[0].to_bits(); }
            // MTP catch-up: pairs (h_{q-1}, x_q) for q in s..s+len, q >= 1
            let nt = ch.len() as u32;
            self.ctx.begin();
            if s > 0 { self.copy(l.hprev, l.mh, e); }
            if nt > 1 { self.copy(l.hrow, l.mh + e, (nt - 1) * e); }
            self.copy(l.hrow + (nt - 1) * e, l.hprev, e);
            self.ctx.barrier(); self.ctx.submit_wait();
            let (toks, pos0, hoff) = if s == 0 { (&ch[1..], 1u32, l.mh + e) } else { (ch, s as u32, l.mh) };
            if !toks.is_empty() { self.mtp_pass(toks, pos0, hoff); }
        }
        self.mpos = n; self.hp = (l.hprev, 1);
        next
    }
    fn mtp_draft(&mut self, toks: &[u32]) -> u32 {
        assert_eq!(toks.len(), self.hp.1);
        let d = self.mtp_pass(toks, self.mpos as u32, self.hp.0);
        self.mpos += toks.len();
        d
    }
    fn mtp_verify(&mut self, x: u32, d: u32) -> (u32, u32) {
        let r = self.run_chunk_v(&[x, d], true, true, true).unwrap();
        self.hp = (self.lay.hrow, 2);
        (r[0].to_bits(), r[1].to_bits())
    }
    fn mtp_reject(&mut self) {
        let c = self.cpu.cfg.clone(); let l = self.lay.clone();
        let csz = ((c.conv_k - 1) * c.conv_dim()) as u32; let ssz = (c.ssm_vheads * c.ssm_state * c.ssm_state) as u32;
        self.ctx.begin();
        for il in 0..c.n_layer { if c.is_recurrent(il) { self.copy(l.ck_conv[il], l.conv_st[il], csz); self.copy(l.ck_ssm[il], l.ssm[il], ssz); } }
        self.ctx.barrier(); self.ctx.submit_wait();
        self.pos -= 1; self.hp = (l.hrow, 1);
    }
    fn reset_all(&mut self) { self.zero_state(); self.mpos = 1; self.hp = (0, 0); }
}
