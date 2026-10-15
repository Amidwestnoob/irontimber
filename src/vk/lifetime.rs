//! Vulkan ownership. Every buffer retains the device, and every device retains
//! its instance/loader. Context field order therefore cannot destroy a parent
//! before a child. Cleanup is also used by partially constructed objects.
use ash::vk;
use std::sync::Arc;

pub(super) struct InstanceOwner {
    pub instance: ash::Instance,
    pub _entry: Option<ash::Entry>,
}
impl Drop for InstanceOwner {
    fn drop(&mut self) { unsafe { self.instance.destroy_instance(None); } }
}
pub(super) struct DeviceOwner {
    pub device: ash::Device,
    pub _instance: Arc<InstanceOwner>,
}
impl Drop for DeviceOwner {
    fn drop(&mut self) { unsafe {
        // Drop cannot propagate an error (including DEVICE_LOST). The supervisor
        // retains actual device faults; cleanup must still release host objects.
        let _ = self.device.device_wait_idle();
        self.device.destroy_device(None);
    } }
}
/// A shader module has no lifetime beyond pipeline construction, including a
/// failed pipeline call. Unlike Box::leak, the entry-point CString is scoped too.
pub(super) struct ShaderModule<'a>(pub &'a ash::Device, pub vk::ShaderModule);
impl Drop for ShaderModule<'_> {
    fn drop(&mut self) { unsafe { self.0.destroy_shader_module(self.1, None); } }
}

impl super::Buf {
    pub(super) fn empty(owner: Arc<DeviceOwner>) -> Self {
        Self { buf: vk::Buffer::null(), mem: vk::DeviceMemory::null(),
            size: 0, ptr: std::ptr::null_mut(), owner }
    }
}
impl Drop for super::Buf {
    fn drop(&mut self) { unsafe {
        let d = &self.owner.device;
        // Also safe on unwinding after a split submission, or when a buffer is
        // dropped independently of its context. No pending dispatch uses freed memory.
        if self.buf != vk::Buffer::null() || self.mem != vk::DeviceMemory::null() {
            let _ = d.device_wait_idle();
        }
        if !self.ptr.is_null() { d.unmap_memory(self.mem); self.ptr = std::ptr::null_mut(); }
        if self.buf != vk::Buffer::null() { d.destroy_buffer(self.buf, None); self.buf = vk::Buffer::null(); }
        if self.mem != vk::DeviceMemory::null() { d.free_memory(self.mem, None); self.mem = vk::DeviceMemory::null(); }
    } }
}
impl super::Ctx {
    pub(super) fn empty(owner: Arc<DeviceOwner>, pdev: vk::PhysicalDevice,
        queue: vk::Queue, mem_props: vk::PhysicalDeviceMemoryProperties,
        props: vk::PhysicalDeviceProperties, rust: bool) -> Self {
        Self {
            instance: owner._instance.instance.clone(), device: owner.device.clone(),
            staging: super::Buf::empty(owner.clone()), owner, pdev, queue, mem_props, props,
            cpool: vk::CommandPool::null(), cb: vk::CommandBuffer::null(), fence: vk::Fence::null(),
            seg: Vec::new(), seg_used: Default::default(), cur: Default::default(),
            dsl: vk::DescriptorSetLayout::null(), layout: vk::PipelineLayout::null(),
            dpool: vk::DescriptorPool::null(), pipes: Default::default(), prof: None,
            rust, trace: std::env::var_os("BONSAI_REVIEW_TRACE").is_some(),
            pending_dispatches: Default::default(), nobar: Default::default(),
            pending_q8: Default::default(), completed_dispatches: Default::default(),
            completed_q8: Default::default(),
        }
    }
}
impl Drop for super::Ctx {
    fn drop(&mut self) { unsafe {
        let d = &self.device;
        let _ = d.device_wait_idle();
        // Command buffers and descriptor sets are owned by their pools. Destroy
        // them before pipelines/layouts/buffers they can reference.
        if self.cpool != vk::CommandPool::null() { d.destroy_command_pool(self.cpool, None); }
        if self.dpool != vk::DescriptorPool::null() { d.destroy_descriptor_pool(self.dpool, None); }
        for (_, p) in self.pipes.drain() { d.destroy_pipeline(p, None); }
        if let Some((pool, _, _)) = self.prof.take() { d.destroy_query_pool(pool, None); }
        if self.layout != vk::PipelineLayout::null() { d.destroy_pipeline_layout(self.layout, None); }
        if self.dsl != vk::DescriptorSetLayout::null() { d.destroy_descriptor_set_layout(self.dsl, None); }
        if self.fence != vk::Fence::null() { d.destroy_fence(self.fence, None); }
        // staging drops automatically. Any externally retained Buf keeps the
        // shared device and instance alive after this context is gone.
    } }
}

#[cfg(test)]
#[path = "lifetime_tests.rs"]
mod tests;
