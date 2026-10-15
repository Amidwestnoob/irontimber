//! CPU-only ash dispatch-table mocks: these exercise production Drop and
//! Ctx::buffer error paths, not a separate model of the ownership algorithm.
use super::*;
use crate::vk::Ctx;
use ash::vk::Handle;
use std::{cell::RefCell, ffi::{c_void, CStr}};
thread_local! {
    static EVENTS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
    static FAIL: RefCell<&'static str> = const { RefCell::new("") };
}
fn event(s: &'static str) { EVENTS.with(|x| x.borrow_mut().push(s)); }
fn events() -> Vec<&'static str> { EVENTS.with(|x| x.borrow().clone()) }
fn reset(fail: &'static str) { EVENTS.with(|x| x.borrow_mut().clear()); FAIL.with(|x| *x.borrow_mut()=fail); }
fn result(s: &'static str) -> vk::Result { event(s); if FAIL.with(|x| *x.borrow()==s) {vk::Result::ERROR_OUT_OF_DEVICE_MEMORY} else {vk::Result::SUCCESS} }
unsafe extern "system" fn idle(_: vk::Device) -> vk::Result { result("idle") }
unsafe extern "system" fn destroy_device(_: vk::Device, _: *const vk::AllocationCallbacks<'_>) { event("device"); }
unsafe extern "system" fn destroy_instance(_: vk::Instance, _: *const vk::AllocationCallbacks<'_>) { event("instance"); }
unsafe extern "system" fn unmap(_: vk::Device, _: vk::DeviceMemory) { event("unmap"); }
macro_rules! destroy {
    ($fn:ident,$ty:ty,$name:literal) => { unsafe extern "system" fn $fn(_: vk::Device, _: $ty, _: *const vk::AllocationCallbacks<'_>) {event($name);} };
}
destroy!(buffer, vk::Buffer, "buffer"); destroy!(memory, vk::DeviceMemory, "memory");
destroy!(cpool, vk::CommandPool, "cpool"); destroy!(dpool, vk::DescriptorPool, "dpool");
destroy!(pipe, vk::Pipeline, "pipe"); destroy!(query, vk::QueryPool, "query");
destroy!(layout, vk::PipelineLayout, "layout"); destroy!(dsl, vk::DescriptorSetLayout, "dsl");
destroy!(fence, vk::Fence, "fence"); destroy!(shader, vk::ShaderModule, "shader");
unsafe extern "system" fn create_buffer(_:vk::Device,_:*const vk::BufferCreateInfo<'_>,_:*const vk::AllocationCallbacks<'_>,out:*mut vk::Buffer)->vk::Result {
    let r=result("create_buffer"); if r==vk::Result::SUCCESS {*out=vk::Buffer::from_raw(20);} r
}
unsafe extern "system" fn requirements(_:vk::Device,_:vk::Buffer,out:*mut vk::MemoryRequirements) {
    *out=vk::MemoryRequirements{size:16,alignment:16,memory_type_bits:1};
}
unsafe extern "system" fn allocate(_:vk::Device,_:*const vk::MemoryAllocateInfo<'_>,_:*const vk::AllocationCallbacks<'_>,out:*mut vk::DeviceMemory)->vk::Result {
    let r=result("allocate");if r==vk::Result::SUCCESS {*out=vk::DeviceMemory::from_raw(21);}r
}
unsafe extern "system" fn bind(_:vk::Device,_:vk::Buffer,_:vk::DeviceMemory,_:vk::DeviceSize)->vk::Result {result("bind")}
unsafe extern "system" fn map(_:vk::Device,_:vk::DeviceMemory,_:vk::DeviceSize,_:vk::DeviceSize,_:vk::MemoryMapFlags,out:*mut *mut c_void)->vk::Result {
    let r=result("map");if r==vk::Result::SUCCESS {*out=std::ptr::dangling_mut();}r
}
fn load(n:&CStr)->*const c_void {
    match n.to_bytes() {
        b"vkDeviceWaitIdle"=>idle as *const c_void,b"vkDestroyDevice"=>destroy_device as *const c_void,
        b"vkDestroyInstance"=>destroy_instance as *const c_void,b"vkUnmapMemory"=>unmap as *const c_void,
        b"vkDestroyBuffer"=>buffer as *const c_void,b"vkFreeMemory"=>memory as *const c_void,
        b"vkDestroyCommandPool"=>cpool as *const c_void,b"vkDestroyDescriptorPool"=>dpool as *const c_void,
        b"vkDestroyPipeline"=>pipe as *const c_void,b"vkDestroyQueryPool"=>query as *const c_void,
        b"vkDestroyPipelineLayout"=>layout as *const c_void,b"vkDestroyDescriptorSetLayout"=>dsl as *const c_void,
        b"vkDestroyFence"=>fence as *const c_void,b"vkDestroyShaderModule"=>shader as *const c_void,
        b"vkCreateBuffer"=>create_buffer as *const c_void,b"vkGetBufferMemoryRequirements"=>requirements as *const c_void,
        b"vkAllocateMemory"=>allocate as *const c_void,b"vkBindBufferMemory"=>bind as *const c_void,
        b"vkMapMemory"=>map as *const c_void,_=>std::ptr::null(),
    }
}
fn context() -> Ctx { unsafe {
    let instance=ash::Instance::load_with(load,vk::Instance::from_raw(1));
    let device=ash::Device::load_with(load,vk::Device::from_raw(2));
    let owner=Arc::new(DeviceOwner{device,_instance:Arc::new(InstanceOwner{instance,_entry:None})});
    let mut mem=vk::PhysicalDeviceMemoryProperties::default();mem.memory_type_count=1;
    mem.memory_types[0].property_flags=vk::MemoryPropertyFlags::HOST_VISIBLE|vk::MemoryPropertyFlags::HOST_COHERENT|vk::MemoryPropertyFlags::DEVICE_LOCAL;
    Ctx::empty(owner,vk::PhysicalDevice::from_raw(3),vk::Queue::from_raw(4),mem,Default::default(),true)
} }
fn count(e:&[&str],s:&str)->usize {e.iter().filter(|x|**x==s).count()}
#[test]
fn buffer_constructor_all_failure_edges() {
    for fail in ["create_buffer","memory_type","allocate","bind","map",""] {
        reset(fail);let mut c=context();if fail=="memory_type" {c.mem_props.memory_type_count=0;}
        let b=c.buffer(16,true);assert_eq!(b.is_ok(),fail.is_empty());drop(b);drop(c);
        let e=events();assert_eq!(count(&e,"buffer"),usize::from(fail!="create_buffer"));
        assert_eq!(count(&e,"memory"),usize::from(matches!(fail,"bind"|"map"|"")));
        assert_eq!(count(&e,"unmap"),usize::from(fail.is_empty()));
        assert_eq!(&e[e.len()-2..],["device","instance"]);
        if let Some(m)=e.iter().position(|x|*x=="memory") {assert!(e.iter().position(|x|*x=="buffer").unwrap()<m);}
    }
}
#[test]
fn buffer_outlives_context_and_option_cleanup_is_once() {
    reset("");let c=context();let mut b=Some(c.buffer(16,true).unwrap());drop(c);
    assert_eq!(count(&events(),"device"),0);drop(b.take());drop(b.take());
    let e=events();for s in ["unmap","buffer","memory","device","instance"] {assert_eq!(count(&e,s),1,"{s}: {e:?}");}
    assert_eq!(&e[e.len()-2..],["device","instance"]);
}
#[test]
fn context_partial_construction_prefixes_and_normal_teardown() {
    // Every fallible constructor prefix, including a shader/pipeline or query
    // failure before staging, uses the identical production Ctx Drop.
    for prefix in 0..=9 {
        reset("");let mut c=context();
        if prefix>=1 {c.cpool=vk::CommandPool::from_raw(5);}
        if prefix>=2 {c.fence=vk::Fence::from_raw(6);}
        if prefix>=3 {c.dsl=vk::DescriptorSetLayout::from_raw(7);}
        if prefix>=4 {c.layout=vk::PipelineLayout::from_raw(8);}
        if prefix>=5 {c.dpool=vk::DescriptorPool::from_raw(9);}
        if prefix>=6 {c.pipes.insert("test",vk::Pipeline::from_raw(10));}
        if prefix>=7 {c.prof=Some((vk::QueryPool::from_raw(11),Default::default(),Default::default()));}
        if prefix>=8 {c.staging=c.buffer(16,true).unwrap();}
        if prefix>=9 {let _m=ShaderModule(&c.device,vk::ShaderModule::from_raw(12));}
        drop(c);let e=events();
        for (i,s) in ["cpool","fence","dsl","layout","dpool","pipe","query","buffer","shader"].iter().enumerate() {assert_eq!(count(&e,s),usize::from(prefix>i),"prefix={prefix}, {e:?}");}
        assert_eq!(&e[e.len()-2..],["device","instance"]);
        if prefix>=8 {assert!(e.iter().position(|x|*x=="cpool")<e.iter().position(|x|*x=="buffer"));}
    }
}
#[test]
fn unwinding_releases_mapped_buffer_shader_and_parents() {
    reset("");assert!(std::panic::catch_unwind(|| {
        let c=context();let _b=c.buffer(16,true).unwrap();let _m=ShaderModule(&c.device,vk::ShaderModule::from_raw(12));panic!("injected constructor/dispatch error");
    }).is_err());let e=events();
    for s in ["shader","unmap","buffer","memory","device","instance"] {assert_eq!(count(&e,s),1,"{e:?}");}
}
#[test]
fn device_lost_cleanup_does_not_panic_or_skip_destruction() {
    reset("idle");let c=context();let b=c.buffer(16,true).unwrap();drop(c);drop(b);
    let e=events();assert_eq!(&e[e.len()-2..],["device","instance"]);assert_eq!(count(&e,"memory"),1);
}
#[test]
fn instance_only_constructor_failure_releases_instance() {
    reset("");let c=context();let instance=c.owner._instance.clone();drop(c);
    assert_eq!(count(&events(),"instance"),0);drop(instance);assert_eq!(count(&events(),"instance"),1);
}
