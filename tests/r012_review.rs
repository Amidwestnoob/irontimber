//! Astra independent r012 CPU guards; no Vulkan entry in these tests.
#[cfg(feature="vulkan")]
mod guards {
 use bonsai::{gguf::Gguf,model::Config,vk::{validate_geometry,decode_kv_readback}};
 fn config()->Config {Config::from_gguf(&Gguf::open(std::env::var("BONSAI_MODEL").expect("actual fixture required")).unwrap()).unwrap()}
 #[test] fn production_geometry() { let c=config(); for ctx in [1,192,260,8192] {validate_geometry(&c,ctx,true,true).unwrap();} }
 #[test] fn unsupported_geometry_cpu_rejection() {let c=config(); for i in 0..8 {let mut b=c.clone();match i {0=>b.head_dim=128,1=>b.n_head_kv=0,2=>b.conv_k=5,3=>b.ssm_state=64,4=>b.had_block=512,5=>b.n_ff=17,6=>b.ssm_groups=0,_=>b.n_layer=0};assert!(validate_geometry(&b,8192,true,true).is_err());} assert!(validate_geometry(&c,0,true,true).is_err());assert!(validate_geometry(&c,8193,true,true).is_err());assert!(validate_geometry(&c,8192,true,false).is_err());}
 #[test] fn half_readback_preserves_values_and_length() {let values=[0.0f32,-0.0,1.0,-2.0,65504.0,0.00006103515625];let b:Vec<u8>=values.iter().flat_map(|&v|half::f16::from_f32(v).to_bits().to_le_bytes()).collect();assert_eq!(decode_kv_readback(&b,true),values);let b:Vec<u8>=values.iter().flat_map(|v|v.to_le_bytes()).collect();assert_eq!(decode_kv_readback(&b,false),values);}
 #[test] fn invalid_context_before_vulkan() {let g=Gguf::open(std::env::var("BONSAI_MODEL").unwrap()).unwrap();let e=bonsai::vk::VkEngine::new(&g,None,0,Some("MUST_NOT_ENTER_VULKAN")).err().unwrap();assert!(e.contains("context capacity"),"{e}");}
}
