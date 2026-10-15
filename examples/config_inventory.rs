//! CPU-only exact model metadata inventory; no Vulkan acquisition.
fn main(){
 let path=std::env::var("BONSAI_MODEL").expect("BONSAI_MODEL");
 let g=bonsai::gguf::Gguf::open(&path).unwrap();let c=bonsai::model::Config::from_gguf(&g).unwrap();
 println!("config={c:?}");
 let mut shapes=std::collections::BTreeMap::new();
 for (name,t) in &g.tensors {if name.starts_with("blk.") && t.dims.len()==2 {shapes.entry(t.dims.clone()).or_insert_with(Vec::new).push(name.clone());}}
 for (dims,names) in shapes {println!("{}",serde_json::json!({"dims":dims,"count":names.len(),"names":names}));}
}
