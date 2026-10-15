#[cfg(feature="vulkan")]
#[test] fn unsupported_prefill_config_cpu_guards() {
    use bonsai::vk::{validate_prefill_config as v, validate_prefill_attention as attn};
    assert!(attn(true,"quad").is_ok());assert!(attn(false,"old").is_ok());assert!(attn(false,"quad").is_err());assert!(attn(true,"unknown").is_err());
    for v in ["quad256","pair256"] {assert!(attn(true,v).is_ok());assert!(attn(false,v).is_err());}
    use bonsai::vk::select_prefill_attention as pick;
    assert_eq!(pick(1,256,"pair256"),("attn_s_h",1));assert_eq!(pick(2,64,"quad256"),("attn_s_h",1));
    assert_eq!(pick(128,256,"pair256"),("attn_prefill_h2_256",2));assert_eq!(pick(64,256,"quad256"),("attn_prefill_h4_256",4));
    assert_eq!(pick(64,512,"pair256"),("attn_prefill_h4",4));assert_eq!(pick(64,1024,"quad256"),("attn_prefill_h4",4));
    for (ctx,rust,kv,tile) in [(0,true,Some("f16"),None),(usize::MAX,true,None,None),(8192,false,Some("f16"),None),(8192,false,None,Some("pf128")),(8192,true,Some("bf16"),None),(8192,true,None,Some("bad"))] {assert!(v(ctx,rust,kv,tile).is_err());}
    for ctx in [1,192,8192] { for tile in [None,Some("old"),Some("pf64"),Some("pf128"),Some("pf128x32"),Some("pf64x32"),Some("pf64k64"),Some("pf512"),Some("h16"),Some("r64"),Some("r128"),Some("ru8"),Some("rb64"),Some("rb128"),Some("col"),Some("col16"),Some("col8")] {assert!(v(ctx,true,Some("f16"),tile).is_ok());} }
    assert!(v(8192,false,Some("f32"),Some("old")).is_ok());
    for d in [0.0f32,-0.0,0.5,-0.5,0.00006103515625,32752.] {assert!(bonsai::vk::lossless_weight_lds_scale(d));}
    for d in [f32::NAN,f32::INFINITY,0.000001,0.1,32768.] {assert!(!bonsai::vk::lossless_weight_lds_scale(d));}
    assert!(v(8192,false,Some("f32"),Some("h16")).is_err());
    assert_eq!(bonsai::vk::attention_scratch_words(8192,24),64*8192*24);
    for b in [None,Some("64"),Some("128"),Some("256"),Some("512")] {assert!(bonsai::vk::validate_prefill_batch(b).is_ok());}
    for b in ["0","65","1024","garbage"] {assert!(bonsai::vk::validate_prefill_batch(Some(b)).is_err());}
    for nt in [1usize,2,22,43,64,128,256,512] {let room=bonsai::vk::attention_dispatch_room(8192,24,nt);assert_eq!(room,nt.div_ceil(64)*64*8192*24);}
    for batch in [64,128,256,512] {for n in [0usize,1,21,22,64,65,129,2091,4191,7416,8192] {
        let rr=bonsai::vk::prefill_chunks(n,batch);let mut ids=vec![];
        for r in rr {assert!(r.len()>0 && r.len()<=batch);ids.extend(r);}
        assert_eq!(ids,(0..n).collect::<Vec<_>>());
    }}
    assert_eq!(bonsai::vk::prefill_chunks(129,512),vec![0..128,128..129]);
    // API harness boundary: -1 is a supported default-budget sentinel, not invalid.
    let model=std::env::var("BONSAI_MODEL").expect("actual canonical model fixture required");
    let g=bonsai::gguf::Gguf::open(&model).unwrap();let tk=bonsai::tokenizer::Tokenizer::from_gguf(&g).unwrap();
    for key in ["n_predict","max_tokens","max_completion_tokens"] {for n in [-2i64,-1,0,1] {
        let mut request=serde_json::json!({"prompt":[760]});request[key]=serde_json::json!(n);
        let parsed=bonsai::server::parse_request(&request,&tk,None,false);
        assert_eq!(parsed.is_ok(),n>=-1);if n== -1 {assert_eq!(parsed.unwrap().n_predict,4096);}
    }}
    assert!(v(8192,true,Some("f16"),Some("dense")).is_ok());for mode in ["dense","col128u4","col128u8"] {assert!(v(8192,true,Some("f16"),Some(mode)).is_ok());assert!(v(8192,false,Some("f32"),Some(mode)).is_err());}
    for ctx in [1usize,16,192,256,260,8192] {
        let room=bonsai::vk::attention_scratch_words(ctx,24) as u32;
        let nt=ctx.min(64) as u32;let total=ctx as u32;let mut c=64u32;
        while c<1024 && nt*24*total.div_ceil(c)*260>room {c*=2;}
        assert!(nt*24*total.div_ceil(c)*260<=room,"ctx={ctx}");
    }
    for (m,k,nt,it) in [(0,5120,64,5),(17408,0,64,5),(17408,5121,64,5),(17408,5120,65,5),(17408,5120,64,0),(17408,5120,64,21)] {
        assert!(bonsai::vk::kparity::prefill_bench(Some("MUST_NOT_QUERY_DEVICE"),m,k,nt,it).is_err());
    }
    for (ctx,nt,it) in [(7001,8,1),(7000,64,1),(16,65,1),(16,1,0),(16,1,6)] {
        assert!(bonsai::vk::kparity::prefill_attention_bench(Some("MUST_NOT_QUERY_DEVICE"),ctx,nt,it).is_err());
    }
}

#[cfg(feature="vulkan")]
#[test] fn prefill_tiles_boundary_and_accumulation_gpu() {
    for m in [1000,1001] {for nt in [1,22,43,64,128] { for accum in [false,true] {
        let r=bonsai::vk::kparity::prefill_bench_accum(Some("HAWAII"),m,1024,nt,1,accum).unwrap();println!("{r}");
    } }}
}
#[cfg(feature="vulkan")]
#[test] fn prefill_attention_query_tail_and_long_kv_gpu() {
    for (ctx,nt) in [(16,1),(16,3),(16,7),(16,8),(16,22),(16,43),(16,64),(500,8),(2000,8),(7000,8)] {
        let r=bonsai::vk::kparity::prefill_attention_bench(Some("HAWAII"),ctx,nt,1).unwrap();println!("{r}");
    }
}
