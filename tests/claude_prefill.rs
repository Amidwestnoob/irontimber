//! CPU-only guards for measured GQA specializations; no device acquisition.
#[cfg(feature="vulkan")]
#[test]
fn column_tail_plan_cpu(){
    use bonsai::vk::{column_prefill_segments as p,prefill_column_kernel as k,validate_prefill_matmul as v};
    for value in [None,Some("base"),Some("shape")] {assert!(v(value).is_ok());}
    for value in ["","fast","unknown","dr512"] {assert!(v(Some(value)).is_err());}
    assert_eq!(k(5120,17408,256,"shape"),("matmul_col_sw16",64,64));
    for nt in [1,2,7,22,43,64,127] {for value in ["shape","dr512"] {assert_eq!(k(5120,17408,nt,value),("matmul_col",64,64));}}
    for nt in [128,129,256] {assert_eq!(k(17408,5120,nt,"dr512"),("matmul_col",64,64));}
    for (m,n) in [(17408,5120),(12288,5120),(5120,6144)] {assert_eq!(k(m,n,256,"shape"),("matmul_col",64,64));}
    for nt in 0..=512 {for peel in [false,true] {
        let plan=p(nt,peel);let ids:Vec<_>=plan.iter().flat_map(|(s,n)|*s..s+n).collect();
        assert_eq!(ids,(0..nt).collect::<Vec<_>>());
        assert!(plan.iter().all(|(_,n)|*n>0&&*n<=256));
        if !peel {assert_eq!(plan,(0..nt).step_by(256).map(|s|(s,(nt-s).min(256))).collect::<Vec<_>>());}
    }}
    assert_eq!(p(129,true),vec![(0,128),(128,1)]);
    assert_eq!(p(257,true),vec![(0,256),(256,1)]);
    assert_eq!(p(248,true),vec![(0,248)]);
    assert!(std::panic::catch_unwind(||p(513,true)).is_err());
}

#[cfg(feature="vulkan")]
#[test]
fn claude_prefill_gqa_cpu_guards(){
    use bonsai::vk::{select_prefill_gqa as s, validate_prefill_attention as v};
    for (value,pipe,heads) in [("gqa6","attn_gqa6",6),("gqa3","attn_gqa3",3),("gqa2","attn_gqa2",2)] {
        assert!(v(true,value).is_ok());assert!(v(false,value).is_err());
        for nt in [8,22,43,64,128,248,256,512]{assert_eq!(s(nt,256,4096,24,4,256,value),(pipe,1,heads));}
        for nt in [1,2,3,7]{assert_eq!(s(nt,256,4096,24,4,256,value),("attn_s_h",1,1));}
        for cn in [64,128,512,1024]{assert!(!s(256,cn,4096,24,4,256,value).0.starts_with("attn_gqa"));}
        for pos in [0,16,500,1023]{assert_eq!(s(256,256,pos,24,4,256,value),("attn_prefill_h4_256",4,1));}
        for (nh,nkv,hd) in [(24,0,256),(24,8,256),(32,4,256),(24,4,128)]{assert!(!s(256,256,4096,nh,nkv,hd,value).0.starts_with("attn_gqa"));}
    }
    assert_eq!(s(256,256,4096,24,4,256,"quad256"),("attn_prefill_h4_256",4,1));
    for v in ["NaN","inf","-1","6","bad"] {
        std::env::set_var("BONSAI_SCREEN_STEADY_SECONDS",v);
        assert!(bonsai::vk::kparity::prefill_bench(Some("MUST_NOT_QUERY_DEVICE"),64,128,64,1).is_err());
    }
    std::env::remove_var("BONSAI_SCREEN_STEADY_SECONDS");
}
