//! Real same-process, full64/full8192/f16 load/run/drop/reload regression.
//! Run only under the externally owned exclusive monitored W9100 lease.
#[cfg(feature="vulkan")]
#[test]
fn sequential_full_engines_release_device_allocations() {
    use bonsai::{gguf::Gguf, tokenizer::Tokenizer, vk::VkEngine};
    fn memory(label:&str)->(u64,u64) {
        let ds:Vec<_>=std::fs::read_dir("/sys/class/drm").unwrap().filter_map(Result::ok)
            .filter(|e|e.file_name().to_string_lossy().starts_with("card"))
            .map(|e|e.path().join("device"))
            .filter(|p|std::fs::read_to_string(p.join("device")).is_ok_and(|x|x.trim()=="0x67a0")).collect();
        assert_eq!(ds.len(),1,"W9100 monitoring identity");
        let read=|n:&str|std::fs::read_to_string(ds[0].join(n)).unwrap().trim().parse::<u64>().unwrap();
        let(total,used)=(read("mem_info_vram_total"),read("mem_info_vram_used"));
        eprintln!("[lifetime] {}",serde_json::json!({"label":label,"pid":std::process::id(),"device":"0x67a0","total":total,"used":used,"free":total-used}));
        (total,used)
    }
    fn released(before:u64,label:&str) {
        // Driver destruction can retire allocations asynchronously. Observe it,
        // without constructing another engine until release has been proved.
        for _ in 0..50 {
            let(total,used)=memory(label);
            if used<=before+(128<<20) && total-used>=10*(1u64<<30) {return;}
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        panic!("recovery capacity boundary: allocations not released after engine drop");
    }
    assert_eq!(std::env::var("BONSAI_KV").unwrap(),"f16");
    let g=Gguf::open(std::env::var("BONSAI_MODEL").unwrap()).unwrap();
    let h=Gguf::open(std::env::var("BONSAI_MTP_HEAD").unwrap()).unwrap();
    let tk=Tokenizer::from_gguf(&g).unwrap();let ids=tk.encode("The capital of France is",true,true);
    for head in [false,true] {
        let(_,before)=memory(&format!("before-head-{head}"));let mut histories=Vec::new();
        for index in 0..2 {
            let mut b=VkEngine::new_with_mtp(&g,None,8192,Some("HAWAII"),if head {Some(&h)}else{None}).unwrap();
            assert!(b.kv16);assert_eq!(b.cpu.cfg.n_layer,64);assert_eq!(b.ctx.props.device_id,0x67a0);
            let(total,used)=memory(&format!("loaded-head-{head}-engine-{index}"));
            assert!(total-used>=1u64<<30,"recovery capacity boundary: 1GiB reserve");
            let mut output=vec![b.eval_argmax(&ids)];for _ in 0..3 {output.push(b.eval_argmax(&[*output.last().unwrap()]));}
            if head {
                use bonsai::mtp::MtpBackend;
                let first=b.mtp_prefill(&ids);let draft=b.mtp_draft(&[first]);let pair=b.mtp_verify(first,draft);
                eprintln!("[lifetime] head first={first} draft={draft} verify={pair:?}");
                assert!(b.ctx.completed_q8.get()>0);
            }
            eprintln!("[lifetime] {}",serde_json::json!({"head":head,"index":index,"input_ids":ids,"output_ids":output,"engine":b.review_identity()}));
            assert!(b.ctx.completed_dispatches.get()>0);histories.push(output);
            drop(b);released(before,&format!("after-drop-head-{head}-engine-{index}"));
        }
        assert_eq!(histories[0],histories[1],"reconstructed full engines differ");
    }
}
