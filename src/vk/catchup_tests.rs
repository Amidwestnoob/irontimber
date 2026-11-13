//! Exact head KV and subsequent draft regression against the original full catch-up.
use super::*;
use crate::mtp::MtpBackend;
#[test]
fn state_only_catchup_preserves_all_head_kv_and_drafts() {
    let g=crate::gguf::Gguf::open(std::env::var("BONSAI_MODEL").unwrap()).unwrap();
    let h=crate::gguf::Gguf::open(std::env::var("BONSAI_MTP_HEAD").unwrap()).unwrap();
    let mut b=VkEngine::new_with_mtp(&g,None,8192,Some("HAWAII"),Some(&h)).unwrap();
    assert_eq!(b.ctx.props.device_id,0x67a0); assert!(b.kv16);
    for n in [1,2,63,64,65,129] {
        let prompt=vec![760;n];
        let before=b.ctx.completed_dispatches.get();
        let next=b.mtp_prefill(&prompt);
        let reduced=b.ctx.completed_dispatches.get()-before;
        let state=b.review_state(true);
        let draft=b.mtp_draft(&[next]);
        let drafted_state=b.review_state(true);
        b.reset_all();
        let before=b.ctx.completed_dispatches.get();
        let e=b.cpu.cfg.n_embd as u32; let l=b.lay.clone();let mut old_next=0;
        // Independent unchanged C10 prefill/catch-up schedule.
        for range in prefill_chunks(n,b.ubatch) {
            let s=range.start;let last=range.end==n;let ch=&prompt[range];
            let r=b.run_chunk(ch,last,true);if last {old_next=r.unwrap()[0].to_bits();}
            let nt=ch.len() as u32;b.ctx.begin();
            if s>0 {b.copy(l.hprev,l.mh,e);}
            if nt>1 {b.copy(l.hrow,l.mh+e,(nt-1)*e);}
            // Oracle deliberately serializes transfers instead of copying C10's
            // missing read-before-overwrite dependency for hprev.
            b.ctx.submit_wait();b.ctx.begin();
            b.copy(l.hrow+(nt-1)*e,l.hprev,e);b.ctx.barrier();b.ctx.submit_wait();
            let(toks,pos0,hoff)=if s==0 {(&ch[1..],1u32,l.mh+e)}else{(ch,s as u32,l.mh)};
            if !toks.is_empty(){b.mtp_pass(toks,pos0,hoff);}
        }
        b.mpos=n;b.hp=(l.hprev,1);
        let original=b.ctx.completed_dispatches.get()-before;
        assert_eq!(next,old_next);
        let bits=|v:Vec<f32>|v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(bits(state),bits(b.review_state(true)),"head KV n={n}");
        assert_eq!(draft,b.mtp_draft(&[next]),"next draft n={n}");
        assert_eq!(bits(drafted_state),bits(b.review_state(true)),"next draft state n={n}");
        if n>1 && n<=b.ubatch {assert!(reduced<original);} else {assert_eq!(reduced,original);}
        eprintln!("[catchup-proof] n={n} original_dispatches={original} candidate_dispatches={reduced} next={next} draft={draft} bitwise_head_state=true");
    }
}
