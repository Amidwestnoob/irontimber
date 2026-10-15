//! Deterministic independent loop/state reference; not real model or GPU evidence.
use bonsai::mtp::{MtpBackend,greedy_mtp};
#[derive(Default)] struct Script { pos:usize, calls:usize, rejected:usize, drafts:usize, bad:bool }
impl MtpBackend for Script {
 fn mtp_single(&mut self,_:u32)->u32 {let a=10+self.pos as u32;self.pos+=1;self.calls+=1;a}
 fn mtp_prefill(&mut self,p:&[u32])->u32 {self.pos=p.len(); 10}
 fn mtp_draft(&mut self,_:&[u32])->u32 {self.drafts+=1;if self.bad {99} else {10+self.pos as u32}}
 fn mtp_verify(&mut self,_:u32,_:u32)->(u32,u32) {let a=10+self.pos as u32;self.pos+=2;self.calls+=2;(a,a+1)}
 fn mtp_reject(&mut self){self.pos-=1;self.rejected+=1;}
 fn reset_all(&mut self){self.pos=0;}
}
#[test] fn final_one_token_does_one_forward(){let mut b=Script::default();let(o,s)=greedy_mtp(&mut b,&[1],2,&|_|false,&mut |_|true);assert_eq!(o,vec![10,11]);assert_eq!(b.calls,1);assert_eq!(s.drafted,0);}
#[test] fn terminal_draft_eos_state_matches_plain(){let mut b=Script::default();let(o,_)=greedy_mtp(&mut b,&[1],9,&|t|t==11,&mut |_|true);assert_eq!(o,vec![10,11]);assert_eq!(b.pos,2);}
#[test] fn all_even_budgets_no_extra_work(){for n in [2,4,6,8]{let mut b=Script::default();let(o,s)=greedy_mtp(&mut b,&[1],n,&|_|false,&mut |_|true);assert_eq!(o.len(),n);assert_eq!(b.calls,n-1);assert_eq!(s.timed_decode_forwards,n-1);}}
#[test] fn zero_budget(){let mut b=Script::default();assert!(greedy_mtp(&mut b,&[1],0,&|_|false,&mut |_|true).0.is_empty());assert_eq!(b.pos,0);}
#[test] fn first_eos(){let mut b=Script::default();assert_eq!(greedy_mtp(&mut b,&[1],9,&|t|t==10,&mut |_|true).0,vec![10]);assert_eq!(b.calls,0);}
#[test] fn accepted_terminal_eos(){let mut b=Script::default();assert_eq!(greedy_mtp(&mut b,&[1],9,&|t|t==11,&mut |_|true).0,vec![10,11]);}
#[test] fn rejected_terminal_eos(){let mut b=Script{bad:true,..Default::default()};assert_eq!(greedy_mtp(&mut b,&[1],9,&|t|t==11,&mut |_|true).0,vec![10,11]);assert_eq!(b.rejected,1);}
#[test] fn exact_odd_budgets(){for n in [1,3,5,7,9]{let mut b=Script::default();let (o,_)=greedy_mtp(&mut b,&[1],n,&|_|false,&mut |_|true);assert_eq!(o,(10..10+n as u32).collect::<Vec<_>>());}}
#[test] fn cancellation(){let mut b=Script::default();let (o,_)=greedy_mtp(&mut b,&[1],9,&|_|false,&mut |t|t<11);assert_eq!(o,vec![10,11]);}
#[test] fn rejection_history(){let mut b=Script{bad:true,..Default::default()};let (o,s)=greedy_mtp(&mut b,&[1],9,&|_|false,&mut |_|true);assert_eq!(o,(10..19).collect::<Vec<_>>());assert_eq!(s.accepted,0);}
