//! Fail-closed parser/converter regressions; never constructs a device.
use bonsai::gguf::{Gguf,raw_u32,write_gguf};
fn header(nt:u64,nkv:u64)->Vec<u8>{let mut b=b"GGUF".to_vec();b.extend(3u32.to_le_bytes());b.extend(nt.to_le_bytes());b.extend(nkv.to_le_bytes());b}
fn string(b:&mut Vec<u8>,s:&str){b.extend((s.len() as u64).to_le_bytes());b.extend(s.as_bytes());}
fn tensor(b:&mut Vec<u8>,dims:&[u64],ty:u32,off:u64){string(b,"head");b.extend((dims.len() as u32).to_le_bytes());for d in dims{b.extend(d.to_le_bytes());}b.extend(ty.to_le_bytes());b.extend(off.to_le_bytes());}
fn bad(name:&str,b:&[u8]){let p=std::env::temp_dir().join(format!("bonsai-r009-{}-{name}.gguf",std::process::id()));std::fs::write(&p,b).unwrap();let result=Gguf::open(&p);std::fs::remove_file(&p).unwrap();assert!(result.is_err(),"accepted {name}");}
#[test]fn truncated_header(){bad("truncated",b"GGUF");}
#[test]fn overflowing_string_length(){let mut b=header(0,1);b.extend(u64::MAX.to_le_bytes());bad("string-overflow",&b);}
#[test]fn duplicate_metadata(){let mut b=header(0,2);for _ in 0..2{string(&mut b,"general.alignment");b.extend(raw_u32(32));}bad("duplicate-kv",&b);}
#[test]fn zero_alignment(){let mut b=header(0,1);string(&mut b,"general.alignment");b.extend(raw_u32(0));bad("alignment",&b);}
#[test]fn dimension_overflow(){let mut b=header(1,0);tensor(&mut b,&[u64::MAX,2],0,0);bad("dims",&b);}
#[test]fn tensor_offset_overflow(){let mut b=header(1,0);tensor(&mut b,&[32],0,u64::MAX);bad("offset",&b);}
#[test]fn quant_row_misalignment(){let mut b=header(1,0);tensor(&mut b,&[31],8,0);b.resize(256,0);bad("quant-alignment",&b);}
#[test]fn duplicate_tensor(){let mut b=header(2,0);tensor(&mut b,&[32],0,0);tensor(&mut b,&[32],0,0);b.resize(256,0);bad("duplicate-tensor",&b);}
#[test]fn converter_never_overwrites_existing(){let p=std::env::temp_dir().join(format!("bonsai-r009-{}-preserve.gguf",std::process::id()));std::fs::write(&p,b"preserve").unwrap();assert!(write_gguf(&p,&[],&[]).is_err());assert_eq!(std::fs::read(&p).unwrap(),b"preserve");std::fs::remove_file(p).unwrap();}
// r010: `serve --mtp` is the persistent MTP API; invalid serve MTP options still fail before opening.
#[test]fn cli_mtp_incompatible_modes_fail_before_open(){for args in [vec!["serve","--mtp","--mtp-default","maybe"],vec!["serve","--mtp-default","on"],vec!["tokenize","--mtp"],vec!["run","--mtp","--temp","0"],vec!["run","--mtp","--stream"],vec!["run","--mtp-head","nonexistent"]]{let r=std::process::Command::new(env!("CARGO_BIN_EXE_bonsai")).args(&args).args(["--model","MUST_NOT_OPEN"]).output().unwrap();assert!(!r.status.success());let s=String::from_utf8_lossy(&r.stderr);assert!(!s.contains("open MUST_NOT_OPEN"),"{s}");}}
