//! MTP speculative decoding: greedy output must be byte-identical with and without MTP.
//! Needs BONSAI_MTP_MODEL (a GGUF with a NextN block, e.g. PTQ1_0-mtp or a grafted PQ2_0-mtp); skipped otherwise.
use std::process::Command;

fn run(model: &str, extra: &[&str]) -> serde_json::Value {
    let out = Command::new(env!("CARGO_BIN_EXE_bonsai"))
        .args(["run", "--model", model, "--ids", "760,6511,314,9338,369,13,1049,374,1100,2014,0,220,17,15", "--n-predict", "16", "--json"])
        .args(extra).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str(s.lines().last().unwrap()).unwrap()
}

fn check(backend: &[&str]) {
    let Ok(h) = std::env::var("BONSAI_MTP_HEAD") else { eprintln!("BONSAI_MTP_HEAD unset: skipped"); return };
    let m = std::env::var("BONSAI_MODEL").expect("baseline model required");
    let plain = run(&m, backend);
    let mut a = backend.to_vec(); a.extend(["--mtp","--mtp-head",h.as_str()]);
    let mtp = run(&m, &a);
    assert_eq!(plain["greedy"], mtp["greedy"], "greedy differs with MTP");
    assert!(mtp["mtp_drafted"].as_u64().unwrap() > 0);
    eprintln!("{backend:?}: accepted {}/{}", mtp["mtp_accepted"], mtp["mtp_drafted"]);
}

#[test] fn mtp_greedy_identical_cpu() { check(&["--backend", "cpu", "--numerics", "f32", "--layers", "2"]); }
#[cfg(feature="vulkan")]
#[test] fn mtp_greedy_identical_vulkan() { check(&["--backend", "vulkan", "--device", "HAWAII", "--ctx", "8192", "--kv", "f16"]); }
