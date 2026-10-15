// Compiles shaders/*.comp (GLSL, prefixed with shaders/common.glsl) to SPIR-V with glslangValidator.
// If glslangValidator is missing, falls back to the checked-in shaders/spv/*.spv.
use std::{env, fs, path::PathBuf, process::Command};
fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let common = fs::read_to_string("shaders/common.glsl").unwrap();
    println!("cargo:rerun-if-changed=shaders");
    let have = Command::new("glslangValidator").arg("--version").output().is_ok();
    for e in fs::read_dir("shaders").unwrap() {
        let p = e.unwrap().path();
        if p.extension().map_or(true, |x| x != "comp") { continue; }
        let name = p.file_stem().unwrap().to_str().unwrap().to_string();
        let dst = out.join(format!("{name}.spv"));
        if have {
            let src = format!("{}\n{}", common, fs::read_to_string(&p).unwrap());
            let tmp = out.join(format!("{name}.comp"));
            fs::write(&tmp, src).unwrap();
            let o = Command::new("glslangValidator").args(["-V", "--target-env", "vulkan1.1", "-o"]).arg(&dst).arg(&tmp).output().unwrap();
            if !o.status.success() { panic!("shader {name} failed:\n{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)); }
            // Build artifacts stay in OUT_DIR; never rewrite the reviewed source snapshot.
        } else {
            fs::copy(format!("shaders/spv/{name}.spv"), &dst).expect("no glslangValidator and no prebuilt spv");
        }
    }
}
