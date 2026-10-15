//! Dequant vectors produced by llama-prism's own ggml (tools/oracle/dequant_oracle).
use bonsai::gguf::GgmlType;
use bonsai::quant::*;
fn f32s(p: &str) -> Vec<f32> { std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
fn check(prefix: &str, rows: usize, cols: usize) {
    let d = env!("CARGO_MANIFEST_DIR");
    for (ext, ty) in [("", GgmlType::PQ2_0), (".ptq1", GgmlType::PTQ1_0)] {
        let raw = std::fs::read(format!("{d}/tests/data/{prefix}{ext}.raw")).unwrap();
        let want = f32s(&format!("{d}/tests/data/{prefix}{ext}.f32"));
        let mut got = vec![0f32; want.len()];
        match ty { GgmlType::PQ2_0 => dequant_pq2_0(&raw, &mut got), _ => dequant_ptq1_0(&raw, &mut got) }
        assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{prefix}{ext} bit-exact");
        // repacked 2-bit layout must decode identically (this is what the GPU consumes)
        let p = Packed2::from_ggml(ty, &raw, rows, cols);
        let mut row = vec![0f32; cols];
        for r in 0..rows { p.row_f32(r, &mut row); assert_eq!(&row[..], &want[r * cols..(r + 1) * cols]); }
    }
}
#[test] fn ssm_out_rows_bit_exact() { check("ssm_out4", 4, 6144); }
#[test] fn attn_k_rows_bit_exact() { check("attn_k3", 3, 5120); }

fn kquant(prefix: &str, ty: GgmlType, rows: usize, cols: usize) {
    let d = env!("CARGO_MANIFEST_DIR");
    let raw = std::fs::read(format!("{d}/tests/data/{prefix}.raw")).unwrap();
    let want = f32s(&format!("{d}/tests/data/{prefix}.f32"));
    let mut got = vec![0f32; want.len()];
    match ty { GgmlType::Q8_0 => dequant_q8_0(&raw, &mut got), GgmlType::Q6_K => dequant_q6_k(&raw, &mut got), _ => dequant_q4_k(&raw, &mut got) }
    assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{prefix} bit-exact vs ggml");
    if ty != GgmlType::Q4_K {
        let p = Q8G16::from_ggml(ty, &raw, rows, cols); let mut row = vec![0f32; cols];
        for r in 0..rows { p.row_f32(r, &mut row); assert_eq!(row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want[r * cols..(r + 1) * cols].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{prefix} Q8G16 repack row {r}"); }
    }
}
#[test] fn mtp_q6_k_bit_exact() { kquant("mtp_q6k3", GgmlType::Q6_K, 3, 5120); }
#[test] fn mtp_q8_0_bit_exact() { kquant("mtp_q80_3", GgmlType::Q8_0, 3, 5120); }
#[test] fn mtp_q4_k_bit_exact() { kquant("mtp_q4k3", GgmlType::Q4_K, 3, 5120); }
