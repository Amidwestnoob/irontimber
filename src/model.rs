//! qwen35 (Qwen3.5/3.8 dense hybrid: Gated DeltaNet + gated full attention) model description.
use crate::gguf::{Gguf, GgmlType, TensorInfo};
use crate::hadamard::Hadamard;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Config {
    pub n_layer: usize, pub n_embd: usize, pub n_ff: usize, pub n_vocab: usize,
    pub n_head: usize, pub n_head_kv: usize, pub head_dim: usize,
    pub n_rot: usize, pub rope_base: f32, pub rope_sections: [usize; 4], pub eps: f32,
    pub conv_k: usize, pub ssm_state: usize, pub ssm_groups: usize, pub ssm_vheads: usize, pub ssm_inner: usize,
    pub full_attn_interval: usize,
    pub had_block: usize, pub had_weights: Vec<String>, pub had_inverse: Vec<String>, pub gdn_v_grouped: bool,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> {
        let arch = g.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("");
        if arch != "qwen35" { return Err(format!("expected qwen35 arch, got {arch}")); }
        let k = |s: &str| format!("qwen35.{s}");
        let secs: Vec<usize> = g.get(&k("rope.dimension_sections")).and_then(|v| v.as_arr()).map(|a| a.iter().map(|v| v.as_u64().unwrap() as usize).collect()).unwrap_or(vec![0; 4]);
        let tok = g.tensor("token_embd.weight")?;
        let strs = |key: &str| -> Vec<String> { g.get(key).and_then(|v| v.as_arr()).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default() };
        if let Some(v) = g.get("prism.hadamard.version") { let v = v.as_u64().unwrap(); if v != 1 { return Err(format!("unsupported prism.hadamard.version {v} (only v1 / untied output)")); } }
        if let Some(t) = g.get("prism.hadamard.transform").and_then(|v| v.as_str()) { if t != "normalized-sylvester-walsh-hadamard" { return Err(format!("unsupported hadamard transform {t}")); } }
        Ok(Config {
            // MTP/NextN blocks are appended after the trunk (blk.{n_layer}..); the trunk excludes them.
            n_layer: g.u(&k("block_count"))? - g.u(&k("nextn_predict_layers")).unwrap_or(0), n_embd: g.u(&k("embedding_length"))?, n_ff: g.u(&k("feed_forward_length"))?,
            n_vocab: tok.dims[1], n_head: g.u(&k("attention.head_count"))?, n_head_kv: g.u(&k("attention.head_count_kv"))?,
            head_dim: g.u(&k("attention.key_length"))?, n_rot: g.u(&k("rope.dimension_count"))?, rope_base: g.f(&k("rope.freq_base"))?,
            rope_sections: [secs[0], secs[1], secs[2], secs[3]], eps: g.f(&k("attention.layer_norm_rms_epsilon"))?,
            conv_k: g.u(&k("ssm.conv_kernel"))?, ssm_state: g.u(&k("ssm.state_size"))?, ssm_groups: g.u(&k("ssm.group_count"))?,
            ssm_vheads: g.u(&k("ssm.time_step_rank"))?, ssm_inner: g.u(&k("ssm.inner_size"))?,
            full_attn_interval: g.u(&k("full_attention_interval")).unwrap_or(4),
            had_block: g.u("prism.hadamard.block_size").unwrap_or(0), had_weights: strs("prism.hadamard.weight_names"),
            had_inverse: strs("prism.hadamard.inverse_weight_names"),
            gdn_v_grouped: g.get("prism.hadamard.gdn_v_grouped").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    }
    pub fn is_recurrent(&self, il: usize) -> bool { (il + 1) % self.full_attn_interval != 0 }
    pub fn qk_dim(&self) -> usize { self.ssm_state * self.ssm_groups }
    pub fn conv_dim(&self) -> usize { 2 * self.qk_dim() + self.ssm_inner }
    pub fn head_v_dim(&self) -> usize { self.ssm_inner / self.ssm_vheads }
    /// truncate the trunk (debug / lavapipe parity runs)
    pub fn with_layers(mut self, n: usize) -> Self { self.n_layer = n.min(self.n_layer); self }
}

pub fn load_hadamard(g: &Gguf) -> Result<Option<Hadamard>, String> {
    let Some(bs) = g.get("prism.hadamard.block_size").and_then(|v| v.as_u64()) else { return Ok(None) };
    let mode = g.get("prism.hadamard.sign_mode").and_then(|v| v.as_str()).unwrap_or("none");
    let mut signs = HashMap::new();
    if mode == "explicit" {
        let w = g.get("prism.hadamard.sign_widths").and_then(|v| v.as_arr()).ok_or("sign_widths")?;
        let v = g.get("prism.hadamard.sign_values").and_then(|v| v.as_arr()).ok_or("sign_values")?;
        let mut off = 0;
        for wv in w { let n = wv.as_u64().unwrap() as usize; signs.insert(n, v[off..off + n].iter().map(|x| x.as_i64().unwrap() as f32).collect()); off += n; }
        if off != v.len() { return Err("hadamard sign length mismatch".into()); }
    } else if mode != "none" { return Err(format!("unsupported sign_mode {mode}")); }
    Ok(Some(Hadamard { block: bs as usize, signs }))
}

/// A weight matrix view straight out of the mmap.
#[derive(Clone, Copy)]
pub struct WRef<'a> { pub ty: GgmlType, pub data: &'a [u8], pub rows: usize, pub cols: usize, pub hadamard: bool }

impl<'a> WRef<'a> {
    pub fn new(g: &'a Gguf, cfg: &Config, name: &str) -> Result<Self, String> {
        let t: &TensorInfo = g.tensor(name)?;
        Ok(WRef { ty: t.ty, data: g.data(t), cols: t.dims[0], rows: t.dims.get(1).copied().unwrap_or(1), hadamard: cfg.had_weights.iter().any(|w| w == name) })
    }
    pub fn row_bytes(&self) -> usize { self.ty.row_bytes(self.cols) }
    pub fn f32_vec(&self) -> Vec<f32> {
        let mut o = vec![0f32; self.rows * self.cols];
        match self.ty {
            GgmlType::F32 => for (i, c) in self.data.chunks_exact(4).enumerate() { o[i] = f32::from_le_bytes(c.try_into().unwrap()); },
            GgmlType::BF16 => for (i, c) in self.data.chunks_exact(2).enumerate() { o[i] = crate::quant::bf16_to_f32(c[0], c[1]); },
            GgmlType::F16 => for (i, c) in self.data.chunks_exact(2).enumerate() { o[i] = crate::quant::f16_to_f32(c[0], c[1]); },
            GgmlType::PQ2_0 => crate::quant::dequant_pq2_0(self.data, &mut o),
            GgmlType::PTQ1_0 => crate::quant::dequant_ptq1_0(self.data, &mut o),
            GgmlType::Q8_0 => crate::quant::dequant_q8_0(self.data, &mut o),
            GgmlType::Q4_K => crate::quant::dequant_q4_k(self.data, &mut o),
            GgmlType::Q6_K => crate::quant::dequant_q6_k(self.data, &mut o),
            t => panic!("f32_vec {t:?}"),
        }
        o
    }
    pub fn row_f32(&self, r: usize) -> Vec<f32> {
        let rb = self.row_bytes();
        WRef { data: &self.data[r * rb..(r + 1) * rb], rows: 1, ..*self }.f32_vec()
    }
}

pub fn layer_names(cfg: &Config, il: usize) -> Vec<String> {
    let p = |s: &str| format!("blk.{il}.{s}");
    let mut v = vec![p("attn_norm.weight"), p("post_attention_norm.weight"), p("ffn_gate.weight"), p("ffn_up.weight"), p("ffn_down.weight")];
    if cfg.is_recurrent(il) {
        for s in ["attn_qkv.weight", "attn_gate.weight", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_beta.weight", "ssm_alpha.weight", "ssm_norm.weight", "ssm_out.weight"] { v.push(p(s)); }
    } else {
        for s in ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"] { v.push(p(s)); }
    }
    v
}
