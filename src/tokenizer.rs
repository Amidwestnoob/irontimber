//! Byte-level BPE tokenizer for the qwen35 vocabulary (tokenizer.ggml.model = gpt2, pre = qwen35).
use crate::gguf::Gguf;
use fancy_regex::Regex;
use std::collections::HashMap;

pub const QWEN35_PRE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    pub tokens: Vec<String>, pub ids: HashMap<String, u32>, pub types: Vec<i32>,
    ranks: HashMap<(String, String), usize>, specials: Vec<(String, u32)>,
    re: Regex, byte_enc: [char; 256], byte_dec: HashMap<char, u8>,
    pub bos: Option<u32>, pub eos: u32, pub add_bos: bool, pub eog: Vec<u32>,
}

fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone(); let mut n = 0;
    for b in 0..256u32 { if !bs.contains(&b) { bs.push(b); cs.push(256 + n); n += 1; } }
    let mut t = [' '; 256];
    for (b, c) in bs.iter().zip(cs) { t[*b as usize] = char::from_u32(c).unwrap(); }
    t
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> {
        let arr = |k: &str| g.get(k).and_then(|v| v.as_arr()).ok_or_else(|| format!("missing {k}"));
        let tokens: Vec<String> = arr("tokenizer.ggml.tokens")?.iter().map(|v| v.as_str().unwrap().to_string()).collect();
        let types: Vec<i32> = arr("tokenizer.ggml.token_type")?.iter().map(|v| v.as_i64().unwrap() as i32).collect();
        let mut ranks = HashMap::new();
        for (i, m) in arr("tokenizer.ggml.merges")?.iter().enumerate() {
            let s = m.as_str().unwrap(); let (a, b) = s.split_once(' ').ok_or("bad merge")?;
            ranks.insert((a.to_string(), b.to_string()), i);
        }
        let ids: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        // control (3) and user-defined (4) tokens are matched verbatim before pre-tokenisation
        let mut specials: Vec<(String, u32)> = types.iter().enumerate().filter(|(_, &t)| t == 3 || t == 4).map(|(i, _)| (tokens[i].clone(), i as u32)).filter(|(s, _)| !s.is_empty()).collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let byte_enc = bytes_to_unicode();
        let byte_dec = byte_enc.iter().enumerate().map(|(i, c)| (*c, i as u8)).collect();
        let eos = g.u("tokenizer.ggml.eos_token_id")? as u32;
        let mut eog = vec![eos];
        for s in ["<|im_end|>", "<|endoftext|>", "<|end_of_text|>"] { if let Some(&i) = ids.get(s) { if !eog.contains(&i) { eog.push(i); } } }
        Ok(Tokenizer { bos: g.u("tokenizer.ggml.bos_token_id").ok().map(|v| v as u32),
            add_bos: g.get("tokenizer.ggml.add_bos_token").and_then(|v| v.as_bool()).unwrap_or(false),
            re: Regex::new(QWEN35_PRE).map_err(|e| e.to_string())?, tokens, ids, types, ranks, specials, byte_enc, byte_dec, eos, eog })
    }

    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        let enc: String = word.bytes().map(|b| self.byte_enc[b as usize]).collect();
        if let Some(&id) = self.ids.get(&enc) { out.push(id); return; }
        let mut parts: Vec<String> = enc.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.map_or(true, |(br, _)| r < br) { best = Some((r, i)); }
                }
            }
            let Some((_, i)) = best else { break };
            let m = format!("{}{}", parts[i], parts[i + 1]);
            parts[i] = m; parts.remove(i + 1);
        }
        for p in parts {
            match self.ids.get(&p) { Some(&id) => out.push(id), None => for ch in p.chars() { if let Some(&id) = self.ids.get(&ch.to_string()) { out.push(id); } } }
        }
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for m in self.re.find_iter(text) { let m = m.expect("regex"); self.bpe(m.as_str(), out); }
    }

    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special && self.add_bos { if let Some(b) = self.bos { out.push(b); } }
        if !parse_special { self.encode_plain(text, &mut out); return out; }
        let mut rest = text;
        while !rest.is_empty() {
            // earliest special occurrence (longest on ties)
            let mut hit: Option<(usize, &str, u32)> = None;
            for (s, id) in &self.specials {
                if let Some(p) = rest.find(s.as_str()) { if hit.map_or(true, |(hp, hs, _)| p < hp || (p == hp && s.len() > hs.len())) { hit = Some((p, s, *id)); } }
            }
            match hit {
                Some((p, s, id)) => { self.encode_plain(&rest[..p], &mut out); out.push(id); rest = &rest[p + s.len()..]; }
                None => { self.encode_plain(rest, &mut out); break; }
            }
        }
        out
    }

    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let t = &self.tokens[id as usize];
        if matches!(self.types.get(id as usize), Some(3) | Some(4)) { return t.as_bytes().to_vec(); }
        t.chars().map(|c| self.byte_dec.get(&c).copied().unwrap_or(b'?')).collect()
    }
    pub fn decode(&self, ids: &[u32]) -> String { String::from_utf8_lossy(&ids.iter().flat_map(|&i| self.token_bytes(i)).collect::<Vec<_>>()).into_owned() }
    pub fn is_eog(&self, id: u32) -> bool { self.eog.contains(&id) }
}
