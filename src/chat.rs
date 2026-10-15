//! Chat formatting with the GGUF's own Jinja template (minijinja + Python-compat methods).
use crate::gguf::Gguf;
use serde_json::Value;

pub struct ChatTemplate { src: String }

impl ChatTemplate {
    pub fn from_gguf(g: &Gguf) -> Option<Self> { g.get("tokenizer.chat_template").and_then(|v| v.as_str()).map(|s| ChatTemplate { src: s.to_string() }) }
    /// `messages`: [{role, content}], `extra`: e.g. {"enable_thinking": false}
    pub fn render(&self, messages: &Value, add_generation_prompt: bool, extra: &Value) -> Result<String, String> {
        let mut env = minijinja::Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |m: String| -> Result<String, minijinja::Error> { Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, m)) });
        env.add_template("chat", &self.src).map_err(|e| e.to_string())?;
        let mut ctx = serde_json::json!({"messages": messages, "add_generation_prompt": add_generation_prompt, "bos_token": "", "eos_token": ""});
        if let (Some(o), Some(e)) = (ctx.as_object_mut(), extra.as_object()) { for (k, v) in e { o.insert(k.clone(), v.clone()); } }
        env.get_template("chat").unwrap().render(minijinja::Value::from_serialize(&ctx)).map_err(|e| format!("{e:#}"))
    }
}
