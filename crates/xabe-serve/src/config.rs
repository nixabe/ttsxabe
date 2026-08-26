//! What the gateway needs to know to run a conversation.
//!
//! Every default here is a measurement or a mistake that was paid for once.
//! The comments say which; changing a number without reading the reason is how
//! the pipeline regressed the first time.

use std::collections::BTreeMap;

/// The chat prompt, the chunking thresholds and where the stages live.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// The system prompt, including its few-shot turns.
    pub system_prompt: String,
    /// What the user is called in the transcript.
    pub person: String,
    /// What the assistant is called in the transcript.
    pub bot: String,
    /// Sampling temperature for the reply.
    pub temperature: f32,
    /// Maximum reply length in tokens.
    pub max_tokens: u32,
    /// How many previous turns to keep in the prompt.
    pub history_turns: usize,
    /// Minimum characters before a *later* chunk is synthesised.
    pub min_chunk: usize,
    /// Minimum characters before the *first* chunk is synthesised.
    pub first_chunk: usize,
    /// Language given to the ASR.
    pub asr_lang: String,
    /// TTS engines by name, as base URLs or the local marker.
    pub tts_engines: BTreeMap<String, String>,
    /// Which engine the page selects on load.
    pub tts_default: String,
    /// When a turn starts and ends. See [`crate::turntaking`].
    pub turn: crate::turntaking::TurnPolicy,
}

/// The name the local, in-process TTS is registered under.
pub const LOCAL_ENGINE: &str = "mms";

impl Default for GatewayConfig {
    fn default() -> Self {
        GatewayConfig {
            system_prompt: direct_taigi_prompt("使用者", "小助理"),
            person: "使用者".into(),
            bot: "小助理".into(),
            // Low but not zero. The reply is spoken, so an occasional identical
            // answer is more noticeable than a slightly varied one.
            temperature: 0.3,
            // The prompt asks for at most two sentences, because the reply is
            // read aloud; this is the backstop for when it does not comply.
            max_tokens: 160,
            history_turns: 6,
            // Six characters is about the shortest Han clause worth a whole
            // synthesis round trip.
            min_chunk: 6,
            // Four for the first, deliberately lower: getting the voice started
            // is worth more than getting the first clause exactly right.
            first_chunk: 4,
            // Never `en`. Breeze-ASR-26 transcribes Taigi speech *into*
            // Mandarin Han, and asking it for English gets a translation.
            asr_lang: "zh".into(),
            tts_engines: BTreeMap::new(),
            tts_default: LOCAL_ENGINE.into(),
            turn: crate::turntaking::TurnPolicy::default(),
        }
    }
}

/// The prompt that makes the chat model answer in Taigi Han itself.
///
/// The few-shot turns are load-bearing, not decoration. Breeze2 writes real
/// Taigi — 毋過, 真濟, 食飽 — only when shown examples; without them it writes
/// Mandarin transliterated into Taigi-looking characters, which the synthesiser
/// then pronounces as nonsense.
///
/// This removes an entire model hop. Measured 3.8 s → 1.6 s on a voice turn,
/// and it frees the 13 B translator's VRAM.
pub fn direct_taigi_prompt(person: &str, bot: &str) -> String {
    format!(
        "以下是 {person} 佮台語助理 {bot} 咧講話的紀錄。\n\
         {bot} 干焦用台語漢字回答，袂使用華語。\n\
         {bot} 的回答愛真短，上濟兩句，因為會唸出聲。\n\
         無註解、無括號、無 Markdown。\n\n\
         {person}: 你好\n{bot}: 你好！有啥物代誌我會使鬥相共？\n\
         {person}: 台北今仔日天氣按怎？\n{bot}: 台北今仔日好天，溫度差不多二十五度。\n\
         {person}: 你食飽未？\n{bot}: 食飽矣，多謝關心。"
    )
}

/// The prompt used when a translator sits between the chat model and the TTS.
pub fn mandarin_prompt(person: &str, bot: &str) -> String {
    format!(
        "以下是 {person} 與一位名為 {bot} 的助理之間的對話。\n\
         {bot} 親切、友善，一律使用台灣繁體中文回答。\n\
         {bot} 的回答必須簡短，最多兩句話，因為內容會被唸出聲音。\n\
         沒有註解、沒有括號、沒有 Markdown。"
    )
}

impl GatewayConfig {
    /// Builds the completion prompt for one turn.
    pub fn build_prompt(&self, history: &[(Role, String)], user_text: &str) -> String {
        let mut lines = vec![self.system_prompt.clone(), String::new()];
        let keep = self.history_turns * 2;
        let start = history.len().saturating_sub(keep);
        for (role, text) in &history[start..] {
            let who = match role {
                Role::User => &self.person,
                Role::Bot => &self.bot,
            };
            lines.push(format!("{who}: {text}"));
        }
        lines.push(format!("{}: {user_text}", self.person));
        lines.push(format!("{}:", self.bot));
        lines.join("\n")
    }

    /// The body for a streamed reply.
    pub fn completion_body(&self, prompt: &str) -> serde_json::Value {
        serde_json::json!({
            "prompt": prompt,
            "stream": true,
            "temperature": self.temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "n_predict": self.max_tokens,
            // Without these the model writes the user's next turn for them.
            "stop": [format!("{}:", self.person), format!("{}:", self.bot), "\n\n"],
        })
    }
}

/// Who said a line of the transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The person.
    User,
    /// The assistant.
    Bot,
}
