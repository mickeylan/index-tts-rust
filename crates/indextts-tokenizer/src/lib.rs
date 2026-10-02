//! IndexTTS-2.5 tokenizer backed by the official tiktoken vocabulary.

use base64::{engine::general_purpose::STANDARD, Engine};
use indextts_core::{IndexTtsError, Language, Result};
use rustc_hash::FxHashMap;
use std::{collections::HashMap, fmt, path::Path};
use tiktoken_rs::CoreBPE;

const PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

const LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
    "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu",
    "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa",
    "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
    "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
    "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw",
    "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh", "common",
];
const AUDIO_EVENTS: &[&str] = &["ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter", "/Laughter", "Applause", "/Applause"];
const EMOTIONS: &[&str] = &["HAPPY", "SAD", "ANGRY", "NEUTRAL"];
const TTS_TOKENS: &[&str] = &["TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H", "TTS/SP01", "TTS/SP02", "TTS/SP03", "TTS/SP04", "TTS/SP05", "TTS/SP06", "TTS/SP07", "TTS/SP08", "TTS/SP09", "TTS/SP10", "TTS/SP11", "TTS/SP12", "TTS/SP13"];

#[derive(Clone)]
pub struct TiktokenTokenizer {
    bpe: CoreBPE,
    special_tokens: HashMap<String, u32>,
    vocab_size: usize,
}

impl fmt::Debug for TiktokenTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TiktokenTokenizer").field("vocab_size", &self.vocab_size).finish()
    }
}

impl TiktokenTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_str(&std::fs::read_to_string(path)?)
    }

    pub fn from_str(content: &str) -> Result<Self> {
        let mut ranks = FxHashMap::default();
        for (line_number, line) in content.lines().enumerate() {
            if line.trim().is_empty() { continue; }
            let mut parts = line.split_whitespace();
            let token = parts.next().ok_or_else(|| IndexTtsError::InvalidModel(format!("invalid tiktoken line {}", line_number + 1)))?;
            let rank: u32 = parts.next().ok_or_else(|| IndexTtsError::InvalidModel(format!("missing rank on tiktoken line {}", line_number + 1)))?
                .parse().map_err(|error| IndexTtsError::InvalidModel(format!("invalid rank on line {}: {error}", line_number + 1)))?;
            // Python's base64 decoder accepts the historical single "=" entry
            // in the official vocabulary and decodes it as empty bytes.
            let bytes = if token == "=" {
                Vec::new()
            } else {
                STANDARD.decode(token).map_err(|error| IndexTtsError::InvalidModel(format!("invalid base64 token on line {}: {error}", line_number + 1)))?
            };
            ranks.insert(bytes, rank);
        }
        if ranks.is_empty() {
            return Err(IndexTtsError::InvalidModel("empty tiktoken vocabulary".into()));
        }
        let mut specials = Vec::new();
        specials.push("<|endoftext|>".to_owned());
        specials.push("<|startoftranscript|>".to_owned());
        specials.extend(LANGUAGES.iter().take(99).map(|value| format!("<|{value}|>")));
        specials.extend(AUDIO_EVENTS.iter().map(|value| format!("<|{value}|>")));
        specials.extend(EMOTIONS.iter().map(|value| format!("<|{value}|>")));
        specials.extend(["translate", "transcribe", "startoflm", "startofprev", "nospeech", "notimestamps"].map(|value| format!("<|{value}|>")));
        specials.extend((1..=30).map(|index| format!("<|SPECIAL_TOKEN_{index}|>")));
        specials.extend(TTS_TOKENS.iter().map(|value| format!("<|{value}|>")));
        specials.extend((0..=1500).map(|index| format!("<|{:.2}|>", index as f32 * 0.02)));
        let base_size = ranks.len();
        let special_tokens: FxHashMap<String, u32> = specials.into_iter().enumerate()
            .map(|(offset, token)| (token, (base_size + offset) as u32)).collect();
        let exposed_specials = special_tokens.iter()
            .map(|(token, rank)| (token.clone(), *rank)).collect();
        let bpe = CoreBPE::new(ranks, special_tokens, PATTERN)
            .map_err(|error| IndexTtsError::InvalidModel(format!("failed to construct tiktoken BPE: {error}")))?;
        Ok(Self { bpe, special_tokens: exposed_specials, vocab_size: base_size })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.bpe.encode_with_special_tokens(text).into_iter().map(|value| value as u32).collect())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.bpe.decode(tokens.to_vec())
            .map_err(|error| IndexTtsError::InvalidText(error.to_string()))
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_ordinary(text).into_iter().map(|value| value as u32).collect()
    }

    pub fn special_token_id(&self, token: &str) -> Option<u32> {
        self.special_tokens.get(token).copied()
    }

    pub fn vocab_size(&self) -> usize { self.vocab_size }
}

#[derive(Debug, Clone)]
pub struct PinyinVocab {
    pinyin_to_id: HashMap<String, u32>,
    id_to_pinyin: HashMap<u32, String>,
}

impl PinyinVocab {
    pub fn from_file(path: &Path) -> Result<Self> { Self::from_str(&std::fs::read_to_string(path)?) }
    pub fn from_str(content: &str) -> Result<Self> {
        let mut pinyin_to_id = HashMap::new();
        let mut id_to_pinyin = HashMap::new();
        for (id, line) in content.lines().enumerate() {
            let value = line.trim();
            if !value.is_empty() {
                pinyin_to_id.insert(value.to_owned(), id as u32);
                id_to_pinyin.insert(id as u32, value.to_owned());
            }
        }
        Ok(Self { pinyin_to_id, id_to_pinyin })
    }
    pub fn get_id(&self, value: &str) -> Option<u32> { self.pinyin_to_id.get(value).copied() }
    pub fn get_pinyin(&self, id: u32) -> Option<&str> { self.id_to_pinyin.get(&id).map(String::as_str) }
    pub fn vocab_size(&self) -> usize { self.pinyin_to_id.len() }
}

#[derive(Debug, Clone)]
pub struct IndexTtsTokenizer {
    tiktoken: TiktokenTokenizer,
    pinyin_vocab: PinyinVocab,
}

impl IndexTtsTokenizer {
    pub fn new(tiktoken: TiktokenTokenizer, pinyin_vocab: PinyinVocab) -> Self { Self { tiktoken, pinyin_vocab } }
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Ok(Self::new(
            TiktokenTokenizer::from_file(&dir.join("multilingual_zh_ja_yue_char_del.tiktoken"))?,
            PinyinVocab::from_file(&dir.join("pinyin.vocab"))?,
        ))
    }
    pub fn tokenize(&self, text: &str, language: Language) -> Result<Vec<u32>> {
        let special = format!("<|{}|>", language.code().to_lowercase());
        let language_id = self.tiktoken.special_token_id(&special)
            .ok_or_else(|| IndexTtsError::InvalidModel(format!("missing language token {special}")))?;
        let mut tokens = vec![language_id];
        tokens.extend(self.tiktoken.encode_ordinary(&format!(" {text}")));
        Ok(tokens)
    }
    /// The GPT model itself adds text start/stop tokens; this returns raw tokenizer IDs.
    pub fn tokenize_for_gpt(&self, text: &str, language: Language) -> Result<Vec<u32>> {
        self.tokenize(text, language)
    }
    pub fn vocab_size(&self) -> usize { self.tiktoken.vocab_size() }
    pub fn pinyin_vocab(&self) -> &PinyinVocab { &self.pinyin_vocab }
}

pub fn language_token_id(language: Language) -> u32 {
    match language { Language::En => 0, Language::Zh => 1, Language::Es => 3, Language::Ja => 7, Language::Ar => 13 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_tokenizer_matches_python() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/model-export/tokenizer-fixture");
        if !dir.exists() { return; }
        let tokenizer = IndexTtsTokenizer::from_dir(&dir).unwrap();
        assert_eq!(tokenizer.tokenize("你好世界", Language::Zh).unwrap(), vec![58839, 220, 48934, 50371, 48721, 53743]);
        assert_eq!(tokenizer.tokenize("Hello world", Language::En).unwrap(), vec![58838, 2415, 1002]);
    }

    #[test]
    fn language_ids_follow_official_dictionary_order() {
        assert_eq!(language_token_id(Language::En), 0);
        assert_eq!(language_token_id(Language::Zh), 1);
        assert_eq!(language_token_id(Language::Ja), 7);
        assert_eq!(language_token_id(Language::Es), 3);
        assert_eq!(language_token_id(Language::Ar), 13);
    }
}
