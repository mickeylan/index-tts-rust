//! Tokenizer for IndexTTS-2.5
//!
//! Handles tokenization of text using tiktoken-style BPE tokenizer
//! and pinyin vocabulary for Chinese text.

use indextts_core::{IndexTtsError, Language, Result, GenerationConfig};
use std::collections::HashMap;
use std::path::Path;

/// Tiktoken-style BPE tokenizer
#[derive(Debug, Clone)]
pub struct TiktokenTokenizer {
    /// Merge ranks (BPE)
    merges: HashMap<(u32, u32), u32>,
    /// Encoder (vocab)
    encoder: HashMap<String, u32>,
    /// Special tokens
    special_tokens: HashMap<String, u32>,
    /// Decoder
    decoder: HashMap<u32, String>,
    /// Special token IDs
    special_ids: HashMap<u32, String>,
}

impl TiktokenTokenizer {
    /// Load tokenizer from tiktoken file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_str(&content)
    }

    /// Parse tiktoken content
    pub fn from_str(content: &str) -> Result<Self> {
        let mut encoder: HashMap<String, u32> = HashMap::new();
        let mut decoder: HashMap<u32, String> = HashMap::new();
        let mut merges: HashMap<(u32, u32), u32> = HashMap::new();
        let mut special_tokens: HashMap<String, u32> = HashMap::new();
        let mut special_ids: HashMap<u32, String> = HashMap::new();

        let mut lines = content.lines();
        
        // Skip header
        while let Some(line) = lines.next() {
            if line.starts_with("|") {
                break;
            }
        }

        // Parse vocab and merges
        for line in lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            if line.starts_with("|") {
                // Special token line: |token|<space>id|
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let token = parts[0].trim_start_matches('|').to_string();
                    if let Ok(id) = parts[1].trim_end_matches('|').parse::<u32>() {
                        encoder.insert(token.clone(), id);
                        decoder.insert(id, token.clone());
                        special_tokens.insert(token, id);
                        special_ids.insert(id, parts[0].trim_start_matches('|').to_string());
                    }
                }
            } else if line.contains(' ') || line.contains('\t') {
                // Merge line: "A B" rank
                let parts: Vec<&str> = if line.contains('\t') {
                    line.split('\t').collect()
                } else {
                    line.split(' ').collect()
                };
                if parts.len() >= 2 {
                    let first = parts[0];
                    let second = parts[1];
                    let rank = if let Ok(r) = parts[parts.len()-1].parse::<u32>() {
                        r
                    } else {
                        continue;
                    };
                    
                    // Get IDs for merge pairs
                    if let (Some(&id1), Some(&id2)) = (encoder.get(first), encoder.get(second)) {
                        merges.insert((id1, id2), rank);
                    }
                }
            } else {
                // Vocab entry: token id
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(id) = parts[1].parse::<u32>() {
                        encoder.insert(parts[0].to_string(), id);
                        decoder.insert(id, parts[0].to_string());
                    }
                }
            }
        }

        Ok(Self {
            merges,
            encoder,
            special_tokens,
            decoder,
            special_ids,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        
        // Simple byte-level encoding with BPE merges
        let bytes: Vec<u8> = text.as_bytes().to_vec();
        let mut token_ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
        
        // Apply BPE merges
        loop {
            let mut best_merge: Option<((u32, u32), u32)> = None;
            
            for i in 0..token_ids.len().saturating_sub(1) {
                let pair = (token_ids[i], token_ids[i + 1]);
                if let Some(&rank) = self.merges.get(&pair) {
                    if best_merge.map(|(_, r)| rank < r).unwrap_or(true) {
                        best_merge = Some((pair, rank));
                    }
                }
            }
            
            match best_merge {
                Some(((first, second), _)) => {
                    let mut new_tokens = Vec::new();
                    let mut i = 0;
                    while i < token_ids.len() {
                        if i < token_ids.len().saturating_sub(1) 
                            && token_ids[i] == first 
                            && token_ids[i + 1] == second {
                            new_tokens.push(first * 256 + second);
                            i += 2;
                        } else {
                            new_tokens.push(token_ids[i]);
                            i += 1;
                        }
                    }
                    token_ids = new_tokens;
                }
                None => break,
            }
        }
        
        tokens.extend(token_ids);
        Ok(tokens)
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut result = String::new();
        for &token in tokens {
            if let Some(text) = self.decoder.get(&token) {
                result.push_str(text);
            } else {
                // Fallback: convert byte
                result.push(token as u8 as char);
            }
        }
        Ok(result)
    }

    /// Get special token ID
    pub fn special_token_id(&self, token: &str) -> Option<u32> {
        self.special_tokens.get(token).copied()
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }
}

/// Pinyin vocabulary for Chinese text
#[derive(Debug, Clone)]
pub struct PinyinVocab {
    /// Pinyin to ID mapping
    pinyin_to_id: HashMap<String, u32>,
    /// ID to pinyin mapping
    id_to_pinyin: HashMap<u32, String>,
}

impl PinyinVocab {
    /// Load from file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_str(&content)
    }

    /// Parse pinyin vocab content
    pub fn from_str(content: &str) -> Result<Self> {
        let mut pinyin_to_id = HashMap::new();
        let mut id_to_pinyin = HashMap::new();
        
        for (id, line) in content.lines().enumerate() {
            let line = line.trim();
            if !line.is_empty() {
                pinyin_to_id.insert(line.to_string(), id as u32);
                id_to_pinyin.insert(id as u32, line.to_string());
            }
        }
        
        Ok(Self {
            pinyin_to_id,
            id_to_pinyin,
        })
    }

    /// Get pinyin ID
    pub fn get_id(&self, pinyin: &str) -> Option<u32> {
        self.pinyin_to_id.get(pinyin).copied()
    }

    /// Get pinyin string
    pub fn get_pinyin(&self, id: u32) -> Option<&str> {
        self.id_to_pinyin.get(&id).map(|s| s.as_str())
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.pinyin_to_id.len()
    }
}

/// Combined tokenizer for IndexTTS
#[derive(Debug, Clone)]
pub struct IndexTtsTokenizer {
    /// Main BPE tokenizer
    tiktoken: TiktokenTokenizer,
    /// Pinyin vocabulary
    pinyin_vocab: PinyinVocab,
}

impl IndexTtsTokenizer {
    /// Create a new tokenizer
    pub fn new(tiktoken: TiktokenTokenizer, pinyin_vocab: PinyinVocab) -> Self {
        Self {
            tiktoken,
            pinyin_vocab,
        }
    }

    /// Load tokenizer from directory
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let tiktoken_path = dir.join("multilingual_zh_ja_yue_char_del.tiktoken");
        let pinyin_path = dir.join("pinyin.vocab");
        
        let tiktoken = TiktokenTokenizer::from_file(&tiktoken_path)?;
        let pinyin_vocab = PinyinVocab::from_file(&pinyin_path)?;
        
        Ok(Self::new(tiktoken, pinyin_vocab))
    }

    /// Tokenize text with language awareness
    pub fn tokenize(&self, text: &str, _language: Language) -> Result<Vec<u32>> {
        // For now, use simple byte-level encoding
        // Full implementation would handle Chinese text with pinyin
        self.tiktoken.encode(text)
    }

    /// Tokenize for GPT input (with special tokens)
    pub fn tokenize_for_gpt(&self, text: &str, language: Language) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        
        // Add language token
        let lang_token = match language {
            Language::Zh => "[ZH]",
            Language::En => "[EN]",
            Language::Ja => "[JA]",
            Language::Es => "[ES]",
            Language::Ar => "[AR]",
        };
        
        // Add start text token
        tokens.push(0); // START_TEXT_TOKEN
        
        // Tokenize text
        let text_tokens = self.tokenize(text, language)?;
        tokens.extend(text_tokens);
        
        // Add stop text token
        tokens.push(1); // STOP_TEXT_TOKEN
        
        Ok(tokens)
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.tiktoken.vocab_size()
    }
}

/// Language dictionary (from Python tokenizer.py)
pub const LANGUAGE_DICT: &[(&str, u32)] = &[
    ("<zh>", 0),
    ("<en>", 1),
    ("<ja>", 2),
    ("<es>", 3),
    ("<ar>", 4),
];

/// Get language token ID
pub fn language_token_id(language: Language) -> u32 {
    match language {
        Language::Zh => 0,
        Language::En => 1,
        Language::Ja => 2,
        Language::Es => 3,
        Language::Ar => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_token_id() {
        assert_eq!(language_token_id(Language::Zh), 0);
        assert_eq!(language_token_id(Language::En), 1);
        assert_eq!(language_token_id(Language::Ja), 2);
        assert_eq!(language_token_id(Language::Es), 3);
        assert_eq!(language_token_id(Language::Ar), 4);
    }
}
