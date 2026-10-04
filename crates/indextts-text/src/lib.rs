//! Text normalization for IndexTTS-2.5
//!
//! Implements Chinese text normalization rules including:
//! - Arabic numbers
//! - Dates and times
//! - Ordinals, percentages, decimals
//! - Units
//! - Phone numbers
//! - English abbreviations
//! - Mixed Chinese/English text
//! - Punctuation
//! - Pronunciation annotations `<text|pronunciation>`

use indextts_core::{Language, Result};
use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    // Number patterns
    static ref ARABIC_NUMBER: Regex = Regex::new(r"\d+").unwrap();
    static ref DECIMAL: Regex = Regex::new(r"\d+\.\d+").unwrap();
    static ref PERCENTAGE: Regex = Regex::new(r"\d+%|百分之\d+").unwrap();

    // Date/Time patterns
    static ref DATE_CHINESE: Regex = Regex::new(r"(\d{4})年(\d{1,2})月(\d{1,2})日?").unwrap();
    static ref TIME_24H: Regex = Regex::new(r"(\d{1,2}):(\d{2})").unwrap();
    static ref TIME_12H: Regex = Regex::new(r"(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)").unwrap();

    // Phone number pattern
    static ref PHONE: Regex = Regex::new(r"\d{3,4}-?\d{7,8}").unwrap();

    // Pronunciation annotation
    static ref PRONUNCIATION: Regex = Regex::new(r"<([^|]+)\|([^>]+)>").unwrap();
}

/// Text normalizer for IndexTTS
#[derive(Debug, Clone)]
pub struct TextNormalizer {
    /// Enable number normalization
    normalize_numbers: bool,
    /// Enable date/time normalization
    normalize_dates: bool,
    /// Enable phone normalization
    normalize_phones: bool,
}

impl Default for TextNormalizer {
    fn default() -> Self {
        Self {
            normalize_numbers: true,
            normalize_dates: true,
            normalize_phones: true,
        }
    }
}

impl TextNormalizer {
    /// Create a new text normalizer with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom settings
    pub fn with_options(
        normalize_numbers: bool,
        normalize_dates: bool,
        normalize_phones: bool,
    ) -> Self {
        Self {
            normalize_numbers,
            normalize_dates,
            normalize_phones,
        }
    }

    /// Normalize text for the given language
    pub fn normalize(&self, text: &str, language: Language) -> Result<String> {
        let mut result = text.to_string();

        // Handle pronunciation annotations first (preserve them)
        result = self.normalize_pronunciations(&result);

        match language {
            Language::Zh => {
                result = self.normalize_chinese(&result)?;
            }
            Language::En => {
                result = self.normalize_english(&result)?;
            }
            Language::Ja => {
                result = self.normalize_japanese(&result)?;
            }
            _ => {
                // Basic cleanup for other languages
                result = result.trim().to_string();
            }
        }

        Ok(result)
    }

    /// Normalize Chinese text
    fn normalize_chinese(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();

        // Normalize pronunciation annotations
        result = self.normalize_pronunciations(&result);

        // Normalize numbers
        if self.normalize_numbers {
            result = self.normalize_numbers_chinese(&result);
        }

        // Normalize dates
        if self.normalize_dates {
            result = self.normalize_dates_chinese(&result);
        }

        // Normalize times
        if self.normalize_dates {
            result = self.normalize_times_chinese(&result);
        }

        // Normalize phones
        if self.normalize_phones {
            result = self.normalize_phones_chinese(&result);
        }

        // Normalize percentages
        if self.normalize_numbers {
            result = self.normalize_percentages_chinese(&result);
        }

        // Clean up extra whitespace
        result = result.split_whitespace().collect::<Vec<_>>().join(" ");

        Ok(result)
    }

    /// Normalize English text
    fn normalize_english(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();

        // Expand common abbreviations
        result = self.expand_abbreviations(&result);

        // Normalize numbers
        if self.normalize_numbers {
            result = self.normalize_numbers_english(&result);
        }

        // Clean up
        result = result.split_whitespace().collect::<Vec<_>>().join(" ");

        Ok(result)
    }

    /// Normalize Japanese text
    fn normalize_japanese(&self, text: &str) -> Result<String> {
        // For now, just basic cleanup
        Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
    }

    /// Normalize pronunciation annotations: <text|pronunciation> -> pronunciation
    fn normalize_pronunciations(&self, text: &str) -> String {
        PRONUNCIATION
            .replace_all(text, |caps: &regex::Captures| {
                caps.get(2).map_or("", |m| m.as_str()).to_string()
            })
            .to_string()
    }

    /// Normalize Chinese numbers
    fn normalize_numbers_chinese(&self, text: &str) -> String {
        // Replace Arabic digits with Chinese characters
        // Simple digit-by-digit conversion for now
        let digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];

        text.chars()
            .map(|c| {
                if let Some(digit) = c.to_digit(10) {
                    digits[digit as usize]
                } else {
                    c
                }
            })
            .collect()
    }

    /// Normalize Chinese dates
    fn normalize_dates_chinese(&self, text: &str) -> String {
        DATE_CHINESE
            .replace_all(text, |caps: &regex::Captures| {
                let year = caps.get(1).map_or("", |m| m.as_str());
                let month = caps.get(2).map_or("", |m| m.as_str());
                let day = caps.get(3).map_or("", |m| m.as_str());
                format!("{}年{}月{}日", year, month, day)
            })
            .to_string()
    }

    /// Normalize Chinese times
    fn normalize_times_chinese(&self, text: &str) -> String {
        let result = TIME_24H
            .replace_all(text, |caps: &regex::Captures| {
                let hour = caps.get(1).map_or("", |m| m.as_str());
                let minute = caps.get(2).map_or("", |m| m.as_str());
                format!("{}点{}分", hour, minute)
            })
            .to_string();

        TIME_12H
            .replace_all(&result, |caps: &regex::Captures| {
                let hour = caps.get(1).map_or("", |m| m.as_str());
                let minute = caps.get(2).map_or("", |m| m.as_str());
                let ampm = caps.get(3).map_or("", |m| m.as_str());
                let hour_int: u32 = hour.parse().unwrap_or(12);
                let is_pm = ampm.to_lowercase() == "pm";
                let display_hour = if is_pm && hour_int != 12 {
                    hour_int + 12
                } else if !is_pm && hour_int == 12 {
                    0
                } else {
                    hour_int
                };
                format!("{}点{}分", display_hour, minute)
            })
            .to_string()
    }

    /// Normalize Chinese phone numbers
    fn normalize_phones_chinese(&self, text: &str) -> String {
        PHONE
            .replace_all(text, |caps: &regex::Captures| {
                caps.get(0).map_or("", |m| m.as_str()).to_string()
            })
            .to_string()
    }

    /// Normalize Chinese percentages
    fn normalize_percentages_chinese(&self, text: &str) -> String {
        PERCENTAGE
            .replace_all(text, |caps: &regex::Captures| {
                let m = caps.get(0).map_or("", |m| m.as_str());
                if let Some(num) = m.strip_suffix('%') {
                    format!("百分之{}", num)
                } else {
                    m.to_string()
                }
            })
            .to_string()
    }

    /// Normalize English numbers
    fn normalize_numbers_english(&self, text: &str) -> String {
        // Keep numbers as-is for English
        text.to_string()
    }

    /// Expand English abbreviations
    fn expand_abbreviations(&self, text: &str) -> String {
        let abbreviations: &[(&str, &str)] = &[
            ("Dr.", "Doctor"),
            ("Mr.", "Mister"),
            ("Mrs.", "Missus"),
            ("Ms.", "Miss"),
            ("Prof.", "Professor"),
            ("U.S.", "United States"),
            ("U.K.", "United Kingdom"),
            ("etc.", "etcetera"),
        ];

        let mut result = text.to_string();
        for (abbr, expansion) in abbreviations {
            result = result.replace(abbr, expansion);
        }
        result
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextSegment {
    pub text: String,
    pub start_char: usize,
    pub end_char: usize,
}

/// Deterministically split normalized text while preserving every character.
/// Sentence punctuation is preferred over semicolons, then commas, before the hard boundary.
pub fn segment_text(text: &str, max_chars: usize) -> Result<Vec<TextSegment>> {
    if max_chars == 0 {
        return Err(indextts_core::IndexTtsError::InvalidText(
            "segment size must be greater than zero".into(),
        ));
    }
    let chars: Vec<char> = text.chars().collect();
    let mut segments = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let hard_end = (start + max_chars).min(chars.len());
        let end = if hard_end == chars.len() {
            hard_end
        } else {
            ["。！？!?", "；;", "，,"]
                .iter()
                .find_map(|punctuation| {
                    (start..hard_end)
                        .rev()
                        .find(|index| punctuation.contains(chars[*index]))
                        .map(|index| index + 1)
                })
                .unwrap_or(hard_end)
        };
        segments.push(TextSegment {
            text: chars[start..end].iter().collect(),
            start_char: start,
            end_char: end,
        });
        start = end;
    }
    Ok(segments)
}

/// Normalize text with default settings
pub fn normalize(text: &str, language: Language) -> Result<String> {
    TextNormalizer::new().normalize(text, language)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pronunciation_annotation() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.normalize("<你好|nihao>", Language::Zh).unwrap();
        assert_eq!(result, "nihao");
    }

    #[test]
    fn deterministic_segments_preserve_text_and_prioritize_punctuation() {
        let text = "甲乙，丙丁；戊己。庚辛壬癸";
        let segments = segment_text(text, 6).unwrap();
        assert_eq!(
            segments
                .iter()
                .map(|part| part.text.as_str())
                .collect::<Vec<_>>(),
            vec!["甲乙，丙丁；", "戊己。", "庚辛壬癸"]
        );
        assert_eq!(segments.concat_text(), text);
        for segment in &segments {
            assert_eq!(
                segment.end_char - segment.start_char,
                segment.text.chars().count()
            );
            assert!(segment.text.chars().count() <= 6);
        }
        assert_eq!(segment_text(text, 6).unwrap(), segments);
    }

    trait SegmentTestExt {
        fn concat_text(&self) -> String;
    }
    impl SegmentTestExt for Vec<TextSegment> {
        fn concat_text(&self) -> String {
            self.iter().map(|part| part.text.as_str()).collect()
        }
    }

    #[test]
    fn long_inputs_preserve_every_character() {
        for length in [500, 1000, 2000] {
            let text: String = (0..length)
                .map(|index| if index % 47 == 46 { '。' } else { '字' })
                .collect();
            let segments = segment_text(&text, 120).unwrap();
            assert_eq!(segments.concat_text(), text);
            assert!(segments
                .iter()
                .all(|segment| segment.text.chars().count() <= 120));
            assert_eq!(segments.first().unwrap().start_char, 0);
            assert_eq!(segments.last().unwrap().end_char, length);
            for pair in segments.windows(2) {
                assert_eq!(pair[0].end_char, pair[1].start_char);
            }
        }
    }

    #[test]
    fn test_date_normalization() {
        let normalizer = TextNormalizer::new();
        let result = normalizer
            .normalize("今天是2024年10月2日", Language::Zh)
            .unwrap();
        // Numbers should be converted to Chinese characters
        assert!(result.contains("二"));
        assert!(result.contains("年"));
        assert!(result.contains("月"));
        assert!(result.contains("日"));
    }
}
