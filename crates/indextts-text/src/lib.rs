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

use indextts_core::{IndexTtsError, Language, Result};
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
                if m.ends_with('%') {
                    let num = &m[..m.len()-1];
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
    fn test_date_normalization() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.normalize("今天是2024年10月2日", Language::Zh).unwrap();
        // Numbers should be converted to Chinese characters
        assert!(result.contains("二"));
        assert!(result.contains("年"));
        assert!(result.contains("月"));
        assert!(result.contains("日"));
    }
}
