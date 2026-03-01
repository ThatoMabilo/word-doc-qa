// ============================================================
// src/inference.rs  – Load model and answer questions
// ============================================================

use crate::{
    data::Vocabulary,
    model::{TransformerQAConfig, TransformerQAModel},
};
use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Int, Tensor},
};

// ── 1. INFERENCE ENGINE ──────────────────────────────────────

pub struct QAInferenceEngine<B: Backend> {
    model: TransformerQAModel<B>,
    vocab: Vocabulary,
    max_seq_len: usize,
    device: B::Device,
}

impl<B: Backend> QAInferenceEngine<B> {
    /// Load a trained model and vocabulary from disk.
    pub fn load(model_path: &str, vocab_path: &str, max_seq_len: usize, device: B::Device) -> Self {
        // Load vocabulary
        let vocab_json = std::fs::read_to_string(vocab_path).expect("Cannot read vocab file");
        let vocab: Vocabulary = serde_json::from_str(&vocab_json).expect("Cannot parse vocab file");

        println!("Loaded vocabulary with {} tokens", vocab.size());

        // Build model config and initialise
        let config = TransformerQAConfig {
            vocab_size: vocab.size(),
            max_seq_len,
            d_model: 256,
            num_heads: 8,
            num_layers: 6,
            d_ff: 512,
            dropout: 0.0, // no dropout at inference time
        };
        let record = CompactRecorder::new()
            .load(model_path.into(), &device)
            .expect("Cannot load model checkpoint");

        let model = TransformerQAModel::new(&config, &device).load_record(record);

        println!("Model loaded from {}", model_path);

        Self {
            model,
            vocab,
            max_seq_len,
            device,
        }
    }

    /// Answer a question given a context string.
    pub fn answer(&self, question: &str, context: &str) -> String {
        // Build input: [CLS] question tokens [SEP] context tokens
        let mut input_ids = vec![Vocabulary::CLS];
        input_ids.extend(self.vocab.encode(question));
        input_ids.push(Vocabulary::SEP);
        input_ids.extend(self.vocab.encode(context));
        input_ids.truncate(self.max_seq_len);
        while input_ids.len() < self.max_seq_len {
            input_ids.push(Vocabulary::PAD);
        }

        // Convert to tensor [1, seq_len]
        let input_data: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let input_tensor = Tensor::<B, 1, Int>::from_ints(
            burn::tensor::TensorData::new(input_data, [self.max_seq_len]),
            &self.device,
        )
        .unsqueeze::<2>();

        // Forward pass → logits [1, seq_len, vocab_size]
        let logits = self.model.forward(input_tensor);
        let [_, seq_len, _] = logits.dims();
        let pred_ids = logits.argmax(2).reshape([seq_len]);

        // Convert predicted token ids back to words
        let pred_vec: Vec<i64> = pred_ids.into_data().iter().collect();
        let pred_usize: Vec<usize> = pred_vec.iter().map(|&x| x as usize).collect();

        // Remove padding and special tokens, return answer
        let answer = self.vocab.decode(&pred_usize);
        if answer.trim().is_empty() {
            "I could not find an answer in the documents.".to_string()
        } else {
            answer
        }
    }
}

// ── 2. SIMPLE KEYWORD SEARCH FALLBACK ────────────────────────
// For questions the model struggles with, we also provide a
// keyword search over the raw document text as a fallback.

pub fn keyword_search(question: &str, document_text: &str) -> String {
    let question_lower = question.to_lowercase();
    let stop_words = vec![
        "when", "what", "how", "many", "did", "the", "is", "are", "was", "will", "does", "has",
        "their", "hold", "times", "meet",
    ];
    let expanded_question = question_lower
        .replace("hdc", "higher degrees committee")
        .replace("emc", "executive management committee")
        .replace("src", "student representative council");
    let cleaned_question: String = expanded_question
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    let words: Vec<&str> = cleaned_question.split_whitespace().collect();
    let keywords: Vec<&str> = words
        .iter()
        .enumerate()
        .filter(|(i, w)| {
            let is_term_number = *i > 0 && words[*i - 1] == "term" && (w.len() == 1);
            let is_year = w.len() == 4 && w.chars().all(|c| c.is_numeric());
            (w.len() > 2 || is_term_number || is_year) && !stop_words.contains(*w)
        })
        .map(|(_, w)| *w)
        .collect();

    println!("Searching for keywords: {:?}", keywords);

    // Check if question contains a specific year
    let year_filter: Option<&str> = ["2024", "2025", "2026"]
        .iter()
        .find(|&&y| cleaned_question.contains(y))
        .copied();

    let mut best_line = String::new();
    let mut best_score = 0usize;
    let mut current_year = String::new();

    for line in document_text.lines() {
        // Track current year section
        if line.len() < 20
            && (line.contains("2024") || line.contains("2025") || line.contains("2026"))
        {
            current_year = line.to_string();
        }

        // Skip lines that don't match year filter
        if let Some(y) = year_filter {
            if !current_year.contains(y) {
                continue;
            }
        }

        let line_lower = line.to_lowercase();
        // Score based on non-year keywords only
        let score = keywords
            .iter()
            .filter(|&&kw| kw != "2024" && kw != "2025" && kw != "2026")
            .filter(|&&kw| line_lower.contains(kw))
            .count();

        // Boost score for lines that also contain the year keyword
        // but only for content lines, not month headers
        let year_boost = if let Some(y) = year_filter {
            if line_lower.contains(y) && line.trim().len() > 15 {
                1
            } else {
                0
            }
        } else {
            0
        };

        // Bonus for exact term number match
        // Bonus for exact term number match — only when "term" is in the question
        let term_bonus = if keywords.contains(&"term") {
            if keywords.contains(&"2") && line_lower.contains("term 2") {
                2
            } else if keywords.contains(&"3") && line_lower.contains("term 3") {
                2
            } else if keywords.contains(&"4") && line_lower.contains("term 4") {
                2
            } else if keywords.contains(&"1") && line_lower.contains("term 1") {
                1
            } else {
                0
            }
        } else {
            0
        };

        // Penalise lines that contain term patterns when question is not about terms
        let term_penalty = if !keywords.contains(&"term") && line_lower.contains("start of term") {
            2
        } else {
            0
        };

        let final_score = score + year_boost + term_bonus;
        if term_penalty < final_score
            && (final_score - term_penalty) > best_score
            && line.trim().len() > 5
        {
            best_score = final_score - term_penalty;
            best_line = line.trim().to_string();
        }
    }

    if best_score > 0 {
        // Patterns where we want the full line (no date prefix needed)
        let full_line_patterns = ["SUMMER GRADUATION"];
        for pattern in &full_line_patterns {
            if best_line.to_uppercase().contains(pattern) {
                return best_line.trim().to_string();
            }
        }

        // Check for specific term number matches
        let term_keywords: Vec<&str> = keywords
            .iter()
            .filter(|&&k| k == "1" || k == "2" || k == "3" || k == "4")
            .copied()
            .collect();

        if !term_keywords.is_empty() {
            let term_num = term_keywords[0];
            let term_pattern = format!("START OF TERM {}", term_num);
            if let Some(pos) = best_line.to_uppercase().find(&term_pattern.to_uppercase()) {
                let start = if pos > 30 { pos - 30 } else { 0 };
                let raw_prefix = best_line[start..pos].trim().to_string();
                let prefix = raw_prefix
                    .split_whitespace()
                    .last()
                    .unwrap_or("")
                    .to_string();
                let pattern_text =
                    &best_line[pos..pos + term_pattern.len().min(best_line.len() - pos)];
                let result = if prefix.len() <= 2 && prefix.chars().all(|c| c.is_numeric()) {
                    format!("{} {}", prefix, pattern_text.trim())
                        .trim()
                        .to_string()
                } else {
                    pattern_text.trim().to_string()
                };
                return result;
            }
        }

        // General patterns with date prefix extraction
        // General patterns with date prefix extraction
        // Only apply term patterns if question is about terms
        let patterns: Vec<&str> = if keywords.contains(&"term") {
            vec![
                "START OF TERM",
                "END OF TERM",
                "HUMAN RIGHTS DAY",
                "CHRISTMAS DAY",
                "DAY OF RECONCILIATION",
            ]
        } else {
            vec!["HUMAN RIGHTS DAY", "CHRISTMAS DAY", "DAY OF RECONCILIATION"]
        };

        let upper = best_line.to_uppercase();
        for pattern in &patterns {
            if let Some(pos) = upper.find(pattern) {
                let start = if pos > 30 { pos - 30 } else { 0 };
                let raw_prefix = best_line[start..pos].trim().to_string();
                let prefix = raw_prefix
                    .split_whitespace()
                    .last()
                    .unwrap_or("")
                    .to_string();
                let pattern_text = &best_line[pos..pos + pattern.len().min(best_line.len() - pos)];
                let result = if prefix.len() <= 2 && prefix.chars().all(|c| c.is_numeric()) {
                    format!("{} {}", prefix, pattern_text.trim())
                        .trim()
                        .to_string()
                } else {
                    pattern_text.trim().to_string()
                };
                return result;
            }
        }

        // fallback
        // fallback - find the part of the line most relevant to the primary keyword
        let primary_keyword = keywords
            .iter()
            .filter(|&&kw| kw != "2024" && kw != "2025" && kw != "2026")
            .next()
            .unwrap_or(&"");

        let parts: Vec<&str> = best_line.split('|').collect();
        let best_part = parts
            .iter()
            .find(|p| p.to_lowercase().contains(primary_keyword))
            .unwrap_or(&parts[0]);

        let clean = best_part.trim().to_string();
        if clean.is_empty() {
            best_line
        } else {
            clean
        }
    } else {
        "No relevant information found.".to_string()
    }
}

pub fn count_occurrences(keyword: &str, document_text: &str, year: Option<&str>) -> String {
    let keyword_lower = keyword.to_lowercase();
    let mut count = 0;
    let mut current_year = String::new();

    for line in document_text.lines() {
        // Track which year section we're in
        if line.contains("2024") || line.contains("2025") || line.contains("2026") {
            if line.len() < 20 {
                // short lines are likely month/year headers
                current_year = line.to_string();
            }
        }
        // Count matches filtered by year if specified
        let year_match = match year {
            Some(y) => current_year.contains(y),
            None => true,
        };
        if year_match && line.to_lowercase().contains(&keyword_lower) {
            count += 1;
        }
    }
    format!(
        "'{}' appears {} time(s) in the documents{}",
        keyword,
        count,
        year.map(|y| format!(" for year {}", y)).unwrap_or_default()
    )
}
