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

    // Extract meaningful keywords, skip common question words
    let stop_words = vec![
        "when", "what", "how", "many", "did", "the", "is", "are", "was", "will", "does", "has",
        "their", "hold", "times",
    ];
    // Expand common abbreviations

    let expanded_question = question_lower
        .replace("hdc", "higher degrees committee")
        .replace("emc", "executive management committee")
        .replace("src", "student representative council");
    let cleaned_question: String = expanded_question
        .chars()
        .map(|c| {
            if c.is_alphabetic() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    let keywords: Vec<&str> = cleaned_question
        .split_whitespace()
        .filter(|w| w.len() > 2 && !stop_words.contains(w))
        .collect();

    println!("Searching for keywords: {:?}", keywords);

    let mut best_line = String::new();
    let mut best_score = 0usize;

    for line in document_text.lines() {
        let line_lower = line.to_lowercase();
        let score = keywords
            .iter()
            .filter(|&&kw| line_lower.contains(kw))
            .count();
        if score > best_score && line.trim().len() > 5 {
            best_score = score;
            best_line = line.trim().to_string();
        }
    }

    if best_score > 0 {
        // Clean up the answer - take only the most relevant part
        let parts: Vec<&str> = best_line.split('|').collect();
        let relevant = parts
            .iter()
            .filter(|p| keywords.iter().any(|&kw| p.to_lowercase().contains(kw)))
            .map(|p| p.trim())
            .collect::<Vec<_>>()
            .join(" | ");
        if relevant.is_empty() {
            best_line
        } else {
            relevant
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
