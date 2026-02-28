// ============================================================
// src/data.rs  – Document loading, tokenisation, dataset
// ============================================================
use burn::data::dataset::Dataset;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs};

// ── 1. RAW TEXT EXTRACTION ───────────────────────────────────

pub fn extract_text_from_docx(path: &str) -> String {
    let bytes = fs::read(path).expect("Cannot read docx file");
    let docx = docx_rs::read_docx(&bytes).expect("Cannot parse docx");
    let mut lines: Vec<String> = Vec::new();

    for child in &docx.document.children {
        match child {
            // Extract text from paragraphs
            docx_rs::DocumentChild::Paragraph(para) => {
                let mut line = String::new();
                for run_child in &para.children {
                    if let docx_rs::ParagraphChild::Run(run) = run_child {
                        for rc in &run.children {
                            if let docx_rs::RunChild::Text(t) = rc {
                                line.push_str(&t.text);
                            }
                        }
                    }
                }
                let trimmed = line.trim().to_string();
                if !trimmed.is_empty() {
                    lines.push(trimmed);
                }
            }
            // Extract text from tables
            docx_rs::DocumentChild::Table(table) => {
                for row in &table.rows {
                    let docx_rs::TableChild::TableRow(tr) = row;
                    let mut row_parts: Vec<String> = Vec::new();
                    for cell in &tr.cells {
                        let docx_rs::TableRowChild::TableCell(tc) = cell;
                        let mut cell_text = String::new();
                        for cell_child in &tc.children {
                            if let docx_rs::TableCellContent::Paragraph(para) = cell_child {
                                for run_child in &para.children {
                                    if let docx_rs::ParagraphChild::Run(run) = run_child {
                                        for rc in &run.children {
                                            if let docx_rs::RunChild::Text(t) = rc {
                                                cell_text.push_str(&t.text);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        let trimmed = cell_text.trim().to_string();
                        if !trimmed.is_empty() {
                            row_parts.push(trimmed);
                        }
                    }
                    if !row_parts.is_empty() {
                        lines.push(row_parts.join(" | "));
                    }
                }
            }
            _ => {}
        }
    }
    lines.join("\n")
}

pub fn load_documents(folder: &str) -> String {
    let mut all_text = String::new();
    for entry in fs::read_dir(folder).expect("Cannot read folder").flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("docx") {
            println!("Loading: {}", path.display());
            all_text.push_str(&extract_text_from_docx(path.to_str().unwrap()));
            all_text.push('\n');
        }
    }

    // Append known text box content not extractable by docx-rs
    // These entries are stored as floating drawing objects in the .docx files
    // and cannot be reached through standard table/paragraph traversal
    all_text.push_str("\nSUMMER GRADUATION December 11-13 2024\n");
    all_text.push_str("\nSUMMER GRADUATION December 10-12 2025\n");
    all_text.push_str("\nSUMMER GRADUATION December 10 2026\n");

    all_text
}

// ── 2. Q&A PAIR GENERATION ───────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAPair {
    pub question: String,
    pub answer: String,
    pub context: String,
}

pub fn generate_qa_pairs(text: &str) -> Vec<QAPair> {
    let mut pairs: Vec<QAPair> = Vec::new();
    let lines: Vec<&str> = text.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        let line = line.trim();
        if line.to_lowercase().contains("graduation") {
            pairs.push(QAPair {
                question: "When is the graduation ceremony?".to_string(),
                answer: line.to_string(),
                context: get_context(&lines, i, 2),
            });
            pairs.push(QAPair {
                question: "What date is the graduation?".to_string(),
                answer: line.to_string(),
                context: get_context(&lines, i, 2),
            });
        }
        if line.to_lowercase().contains("start of term") {
            pairs.push(QAPair {
                question: "When does the term start?".to_string(),
                answer: line.to_string(),
                context: get_context(&lines, i, 2),
            });
        }
        if line.to_lowercase().contains("end of term") {
            pairs.push(QAPair {
                question: "When does the term end?".to_string(),
                answer: line.to_string(),
                context: get_context(&lines, i, 2),
            });
        }
        if line.to_lowercase().contains("higher degrees committee") {
            pairs.push(QAPair {
                question: "When does the Higher Degrees Committee meet?".to_string(),
                answer: line.to_string(),
                context: get_context(&lines, i, 2),
            });
        }
        if line.to_lowercase().contains("council") && line.to_lowercase().contains("09:00") {
            pairs.push(QAPair {
                question: "When is the Council meeting?".to_string(),
                answer: line.to_string(),
                context: get_context(&lines, i, 2),
            });
        }
        if line.to_uppercase() == *line && line.len() > 4 && !line.contains('|') {
            pairs.push(QAPair {
                question: format!("When is {}?", line),
                answer: line.to_string(),
                context: get_context(&lines, i, 1),
            });
        }
        if line.to_lowercase().contains("senate (12:00)") {
            pairs.push(QAPair {
                question: "When does the Senate meet?".to_string(),
                answer: line.to_string(),
                context: get_context(&lines, i, 2),
            });
        }
    }
    println!("Generated {} Q&A pairs", pairs.len());
    pairs
}

fn get_context(lines: &[&str], i: usize, radius: usize) -> String {
    let start = i.saturating_sub(radius);
    let end = (i + radius + 1).min(lines.len());
    lines[start..end].join(" ")
}

// ── 3. VOCABULARY ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    pub word_to_id: HashMap<String, usize>,
    pub id_to_word: Vec<String>,
}

impl Vocabulary {
    pub const PAD: usize = 0;
    pub const UNK: usize = 1;
    pub const CLS: usize = 2;
    pub const SEP: usize = 3;

    pub fn new() -> Self {
        let specials = vec![
            "<PAD>".to_string(),
            "<UNK>".to_string(),
            "<CLS>".to_string(),
            "<SEP>".to_string(),
        ];
        let word_to_id = specials
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, w)| (w, i))
            .collect();
        Self {
            id_to_word: specials,
            word_to_id,
        }
    }

    pub fn build_from_texts(texts: &[String], min_freq: usize) -> Self {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for text in texts {
            for word in tokenize_str(text) {
                *freq.entry(word).or_insert(0) += 1;
            }
        }
        let mut vocab = Self::new();
        let mut words: Vec<_> = freq.into_iter().filter(|(_, f)| *f >= min_freq).collect();
        words.sort_by(|a, b| b.1.cmp(&a.1));
        for (word, _) in words {
            let id = vocab.id_to_word.len();
            vocab.id_to_word.push(word.clone());
            vocab.word_to_id.insert(word, id);
        }
        vocab
    }

    pub fn size(&self) -> usize {
        self.id_to_word.len()
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        tokenize_str(text)
            .into_iter()
            .map(|w| *self.word_to_id.get(&w).unwrap_or(&Self::UNK))
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter(|&&id| id != Self::PAD && id != Self::CLS && id != Self::SEP)
            .map(|&id| {
                self.id_to_word
                    .get(id)
                    .map(|s| s.as_str())
                    .unwrap_or("<UNK>")
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

pub fn tokenize_str(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| c.is_whitespace() || ".,;:!?()[]{}\"'".contains(c))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

// ── 4. BURN DATASET ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QAItem {
    pub input_ids: Vec<usize>,
    pub target_ids: Vec<usize>,
}

pub struct QADataset {
    pub items: Vec<QAItem>,
}

impl QADataset {
    pub fn new(pairs: &[QAPair], vocab: &Vocabulary, max_len: usize) -> Self {
        let items = pairs
            .iter()
            .map(|pair| {
                let mut input = vec![Vocabulary::CLS];
                input.extend(vocab.encode(&pair.question));
                input.push(Vocabulary::SEP);
                input.extend(vocab.encode(&pair.context));
                input.truncate(max_len);
                while input.len() < max_len {
                    input.push(Vocabulary::PAD);
                }
                let mut target = vocab.encode(&pair.answer);
                target.truncate(max_len);
                while target.len() < max_len {
                    target.push(Vocabulary::PAD);
                }
                QAItem {
                    input_ids: input,
                    target_ids: target,
                }
            })
            .collect();
        Self { items }
    }

    pub fn split(self, train_ratio: f64) -> (Self, Self) {
        let split_at = (self.items.len() as f64 * train_ratio) as usize;
        let mut items = self.items;
        let val_items = items.split_off(split_at);
        (Self { items }, Self { items: val_items })
    }
}

impl Dataset<QAItem> for QADataset {
    fn get(&self, index: usize) -> Option<QAItem> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}
