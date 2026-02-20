// ============================================================
// src/main.rs  – CLI entry point
// ============================================================

mod data;
mod inference;
mod model;
mod training;

use crate::data::load_documents;
use crate::inference::{keyword_search, QAInferenceEngine};
use crate::training::{train, TrainingConfig};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};

type MyBackend = NdArray;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help();
        return;
    }

    match args[1].as_str() {
        "train" => {
            let docs_folder = args.get(2).map(|s| s.as_str()).unwrap_or("documents");
            let output_dir = args.get(3).map(|s| s.as_str()).unwrap_or("model_output");

            println!("=== Training Mode ===");
            println!("Documents folder : {}", docs_folder);
            println!("Output directory : {}", output_dir);

            let config = TrainingConfig::default_config(docs_folder, output_dir);
            let device = NdArrayDevice::Cpu;
            train::<MyAutodiffBackend>(&config, device);
        }

        "ask" => {
            let model_dir = args.get(2).map(|s| s.as_str()).unwrap_or("model_output");
            let docs_folder = args.get(3).map(|s| s.as_str()).unwrap_or("documents");
            let question = args
                .get(4)
                .map(|s| s.as_str())
                .unwrap_or("What events are scheduled?");

            println!("=== Inference Mode ===");
            println!("Question: {}", question);

            let doc_text = load_documents(docs_folder);
            let model_path = format!("{}/model_final", model_dir);
            let vocab_path = format!("{}/vocab.json", model_dir);

            if std::path::Path::new(&model_path).exists() {
                let device = NdArrayDevice::Cpu;
                let engine =
                    QAInferenceEngine::<MyBackend>::load(&model_path, &vocab_path, 128, device);
                let answer = engine.answer(question, &doc_text);
                println!("\nNeural Model Answer:\n  {}", answer);
            } else {
                println!(
                    "No trained model found at {}. Using keyword search.",
                    model_path
                );
            }

            let kw_answer = keyword_search(question, &doc_text);
            println!("\nKeyword Search Answer:\n  {}", kw_answer);
        }

        "search" => {
            let docs_folder = args.get(2).map(|s| s.as_str()).unwrap_or("documents");
            let question = args.get(3).map(|s| s.as_str()).unwrap_or("graduation");

            println!("=== Keyword Search Mode ===");
            println!("Question: {}", question);

            let doc_text = load_documents(docs_folder);
            let answer = keyword_search(question, &doc_text);
            println!("\nAnswer:\n  {}", answer);
        }

        "debug" => {
            let docs_folder = args.get(2).map(|s| s.as_str()).unwrap_or("documents");
            let text = load_documents(docs_folder);
            let lines: Vec<&str> = text.lines().take(30).collect();
            println!("=== First 30 lines extracted ===");
            for (i, line) in lines.iter().enumerate() {
                println!("{}: {}", i, line);
            }
            println!("Total characters extracted: {}", text.len());
        }

        _ => print_help(),
    }
}

fn print_help() {
    println!(
        "
Word Document Q&A System
========================
USAGE:
  cargo run train  <docs_folder> <output_dir>
      Train the model on .docx files in docs_folder.
      Example: cargo run train documents model_output

  cargo run ask  <model_dir> <docs_folder> \"your question\"
      Answer a question using the trained model.
      Example: cargo run ask model_output documents \"When is graduation?\"

  cargo run search  <docs_folder> \"your question\"
      Fast keyword search without neural model.
      Example: cargo run search documents \"Higher Degrees Committee\"
    "
    );
}
