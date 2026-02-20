// ============================================================
// src/training.rs  – Training loop, loss, checkpoints
// ============================================================

use crate::{
    data::{generate_qa_pairs, load_documents, QADataset, QAItem, Vocabulary},
    model::{TransformerQAConfig, TransformerQAModel},
};
use burn::{
    config::Config,
    data::dataset::Dataset,
    module::Module,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{CompactRecorder, Recorder},
    tensor::{backend::AutodiffBackend, ElementConversion, Int, Tensor},
};

// ── 1. TRAINING CONFIG ───────────────────────────────────────

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub d_ff: usize,
    pub dropout: f64,
    pub docs_folder: String,
    pub output_dir: String,
}

impl TrainingConfig {
    pub fn default_config(docs_folder: &str, output_dir: &str) -> Self {
        Self {
            num_epochs: 10,
            batch_size: 8,
            learning_rate: 1e-4,
            max_seq_len: 128,
            d_model: 256,
            num_heads: 8,
            num_layers: 6,
            d_ff: 512,
            dropout: 0.1,
            docs_folder: docs_folder.to_string(),
            output_dir: output_dir.to_string(),
        }
    }
}

// ── 2. BATCHER ───────────────────────────────────────────────

/// Collects individual QAItems into padded tensors for a batch.
pub struct QABatcher<B: burn::tensor::backend::Backend> {
    device: B::Device,
}

impl<B: burn::tensor::backend::Backend> QABatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn batch(&self, items: Vec<QAItem>) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        let batch_size = items.len();
        let seq_len = items[0].input_ids.len();

        let input_flat: Vec<i64> = items
            .iter()
            .flat_map(|it| it.input_ids.iter().map(|&x| x as i64))
            .collect();
        let target_flat: Vec<i64> = items
            .iter()
            .flat_map(|it| it.target_ids.iter().map(|&x| x as i64))
            .collect();

        let inputs = Tensor::<B, 1, Int>::from_ints(
            burn::tensor::TensorData::new(input_flat, [batch_size * seq_len]),
            &self.device,
        )
        .reshape([batch_size, seq_len]);

        let targets = Tensor::<B, 1, Int>::from_ints(
            burn::tensor::TensorData::new(target_flat, [batch_size * seq_len]),
            &self.device,
        )
        .reshape([batch_size, seq_len]);

        (inputs, targets)
    }
}

// ── 3. LOSS FUNCTION ─────────────────────────────────────────

/// Cross-entropy loss over the vocabulary dimension.
/// logits:  [B, S, vocab_size]
/// targets: [B, S]
fn cross_entropy_loss<B: AutodiffBackend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    vocab_size: usize,
) -> Tensor<B, 1> {
    let [batch, seq_len, _] = logits.dims();
    let logits_2d = logits.reshape([batch * seq_len, vocab_size]);
    let targets_1d = targets.reshape([batch * seq_len]);

    // Gather the logit for the correct token
    let correct_logits = logits_2d.clone().gather(1, targets_1d.unsqueeze_dim(1));

    // log(sum(exp(all logits))) for each row — logsumexp
    let logsumexp = logits_2d.exp().mean_dim(1).log();

    // NLL loss = logsumexp - correct_logit
    let [n, _] = correct_logits.dims();
    let loss = logsumexp.reshape([n]) - correct_logits.reshape([n]);
    loss.mean()
}

// ── 4. ACCURACY HELPER ───────────────────────────────────────

fn compute_accuracy<B: AutodiffBackend>(logits: &Tensor<B, 3>, targets: &Tensor<B, 2, Int>) -> f64 {
    let [batch, seq_len, _] = logits.dims();
    let preds = logits.clone().argmax(2).reshape([batch * seq_len]);
    let targets_flat = targets.clone().reshape([batch * seq_len]);
    let correct: i64 = preds
        .equal(targets_flat)
        .int()
        .sum()
        .into_scalar()
        .elem::<i64>();
    correct as f64 / (batch * seq_len) as f64
}

// ── 5. MAIN TRAINING FUNCTION ────────────────────────────────

pub fn train<B: AutodiffBackend>(config: &TrainingConfig, device: B::Device)
where
    B::InnerBackend: burn::tensor::backend::Backend,
{
    // ── Load and process documents ───────────────────────────
    println!("\n=== Loading documents from '{}' ===", config.docs_folder);
    let text = load_documents(&config.docs_folder);
    let pairs = generate_qa_pairs(&text);

    if pairs.is_empty() {
        panic!("No Q&A pairs generated! Check your documents folder.");
    }

    // ── Build vocabulary ─────────────────────────────────────
    let all_texts: Vec<String> = pairs
        .iter()
        .flat_map(|p| vec![p.question.clone(), p.answer.clone(), p.context.clone()])
        .collect();
    let vocab = Vocabulary::build_from_texts(&all_texts, 1);
    println!("Vocabulary size: {}", vocab.size());

    // Save vocabulary for inference later
    std::fs::create_dir_all(&config.output_dir).unwrap();
    let vocab_path = format!("{}/vocab.json", config.output_dir);
    let vocab_json = serde_json::to_string(&vocab).unwrap();
    std::fs::write(&vocab_path, vocab_json).unwrap();
    println!("Vocabulary saved to {}", vocab_path);

    // ── Build dataset and split ──────────────────────────────
    let dataset = QADataset::new(&pairs, &vocab, config.max_seq_len);
    let (train_set, val_set) = dataset.split(0.8);
    println!(
        "Train samples: {}  |  Val samples: {}",
        train_set.len(),
        val_set.len()
    );

    // ── Build model ──────────────────────────────────────────
    let model_config = TransformerQAConfig {
        vocab_size: vocab.size(),
        max_seq_len: config.max_seq_len,
        d_model: config.d_model,
        num_heads: config.num_heads,
        num_layers: config.num_layers,
        d_ff: config.d_ff,
        dropout: config.dropout,
    };
    let mut model: TransformerQAModel<B> = TransformerQAModel::new(&model_config, &device);
    let mut optim = AdamConfig::new().with_epsilon(1e-8).init();
    let batcher = QABatcher::<B>::new(device.clone());

    // ── Training loop ────────────────────────────────────────
    println!(
        "\n=== Starting training for {} epochs ===",
        config.num_epochs
    );
    for epoch in 1..=config.num_epochs {
        // ── Train ────────────────────────────────────────────
        let mut train_loss = 0.0_f64;
        let mut train_acc = 0.0_f64;
        let mut num_batches = 0usize;

        let train_items: Vec<QAItem> = (0..train_set.len())
            .filter_map(|i| burn::data::dataset::Dataset::get(&train_set, i))
            .collect();

        for chunk in train_items.chunks(config.batch_size) {
            let (inputs, targets) = batcher.batch(chunk.to_vec());
            let logits = model.forward(inputs);
            let loss = cross_entropy_loss::<B>(logits.clone(), targets.clone(), vocab.size());
            let acc = compute_accuracy::<B>(&logits, &targets);

            train_loss += loss.clone().into_scalar().elem::<f64>();
            train_acc += acc;
            num_batches += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads);
        }

        let avg_train_loss = train_loss / num_batches as f64;
        let avg_train_acc = train_acc / num_batches as f64;

        // ── Validate ─────────────────────────────────────────
        let mut val_loss = 0.0_f64;
        let mut val_acc = 0.0_f64;
        let mut val_batches = 0usize;

        let val_items: Vec<QAItem> = (0..val_set.len())
            .filter_map(|i| burn::data::dataset::Dataset::get(&val_set, i))
            .collect();

        for chunk in val_items.chunks(config.batch_size) {
            let (inputs, targets) = batcher.batch(chunk.to_vec());
            let logits = model.forward(inputs);
            let loss = cross_entropy_loss::<B>(logits.clone(), targets.clone(), vocab.size());
            let acc = compute_accuracy::<B>(&logits, &targets);
            val_loss += loss.into_scalar().elem::<f64>();
            val_acc += acc;
            val_batches += 1;
        }

        let avg_val_loss = if val_batches > 0 {
            val_loss / val_batches as f64
        } else {
            0.0
        };
        let avg_val_acc = if val_batches > 0 {
            val_acc / val_batches as f64
        } else {
            0.0
        };

        println!(
            "Epoch {:>3}/{} | Train Loss: {:.4}  Acc: {:.2}% | Val Loss: {:.4}  Acc: {:.2}%",
            epoch,
            config.num_epochs,
            avg_train_loss,
            avg_train_acc * 100.0,
            avg_val_loss,
            avg_val_acc * 100.0,
        );

        // ── Save checkpoint every epoch ───────────────────────
        let checkpoint_path = format!("{}/checkpoint_epoch_{}", config.output_dir, epoch);
        CompactRecorder::new()
            .record(model.clone().into_record(), checkpoint_path.clone().into())
            .unwrap();
        println!("Checkpoint saved: {}", checkpoint_path);
    }

    // ── Save final model ─────────────────────────────────────
    let final_path = format!("{}/model_final", config.output_dir);
    CompactRecorder::new()
        .record(model.into_record(), final_path.clone().into())
        .unwrap();
    println!(
        "\n=== Training complete! Final model saved to {} ===",
        final_path
    );
}
