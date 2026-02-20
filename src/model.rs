// ============================================================
// src/model.rs  – Transformer Q&A Model
// ============================================================
// Implements:
//   - Token embeddings
//   - Positional embeddings
//   - Multi-layer Transformer encoder (6 layers)
//   - Output projection layer
// Generic over Burn's Backend trait as required.
// ============================================================

use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{MultiHeadAttention, MultiHeadAttentionConfig},
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
};

// ── 1. CONFIG ────────────────────────────────────────────────

/// All hyperparameters for the model in one place.
#[derive(Config, Debug)]
pub struct TransformerQAConfig {
    pub vocab_size: usize,  // number of tokens in vocabulary
    pub max_seq_len: usize, // maximum input length
    pub d_model: usize,     // embedding / hidden dimension
    pub num_heads: usize,   // attention heads per layer
    pub num_layers: usize,  // number of transformer encoder layers
    pub d_ff: usize,        // feed-forward inner dimension
    pub dropout: f64,       // dropout probability
}

impl TransformerQAConfig {
    /// Sensible defaults for our calendar Q&A task.
    #[allow(dead_code)]
    pub fn default_for_qa(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len: 128,
            d_model: 256,
            num_heads: 8,
            num_layers: 6, // minimum required by assignment
            d_ff: 512,
            dropout: 0.1,
        }
    }
}

// ── 2. SINGLE TRANSFORMER ENCODER LAYER ──────────────────────

#[derive(Module, Debug)]
pub struct TransformerLayer<B: Backend> {
    self_attn: MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    ff1: Linear<B>,
    ff2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> TransformerLayer<B> {
    pub fn new(config: &TransformerQAConfig, device: &B::Device) -> Self {
        let attn_config = MultiHeadAttentionConfig::new(config.d_model, config.num_heads)
            .with_dropout(config.dropout);
        Self {
            self_attn: attn_config.init(device),
            norm1: LayerNormConfig::new(config.d_model).init(device),
            norm2: LayerNormConfig::new(config.d_model).init(device),
            ff1: LinearConfig::new(config.d_model, config.d_ff).init(device),
            ff2: LinearConfig::new(config.d_ff, config.d_model).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    /// Forward pass: self-attention + feed-forward, both with residual + norm.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Self-attention block
        let attn_input = burn::nn::attention::MhaInput::self_attn(x.clone());
        let attn_out = self.self_attn.forward(attn_input).context;
        let x = self.norm1.forward(x + self.dropout.forward(attn_out));

        // Feed-forward block
        let ff_out = self.ff2.forward(
            self.dropout
                .forward(burn::tensor::activation::relu(self.ff1.forward(x.clone()))),
        );
        self.norm2.forward(x + self.dropout.forward(ff_out))
    }
}

// ── 3. FULL MODEL ────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct TransformerQAModel<B: Backend> {
    token_embed: Embedding<B>,
    pos_embed: Embedding<B>,
    layers: Vec<TransformerLayer<B>>,
    output_proj: Linear<B>,
    dropout: Dropout,
    d_model: usize,
    max_seq_len: usize,
}

impl<B: Backend> TransformerQAModel<B> {
    /// Build the model from config.
    pub fn new(config: &TransformerQAConfig, device: &B::Device) -> Self {
        let layers = (0..config.num_layers)
            .map(|_| TransformerLayer::new(config, device))
            .collect();

        Self {
            token_embed: EmbeddingConfig::new(config.vocab_size, config.d_model).init(device),
            pos_embed: EmbeddingConfig::new(config.max_seq_len, config.d_model).init(device),
            layers,
            output_proj: LinearConfig::new(config.d_model, config.vocab_size).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
            d_model: config.d_model,
            max_seq_len: config.max_seq_len,
        }
    }

    /// Forward pass.
    /// input_ids: [batch, seq_len]  (integer token ids)
    /// Returns logits: [batch, seq_len, vocab_size]
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq_len] = input_ids.dims();
        let device = input_ids.device();

        // Token embeddings
        let tok_emb = self.token_embed.forward(input_ids); // [B, S, d_model]

        // Positional embeddings: 0, 1, 2, ... seq_len-1
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>()
            .expand([batch, seq_len]);
        let pos_emb = self.pos_embed.forward(positions); // [B, S, d_model]

        // Combine and apply dropout
        let mut x = self.dropout.forward(tok_emb + pos_emb);

        // Pass through each transformer layer
        for layer in &self.layers {
            x = layer.forward(x);
        }

        // Project to vocabulary size → logits
        self.output_proj.forward(x) // [B, S, vocab_size]
    }
}
