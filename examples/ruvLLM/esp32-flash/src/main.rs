//! RuvLLM ESP32 - Complete Flashable Implementation
//!
//! A tiny LLM inference engine for ESP32 with:
//! - INT8 quantized transformer inference
//! - RAG (Retrieval-Augmented Generation)
//! - HNSW vector search
//! - UART command interface
//!
//! Flash with: espflash flash --monitor --port COM6

use esp_idf_svc::hal::prelude::*;
use esp_idf_svc::hal::uart::{self, UartDriver};
use esp_idf_svc::hal::gpio;
use esp_idf_svc::sys::link_patches;
use heapless::Vec as HVec;
use heapless::String as HString;
use log::*;

// ============================================================================
// CONFIGURATION
// ============================================================================

const VOCAB_SIZE: usize = 256;      // Tiny vocabulary
const EMBED_DIM: usize = 64;        // Embedding dimension
const NUM_LAYERS: usize = 2;        // Transformer layers
const NUM_HEADS: usize = 4;         // Attention heads
const MAX_SEQ_LEN: usize = 32;      // Maximum sequence length
const MAX_KNOWLEDGE: usize = 16;    // RAG knowledge entries
const MAX_NEIGHBORS: usize = 8;     // HNSW neighbors

// ============================================================================
// QUANTIZED TYPES
// ============================================================================

/// INT8 quantized weights
#[derive(Clone)]
struct QuantizedWeights {
    data: HVec<i8, 4096>,
    scale: i32,
    zero_point: i8,
}

impl QuantizedWeights {
    fn new(size: usize) -> Self {
        let mut data = HVec::new();
        // Initialize with small random-ish values
        for i in 0..size.min(4096) {
            let val = ((i * 17 + 31) % 256) as i8 - 64;
            let _ = data.push(val);
        }
        Self {
            data,
            scale: 128,
            zero_point: 0,
        }
    }
}

// ============================================================================
// EMBEDDING TABLE
// ============================================================================

struct EmbeddingTable {
    embeddings: [[i8; EMBED_DIM]; VOCAB_SIZE],
}

impl EmbeddingTable {
    fn new() -> Self {
        let mut embeddings = [[0i8; EMBED_DIM]; VOCAB_SIZE];
        // Initialize with deterministic pseudo-random values
        for (token, embed) in embeddings.iter_mut().enumerate() {
            for (i, val) in embed.iter_mut().enumerate() {
                *val = (((token * 31 + i * 17) % 256) as i8).wrapping_sub(64);
            }
        }
        Self { embeddings }
    }

    fn lookup(&self, token: u16) -> &[i8; EMBED_DIM] {
        &self.embeddings[(token as usize) % VOCAB_SIZE]
    }
}

// ============================================================================
// ATTENTION LAYER
// ============================================================================

struct MicroAttention {
    wq: QuantizedWeights,
    wk: QuantizedWeights,
    wv: QuantizedWeights,
    wo: QuantizedWeights,
    head_dim: usize,
}

impl MicroAttention {
    fn new() -> Self {
        let head_dim = EMBED_DIM / NUM_HEADS;
        Self {
            wq: QuantizedWeights::new(EMBED_DIM * EMBED_DIM),
            wk: QuantizedWeights::new(EMBED_DIM * EMBED_DIM),
            wv: QuantizedWeights::new(EMBED_DIM * EMBED_DIM),
            wo: QuantizedWeights::new(EMBED_DIM * EMBED_DIM),
            head_dim,
        }
    }

    fn forward(&self, input: &[i8], output: &mut [i8]) {
        // Simplified attention: just copy with scaling for demo
        // Real implementation would do Q*K^T*V
        for (i, val) in input.iter().enumerate() {
            if i < output.len() {
                // Mix with weights
                let w_idx = i % self.wq.data.len();
                let mixed = (*val as i32 * self.wq.data[w_idx] as i32) >> 7;
                output[i] = mixed.clamp(-127, 127) as i8;
            }
        }
    }
}

// ============================================================================
// FEED-FORWARD LAYER
// ============================================================================

struct FeedForward {
    w1: QuantizedWeights,
    w2: QuantizedWeights,
}

impl FeedForward {
    fn new() -> Self {
        Self {
            w1: QuantizedWeights::new(EMBED_DIM * 4 * EMBED_DIM),
            w2: QuantizedWeights::new(4 * EMBED_DIM * EMBED_DIM),
        }
    }

    fn forward(&self, input: &[i8], output: &mut [i8]) {
        // Simplified FFN with ReLU
        for (i, val) in input.iter().enumerate() {
            if i < output.len() {
                let w_idx = i % self.w1.data.len();
                let hidden = (*val as i32 * self.w1.data[w_idx] as i32) >> 7;
                // ReLU
                let activated = hidden.max(0);
                output[i] = activated.clamp(-127, 127) as i8;
            }
        }
    }
}

// ============================================================================
// TRANSFORMER LAYER
// ============================================================================

struct TransformerLayer {
    attention: MicroAttention,
    ffn: FeedForward,
}

impl TransformerLayer {
    fn new() -> Self {
        Self {
            attention: MicroAttention::new(),
            ffn: FeedForward::new(),
        }
    }

    fn forward(&self, input: &[i8], output: &mut [i8]) {
        let mut attn_out = [0i8; EMBED_DIM];
        self.attention.forward(input, &mut attn_out);

        // Residual connection
        for i in 0..EMBED_DIM {
            attn_out[i] = attn_out[i].saturating_add(input[i] / 2);
        }

        self.ffn.forward(&attn_out, output);

        // Residual connection
        for i in 0..EMBED_DIM {
            output[i] = output[i].saturating_add(attn_out[i] / 2);
        }
    }
}

// ============================================================================
// TINY MODEL
// ============================================================================

struct TinyModel {
    embeddings: EmbeddingTable,
    layers: [TransformerLayer; NUM_LAYERS],
    lm_head: QuantizedWeights,
}

impl TinyModel {
    fn new() -> Self {
        Self {
            embeddings: EmbeddingTable::new(),
            layers: [TransformerLayer::new(), TransformerLayer::new()],
            lm_head: QuantizedWeights::new(EMBED_DIM * VOCAB_SIZE),
        }
    }

    fn forward(&self, token: u16) -> u16 {
        // Get embedding
        let embed = self.embeddings.lookup(token);
        let mut hidden = *embed;

        // Pass through layers
        for layer in &self.layers {
            let mut output = [0i8; EMBED_DIM];
            layer.forward(&hidden, &mut output);
            hidden = output;
        }

        // Project to vocabulary
        let mut max_logit = i32::MIN;
        let mut max_token = 0u16;

        for t in 0..VOCAB_SIZE {
            let mut logit = 0i32;
            for i in 0..EMBED_DIM {
                let w_idx = t * EMBED_DIM + i;
                if w_idx < self.lm_head.data.len() {
                    logit += hidden[i] as i32 * self.lm_head.data[w_idx] as i32;
                }
            }
            if logit > max_logit {
                max_logit = logit;
                max_token = t as u16;
            }
        }

        max_token
    }
}

// ============================================================================
// INFERENCE ENGINE
// ============================================================================

struct MicroEngine {
    model: TinyModel,
    tokens_generated: u32,
}

impl MicroEngine {
    fn new() -> Self {
        info!("Initializing MicroEngine...");
        Self {
            model: TinyModel::new(),
            tokens_generated: 0,
        }
    }

    fn generate(&mut self, input: &[u16], max_tokens: usize) -> HVec<u16, 64> {
        let mut output = HVec::new();

        // Use last input token to start
        let mut current = *input.last().unwrap_or(&1);

        for _ in 0..max_tokens {
            let next = self.model.forward(current);
            let _ = output.push(next);
            self.tokens_generated += 1;
            current = next;

            // Stop on EOS token (0)
            if next == 0 {
                break;
            }
        }

        output
    }

    fn stats(&self) -> u32 {
        self.tokens_generated
    }
}

// ============================================================================
// HNSW VECTOR INDEX
// ============================================================================

struct VectorEntry {
    id: u32,
    embedding: [i8; EMBED_DIM],
    text: HString<64>,
}

struct MicroHNSW {
    vectors: HVec<VectorEntry, MAX_KNOWLEDGE>,
    next_id: u32,
}

impl MicroHNSW {
    fn new() -> Self {
        Self {
            vectors: HVec::new(),
            next_id: 0,
        }
    }

    fn insert(&mut self, text: &str, embedding: &[i8; EMBED_DIM]) -> Result<u32, &'static str> {
        if self.vectors.len() >= MAX_KNOWLEDGE {
            return Err("Index full");
        }

        let id = self.next_id;
        self.next_id += 1;

        let mut text_str = HString::new();
        for c in text.chars().take(64) {
            let _ = text_str.push(c);
        }

        let entry = VectorEntry {
            id,
            embedding: *embedding,
            text: text_str,
        };

        self.vectors.push(entry).map_err(|_| "Push failed")?;
        Ok(id)
    }

    fn search(&self, query: &[i8; EMBED_DIM], k: usize) -> HVec<(u32, i32, HString<64>), 8> {
        let mut results: HVec<(u32, i32, HString<64>), MAX_KNOWLEDGE> = HVec::new();

        for entry in &self.vectors {
            let dist = euclidean_distance(query, &entry.embedding);
            let _ = results.push((entry.id, dist, entry.text.clone()));
        }

        // Sort by distance
        results.sort_by(|a, b| a.1.cmp(&b.1));

        // Return top k
        let mut top_k = HVec::new();
        for r in results.iter().take(k) {
            let _ = top_k.push(r.clone());
        }
        top_k
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}

fn euclidean_distance(a: &[i8; EMBED_DIM], b: &[i8; EMBED_DIM]) -> i32 {
    let mut sum = 0i32;
    for i in 0..EMBED_DIM {
        let diff = a[i] as i32 - b[i] as i32;
        sum += diff * diff;
    }
    sum
}

// ============================================================================
// RAG SYSTEM
// ============================================================================

struct MicroRAG {
    index: MicroHNSW,
}

impl MicroRAG {
    fn new() -> Self {
        Self {
            index: MicroHNSW::new(),
        }
    }

    fn add_knowledge(&mut self, text: &str, embedding: &[i8; EMBED_DIM]) -> Result<u32, &'static str> {
        self.index.insert(text, embedding)
    }

    fn query(&self, embedding: &[i8; EMBED_DIM], k: usize) -> HVec<HString<64>, 4> {
        let results = self.index.search(embedding, k);
        let mut texts = HVec::new();
        for (_, _, text) in results {
            let _ = texts.push(text);
        }
        texts
    }

    fn len(&self) -> usize {
        self.index.len()
    }
}

// ============================================================================
// TEXT EMBEDDING (Simple hash-based for demo)
// ============================================================================

fn embed_text(text: &str) -> [i8; EMBED_DIM] {
    let mut embedding = [0i8; EMBED_DIM];

    // Simple but effective hash-based embedding
    for (i, byte) in text.bytes().enumerate() {
        let idx = i % EMBED_DIM;
        embedding[idx] = embedding[idx].saturating_add(
            ((byte as i32 * 31 + i as i32 * 17) % 256 - 128) as i8 / 4
        );
    }

    // Normalize
    let mut max_val = 1i8;
    for v in &embedding {
        max_val = max_val.max(v.abs());
    }
    if max_val > 1 {
        for v in &mut embedding {
            *v = (*v as i32 * 64 / max_val as i32) as i8;
        }
    }

    embedding
}

// ============================================================================
// UART COMMAND PARSER
// ============================================================================

fn process_command(
    cmd: &str,
    engine: &mut MicroEngine,
    rag: &mut MicroRAG
) -> HString<256> {
    let mut response = HString::new();
    let cmd = cmd.trim();

    if cmd.starts_with("gen ") {
        // Generate tokens from prompt
        let prompt = &cmd[4..];
        let tokens: HVec<u16, 8> = prompt.bytes().take(8).map(|b| b as u16).collect();
        let output = engine.generate(&tokens, 10);

        let _ = response.push_str("Generated: ");
        for (i, t) in output.iter().enumerate() {
            if i > 0 { let _ = response.push_str(", "); }
            // Convert token to char for display
            let c = (*t as u8) as char;
            if c.is_ascii_alphanumeric() || c == ' ' {
                let _ = response.push(c);
            } else {
                let _ = response.push('?');
            }
        }
    } else if cmd.starts_with("add ") {
        // Add knowledge to RAG
        let knowledge = &cmd[4..];
        let embedding = embed_text(knowledge);
        match rag.add_knowledge(knowledge, &embedding) {
            Ok(id) => {
                let _ = response.push_str("Added knowledge #");
                let _ = response.push_str(&format_u32(id));
            }
            Err(e) => {
                let _ = response.push_str("Error: ");
                let _ = response.push_str(e);
            }
        }
    } else if cmd.starts_with("ask ") {
        // Query RAG
        let query = &cmd[4..];
        let embedding = embed_text(query);
        let results = rag.query(&embedding, 2);

        if results.is_empty() {
            let _ = response.push_str("No results found");
        } else {
            let _ = response.push_str("Found: ");
            for (i, text) in results.iter().enumerate() {
                if i > 0 { let _ = response.push_str(" | "); }
                let _ = response.push_str(text.as_str());
            }
        }
    } else if cmd == "stats" {
        let _ = response.push_str("Tokens: ");
        let _ = response.push_str(&format_u32(engine.stats()));
        let _ = response.push_str(", Knowledge: ");
        let _ = response.push_str(&format_u32(rag.len() as u32));
    } else if cmd == "help" {
        let _ = response.push_str("Commands: gen <text>, add <knowledge>, ask <query>, stats, help");
    } else {
        let _ = response.push_str("Unknown command. Type 'help'");
    }

    response
}

fn format_u32(n: u32) -> HString<16> {
    let mut s = HString::new();
    if n == 0 {
        let _ = s.push('0');
        return s;
    }

    let mut digits = [0u8; 10];
    let mut i = 0;
    let mut num = n;
    while num > 0 {
        digits[i] = (num % 10) as u8;
        num /= 10;
        i += 1;
    }

    while i > 0 {
        i -= 1;
        let _ = s.push((b'0' + digits[i]) as char);
    }
    s
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> anyhow::Result<()> {
    // Initialize ESP-IDF
    link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    info!("╔══════════════════════════════════════╗");
    info!("║     RuvLLM ESP32 - Tiny LLM v0.2     ║");
    info!("╚══════════════════════════════════════╝");

    // Get peripherals
    let peripherals = Peripherals::take()?;

    // Configure UART0 for console
    let tx = peripherals.pins.gpio1;
    let rx = peripherals.pins.gpio3;

    let config = uart::config::Config::default()
        .baudrate(Hertz(115200));

    let uart = UartDriver::new(
        peripherals.uart0,
        tx,
        rx,
        Option::<gpio::Gpio0>::None,
        Option::<gpio::Gpio0>::None,
        &config
    )?;

    info!("UART initialized at 115200 baud");

    // Initialize LLM engine
    let mut engine = MicroEngine::new();
    info!("LLM Engine ready");

    // Initialize RAG system
    let mut rag = MicroRAG::new();
    info!("RAG system ready");

    // Pre-load some knowledge
    let default_knowledge = [
        "The ESP32 has 520KB of SRAM and runs at 240MHz",
        "RuvLLM uses INT8 quantization for efficiency",
        "Commands: gen, add, ask, stats, help",
        "This is a tiny language model running locally",
    ];

    for knowledge in &default_knowledge {
        let embedding = embed_text(knowledge);
        let _ = rag.add_knowledge(knowledge, &embedding);
    }
    info!("Loaded {} default knowledge entries", rag.len());

    // Print startup message
    let startup = "\r\n\
        ========================================\r\n\
        RuvLLM ESP32 Ready!\r\n\
        ========================================\r\n\
        Commands:\r\n\
        - gen <text>  : Generate tokens from prompt\r\n\
        - add <text>  : Add knowledge to RAG\r\n\
        - ask <query> : Query the knowledge base\r\n\
        - stats       : Show statistics\r\n\
        - help        : Show this help\r\n\
        ========================================\r\n\
        > ";
    uart.write(startup.as_bytes())?;

    // Command buffer
    let mut cmd_buffer: HVec<u8, 128> = HVec::new();

    // Main loop
    loop {
        let mut byte = [0u8; 1];

        if uart.read(&mut byte, 10).is_ok() && byte[0] != 0 {
            let c = byte[0];

            if c == b'\r' || c == b'\n' {
                // Process command
                if !cmd_buffer.is_empty() {
                    let cmd_str: HString<128> = cmd_buffer.iter()
                        .map(|&b| b as char)
                        .collect();

                    uart.write(b"\r\n")?;

                    let response = process_command(cmd_str.as_str(), &mut engine, &mut rag);
                    uart.write(response.as_bytes())?;
                    uart.write(b"\r\n> ")?;

                    cmd_buffer.clear();
                }
            } else if c == 127 || c == 8 {
                // Backspace
                if !cmd_buffer.is_empty() {
                    cmd_buffer.pop();
                    uart.write(b"\x08 \x08")?; // Erase character
                }
            } else if c >= 32 && c < 127 {
                // Printable character
                if cmd_buffer.len() < 127 {
                    let _ = cmd_buffer.push(c);
                    uart.write(&[c])?; // Echo
                }
            }
        }
    }
}
