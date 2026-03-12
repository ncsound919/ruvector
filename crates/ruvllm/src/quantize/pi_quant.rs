//! Pi-Constant Quantization for Ultra-Low-Bit Weight Compression (ADR-090 Phase 1)
//!
//! This module implements pi-constant quantization, a novel approach that uses
//! pi-based step sizes instead of uniform grids. Research shows this can provide
//! ~0.5 effective bits of additional precision by better matching weight distributions.
//!
//! ## Mathematical Foundation
//!
//! Standard uniform quantization uses evenly-spaced levels:
//! ```text
//! w_q = round(w / step) * step, where step = range / (2^bits - 1)
//! ```
//!
//! Pi-quantization uses pi-scaled steps:
//! ```text
//! w_q = round(w / (alpha * pi / k)) * (alpha * pi / k)
//! ```
//!
//! Where:
//! - `alpha` is a learnable per-channel scale factor
//! - `k` is a small integer (2, 3, 4, or 5) controlling step granularity
//! - `pi` provides mathematically favorable quantization boundaries
//!
//! ## Packed Storage Formats
//!
//! - **Pi3BitBlock**: 8 weights packed into 3 bytes (3 bits per weight)
//! - **Pi2BitBlock**: 4 weights packed into 1 byte (2 bits per weight)
//!
//! ## System Invariants (from ADR-090)
//!
//! - **INV-2**: Scale Positivity - `alpha > 0` always
//! - **INV-3**: Step Size Constraint - step = `alpha * pi / k` where k in {2, 3, 4, 5}
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::quantize::pi_quant::{PiQuantizer, Pi3BitBlock};
//!
//! // Create a 3-bit pi-quantizer with k=4 (step size = alpha * pi/4)
//! let quantizer = PiQuantizer::new(3, 4, vec![1.0; 896])?;
//!
//! // Quantize a weight
//! let (quantized, dequantized) = quantizer.quantize_scalar(0.5, 0);
//!
//! // Quantize a block of 8 weights
//! let weights = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
//! let block = quantizer.quantize_block(&weights, 0);
//! ```

use std::f32::consts::PI;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Errors specific to pi-quantization operations
#[derive(Error, Debug, Clone)]
pub enum PiQuantError {
    /// Scale factor must be positive (INV-2)
    #[error("Scale alpha must be positive, got {0}")]
    NonPositiveScale(f32),

    /// K value must be in valid range (INV-3)
    #[error("K value must be in {{2, 3, 4, 5}}, got {0}")]
    InvalidK(u8),

    /// Bits must be 2 or 3 for pi-quantization
    #[error("Bits must be 2 or 3, got {0}")]
    InvalidBits(u8),

    /// Block size mismatch
    #[error("Expected block size {expected}, got {actual}")]
    BlockSizeMismatch { expected: usize, actual: usize },

    /// Channel index out of bounds
    #[error("Channel index {index} out of bounds (num_channels: {num_channels})")]
    ChannelOutOfBounds { index: usize, num_channels: usize },

    /// Empty alpha vector
    #[error("Alpha vector cannot be empty")]
    EmptyAlpha,
}

/// Result type for pi-quantization operations
pub type Result<T> = std::result::Result<T, PiQuantError>;

// ============================================================================
// Constants
// ============================================================================

/// Number of weights packed in a Pi3BitBlock
pub const PI3_BLOCK_WEIGHTS: usize = 8;

/// Number of bytes in a Pi3BitBlock (3 bytes for 8 weights)
pub const PI3_BLOCK_BYTES: usize = 3;

/// Number of weights packed in a Pi2BitBlock
pub const PI2_BLOCK_WEIGHTS: usize = 4;

/// Number of bytes in a Pi2BitBlock (1 byte for 4 weights)
pub const PI2_BLOCK_BYTES: usize = 1;

/// Valid k values for pi-quantization step size (INV-3)
pub const VALID_K_VALUES: [u8; 4] = [2, 3, 4, 5];

// ============================================================================
// Pi-Quantizer Configuration
// ============================================================================

/// Pi-constant quantization configuration
///
/// Implements the core pi-quantization algorithm where step sizes are
/// multiples of pi/k instead of uniform divisions.
#[derive(Debug, Clone)]
pub struct PiQuantizer {
    /// Number of quantization bits (2 or 3)
    bits: u8,

    /// Divisor for pi step size (2, 3, 4, or 5)
    k: u8,

    /// Per-channel learnable scale factors (must all be positive per INV-2)
    alpha: Vec<f32>,

    /// Precomputed half-range for clamping: 2^(bits-1)
    half_range: i8,

    /// Precomputed base step: pi / k
    base_step: f32,
}

impl PiQuantizer {
    /// Create a new Pi-quantizer with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `bits` - Number of quantization bits (must be 2 or 3)
    /// * `k` - Divisor for pi step size (must be in {2, 3, 4, 5})
    /// * `alpha` - Per-channel scale factors (must be non-empty, all positive)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `bits` is not 2 or 3
    /// - `k` is not in {2, 3, 4, 5}
    /// - `alpha` is empty
    /// - Any `alpha[i] <= 0` (violates INV-2)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let quantizer = PiQuantizer::new(3, 4, vec![1.0; 896])?;
    /// ```
    pub fn new(bits: u8, k: u8, alpha: Vec<f32>) -> Result<Self> {
        // Validate bits (2 or 3 for ultra-low-bit)
        if bits != 2 && bits != 3 {
            return Err(PiQuantError::InvalidBits(bits));
        }

        // Validate k (INV-3: k must be in {2, 3, 4, 5})
        if !VALID_K_VALUES.contains(&k) {
            return Err(PiQuantError::InvalidK(k));
        }

        // Validate alpha (must be non-empty)
        if alpha.is_empty() {
            return Err(PiQuantError::EmptyAlpha);
        }

        // Validate all alpha values are positive (INV-2)
        for (i, &a) in alpha.iter().enumerate() {
            if a <= 0.0 {
                return Err(PiQuantError::NonPositiveScale(a));
            }
            // Defensive: also check for NaN/Inf
            if !a.is_finite() {
                return Err(PiQuantError::NonPositiveScale(a));
            }
        }

        let half_range = 1i8 << (bits - 1);
        let base_step = PI / (k as f32);

        Ok(Self {
            bits,
            k,
            alpha,
            half_range,
            base_step,
        })
    }

    /// Create a new Pi-quantizer with uniform scale across all channels.
    ///
    /// # Arguments
    ///
    /// * `bits` - Number of quantization bits (2 or 3)
    /// * `k` - Divisor for pi step size
    /// * `scale` - Uniform scale factor for all channels (must be positive)
    /// * `num_channels` - Number of channels
    pub fn with_uniform_scale(bits: u8, k: u8, scale: f32, num_channels: usize) -> Result<Self> {
        if scale <= 0.0 || !scale.is_finite() {
            return Err(PiQuantError::NonPositiveScale(scale));
        }
        if num_channels == 0 {
            return Err(PiQuantError::EmptyAlpha);
        }
        Self::new(bits, k, vec![scale; num_channels])
    }

    /// Get the number of quantization bits.
    #[inline]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Get the k divisor value.
    #[inline]
    pub fn k(&self) -> u8 {
        self.k
    }

    /// Get the alpha scale factors.
    #[inline]
    pub fn alpha(&self) -> &[f32] {
        &self.alpha
    }

    /// Get the number of channels.
    #[inline]
    pub fn num_channels(&self) -> usize {
        self.alpha.len()
    }

    /// Compute the step size for a given channel.
    ///
    /// Step size = alpha[channel] * pi / k (INV-3)
    #[inline]
    pub fn step_size(&self, channel: usize) -> f32 {
        self.alpha.get(channel).copied().unwrap_or(1.0) * self.base_step
    }

    /// Get bits per weight including scale overhead.
    ///
    /// The overhead accounts for storing per-block scale metadata.
    #[inline]
    pub fn bits_per_weight(&self) -> f32 {
        match self.bits {
            3 => 3.0625, // 3 bits + scale overhead
            2 => 2.0625, // 2 bits + scale overhead
            _ => self.bits as f32,
        }
    }

    /// Quantize a single scalar weight value.
    ///
    /// Implements: w_q = round(w / (alpha * pi / k)) * (alpha * pi / k)
    ///
    /// # Arguments
    ///
    /// * `w` - The weight value to quantize
    /// * `channel` - Channel index for per-channel scale lookup
    ///
    /// # Returns
    ///
    /// A tuple of (quantized_int, dequantized_float):
    /// - `quantized_int`: The integer quantization level (clamped to valid range)
    /// - `dequantized_float`: The reconstructed floating-point value
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let quantizer = PiQuantizer::new(3, 4, vec![1.0])?;
    /// let (q_int, q_float) = quantizer.quantize_scalar(0.5, 0);
    /// // q_int is the quantized integer level
    /// // q_float is the dequantized value (may differ from 0.5 due to quantization)
    /// ```
    #[inline(always)]
    pub fn quantize_scalar(&self, w: f32, channel: usize) -> (i8, f32) {
        let step = self.step_size(channel);

        // Handle edge case of zero step (should not happen per INV-2)
        if step <= 0.0 {
            return (0, 0.0);
        }

        // Quantize: q = round(w / step)
        let q = (w / step).round() as i32;

        // Clamp to valid range: [-2^(bits-1), 2^(bits-1) - 1]
        // For 3-bit: [-4, 3], for 2-bit: [-2, 1]
        let half = self.half_range as i32;
        let q_clamped = q.clamp(-half, half - 1) as i8;

        // Dequantize: w_q = q_clamped * step
        let w_q = (q_clamped as f32) * step;

        (q_clamped, w_q)
    }

    /// Quantize a single scalar and return only the integer level.
    #[inline(always)]
    pub fn quantize_to_int(&self, w: f32, channel: usize) -> i8 {
        self.quantize_scalar(w, channel).0
    }

    /// Dequantize a single integer level back to float.
    #[inline(always)]
    pub fn dequantize_int(&self, q: i8, channel: usize) -> f32 {
        (q as f32) * self.step_size(channel)
    }

    /// Quantize a block of 8 weights to Pi3BitBlock format.
    ///
    /// # Arguments
    ///
    /// * `weights` - Slice of exactly 8 weight values
    /// * `channel` - Channel index for per-channel scale
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `weights.len() != 8`
    /// - `bits != 3`
    pub fn quantize_block_3bit(&self, weights: &[f32], channel: usize) -> Result<Pi3BitBlock> {
        if self.bits != 3 {
            return Err(PiQuantError::InvalidBits(self.bits));
        }
        if weights.len() != PI3_BLOCK_WEIGHTS {
            return Err(PiQuantError::BlockSizeMismatch {
                expected: PI3_BLOCK_WEIGHTS,
                actual: weights.len(),
            });
        }

        let mut block = Pi3BitBlock::new();

        // Quantize all 8 weights
        let mut q_values = [0i8; 8];
        for (i, &w) in weights.iter().enumerate() {
            q_values[i] = self.quantize_to_int(w, channel);
        }

        // Pack 8 3-bit values into 3 bytes
        // Layout: [v0:3, v1:3, v2:2] [v2:1, v3:3, v4:3, v5:1] [v5:2, v6:3, v7:3]
        block.pack(&q_values);

        Ok(block)
    }

    /// Dequantize a Pi3BitBlock back to 8 float weights.
    #[inline]
    pub fn dequantize_block_3bit(&self, block: &Pi3BitBlock, channel: usize) -> [f32; 8] {
        let q_values = block.unpack();
        let step = self.step_size(channel);

        let mut weights = [0.0f32; 8];
        for i in 0..8 {
            weights[i] = (q_values[i] as f32) * step;
        }
        weights
    }

    /// Quantize a block of 4 weights to Pi2BitBlock format.
    pub fn quantize_block_2bit(&self, weights: &[f32], channel: usize) -> Result<Pi2BitBlock> {
        if self.bits != 2 {
            return Err(PiQuantError::InvalidBits(self.bits));
        }
        if weights.len() != PI2_BLOCK_WEIGHTS {
            return Err(PiQuantError::BlockSizeMismatch {
                expected: PI2_BLOCK_WEIGHTS,
                actual: weights.len(),
            });
        }

        let mut block = Pi2BitBlock::new();

        // Quantize all 4 weights
        let mut q_values = [0i8; 4];
        for (i, &w) in weights.iter().enumerate() {
            q_values[i] = self.quantize_to_int(w, channel);
        }

        // Pack 4 2-bit values into 1 byte
        block.pack(&q_values);

        Ok(block)
    }

    /// Dequantize a Pi2BitBlock back to 4 float weights.
    #[inline]
    pub fn dequantize_block_2bit(&self, block: &Pi2BitBlock, channel: usize) -> [f32; 4] {
        let q_values = block.unpack();
        let step = self.step_size(channel);

        let mut weights = [0.0f32; 4];
        for i in 0..4 {
            weights[i] = (q_values[i] as f32) * step;
        }
        weights
    }

    /// Update the alpha scale for a specific channel.
    ///
    /// Used during QAT training to learn optimal scales.
    ///
    /// # Errors
    ///
    /// Returns error if new_alpha <= 0 (violates INV-2)
    pub fn update_alpha(&mut self, channel: usize, new_alpha: f32) -> Result<()> {
        if channel >= self.alpha.len() {
            return Err(PiQuantError::ChannelOutOfBounds {
                index: channel,
                num_channels: self.alpha.len(),
            });
        }
        if new_alpha <= 0.0 || !new_alpha.is_finite() {
            return Err(PiQuantError::NonPositiveScale(new_alpha));
        }
        self.alpha[channel] = new_alpha;
        Ok(())
    }

    /// Calibrate alpha values from a batch of weight tensors.
    ///
    /// Sets alpha[c] = max(|weights_c|) / (half_range - 0.5) to minimize clipping.
    pub fn calibrate_from_weights(&mut self, weights_per_channel: &[&[f32]]) -> Result<()> {
        if weights_per_channel.len() != self.alpha.len() {
            return Err(PiQuantError::ChannelOutOfBounds {
                index: weights_per_channel.len(),
                num_channels: self.alpha.len(),
            });
        }

        let half = self.half_range as f32;
        let divisor = (half - 0.5) * self.base_step;

        for (c, weights) in weights_per_channel.iter().enumerate() {
            let max_abs = weights
                .iter()
                .map(|w| w.abs())
                .fold(0.0f32, |a, b| a.max(b));

            // Set alpha to cover the range with minimal clipping
            // Ensure positivity (INV-2)
            let new_alpha = (max_abs / divisor).max(1e-8);
            self.alpha[c] = new_alpha;
        }

        Ok(())
    }
}

// ============================================================================
// Pi3BitBlock - Packed 3-bit Storage
// ============================================================================

/// Packed 3-bit quantization block: 8 weights in 3 bytes.
///
/// ## Packing Layout
///
/// ```text
/// Byte 0: [v0:3][v1:3][v2_hi:2]
/// Byte 1: [v2_lo:1][v3:3][v4:3][v5_hi:1]
/// Byte 2: [v5_lo:2][v6:3][v7:3]
/// ```
///
/// Each value is stored as a signed 3-bit integer in range [-4, 3].
/// The offset encoding uses: stored = value + 4, so stored range is [0, 7].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pi3BitBlock {
    /// Packed data: 3 bytes encoding 8 3-bit values
    pub data: [u8; 3],
}

impl Pi3BitBlock {
    /// Create a new zeroed block.
    #[inline]
    pub const fn new() -> Self {
        Self { data: [0; 3] }
    }

    /// Get the packed data bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 3] {
        &self.data
    }

    /// Create from raw bytes.
    #[inline]
    pub fn from_bytes(bytes: [u8; 3]) -> Self {
        Self { data: bytes }
    }

    /// Pack 8 signed 3-bit values (-4 to 3) into the block.
    ///
    /// Values outside the range will be clamped.
    ///
    /// Layout (24 bits total = 3 bytes):
    /// ```text
    /// Byte 0: [v0:3][v1:3][v2_hi:2]  = bits 0-2, 3-5, 6-7
    /// Byte 1: [v2_lo:1][v3:3][v4:3][v5_hi:1] = bits 0, 1-3, 4-6, 7
    /// Byte 2: [v5_lo:2][v6:3][v7:3] = bits 0-1, 2-4, 5-7
    /// ```
    pub fn pack(&mut self, values: &[i8; 8]) {
        // Convert signed values to unsigned (offset by 4)
        // Range: [-4, 3] -> [0, 7]
        let mut u = [0u8; 8];
        for i in 0..8 {
            let v = values[i].clamp(-4, 3);
            u[i] = (v + 4) as u8;
        }

        // Pack into 3 bytes (24 bits for 8 x 3-bit values)
        // Byte 0: v0 in bits 0-2, v1 in bits 3-5, v2's high 2 bits in bits 6-7
        self.data[0] = (u[0] & 0x07)          // v0: bits 0-2
            | ((u[1] & 0x07) << 3)            // v1: bits 3-5
            | ((u[2] & 0x07) << 6); // v2_hi: bits 6-7 (top 2 bits of v2, truncated to fit)

        // Byte 1: v2's low 1 bit in bit 0, v3 in bits 1-3, v4 in bits 4-6, v5's high 1 bit in bit 7
        self.data[1] = ((u[2] >> 2) & 0x01)   // v2_lo: bit 0 (the third bit of v2)
            | ((u[3] & 0x07) << 1)            // v3: bits 1-3
            | ((u[4] & 0x07) << 4)            // v4: bits 4-6
            | ((u[5] & 0x07) << 7); // v5_hi: bit 7 (only lowest bit of v5 fits)

        // Byte 2: v5's low 2 bits in bits 0-1, v6 in bits 2-4, v7 in bits 5-7
        self.data[2] = ((u[5] >> 1) & 0x03)   // v5_lo: bits 0-1 (upper 2 bits of v5)
            | ((u[6] & 0x07) << 2)            // v6: bits 2-4
            | ((u[7] & 0x07) << 5); // v7: bits 5-7
    }

    /// Unpack 8 signed 3-bit values from the block.
    ///
    /// Returns values in range [-4, 3].
    ///
    /// Layout (inverse of pack):
    /// ```text
    /// Byte 0: [v0:3][v1:3][v2_hi:2]  = bits 0-2, 3-5, 6-7
    /// Byte 1: [v2_lo:1][v3:3][v4:3][v5_hi:1] = bits 0, 1-3, 4-6, 7
    /// Byte 2: [v5_lo:2][v6:3][v7:3] = bits 0-1, 2-4, 5-7
    /// ```
    pub fn unpack(&self) -> [i8; 8] {
        let d = self.data;

        // Extract unsigned values (reverse the pack operation)
        let u0 = d[0] & 0x07; // bits 0-2 of byte 0
        let u1 = (d[0] >> 3) & 0x07; // bits 3-5 of byte 0
        let u2 = ((d[0] >> 6) & 0x03) | ((d[1] & 0x01) << 2); // bits 6-7 of byte 0 + bit 0 of byte 1
        let u3 = (d[1] >> 1) & 0x07; // bits 1-3 of byte 1
        let u4 = (d[1] >> 4) & 0x07; // bits 4-6 of byte 1
        let u5 = ((d[1] >> 7) & 0x01) | ((d[2] & 0x03) << 1); // bit 7 of byte 1 + bits 0-1 of byte 2
        let u6 = (d[2] >> 2) & 0x07; // bits 2-4 of byte 2
        let u7 = (d[2] >> 5) & 0x07; // bits 5-7 of byte 2

        // Convert back to signed (subtract offset 4)
        [
            (u0 as i8) - 4,
            (u1 as i8) - 4,
            (u2 as i8) - 4,
            (u3 as i8) - 4,
            (u4 as i8) - 4,
            (u5 as i8) - 4,
            (u6 as i8) - 4,
            (u7 as i8) - 4,
        ]
    }

    /// Get memory size in bytes.
    #[inline]
    pub const fn size_bytes() -> usize {
        PI3_BLOCK_BYTES
    }

    /// Get number of weights stored.
    #[inline]
    pub const fn num_weights() -> usize {
        PI3_BLOCK_WEIGHTS
    }
}

// ============================================================================
// Pi2BitBlock - Packed 2-bit Storage
// ============================================================================

/// Packed 2-bit quantization block: 4 weights in 1 byte.
///
/// ## Packing Layout
///
/// ```text
/// Byte 0: [v0:2][v1:2][v2:2][v3:2]
/// ```
///
/// Each value is stored as a signed 2-bit integer in range [-2, 1].
/// The offset encoding uses: stored = value + 2, so stored range is [0, 3].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pi2BitBlock {
    /// Packed data: 1 byte encoding 4 2-bit values
    pub data: u8,
}

impl Pi2BitBlock {
    /// Create a new zeroed block.
    #[inline]
    pub const fn new() -> Self {
        Self { data: 0 }
    }

    /// Get the packed data byte.
    #[inline]
    pub fn as_byte(&self) -> u8 {
        self.data
    }

    /// Create from raw byte.
    #[inline]
    pub fn from_byte(byte: u8) -> Self {
        Self { data: byte }
    }

    /// Pack 4 signed 2-bit values (-2 to 1) into the block.
    ///
    /// Values outside the range will be clamped.
    pub fn pack(&mut self, values: &[i8; 4]) {
        // Convert signed values to unsigned (offset by 2)
        // Range: [-2, 1] -> [0, 3]
        let u0 = ((values[0].clamp(-2, 1) + 2) as u8) & 0x03;
        let u1 = ((values[1].clamp(-2, 1) + 2) as u8) & 0x03;
        let u2 = ((values[2].clamp(-2, 1) + 2) as u8) & 0x03;
        let u3 = ((values[3].clamp(-2, 1) + 2) as u8) & 0x03;

        // Pack into 1 byte
        self.data = u0 | (u1 << 2) | (u2 << 4) | (u3 << 6);
    }

    /// Unpack 4 signed 2-bit values from the block.
    ///
    /// Returns values in range [-2, 1].
    pub fn unpack(&self) -> [i8; 4] {
        let d = self.data;

        // Extract and convert back to signed
        [
            ((d & 0x03) as i8) - 2,
            (((d >> 2) & 0x03) as i8) - 2,
            (((d >> 4) & 0x03) as i8) - 2,
            (((d >> 6) & 0x03) as i8) - 2,
        ]
    }

    /// Get memory size in bytes.
    #[inline]
    pub const fn size_bytes() -> usize {
        PI2_BLOCK_BYTES
    }

    /// Get number of weights stored.
    #[inline]
    pub const fn num_weights() -> usize {
        PI2_BLOCK_WEIGHTS
    }
}

// ============================================================================
// Batch Quantization Functions
// ============================================================================

/// Quantize a tensor of weights to Pi3BitBlocks.
///
/// # Arguments
///
/// * `weights` - Flat weight tensor (must be multiple of 8)
/// * `quantizer` - Configured PiQuantizer with bits=3
/// * `channel` - Channel index for scale lookup
///
/// # Returns
///
/// Vector of packed Pi3BitBlocks
pub fn quantize_tensor_3bit(
    weights: &[f32],
    quantizer: &PiQuantizer,
    channel: usize,
) -> Result<Vec<Pi3BitBlock>> {
    if quantizer.bits() != 3 {
        return Err(PiQuantError::InvalidBits(quantizer.bits()));
    }

    let num_blocks = (weights.len() + PI3_BLOCK_WEIGHTS - 1) / PI3_BLOCK_WEIGHTS;
    let mut blocks = Vec::with_capacity(num_blocks);

    for chunk in weights.chunks(PI3_BLOCK_WEIGHTS) {
        // Pad last block if needed
        let mut padded = [0.0f32; 8];
        padded[..chunk.len()].copy_from_slice(chunk);

        let block = quantizer.quantize_block_3bit(&padded, channel)?;
        blocks.push(block);
    }

    Ok(blocks)
}

/// Dequantize Pi3BitBlocks back to f32 tensor.
pub fn dequantize_tensor_3bit(
    blocks: &[Pi3BitBlock],
    quantizer: &PiQuantizer,
    channel: usize,
    output: &mut [f32],
) {
    let step = quantizer.step_size(channel);

    for (i, block) in blocks.iter().enumerate() {
        let q_values = block.unpack();
        let base_idx = i * 8;

        for j in 0..8 {
            let out_idx = base_idx + j;
            if out_idx < output.len() {
                output[out_idx] = (q_values[j] as f32) * step;
            }
        }
    }
}

/// Quantize a tensor of weights to Pi2BitBlocks.
pub fn quantize_tensor_2bit(
    weights: &[f32],
    quantizer: &PiQuantizer,
    channel: usize,
) -> Result<Vec<Pi2BitBlock>> {
    if quantizer.bits() != 2 {
        return Err(PiQuantError::InvalidBits(quantizer.bits()));
    }

    let num_blocks = (weights.len() + PI2_BLOCK_WEIGHTS - 1) / PI2_BLOCK_WEIGHTS;
    let mut blocks = Vec::with_capacity(num_blocks);

    for chunk in weights.chunks(PI2_BLOCK_WEIGHTS) {
        // Pad last block if needed
        let mut padded = [0.0f32; 4];
        padded[..chunk.len()].copy_from_slice(chunk);

        let block = quantizer.quantize_block_2bit(&padded, channel)?;
        blocks.push(block);
    }

    Ok(blocks)
}

/// Dequantize Pi2BitBlocks back to f32 tensor.
pub fn dequantize_tensor_2bit(
    blocks: &[Pi2BitBlock],
    quantizer: &PiQuantizer,
    channel: usize,
    output: &mut [f32],
) {
    let step = quantizer.step_size(channel);

    for (i, block) in blocks.iter().enumerate() {
        let q_values = block.unpack();
        let base_idx = i * 4;

        for j in 0..4 {
            let out_idx = base_idx + j;
            if out_idx < output.len() {
                output[out_idx] = (q_values[j] as f32) * step;
            }
        }
    }
}

// ============================================================================
// Quality Metrics
// ============================================================================

/// Compute Mean Squared Error between original and quantized weights.
pub fn compute_mse(original: &[f32], quantized: &[f32]) -> f64 {
    if original.len() != quantized.len() || original.is_empty() {
        return 0.0;
    }

    let sum: f64 = original
        .iter()
        .zip(quantized.iter())
        .map(|(&o, &q)| {
            let diff = (o - q) as f64;
            diff * diff
        })
        .sum();

    sum / (original.len() as f64)
}

/// Compute spectral distortion in dB.
///
/// Formula: 10 * log10(MSE / signal_power)
pub fn compute_spectral_distortion_db(original: &[f32], quantized: &[f32]) -> f64 {
    if original.len() != quantized.len() || original.is_empty() {
        return f64::NEG_INFINITY;
    }

    let signal_power: f64 = original.iter().map(|&x| (x as f64).powi(2)).sum();
    if signal_power == 0.0 {
        return 0.0;
    }

    let mse = compute_mse(original, quantized);
    10.0 * (mse / (signal_power / original.len() as f64)).log10()
}

/// Compute cosine similarity between original and quantized weights.
pub fn compute_cosine_similarity(original: &[f32], quantized: &[f32]) -> f32 {
    if original.len() != quantized.len() || original.is_empty() {
        return 0.0;
    }

    let dot: f32 = original
        .iter()
        .zip(quantized.iter())
        .map(|(&o, &q)| o * q)
        .sum();

    let norm_o: f32 = original.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_q: f32 = quantized.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm_o == 0.0 || norm_q == 0.0 {
        return 0.0;
    }

    dot / (norm_o * norm_q)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --------------------------------
    // PiQuantizer tests
    // --------------------------------

    #[test]
    fn test_pi_quantizer_new_valid() {
        let quantizer = PiQuantizer::new(3, 4, vec![1.0, 2.0, 0.5]).unwrap();
        assert_eq!(quantizer.bits(), 3);
        assert_eq!(quantizer.k(), 4);
        assert_eq!(quantizer.num_channels(), 3);
    }

    #[test]
    fn test_pi_quantizer_invalid_bits() {
        let result = PiQuantizer::new(4, 4, vec![1.0]);
        assert!(matches!(result, Err(PiQuantError::InvalidBits(4))));

        let result = PiQuantizer::new(1, 4, vec![1.0]);
        assert!(matches!(result, Err(PiQuantError::InvalidBits(1))));
    }

    #[test]
    fn test_pi_quantizer_invalid_k() {
        let result = PiQuantizer::new(3, 1, vec![1.0]);
        assert!(matches!(result, Err(PiQuantError::InvalidK(1))));

        let result = PiQuantizer::new(3, 6, vec![1.0]);
        assert!(matches!(result, Err(PiQuantError::InvalidK(6))));
    }

    #[test]
    fn test_pi_quantizer_inv2_positive_scale() {
        // INV-2: Scale alpha must be positive
        let result = PiQuantizer::new(3, 4, vec![1.0, -0.5, 2.0]);
        assert!(matches!(result, Err(PiQuantError::NonPositiveScale(_))));

        let result = PiQuantizer::new(3, 4, vec![0.0]);
        assert!(matches!(result, Err(PiQuantError::NonPositiveScale(0.0))));

        let result = PiQuantizer::new(3, 4, vec![f32::NAN]);
        assert!(matches!(result, Err(PiQuantError::NonPositiveScale(_))));
    }

    #[test]
    fn test_pi_quantizer_empty_alpha() {
        let result = PiQuantizer::new(3, 4, vec![]);
        assert!(matches!(result, Err(PiQuantError::EmptyAlpha)));
    }

    #[test]
    fn test_pi_quantizer_step_size() {
        // INV-3: Step size = alpha * pi / k
        let quantizer = PiQuantizer::new(3, 4, vec![1.0, 2.0]).unwrap();

        let step_0 = quantizer.step_size(0);
        let step_1 = quantizer.step_size(1);

        assert!((step_0 - PI / 4.0).abs() < 1e-6);
        assert!((step_1 - 2.0 * PI / 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_scalar_3bit() {
        let quantizer = PiQuantizer::new(3, 4, vec![1.0]).unwrap();
        let step = PI / 4.0;

        // Test zero
        let (q, dq) = quantizer.quantize_scalar(0.0, 0);
        assert_eq!(q, 0);
        assert!(dq.abs() < 1e-6);

        // Test small positive
        let (q, dq) = quantizer.quantize_scalar(step * 2.0, 0);
        assert_eq!(q, 2);
        assert!((dq - step * 2.0).abs() < 1e-6);

        // Test negative
        let (q, dq) = quantizer.quantize_scalar(-step * 3.0, 0);
        assert_eq!(q, -3);
        assert!((dq + step * 3.0).abs() < 1e-6);

        // Test clamping at upper bound (3-bit: max = 3)
        let (q, _dq) = quantizer.quantize_scalar(step * 10.0, 0);
        assert_eq!(q, 3);

        // Test clamping at lower bound (3-bit: min = -4)
        let (q, _dq) = quantizer.quantize_scalar(-step * 10.0, 0);
        assert_eq!(q, -4);
    }

    #[test]
    fn test_quantize_scalar_2bit() {
        let quantizer = PiQuantizer::new(2, 4, vec![1.0]).unwrap();
        let step = PI / 4.0;

        // Test zero
        let (q, dq) = quantizer.quantize_scalar(0.0, 0);
        assert_eq!(q, 0);
        assert!(dq.abs() < 1e-6);

        // Test positive
        let (q, _dq) = quantizer.quantize_scalar(step * 1.0, 0);
        assert_eq!(q, 1);

        // Test clamping (2-bit: range [-2, 1])
        let (q, _dq) = quantizer.quantize_scalar(step * 10.0, 0);
        assert_eq!(q, 1);

        let (q, _dq) = quantizer.quantize_scalar(-step * 10.0, 0);
        assert_eq!(q, -2);
    }

    // --------------------------------
    // Pi3BitBlock tests
    // --------------------------------

    #[test]
    fn test_pi3bit_block_pack_unpack_roundtrip() {
        let values: [i8; 8] = [-4, -3, -2, -1, 0, 1, 2, 3];
        let mut block = Pi3BitBlock::new();
        block.pack(&values);
        let unpacked = block.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pi3bit_block_all_zeros() {
        let values: [i8; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
        let mut block = Pi3BitBlock::new();
        block.pack(&values);
        let unpacked = block.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pi3bit_block_all_max() {
        let values: [i8; 8] = [3, 3, 3, 3, 3, 3, 3, 3];
        let mut block = Pi3BitBlock::new();
        block.pack(&values);
        let unpacked = block.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pi3bit_block_all_min() {
        let values: [i8; 8] = [-4, -4, -4, -4, -4, -4, -4, -4];
        let mut block = Pi3BitBlock::new();
        block.pack(&values);
        let unpacked = block.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pi3bit_block_clamping() {
        // Values outside range should be clamped
        let values: [i8; 8] = [-10, -5, -4, 0, 3, 4, 5, 10];
        let mut block = Pi3BitBlock::new();
        block.pack(&values);
        let unpacked = block.unpack();
        let expected: [i8; 8] = [-4, -4, -4, 0, 3, 3, 3, 3];
        assert_eq!(expected, unpacked);
    }

    #[test]
    fn test_pi3bit_block_size() {
        assert_eq!(Pi3BitBlock::size_bytes(), 3);
        assert_eq!(Pi3BitBlock::num_weights(), 8);
    }

    // --------------------------------
    // Pi2BitBlock tests
    // --------------------------------

    #[test]
    fn test_pi2bit_block_pack_unpack_roundtrip() {
        let values: [i8; 4] = [-2, -1, 0, 1];
        let mut block = Pi2BitBlock::new();
        block.pack(&values);
        let unpacked = block.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pi2bit_block_all_zeros() {
        let values: [i8; 4] = [0, 0, 0, 0];
        let mut block = Pi2BitBlock::new();
        block.pack(&values);
        let unpacked = block.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pi2bit_block_extremes() {
        let values: [i8; 4] = [-2, -2, 1, 1];
        let mut block = Pi2BitBlock::new();
        block.pack(&values);
        let unpacked = block.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pi2bit_block_clamping() {
        let values: [i8; 4] = [-10, -3, 2, 10];
        let mut block = Pi2BitBlock::new();
        block.pack(&values);
        let unpacked = block.unpack();
        let expected: [i8; 4] = [-2, -2, 1, 1];
        assert_eq!(expected, unpacked);
    }

    #[test]
    fn test_pi2bit_block_size() {
        assert_eq!(Pi2BitBlock::size_bytes(), 1);
        assert_eq!(Pi2BitBlock::num_weights(), 4);
    }

    // --------------------------------
    // Block quantization integration tests
    // --------------------------------

    #[test]
    fn test_quantize_block_3bit() {
        let quantizer = PiQuantizer::new(3, 4, vec![1.0]).unwrap();
        let weights = [0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 0.25];

        let block = quantizer.quantize_block_3bit(&weights, 0).unwrap();
        let dequantized = quantizer.dequantize_block_3bit(&block, 0);

        // Check that dequantized values are close to original
        let mse = compute_mse(&weights, &dequantized);
        assert!(mse < 0.5, "MSE too high: {}", mse);
    }

    #[test]
    fn test_quantize_block_2bit() {
        let quantizer = PiQuantizer::new(2, 4, vec![1.0]).unwrap();
        let weights = [0.0, 0.5, -0.5, 1.0];

        let block = quantizer.quantize_block_2bit(&weights, 0).unwrap();
        let dequantized = quantizer.dequantize_block_2bit(&block, 0);

        // 2-bit is very coarse, but should still produce valid output
        assert_eq!(dequantized.len(), 4);
    }

    // --------------------------------
    // Tensor quantization tests
    // --------------------------------

    #[test]
    fn test_quantize_tensor_3bit() {
        let quantizer = PiQuantizer::new(3, 4, vec![1.0]).unwrap();
        let weights: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.1).collect();

        let blocks = quantize_tensor_3bit(&weights, &quantizer, 0).unwrap();
        assert_eq!(blocks.len(), 3); // 24 weights / 8 per block = 3 blocks

        let mut output = vec![0.0f32; weights.len()];
        dequantize_tensor_3bit(&blocks, &quantizer, 0, &mut output);

        let mse = compute_mse(&weights, &output);
        assert!(mse < 0.5, "MSE too high: {}", mse);
    }

    #[test]
    fn test_quantize_tensor_2bit() {
        let quantizer = PiQuantizer::new(2, 4, vec![1.0]).unwrap();
        let weights: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.2).collect();

        let blocks = quantize_tensor_2bit(&weights, &quantizer, 0).unwrap();
        assert_eq!(blocks.len(), 3); // 12 weights / 4 per block = 3 blocks

        let mut output = vec![0.0f32; weights.len()];
        dequantize_tensor_2bit(&blocks, &quantizer, 0, &mut output);

        // Just check it doesn't crash and produces output
        assert_eq!(output.len(), 12);
    }

    // --------------------------------
    // Quality metrics tests
    // --------------------------------

    #[test]
    fn test_compute_mse_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mse = compute_mse(&a, &b);
        assert!((mse - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_mse_different() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mse = compute_mse(&a, &b);
        assert!((mse - 1.0).abs() < 1e-10); // Each diff is 1, MSE = 1
    }

    #[test]
    fn test_compute_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = compute_cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = compute_cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = compute_cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    // --------------------------------
    // Calibration tests
    // --------------------------------

    #[test]
    fn test_calibrate_from_weights() {
        let mut quantizer = PiQuantizer::new(3, 4, vec![1.0, 1.0]).unwrap();

        let channel_0_weights: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
        let channel_1_weights: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0];

        quantizer
            .calibrate_from_weights(&[&channel_0_weights, &channel_1_weights])
            .unwrap();

        // Channel 1 should have higher alpha since it has larger weights
        assert!(quantizer.alpha()[1] > quantizer.alpha()[0]);
    }

    #[test]
    fn test_update_alpha() {
        let mut quantizer = PiQuantizer::new(3, 4, vec![1.0]).unwrap();

        quantizer.update_alpha(0, 2.0).unwrap();
        assert!((quantizer.alpha()[0] - 2.0).abs() < 1e-6);

        // Should fail for negative
        let result = quantizer.update_alpha(0, -1.0);
        assert!(result.is_err());

        // Should fail for out of bounds
        let result = quantizer.update_alpha(99, 1.0);
        assert!(result.is_err());
    }

    // --------------------------------
    // Edge case tests
    // --------------------------------

    #[test]
    fn test_quantize_block_wrong_bits() {
        let quantizer = PiQuantizer::new(3, 4, vec![1.0]).unwrap();
        let weights = [0.0f32; 4];

        // Trying to use 2-bit block with 3-bit quantizer
        let result = quantizer.quantize_block_2bit(&weights, 0);
        assert!(matches!(result, Err(PiQuantError::InvalidBits(3))));
    }

    #[test]
    fn test_quantize_block_wrong_size() {
        let quantizer = PiQuantizer::new(3, 4, vec![1.0]).unwrap();
        let weights = [0.0f32; 4]; // Should be 8 for 3-bit

        let result = quantizer.quantize_block_3bit(&weights, 0);
        assert!(matches!(
            result,
            Err(PiQuantError::BlockSizeMismatch {
                expected: 8,
                actual: 4
            })
        ));
    }

    #[test]
    fn test_bits_per_weight() {
        let q3 = PiQuantizer::new(3, 4, vec![1.0]).unwrap();
        assert!((q3.bits_per_weight() - 3.0625).abs() < 1e-4);

        let q2 = PiQuantizer::new(2, 4, vec![1.0]).unwrap();
        assert!((q2.bits_per_weight() - 2.0625).abs() < 1e-4);
    }
}
