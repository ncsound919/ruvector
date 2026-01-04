//! Witness logging for audit trails and ReasoningBank integration
//!
//! Every inference produces a small witness bundle that records
//! what happened and enables verification and replay.

pub mod log;
pub mod hash;

// Re-export WitnessLog from types as the canonical location
pub use crate::types::WitnessLog;
pub use log::{WitnessBuilder, WitnessAggregator};
pub use hash::{compute_witness_hash, verify_witness_hash};
