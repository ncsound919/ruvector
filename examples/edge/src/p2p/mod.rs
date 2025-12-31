//! P2P Swarm v2 - Production Grade Rust Implementation
//!
//! Features:
//! - Ed25519 identity keys + X25519 ephemeral keys for ECDH
//! - AES-256-GCM authenticated encryption
//! - Message replay protection (nonces, counters, timestamps)
//! - GUN-based signaling (no external PeerServer)
//! - IPFS CID pointers for large payloads
//! - Ed25519 signatures on all messages
//! - Relay health monitoring
//! - Task execution envelope with resource budgets
//! - WASM compatible

mod identity;
mod crypto;
mod relay;
mod artifact;
mod envelope;
mod swarm;

pub use identity::{IdentityManager, KeyPair};
pub use crypto::{CryptoV2, EncryptedPayload};
pub use relay::RelayManager;
pub use artifact::ArtifactStore;
pub use envelope::{SignedEnvelope, TaskEnvelope, TaskReceipt, ArtifactPointer};
pub use swarm::{P2PSwarmV2, SwarmStatus};
