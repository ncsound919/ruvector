//! # neural-trader-replay
//!
//! Witnessable replay segments, RVF serialization stubs, and audit
//! receipt logging for the RuVector Neural Trader (ADR-084).

use neural_trader_coherence::{CoherenceDecision, WitnessReceipt};
use neural_trader_core::MarketEvent;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Replay segment
// ---------------------------------------------------------------------------

/// A sealed, signed replay segment containing a compact subgraph window,
/// embeddings, labels, and coherence statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaySegment {
    pub segment_id: u64,
    pub symbol_id: u32,
    pub start_ts_ns: u64,
    pub end_ts_ns: u64,
    pub segment_kind: SegmentKind,
    /// Serialized events in this window.
    pub events: Vec<MarketEvent>,
    /// Embedding snapshot at segment creation.
    pub embedding: Option<Vec<f32>>,
    /// Realized labels (e.g. mid-price move, fill outcome).
    pub labels: serde_json::Value,
    /// Coherence statistics at write time.
    pub coherence_stats: CoherenceStats,
    /// Lineage metadata.
    pub lineage: SegmentLineage,
    /// Witness hash for tamper detection.
    pub witness_hash: [u8; 16],
}

/// Segment classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentKind {
    HighUncertainty,
    LargeImpact,
    RegimeTransition,
    StructuralAnomaly,
    RareQueuePattern,
    HeadDisagreement,
    Routine,
}

/// Coherence statistics captured at segment write time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceStats {
    pub mincut_value: u64,
    pub partition_hash: [u8; 16],
    pub drift_score: f32,
    pub cusum_score: f32,
}

/// Lineage metadata linking a segment to its origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentLineage {
    pub model_id: String,
    pub policy_version: String,
    pub ingest_batch_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Memory store trait
// ---------------------------------------------------------------------------

/// Query for memory retrieval.
#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub symbol_id: u32,
    pub embedding: Vec<f32>,
    pub regime: Option<String>,
    pub limit: usize,
}

/// Selective, bounded memory store.
pub trait MemoryStore {
    fn retrieve(&self, query: &MemoryQuery) -> anyhow::Result<Vec<ReplaySegment>>;

    /// Attempts to write a segment. Returns `true` if the gate allowed
    /// admission, `false` if rejected.
    fn maybe_write(
        &mut self,
        seg: ReplaySegment,
        gate: &CoherenceDecision,
    ) -> anyhow::Result<bool>;
}

// ---------------------------------------------------------------------------
// In-memory receipt log
// ---------------------------------------------------------------------------

/// Simple in-memory witness logger for testing and research.
pub struct InMemoryReceiptLog {
    pub receipts: Vec<WitnessReceipt>,
}

impl InMemoryReceiptLog {
    pub fn new() -> Self {
        Self {
            receipts: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.receipts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.receipts.is_empty()
    }
}

impl Default for InMemoryReceiptLog {
    fn default() -> Self {
        Self::new()
    }
}

impl neural_trader_coherence::WitnessLogger for InMemoryReceiptLog {
    fn append_receipt(&mut self, receipt: WitnessReceipt) -> anyhow::Result<()> {
        self.receipts.push(receipt);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// In-memory reservoir store
// ---------------------------------------------------------------------------

/// Simple bounded reservoir memory store for research use.
pub struct ReservoirStore {
    pub segments: Vec<ReplaySegment>,
    pub max_size: usize,
}

impl ReservoirStore {
    pub fn new(max_size: usize) -> Self {
        Self {
            segments: Vec::new(),
            max_size,
        }
    }
}

impl MemoryStore for ReservoirStore {
    fn retrieve(&self, query: &MemoryQuery) -> anyhow::Result<Vec<ReplaySegment>> {
        // Naive: return segments matching symbol_id, up to limit.
        let results: Vec<_> = self
            .segments
            .iter()
            .filter(|s| s.symbol_id == query.symbol_id)
            .take(query.limit)
            .cloned()
            .collect();
        Ok(results)
    }

    fn maybe_write(
        &mut self,
        seg: ReplaySegment,
        gate: &CoherenceDecision,
    ) -> anyhow::Result<bool> {
        if !gate.allow_write {
            return Ok(false);
        }
        if self.segments.len() >= self.max_size {
            // Evict oldest (simplistic; real impl uses reservoir sampling).
            self.segments.remove(0);
        }
        self.segments.push(seg);
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use neural_trader_coherence::WitnessLogger;

    #[test]
    fn receipt_log_append() {
        let mut log = InMemoryReceiptLog::new();
        assert!(log.is_empty());

        let receipt = WitnessReceipt {
            ts_ns: 1_000_000,
            model_id: "test-v1".into(),
            input_segment_hash: [0u8; 16],
            coherence_witness_hash: [1u8; 16],
            policy_hash: [2u8; 16],
            action_intent: "place_bid".into(),
            verified_token_id: [3u8; 16],
            resulting_state_hash: [4u8; 16],
        };
        log.append_receipt(receipt).unwrap();
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn reservoir_respects_gate() {
        let mut store = ReservoirStore::new(100);
        let seg = make_test_segment(1);

        let blocked = CoherenceDecision {
            allow_retrieve: true,
            allow_write: false,
            allow_learn: false,
            allow_act: false,
            mincut_value: 3,
            partition_hash: [0u8; 16],
            drift_score: 0.9,
            cusum_score: 5.0,
            reasons: vec!["test block".into()],
        };
        assert!(!store.maybe_write(seg.clone(), &blocked).unwrap());
        assert!(store.segments.is_empty());

        let allowed = CoherenceDecision {
            allow_retrieve: true,
            allow_write: true,
            allow_learn: true,
            allow_act: true,
            mincut_value: 15,
            partition_hash: [0u8; 16],
            drift_score: 0.1,
            cusum_score: 1.0,
            reasons: vec![],
        };
        assert!(store.maybe_write(seg, &allowed).unwrap());
        assert_eq!(store.segments.len(), 1);
    }

    #[test]
    fn reservoir_evicts_when_full() {
        let mut store = ReservoirStore::new(2);
        let gate = CoherenceDecision {
            allow_retrieve: true,
            allow_write: true,
            allow_learn: true,
            allow_act: true,
            mincut_value: 15,
            partition_hash: [0u8; 16],
            drift_score: 0.1,
            cusum_score: 1.0,
            reasons: vec![],
        };
        store.maybe_write(make_test_segment(1), &gate).unwrap();
        store.maybe_write(make_test_segment(2), &gate).unwrap();
        store.maybe_write(make_test_segment(3), &gate).unwrap();
        assert_eq!(store.segments.len(), 2);
        // First segment (id=1) was evicted.
        assert_eq!(store.segments[0].segment_id, 2);
    }

    fn make_test_segment(id: u64) -> ReplaySegment {
        ReplaySegment {
            segment_id: id,
            symbol_id: 42,
            start_ts_ns: 0,
            end_ts_ns: 1_000_000,
            segment_kind: SegmentKind::Routine,
            events: vec![],
            embedding: None,
            labels: serde_json::json!({}),
            coherence_stats: CoherenceStats {
                mincut_value: 10,
                partition_hash: [0u8; 16],
                drift_score: 0.1,
                cusum_score: 1.0,
            },
            lineage: SegmentLineage {
                model_id: "test".into(),
                policy_version: "v1".into(),
                ingest_batch_id: None,
            },
            witness_hash: [0u8; 16],
        }
    }
}
