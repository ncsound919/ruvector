use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Decision produced by the cognitive coherence pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoherenceDecision {
    Pass,
    Fail { severity: u8 },
    Inconclusive,
}

/// A single receipt in the witness chain, linking the current epoch's
/// computation to the previous receipt via `prev_hash`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerWitnessReceipt {
    pub epoch: u64,
    pub prev_hash: [u8; 32],
    pub input_hash: [u8; 32],
    pub mincut_hash: [u8; 32],
    /// Spectral coherence score as fixed-point 32.32.
    pub spectral_scs: u64,
    pub evidence_hash: [u8; 32],
    pub decision: CoherenceDecision,
    /// Hash of all other fields in this receipt.
    pub receipt_hash: [u8; 32],
}

impl ContainerWitnessReceipt {
    /// Returns the concatenation of all fields except `receipt_hash`,
    /// suitable for hashing or signing.
    pub fn signable_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + 32 * 4 + 8 + 2);
        buf.extend_from_slice(&self.epoch.to_le_bytes());
        buf.extend_from_slice(&self.prev_hash);
        buf.extend_from_slice(&self.input_hash);
        buf.extend_from_slice(&self.mincut_hash);
        buf.extend_from_slice(&self.spectral_scs.to_le_bytes());
        buf.extend_from_slice(&self.evidence_hash);

        // Encode decision deterministically
        match self.decision {
            CoherenceDecision::Pass => buf.push(0),
            CoherenceDecision::Fail { severity } => {
                buf.push(1);
                buf.push(severity);
            }
            CoherenceDecision::Inconclusive => buf.push(2),
        }

        buf
    }

    /// Computes and sets `receipt_hash` from all other fields.
    pub fn compute_hash(&mut self) {
        self.receipt_hash = deterministic_hash(&self.signable_bytes());
    }

    /// Verifies that `receipt_hash` matches the hash of the signable bytes.
    pub fn verify(&self) -> bool {
        let expected = deterministic_hash(&self.signable_bytes());
        self.receipt_hash == expected
    }
}

/// Result of verifying a witness chain.
#[derive(Debug, Clone)]
pub enum VerificationResult {
    Valid {
        chain_length: usize,
        first_epoch: u64,
        last_epoch: u64,
    },
    Empty,
    BrokenChain {
        epoch: u64,
    },
    EpochGap {
        expected: u64,
        got: u64,
    },
}

/// An append-only chain of witness receipts with hash linking.
///
/// Receipts are stored in a ring buffer bounded by `max_receipts`.
/// Each receipt's `prev_hash` points to the preceding receipt's
/// `receipt_hash`, forming a verifiable chain.
pub struct WitnessChain {
    current_epoch: u64,
    prev_hash: [u8; 32],
    receipts: Vec<ContainerWitnessReceipt>,
    max_receipts: usize,
}

impl WitnessChain {
    /// Creates a new witness chain that retains up to `max_receipts` entries.
    pub fn new(max_receipts: usize) -> Self {
        Self {
            current_epoch: 0,
            prev_hash: [0u8; 32],
            receipts: Vec::with_capacity(max_receipts.min(1024)),
            max_receipts,
        }
    }

    /// Generates a new receipt, appends it to the chain, and advances the epoch.
    pub fn generate_receipt(
        &mut self,
        input_deltas: &[u8],
        mincut_data: &[u8],
        spectral_scs: f64,
        evidence_data: &[u8],
        decision: CoherenceDecision,
    ) -> ContainerWitnessReceipt {
        let scs_fixed = f64_to_fixed_32_32(spectral_scs);

        let mut receipt = ContainerWitnessReceipt {
            epoch: self.current_epoch,
            prev_hash: self.prev_hash,
            input_hash: deterministic_hash(input_deltas),
            mincut_hash: deterministic_hash(mincut_data),
            spectral_scs: scs_fixed,
            evidence_hash: deterministic_hash(evidence_data),
            decision,
            receipt_hash: [0u8; 32],
        };

        receipt.compute_hash();

        // Advance chain state
        self.prev_hash = receipt.receipt_hash;
        self.current_epoch += 1;

        // Append to ring buffer
        if self.receipts.len() >= self.max_receipts {
            self.receipts.remove(0);
        }
        self.receipts.push(receipt.clone());

        receipt
    }

    /// Returns the current epoch counter.
    pub fn current_epoch(&self) -> u64 {
        self.current_epoch
    }

    /// Returns the most recent receipt, if any.
    pub fn latest_receipt(&self) -> Option<&ContainerWitnessReceipt> {
        self.receipts.last()
    }

    /// Returns all stored receipts in chronological order.
    pub fn receipt_chain(&self) -> &[ContainerWitnessReceipt] {
        &self.receipts
    }

    /// Verifies the integrity of a receipt chain.
    ///
    /// Checks that:
    /// - Each receipt's `receipt_hash` matches its contents.
    /// - Each receipt's `prev_hash` matches the previous receipt's `receipt_hash`.
    /// - Epochs are strictly monotonically increasing with step 1.
    pub fn verify_chain(receipts: &[ContainerWitnessReceipt]) -> VerificationResult {
        if receipts.is_empty() {
            return VerificationResult::Empty;
        }

        // Verify each receipt's self-hash
        for receipt in receipts {
            if !receipt.verify() {
                return VerificationResult::BrokenChain {
                    epoch: receipt.epoch,
                };
            }
        }

        // Verify chain linkage and epoch monotonicity
        for window in receipts.windows(2) {
            let prev = &window[0];
            let curr = &window[1];

            // Check hash linkage
            if curr.prev_hash != prev.receipt_hash {
                return VerificationResult::BrokenChain {
                    epoch: curr.epoch,
                };
            }

            // Check epoch monotonicity
            if curr.epoch != prev.epoch + 1 {
                return VerificationResult::EpochGap {
                    expected: prev.epoch + 1,
                    got: curr.epoch,
                };
            }
        }

        VerificationResult::Valid {
            chain_length: receipts.len(),
            first_epoch: receipts[0].epoch,
            last_epoch: receipts[receipts.len() - 1].epoch,
        }
    }
}

/// Converts an `f64` to a 32.32 fixed-point representation.
fn f64_to_fixed_32_32(value: f64) -> u64 {
    let clamped = value.clamp(0.0, (u32::MAX as f64) + 0.999_999_999);
    (clamped * (1u64 << 32) as f64) as u64
}

/// Deterministic hash producing 32 bytes.
///
/// Uses `std::hash::DefaultHasher` (SipHash-1-3) with four different seeds
/// to fill all 32 bytes. This is NOT cryptographic but is deterministic
/// across runs on the same platform.
pub(crate) fn deterministic_hash(data: &[u8]) -> [u8; 32] {
    let mut result = [0u8; 32];

    for i in 0u64..4 {
        let mut hasher = DefaultHasher::new();
        // Mix in a seed to produce different 8-byte segments
        i.hash(&mut hasher);
        data.hash(&mut hasher);
        let hash_val = hasher.finish();
        let offset = (i as usize) * 8;
        result[offset..offset + 8].copy_from_slice(&hash_val.to_le_bytes());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_hash_consistency() {
        let data = b"hello witness chain";
        let h1 = deterministic_hash(data);
        let h2 = deterministic_hash(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_deterministic_hash_distinct() {
        let h1 = deterministic_hash(b"input A");
        let h2 = deterministic_hash(b"input B");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_witness_chain_integrity() {
        let mut chain = WitnessChain::new(100);

        // Generate a series of receipts
        for i in 0..5 {
            chain.generate_receipt(
                &[i as u8; 16],
                &[i as u8; 8],
                0.95 + (i as f64) * 0.01,
                &[i as u8; 4],
                CoherenceDecision::Pass,
            );
        }

        assert_eq!(chain.current_epoch(), 5);
        assert_eq!(chain.receipt_chain().len(), 5);

        // The chain should verify as valid
        let result = WitnessChain::verify_chain(chain.receipt_chain());
        match result {
            VerificationResult::Valid {
                chain_length,
                first_epoch,
                last_epoch,
            } => {
                assert_eq!(chain_length, 5);
                assert_eq!(first_epoch, 0);
                assert_eq!(last_epoch, 4);
            }
            other => panic!("Expected Valid, got {:?}", other),
        }
    }

    #[test]
    fn test_witness_chain_epoch_monotonicity() {
        let mut chain = WitnessChain::new(100);

        for _ in 0..3 {
            chain.generate_receipt(
                b"data",
                b"mincut",
                0.5,
                b"evidence",
                CoherenceDecision::Pass,
            );
        }

        let receipts = chain.receipt_chain();
        for window in receipts.windows(2) {
            assert_eq!(window[1].epoch, window[0].epoch + 1);
        }
    }

    #[test]
    fn test_witness_receipt_self_verify() {
        let mut chain = WitnessChain::new(10);
        let receipt = chain.generate_receipt(
            b"test input",
            b"test mincut",
            0.42,
            b"test evidence",
            CoherenceDecision::Fail { severity: 3 },
        );
        assert!(receipt.verify());
    }

    #[test]
    fn test_verification_detects_tampering() {
        let mut chain = WitnessChain::new(100);

        for i in 0..3 {
            chain.generate_receipt(
                &[i; 8],
                &[i; 8],
                0.5,
                &[i; 8],
                CoherenceDecision::Pass,
            );
        }

        // Tamper with the middle receipt's data
        let mut tampered: Vec<ContainerWitnessReceipt> =
            chain.receipt_chain().to_vec();
        tampered[1].spectral_scs = 999_999;

        let result = WitnessChain::verify_chain(&tampered);
        match result {
            VerificationResult::BrokenChain { epoch } => {
                assert_eq!(epoch, 1);
            }
            other => panic!("Expected BrokenChain, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_chain_verification() {
        let result = WitnessChain::verify_chain(&[]);
        assert!(matches!(result, VerificationResult::Empty));
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let mut chain = WitnessChain::new(3);

        for i in 0..5 {
            chain.generate_receipt(
                &[i as u8; 4],
                &[i as u8; 4],
                0.5,
                &[i as u8; 4],
                CoherenceDecision::Pass,
            );
        }

        // Only the last 3 receipts should be retained
        assert_eq!(chain.receipt_chain().len(), 3);
        assert_eq!(chain.receipt_chain()[0].epoch, 2);
        assert_eq!(chain.receipt_chain()[2].epoch, 4);
    }

    #[test]
    fn test_f64_to_fixed_32_32() {
        // 1.0 should map to 2^32
        let fixed = f64_to_fixed_32_32(1.0);
        assert_eq!(fixed, 1u64 << 32);

        // 0.0 should map to 0
        let fixed = f64_to_fixed_32_32(0.0);
        assert_eq!(fixed, 0);

        // 0.5 should map to 2^31
        let fixed = f64_to_fixed_32_32(0.5);
        assert_eq!(fixed, 1u64 << 31);
    }

    #[test]
    fn test_coherence_decision_serialization() {
        let decisions = vec![
            CoherenceDecision::Pass,
            CoherenceDecision::Fail { severity: 5 },
            CoherenceDecision::Inconclusive,
        ];

        for decision in &decisions {
            let json = serde_json::to_string(decision).unwrap();
            let deserialized: CoherenceDecision = serde_json::from_str(&json).unwrap();
            assert_eq!(*decision, deserialized);
        }
    }
}
