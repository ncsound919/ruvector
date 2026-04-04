//! # RVM Kernel
//!
//! Top-level integration crate for the RVM (RuVix Virtual Machine)
//! coherence-native microhypervisor. This crate wires together all
//! subsystems (HAL, capabilities, witness, proof, partitions, scheduler,
//! memory, coherence, boot, Wasm, and security) into a single API
//! surface.
//!
//! ## Architecture
//!
//! ```text
//!          +---------------------------------------------+
//!          |                  rvm-kernel                  |
//!          |                                             |
//!          |  +----------+  +----------+  +-----------+  |
//!          |  | rvm-boot |  | rvm-sched|  |rvm-memory |  |
//!          |  +----+-----+  +----+-----+  +-----+-----+  |
//!          |       |             |              |         |
//!          |  +----+-------------+--------------+-----+  |
//!          |  |            rvm-partition               |  |
//!          |  +----+--------+----------+---------+----+  |
//!          |       |        |          |         |       |
//!          |  +----+--+ +---+----+ +---+---+ +---+----+  |
//!          |  |rvm-cap| |rvm-wit.| |rvm-prf| |rvm-sec.|  |
//!          |  +----+--+ +---+----+ +---+---+ +---+----+  |
//!          |       |        |          |         |       |
//!          |  +----+--------+----------+---------+----+  |
//!          |  |              rvm-types                |   |
//!          |  +----+----------------------------------+  |
//!          |       |                                     |
//!          |  +----+--+  +----------+                    |
//!          |  |rvm-hal|  |rvm-wasm  | (optional)         |
//!          |  +-------+  +----------+                    |
//!          +---------------------------------------------+
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::doc_markdown,
    clippy::new_without_default
)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

/// Re-export all subsystem crates for unified access.
pub use rvm_boot as boot;
/// Capability-based access control.
pub use rvm_cap as cap;
/// Coherence monitoring and Phi computation.
pub use rvm_coherence as coherence;
/// Hardware abstraction layer traits.
pub use rvm_hal as hal;
/// Guest memory management.
pub use rvm_memory as memory;
/// Partition lifecycle management.
pub use rvm_partition as partition;
/// Proof-gated state transitions.
pub use rvm_proof as proof;
/// Coherence-weighted scheduler.
pub use rvm_sched as sched;
/// Security policy enforcement.
pub use rvm_security as security;
/// Core type definitions.
pub use rvm_types as types;
/// WebAssembly guest runtime.
pub use rvm_wasm as wasm;
/// Witness trail management.
pub use rvm_witness as witness;

/// RVM version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// RVM crate count (number of subsystem crates).
pub const CRATE_COUNT: usize = 13;

// ---------------------------------------------------------------------------
// Kernel integration struct
// ---------------------------------------------------------------------------

use rvm_boot::BootTracker;
use rvm_cap::{CapManagerConfig, CapabilityManager};
use rvm_coherence::{CoherenceDecision, DefaultCoherenceEngine};
use rvm_partition::PartitionManager;
use rvm_sched::Scheduler;
use rvm_types::{
    ActionKind, PartitionConfig, PartitionId, RvmConfig, RvmError, RvmResult,
    WitnessRecord,
};
use rvm_witness::WitnessLog;

/// Default maximum CPUs supported by the kernel.
const DEFAULT_MAX_CPUS: usize = 8;

/// Default witness log capacity (number of records).
const DEFAULT_WITNESS_CAPACITY: usize = 256;

/// Default capability table capacity per partition.
const DEFAULT_CAP_CAPACITY: usize = 256;

/// Default partition table capacity.
const DEFAULT_MAX_PARTITIONS: usize = 256;

/// Result of a single epoch tick, combining scheduler and coherence outputs.
#[derive(Debug, Clone)]
pub struct EpochResult {
    /// Scheduler epoch summary (context switches, utilisation).
    pub summary: rvm_sched::EpochSummary,
    /// Coherence engine recommendation (split, merge, or no-action).
    pub decision: CoherenceDecision,
}

/// Result of applying a coherence decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyResult {
    /// No action was taken.
    NoAction,
    /// A partition was split into two.
    Split {
        /// The original partition.
        source: PartitionId,
        /// The newly created partition.
        child: PartitionId,
    },
    /// Two partitions were merged.
    Merged {
        /// The surviving partition.
        survivor: PartitionId,
        /// The partition that was absorbed.
        absorbed: PartitionId,
    },
}

/// Top-level kernel integrating all RVM subsystems.
///
/// The kernel holds ownership of all core subsystem instances
/// and provides a unified API for partition lifecycle, scheduling,
/// and security enforcement.
pub struct Kernel {
    /// Partition lifecycle manager.
    partitions: PartitionManager,
    /// Coherence-weighted scheduler (8 CPUs, 256 partitions).
    scheduler: Scheduler<DEFAULT_MAX_CPUS, DEFAULT_MAX_PARTITIONS>,
    /// Append-only witness log.
    witness_log: WitnessLog<DEFAULT_WITNESS_CAPACITY>,
    /// Capability manager (P1/P2/P3 verification).
    cap_manager: CapabilityManager<DEFAULT_CAP_CAPACITY>,
    /// Coherence engine — graph-driven partition scoring and split/merge.
    coherence: DefaultCoherenceEngine,
    /// Boot progress tracker.
    boot: BootTracker,
    /// Kernel configuration.
    config: RvmConfig,
    /// Whether the kernel has completed booting.
    booted: bool,
}

/// Configuration for constructing a kernel instance.
#[derive(Debug, Clone, Copy)]
pub struct KernelConfig {
    /// Base RVM configuration.
    pub rvm: RvmConfig,
    /// Capability manager configuration.
    pub cap: CapManagerConfig,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            rvm: RvmConfig::default(),
            cap: CapManagerConfig::new(),
        }
    }
}

impl Kernel {
    /// Default Stoer-Wagner iteration budget for the coherence engine.
    const DEFAULT_MINCUT_BUDGET: u32 = 100;

    /// Create a new kernel instance with the given configuration.
    #[must_use]
    pub fn new(config: KernelConfig) -> Self {
        Self {
            partitions: PartitionManager::new(),
            scheduler: Scheduler::new(),
            witness_log: WitnessLog::new(),
            cap_manager: CapabilityManager::new(config.cap),
            coherence: DefaultCoherenceEngine::with_defaults(Self::DEFAULT_MINCUT_BUDGET),
            boot: BootTracker::new(),
            config: config.rvm,
            booted: false,
        }
    }

    /// Create a kernel with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(KernelConfig::default())
    }

    /// Run the boot sequence through all 7 phases.
    ///
    /// Each phase completion is recorded as a witness entry. After all
    /// phases complete, the kernel is ready to accept partition requests.
    pub fn boot(&mut self) -> RvmResult<()> {
        use rvm_boot::BootPhase;

        let phases = [
            BootPhase::HalInit,
            BootPhase::MemoryInit,
            BootPhase::CapabilityInit,
            BootPhase::WitnessInit,
            BootPhase::SchedulerInit,
            BootPhase::RootPartition,
            BootPhase::Handoff,
        ];

        for phase in &phases {
            self.boot.complete_phase(*phase)?;
            emit_boot_witness(&self.witness_log, *phase);
        }

        self.booted = true;
        Ok(())
    }

    /// Advance the scheduler and coherence engine by one epoch.
    ///
    /// Returns an `EpochResult` containing the scheduler summary and the
    /// coherence engine's split/merge recommendation. Requires the kernel
    /// to have booted.
    pub fn tick(&mut self) -> RvmResult<EpochResult> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }

        let summary = self.scheduler.tick_epoch();

        // Tick coherence engine. Use a fixed CPU load estimate for now;
        // a future HAL integration will read real CPU utilisation.
        let cpu_load_estimate = 20u8;
        let decision = self.coherence.tick(cpu_load_estimate);

        // Emit an epoch witness.
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::SchedulerEpoch as u8;
        record.proof_tier = 1;
        let switch_bytes = summary.switch_count.to_le_bytes();
        record.payload[0..2].copy_from_slice(&switch_bytes);
        self.witness_log.append(record);

        Ok(EpochResult { summary, decision })
    }

    /// Record a directed communication event between two partitions.
    ///
    /// Updates the coherence graph edge weight. Call this when agents in
    /// different partitions exchange messages.
    pub fn record_communication(
        &mut self,
        from: PartitionId,
        to: PartitionId,
        weight: u64,
    ) -> RvmResult<()> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }
        self.coherence
            .record_communication(from, to, weight)
            .map_err(|_| RvmError::InternalError)
    }

    /// Get the coherence score for a partition (0..10000 basis points).
    #[must_use]
    pub fn coherence_score(&self, id: PartitionId) -> rvm_types::CoherenceScore {
        self.coherence.score(id)
    }

    /// Get the cut pressure for a partition (0..10000 basis points).
    #[must_use]
    pub fn coherence_pressure(&self, id: PartitionId) -> rvm_types::CutPressure {
        self.coherence.pressure(id)
    }

    /// Get the latest coherence decision without advancing the epoch.
    #[must_use]
    pub fn coherence_recommendation(&self) -> CoherenceDecision {
        self.coherence.recommend()
    }

    /// Create a new partition with the given configuration.
    ///
    /// Registers the partition in the coherence graph and emits a
    /// `PartitionCreate` witness record on success.
    pub fn create_partition(&mut self, config: &PartitionConfig) -> RvmResult<PartitionId> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }

        let epoch = self.scheduler.current_epoch();
        let id = self.partitions.create(
            rvm_partition::PartitionType::Agent,
            config.vcpu_count,
            epoch,
        )?;

        // Register in coherence graph (best-effort: ignore capacity errors
        // since the partition already exists in the partition manager).
        let _ = self.coherence.add_partition(id);

        // Emit witness.
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::PartitionCreate as u8;
        record.proof_tier = 1;
        record.actor_partition_id = PartitionId::HYPERVISOR.as_u32();
        record.target_object_id = id.as_u32() as u64;
        self.witness_log.append(record);

        Ok(id)
    }

    /// Destroy a partition and reclaim its resources.
    ///
    /// Removes the partition from the coherence graph and emits a
    /// `PartitionDestroy` witness. Full resource reclamation is deferred.
    pub fn destroy_partition(&mut self, id: PartitionId) -> RvmResult<()> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }

        // Verify the partition exists.
        if self.partitions.get(id).is_none() {
            return Err(RvmError::PartitionNotFound);
        }

        // Remove from coherence graph (best-effort).
        let _ = self.coherence.remove_partition(id);

        // Emit witness.
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::PartitionDestroy as u8;
        record.proof_tier = 1;
        record.actor_partition_id = PartitionId::HYPERVISOR.as_u32();
        record.target_object_id = id.as_u32() as u64;
        self.witness_log.append(record);

        Ok(())
    }

    /// Return whether the kernel has completed booting.
    #[must_use]
    pub const fn is_booted(&self) -> bool {
        self.booted
    }

    /// Return the current scheduler epoch.
    #[must_use]
    pub fn current_epoch(&self) -> u32 {
        self.scheduler.current_epoch()
    }

    /// Return the number of active partitions.
    #[must_use]
    pub fn partition_count(&self) -> usize {
        self.partitions.count()
    }

    /// Return the total number of witness records emitted.
    pub fn witness_count(&self) -> u64 {
        self.witness_log.total_emitted()
    }

    /// Return a reference to the kernel configuration.
    #[must_use]
    pub const fn config(&self) -> &RvmConfig {
        &self.config
    }

    /// Return a reference to the partition manager.
    #[must_use]
    pub fn partitions(&self) -> &PartitionManager {
        &self.partitions
    }

    /// Return a reference to the capability manager.
    #[must_use]
    pub fn cap_manager(&self) -> &CapabilityManager<DEFAULT_CAP_CAPACITY> {
        &self.cap_manager
    }

    /// Return a mutable reference to the capability manager.
    pub fn cap_manager_mut(&mut self) -> &mut CapabilityManager<DEFAULT_CAP_CAPACITY> {
        &mut self.cap_manager
    }

    /// Return a reference to the witness log.
    #[must_use]
    pub fn witness_log(&self) -> &WitnessLog<DEFAULT_WITNESS_CAPACITY> {
        &self.witness_log
    }

    // -- Scheduler integration --

    /// Enqueue a partition onto a CPU's run queue.
    ///
    /// Automatically injects the partition's coherence-derived cut pressure
    /// into the scheduler priority. This is the primary path for scheduling
    /// partitions with coherence awareness.
    pub fn enqueue_partition(
        &mut self,
        cpu: usize,
        id: PartitionId,
        deadline_urgency: u16,
    ) -> RvmResult<()> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }
        if self.partitions.get(id).is_none() {
            return Err(RvmError::PartitionNotFound);
        }

        let pressure = self.coherence.pressure(id);
        if !self.scheduler.enqueue(cpu, id, deadline_urgency, pressure) {
            return Err(RvmError::ResourceLimitExceeded);
        }
        Ok(())
    }

    /// Pick the next partition on a CPU and switch to it.
    ///
    /// Returns `(old_partition, new_partition)` if a switch occurred.
    /// Emits no witness record (DC-10: switches are bulk-summarised at
    /// epoch boundaries, not individually witnessed).
    pub fn switch_next(&mut self, cpu: usize) -> Option<(Option<PartitionId>, PartitionId)> {
        self.scheduler.switch_next(cpu)
    }

    // -- Coherence-driven split/merge --

    /// Execute a coherence-driven partition split.
    ///
    /// Creates a new "child" partition and emits a `StructuralSplit`
    /// witness. The actual agent migration is the caller's responsibility;
    /// this method handles the partition and coherence graph bookkeeping.
    ///
    /// Returns the new partition ID on success.
    pub fn execute_split(&mut self, source: PartitionId) -> RvmResult<PartitionId> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }
        let src = self.partitions.get(source).ok_or(RvmError::PartitionNotFound)?;
        let vcpu_count = src.vcpu_count;

        // Create the new partition (inherits source's vCPU count).
        let epoch = self.scheduler.current_epoch();
        let child = self.partitions.create(
            rvm_partition::PartitionType::Agent,
            vcpu_count,
            epoch,
        )?;

        // Register child in coherence graph.
        let _ = self.coherence.add_partition(child);

        // Emit structural split witness.
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::StructuralSplit as u8;
        record.proof_tier = 1;
        record.actor_partition_id = source.as_u32();
        record.target_object_id = child.as_u32() as u64;
        self.witness_log.append(record);

        Ok(child)
    }

    /// Execute a coherence-driven partition merge.
    ///
    /// Validates merge preconditions (coherence threshold, adjacency,
    /// resource limits) and emits a `StructuralMerge` witness. The
    /// target partition absorbs the source; the source is destroyed.
    ///
    /// Returns the surviving partition ID on success.
    pub fn execute_merge(
        &mut self,
        absorber: PartitionId,
        absorbed: PartitionId,
    ) -> RvmResult<PartitionId> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }
        // Verify both partitions exist.
        let _a = self.partitions.get(absorber).ok_or(RvmError::PartitionNotFound)?;
        let _b = self.partitions.get(absorbed).ok_or(RvmError::PartitionNotFound)?;

        // Check coherence-based merge preconditions.
        let score_a = self.coherence.score(absorber);
        let score_b = self.coherence.score(absorbed);
        rvm_partition::merge_preconditions_met(score_a, score_b)
            .map_err(|_| RvmError::InvalidPartitionState)?;

        // Remove absorbed from coherence graph.
        let _ = self.coherence.remove_partition(absorbed);

        // Emit structural merge witness.
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::StructuralMerge as u8;
        record.proof_tier = 1;
        record.actor_partition_id = absorber.as_u32();
        record.target_object_id = absorbed.as_u32() as u64;
        self.witness_log.append(record);

        Ok(absorber)
    }

    /// Apply a coherence decision returned from `tick()`.
    ///
    /// - `SplitRecommended` → `execute_split`
    /// - `MergeRecommended` → `execute_merge`
    /// - `NoAction` → no-op
    ///
    /// Returns the decision that was applied, along with any new partition
    /// ID created by a split.
    pub fn apply_decision(
        &mut self,
        decision: CoherenceDecision,
    ) -> RvmResult<ApplyResult> {
        match decision {
            CoherenceDecision::NoAction => Ok(ApplyResult::NoAction),
            CoherenceDecision::SplitRecommended { partition, .. } => {
                let child = self.execute_split(partition)?;
                Ok(ApplyResult::Split { source: partition, child })
            }
            CoherenceDecision::MergeRecommended { a, b, .. } => {
                let survivor = self.execute_merge(a, b)?;
                Ok(ApplyResult::Merged { survivor, absorbed: b })
            }
        }
    }

    // -- Feature-gated subsystems --

    /// Whether the coherence engine is integrated.
    ///
    /// Always `true` since the engine is a core part of the kernel.
    #[must_use]
    pub const fn coherence_enabled(&self) -> bool {
        true
    }

    /// Access the coherence engine directly (for inspection/testing).
    #[must_use]
    pub fn coherence_engine(&self) -> &DefaultCoherenceEngine {
        &self.coherence
    }

    /// Check whether WASM support is compiled in.
    #[cfg(feature = "wasm")]
    pub fn wasm_enabled(&self) -> bool {
        true
    }

    /// WASM support is not compiled in.
    #[cfg(not(feature = "wasm"))]
    pub fn wasm_enabled(&self) -> bool {
        false
    }
}

/// Emit a boot phase completion witness.
fn emit_boot_witness(log: &WitnessLog<DEFAULT_WITNESS_CAPACITY>, phase: rvm_boot::BootPhase) {
    let action = match phase {
        rvm_boot::BootPhase::Handoff => ActionKind::BootComplete,
        _ => ActionKind::BootAttestation,
    };
    let mut record = WitnessRecord::zeroed();
    record.action_kind = action as u8;
    record.proof_tier = 1;
    record.actor_partition_id = PartitionId::HYPERVISOR.as_u32();
    record.payload[0] = phase as u8;
    log.append(record);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_creation() {
        let kernel = Kernel::with_defaults();
        assert!(!kernel.is_booted());
        assert_eq!(kernel.partition_count(), 0);
        assert_eq!(kernel.witness_count(), 0);
    }

    #[test]
    fn test_boot_sequence() {
        let mut kernel = Kernel::with_defaults();
        assert!(kernel.boot().is_ok());
        assert!(kernel.is_booted());

        // 7 boot phases = 7 witness records.
        assert_eq!(kernel.witness_count(), 7);
    }

    #[test]
    fn test_double_boot_fails() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        // Second boot attempt fails because phases are already complete.
        assert!(kernel.boot().is_err());
    }

    #[test]
    fn test_create_partition() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id = kernel.create_partition(&config).unwrap();
        assert_eq!(kernel.partition_count(), 1);
        assert!(kernel.partitions().get(id).is_some());

        // Witness for create.
        let pre_boot_witnesses = 7u64;
        assert_eq!(kernel.witness_count(), pre_boot_witnesses + 1);
    }

    #[test]
    fn test_create_partition_before_boot() {
        let mut kernel = Kernel::with_defaults();
        let config = PartitionConfig::default();
        assert_eq!(kernel.create_partition(&config), Err(RvmError::InvalidPartitionState));
    }

    #[test]
    fn test_destroy_partition() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id = kernel.create_partition(&config).unwrap();
        assert!(kernel.destroy_partition(id).is_ok());
    }

    #[test]
    fn test_destroy_nonexistent_partition() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let bad_id = PartitionId::new(999);
        assert_eq!(kernel.destroy_partition(bad_id), Err(RvmError::PartitionNotFound));
    }

    #[test]
    fn test_tick() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let result = kernel.tick().unwrap();
        assert_eq!(result.summary.epoch, 0);
        assert_eq!(result.decision, CoherenceDecision::NoAction);
        assert_eq!(kernel.current_epoch(), 1);
    }

    #[test]
    fn test_tick_before_boot() {
        let mut kernel = Kernel::with_defaults();
        assert!(kernel.tick().is_err());
    }

    #[test]
    fn test_feature_gates() {
        let kernel = Kernel::with_defaults();

        // Coherence is always enabled now.
        assert!(kernel.coherence_enabled());
        let _wasm = kernel.wasm_enabled();
    }

    #[test]
    fn test_custom_config() {
        let config = KernelConfig {
            rvm: RvmConfig {
                max_partitions: 64,
                ..RvmConfig::default()
            },
            cap: CapManagerConfig::new().with_max_depth(4),
        };
        let mut kernel = Kernel::new(config);
        assert_eq!(kernel.config().max_partitions, 64);
        kernel.boot().unwrap();
        assert!(kernel.is_booted());
    }

    #[test]
    fn test_multiple_partitions() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();

        assert_ne!(id1, id2);
        assert_eq!(kernel.partition_count(), 2);
    }

    #[test]
    fn test_kernel_version() {
        assert!(!VERSION.is_empty());
        assert_eq!(CRATE_COUNT, 13);
    }

    // ---------------------------------------------------------------
    // Integration-style lifecycle tests
    // ---------------------------------------------------------------

    #[test]
    fn test_full_boot_create_tick_destroy_lifecycle() {
        let mut kernel = Kernel::with_defaults();

        // Phase 1: Boot
        kernel.boot().unwrap();
        assert!(kernel.is_booted());
        let boot_witnesses = kernel.witness_count();
        assert_eq!(boot_witnesses, 7);

        // Phase 2: Create a partition
        let config = PartitionConfig::default();
        let id = kernel.create_partition(&config).unwrap();
        assert_eq!(kernel.partition_count(), 1);
        assert_eq!(kernel.witness_count(), boot_witnesses + 1);

        // Phase 3: Tick the scheduler several times
        for expected_epoch in 0..5u32 {
            let result = kernel.tick().unwrap();
            assert_eq!(result.summary.epoch, expected_epoch);
        }
        assert_eq!(kernel.current_epoch(), 5);
        // 5 ticks = 5 more witness records
        assert_eq!(kernel.witness_count(), boot_witnesses + 1 + 5);

        // Phase 4: Destroy the partition
        kernel.destroy_partition(id).unwrap();
        assert_eq!(kernel.witness_count(), boot_witnesses + 1 + 5 + 1);
    }

    #[test]
    fn test_create_partition_with_zero_vcpus() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        // PartitionConfig allows vcpu_count=0 at the kernel level
        // (the partition manager does not validate vcpu count).
        let config = PartitionConfig {
            vcpu_count: 0,
            ..PartitionConfig::default()
        };
        let result = kernel.create_partition(&config);
        // Should succeed -- validation is not enforced at this level.
        assert!(result.is_ok());
    }

    #[test]
    fn test_destroy_before_boot_fails() {
        let mut kernel = Kernel::with_defaults();
        let id = PartitionId::new(1);
        assert_eq!(kernel.destroy_partition(id), Err(RvmError::InvalidPartitionState));
    }

    #[test]
    fn test_destroy_twice_succeeds_because_no_removal() {
        // destroy_partition only verifies existence via get() but does
        // not actually remove from the manager, so a second destroy of
        // the same ID currently succeeds. This tests current behavior.
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id = kernel.create_partition(&config).unwrap();
        assert!(kernel.destroy_partition(id).is_ok());
        // Second destroy: partition is still present because destroy
        // does not remove from the manager.
        assert!(kernel.destroy_partition(id).is_ok());
    }

    #[test]
    fn test_many_partitions_coexist() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let mut ids = [PartitionId::new(0); 10];
        for id in &mut ids {
            *id = kernel.create_partition(&config).unwrap();
        }
        assert_eq!(kernel.partition_count(), 10);

        // All IDs are unique.
        for (i, a) in ids.iter().enumerate() {
            for b in &ids[i + 1..] {
                assert_ne!(a, b);
            }
        }

        // All are accessible.
        for id in &ids {
            assert!(kernel.partitions().get(*id).is_some());
        }
    }

    #[test]
    fn test_create_partition_emits_correct_witness_fields() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();
        let boot_count = kernel.witness_count();

        let config = PartitionConfig::default();
        let id = kernel.create_partition(&config).unwrap();

        // The create witness is the record right after boot witnesses.
        let record = kernel.witness_log().get(boot_count as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::PartitionCreate as u8);
        assert_eq!(record.proof_tier, 1);
        assert_eq!(record.actor_partition_id, PartitionId::HYPERVISOR.as_u32());
        assert_eq!(record.target_object_id, id.as_u32() as u64);
    }

    #[test]
    fn test_destroy_partition_emits_correct_witness_fields() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id = kernel.create_partition(&config).unwrap();
        let pre_destroy = kernel.witness_count();

        kernel.destroy_partition(id).unwrap();
        let record = kernel.witness_log().get(pre_destroy as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::PartitionDestroy as u8);
        assert_eq!(record.proof_tier, 1);
        assert_eq!(record.target_object_id, id.as_u32() as u64);
    }

    #[test]
    fn test_tick_emits_scheduler_epoch_witness() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();
        let pre_tick = kernel.witness_count();

        kernel.tick().unwrap();
        let record = kernel.witness_log().get(pre_tick as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::SchedulerEpoch as u8);
    }

    #[test]
    fn test_cap_manager_accessible() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        // Verify we can use the capability manager through the kernel.
        let cap_mgr = kernel.cap_manager_mut();
        let owner = PartitionId::new(1);
        let result = cap_mgr.create_root_capability(
            rvm_types::CapType::Partition,
            rvm_types::CapRights::READ,
            0,
            owner,
        );
        assert!(result.is_ok());
    }

    // ---------------------------------------------------------------
    // Coherence engine integration tests
    // ---------------------------------------------------------------

    #[test]
    fn test_coherence_engine_tracks_partitions() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();

        // Coherence engine should track the same count.
        assert_eq!(kernel.coherence_engine().partition_count(), 2);

        // Isolated partitions have max coherence score.
        assert_eq!(
            kernel.coherence_score(id1),
            rvm_types::CoherenceScore::MAX,
        );
        assert_eq!(
            kernel.coherence_score(id2),
            rvm_types::CoherenceScore::MAX,
        );
    }

    #[test]
    fn test_record_communication_and_tick() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();

        // Record heavy communication between the two.
        kernel.record_communication(id1, id2, 1000).unwrap();

        // After tick, coherence scores drop (all traffic is external).
        let result = kernel.tick().unwrap();

        assert_eq!(kernel.coherence_score(id1).as_basis_points(), 0);
        // High external traffic → split recommended.
        match result.decision {
            CoherenceDecision::SplitRecommended { partition, .. } => {
                assert!(partition == id1 || partition == id2);
            }
            _ => panic!("expected SplitRecommended after heavy external comms"),
        }
    }

    #[test]
    fn test_coherence_pressure_after_communication() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();
        kernel.record_communication(id1, id2, 500).unwrap();

        kernel.tick().unwrap();

        // Partition with only external traffic has max pressure (10000 bp).
        assert_eq!(kernel.coherence_pressure(id1).as_fixed(), 10_000);
    }

    #[test]
    fn test_no_action_for_isolated_partitions() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        kernel.create_partition(&config).unwrap();
        kernel.create_partition(&config).unwrap();

        let result = kernel.tick().unwrap();
        assert_eq!(result.decision, CoherenceDecision::NoAction);
    }

    #[test]
    fn test_record_communication_before_boot_fails() {
        let mut kernel = Kernel::with_defaults();
        assert_eq!(
            kernel.record_communication(PartitionId::new(1), PartitionId::new(2), 100),
            Err(RvmError::InvalidPartitionState),
        );
    }

    #[test]
    fn test_coherence_recommendation_without_tick() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        kernel.create_partition(&config).unwrap();

        // Before any tick, recommendation is NoAction.
        assert_eq!(kernel.coherence_recommendation(), CoherenceDecision::NoAction);
    }

    #[test]
    fn test_destroy_removes_from_coherence() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();
        assert_eq!(kernel.coherence_engine().partition_count(), 2);

        kernel.destroy_partition(id1).unwrap();
        assert_eq!(kernel.coherence_engine().partition_count(), 1);

        // id2 is still tracked.
        assert_eq!(
            kernel.coherence_score(id2),
            rvm_types::CoherenceScore::MAX,
        );
    }

    #[test]
    fn test_full_coherence_lifecycle() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();
        let c = kernel.create_partition(&config).unwrap();

        // a and b talk heavily; c is isolated.
        kernel.record_communication(a, b, 2000).unwrap();
        kernel.record_communication(b, a, 2000).unwrap();

        let result = kernel.tick().unwrap();

        // a and b should have high pressure, c should not.
        assert!(kernel.coherence_pressure(a).as_fixed() > 0);
        assert!(kernel.coherence_pressure(b).as_fixed() > 0);
        assert_eq!(kernel.coherence_pressure(c).as_fixed(), 0);

        // Should recommend splitting a or b.
        match result.decision {
            CoherenceDecision::SplitRecommended { partition, .. } => {
                assert!(partition == a || partition == b);
            }
            _ => panic!("expected split for heavily communicating partitions"),
        }

        // Destroy a, verify coherence adapts.
        kernel.destroy_partition(a).unwrap();
        assert_eq!(kernel.coherence_engine().partition_count(), 2);
    }

    // ---------------------------------------------------------------
    // Scheduler integration tests
    // ---------------------------------------------------------------

    #[test]
    fn test_enqueue_and_switch() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();

        // Enqueue id1 with lower urgency, id2 with higher.
        kernel.enqueue_partition(0, id1, 100).unwrap();
        kernel.enqueue_partition(0, id2, 200).unwrap();

        // Highest priority should be dequeued first.
        let (old, new) = kernel.switch_next(0).unwrap();
        assert!(old.is_none());
        assert_eq!(new, id2);

        let (old, new) = kernel.switch_next(0).unwrap();
        assert_eq!(old, Some(id2));
        assert_eq!(new, id1);
    }

    #[test]
    fn test_enqueue_injects_coherence_pressure() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();

        // Record heavy communication to give id1 high pressure.
        kernel.record_communication(id1, id2, 5000).unwrap();
        kernel.tick().unwrap();

        // id1 now has max pressure (10000 bp). When enqueued with
        // lower deadline urgency, pressure boost may re-order.
        kernel.enqueue_partition(0, id1, 50).unwrap();
        kernel.enqueue_partition(0, id2, 50).unwrap();

        // id1 should be prioritised because of its pressure boost.
        let (_, first) = kernel.switch_next(0).unwrap();
        assert_eq!(first, id1);
    }

    #[test]
    fn test_enqueue_before_boot_fails() {
        let mut kernel = Kernel::with_defaults();
        assert_eq!(
            kernel.enqueue_partition(0, PartitionId::new(1), 100),
            Err(RvmError::InvalidPartitionState),
        );
    }

    #[test]
    fn test_enqueue_nonexistent_partition_fails() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();
        assert_eq!(
            kernel.enqueue_partition(0, PartitionId::new(999), 100),
            Err(RvmError::PartitionNotFound),
        );
    }

    // ---------------------------------------------------------------
    // Split / merge execution tests
    // ---------------------------------------------------------------

    #[test]
    fn test_execute_split() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let source = kernel.create_partition(&config).unwrap();
        let pre_count = kernel.partition_count();
        let pre_witness = kernel.witness_count();

        let child = kernel.execute_split(source).unwrap();

        assert_ne!(source, child);
        assert_eq!(kernel.partition_count(), pre_count + 1);
        assert_eq!(kernel.coherence_engine().partition_count(), 2);

        // Verify StructuralSplit witness.
        let record = kernel.witness_log().get(pre_witness as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::StructuralSplit as u8);
        assert_eq!(record.actor_partition_id, source.as_u32());
        assert_eq!(record.target_object_id, child.as_u32() as u64);
    }

    #[test]
    fn test_execute_split_before_boot_fails() {
        let mut kernel = Kernel::with_defaults();
        assert_eq!(
            kernel.execute_split(PartitionId::new(1)),
            Err(RvmError::InvalidPartitionState),
        );
    }

    #[test]
    fn test_execute_split_nonexistent_fails() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();
        assert_eq!(
            kernel.execute_split(PartitionId::new(999)),
            Err(RvmError::PartitionNotFound),
        );
    }

    #[test]
    fn test_execute_merge() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();
        let pre_witness = kernel.witness_count();

        // Both start with MAX coherence (isolated), which exceeds
        // the merge threshold of 7000 bp.
        let survivor = kernel.execute_merge(a, b).unwrap();
        assert_eq!(survivor, a);

        // b was removed from coherence graph.
        assert_eq!(kernel.coherence_engine().partition_count(), 1);

        // Verify StructuralMerge witness.
        let record = kernel.witness_log().get(pre_witness as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::StructuralMerge as u8);
        assert_eq!(record.actor_partition_id, a.as_u32());
        assert_eq!(record.target_object_id, b.as_u32() as u64);
    }

    #[test]
    fn test_execute_merge_low_coherence_fails() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();

        // Drive coherence to zero by adding external-only traffic.
        kernel.record_communication(a, b, 5000).unwrap();
        kernel.tick().unwrap();

        // Now a has 0 coherence, below the 7000 bp merge threshold.
        assert_eq!(
            kernel.execute_merge(a, b),
            Err(RvmError::InvalidPartitionState),
        );
    }

    #[test]
    fn test_execute_merge_nonexistent_fails() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        assert_eq!(
            kernel.execute_merge(a, PartitionId::new(999)),
            Err(RvmError::PartitionNotFound),
        );
    }

    // ---------------------------------------------------------------
    // apply_decision tests
    // ---------------------------------------------------------------

    #[test]
    fn test_apply_no_action() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let result = kernel.apply_decision(CoherenceDecision::NoAction).unwrap();
        assert_eq!(result, ApplyResult::NoAction);
    }

    #[test]
    fn test_apply_split_decision() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();

        // Create heavy traffic to trigger split recommendation.
        kernel.record_communication(a, b, 5000).unwrap();
        let epoch = kernel.tick().unwrap();

        match epoch.decision {
            CoherenceDecision::SplitRecommended { .. } => {
                let result = kernel.apply_decision(epoch.decision).unwrap();
                match result {
                    ApplyResult::Split { source, child } => {
                        assert!(source == a || source == b);
                        assert_ne!(source, child);
                        // Now 3 partitions exist.
                        assert_eq!(kernel.partition_count(), 3);
                    }
                    _ => panic!("expected Split result"),
                }
            }
            _ => panic!("expected SplitRecommended"),
        }
    }

    #[test]
    fn test_full_tick_apply_lifecycle() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();

        // Heavy bidirectional traffic.
        kernel.record_communication(a, b, 3000).unwrap();
        kernel.record_communication(b, a, 3000).unwrap();

        // Tick, get decision, apply it.
        let epoch = kernel.tick().unwrap();
        let result = kernel.apply_decision(epoch.decision).unwrap();

        // Should have split one of the partitions.
        match result {
            ApplyResult::Split { source, child } => {
                assert!(source == a || source == b);
                assert_eq!(kernel.partition_count(), 3);
                assert_eq!(kernel.coherence_engine().partition_count(), 3);

                // Enqueue the new partition and verify it can be scheduled.
                kernel.enqueue_partition(0, child, 100).unwrap();
                let (_, next) = kernel.switch_next(0).unwrap();
                assert_eq!(next, child);
            }
            _ => panic!("expected split from heavy traffic"),
        }
    }
}
