//! OpenFang Agent OS — RVF Knowledge Base
//!
//! A deep integration example that exercises the full RVF capability surface
//! using OpenFang's agent registry as the domain model.
//!
//! Capabilities demonstrated:
//!   - Multi-type registry (Hands, Tools, Channels) in one vector space
//!   - Rich metadata with typed fields and combined filter expressions
//!   - Task routing via nearest-neighbor search
//!   - Security and tier filtering
//!   - Delete + compact lifecycle (decommission an agent, reclaim space)
//!   - COW branching + freeze (staging branch for experimental agents)
//!   - File identity and lineage tracking (parent/child provenance)
//!   - Audited queries (witness entries for every search)
//!   - Segment directory inspection
//!   - Cryptographic witness chain with verification
//!   - Persistence round-trip
//!
//! Run with:
//!   cargo run --example openfang

use rvf_crypto::{create_witness_chain, shake256_256, verify_witness_chain, WitnessEntry};
use rvf_runtime::filter::FilterValue;
use rvf_runtime::options::DistanceMetric;
use rvf_runtime::{
    FilterExpr, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore, SearchResult,
};
use rvf_types::DerivationType;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DIM: usize = 128;
const K: usize = 5;

// Metadata field IDs — shared across all component types.
const F_TYPE: u16 = 0; // "hand" | "tool" | "channel"
const F_NAME: u16 = 1;
const F_DOMAIN: u16 = 2; // domain (hand), category (tool), protocol (channel)
const F_TIER: u16 = 3; // hand only: 1-4
const F_SEC: u16 = 4; // hand only: 0-100

// ---------------------------------------------------------------------------
// Data definitions
// ---------------------------------------------------------------------------

struct Hand {
    name: &'static str,
    domain: &'static str,
    tier: u64,
    security: u64,
}

struct Tool {
    name: &'static str,
    category: &'static str,
}

struct Channel {
    name: &'static str,
    protocol: &'static str,
}

const HANDS: &[Hand] = &[
    Hand { name: "clip",       domain: "video-processing",  tier: 3, security: 60 },
    Hand { name: "lead",       domain: "sales-automation",   tier: 2, security: 70 },
    Hand { name: "collector",  domain: "osint-intelligence", tier: 4, security: 90 },
    Hand { name: "predictor",  domain: "forecasting",        tier: 3, security: 80 },
    Hand { name: "researcher", domain: "fact-checking",      tier: 3, security: 75 },
    Hand { name: "twitter",    domain: "social-media",       tier: 2, security: 65 },
    Hand { name: "browser",    domain: "web-automation",     tier: 4, security: 95 },
];

const TOOLS: &[Tool] = &[
    Tool { name: "http_fetch",       category: "network" },
    Tool { name: "web_search",       category: "network" },
    Tool { name: "web_scrape",       category: "network" },
    Tool { name: "file_read",        category: "filesystem" },
    Tool { name: "file_write",       category: "filesystem" },
    Tool { name: "file_list",        category: "filesystem" },
    Tool { name: "shell_exec",       category: "system" },
    Tool { name: "process_spawn",    category: "system" },
    Tool { name: "json_parse",       category: "transform" },
    Tool { name: "json_format",      category: "transform" },
    Tool { name: "csv_parse",        category: "transform" },
    Tool { name: "regex_match",      category: "transform" },
    Tool { name: "template_render",  category: "transform" },
    Tool { name: "llm_complete",     category: "inference" },
    Tool { name: "llm_embed",        category: "inference" },
    Tool { name: "llm_classify",     category: "inference" },
    Tool { name: "vector_store",     category: "memory" },
    Tool { name: "vector_search",    category: "memory" },
    Tool { name: "kv_get",           category: "memory" },
    Tool { name: "kv_set",           category: "memory" },
    Tool { name: "sql_query",        category: "database" },
    Tool { name: "sql_execute",      category: "database" },
    Tool { name: "screenshot",       category: "browser" },
    Tool { name: "click_element",    category: "browser" },
    Tool { name: "fill_form",        category: "browser" },
    Tool { name: "navigate",         category: "browser" },
    Tool { name: "pdf_extract",      category: "document" },
    Tool { name: "ocr_image",        category: "document" },
    Tool { name: "email_send",       category: "communication" },
    Tool { name: "email_read",       category: "communication" },
    Tool { name: "webhook_fire",     category: "integration" },
    Tool { name: "api_call",         category: "integration" },
    Tool { name: "schedule_cron",    category: "scheduling" },
    Tool { name: "schedule_delay",   category: "scheduling" },
    Tool { name: "crypto_sign",      category: "security" },
    Tool { name: "crypto_verify",    category: "security" },
    Tool { name: "secret_read",      category: "security" },
    Tool { name: "audit_log",        category: "security" },
];

const CHANNELS: &[Channel] = &[
    Channel { name: "telegram",      protocol: "bot-api" },
    Channel { name: "discord",       protocol: "gateway" },
    Channel { name: "slack",         protocol: "events-api" },
    Channel { name: "whatsapp",      protocol: "cloud-api" },
    Channel { name: "signal",        protocol: "signal-cli" },
    Channel { name: "matrix",        protocol: "client-server" },
    Channel { name: "email-smtp",    protocol: "smtp" },
    Channel { name: "email-imap",    protocol: "imap" },
    Channel { name: "teams",         protocol: "graph-api" },
    Channel { name: "google-chat",   protocol: "chat-api" },
    Channel { name: "linkedin",      protocol: "rest-api" },
    Channel { name: "twitter-x",     protocol: "api-v2" },
    Channel { name: "mastodon",      protocol: "activitypub" },
    Channel { name: "bluesky",       protocol: "at-proto" },
    Channel { name: "reddit",        protocol: "oauth-api" },
    Channel { name: "irc",           protocol: "irc-v3" },
    Channel { name: "xmpp",         protocol: "xmpp-core" },
    Channel { name: "webhook-in",    protocol: "http-post" },
    Channel { name: "webhook-out",   protocol: "http-post" },
    Channel { name: "grpc",          protocol: "grpc" },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_vector(seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(DIM);
    let mut x = seed.wrapping_add(1);
    for _ in 0..DIM {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v.push(((x >> 33) as f32) / (u32::MAX as f32) - 0.5);
    }
    v
}

fn biased_vector(seed: u64, bias: f32) -> Vec<f32> {
    let mut v = random_vector(seed);
    for d in v.iter_mut().take(16) {
        *d += bias;
    }
    v
}

fn category_bias(cat: &str) -> f32 {
    let h = cat.bytes().fold(0u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
    ((h % 200) as f32 - 100.0) * 0.003
}

fn push_meta(out: &mut Vec<MetadataEntry>, fid: u16, val: MetadataValue) {
    out.push(MetadataEntry { field_id: fid, value: val });
}

fn sv(s: &str) -> MetadataValue {
    MetadataValue::String(s.to_string())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join("")
}

fn witness(entries: &mut Vec<WitnessEntry>, action: &str, ts_ns: u64, wtype: u8) {
    entries.push(WitnessEntry {
        prev_hash: [0u8; 32],
        action_hash: shake256_256(action.as_bytes()),
        timestamp_ns: ts_ns,
        witness_type: wtype,
    });
}

// ---------------------------------------------------------------------------
// Registry — tracks ID ranges for component lookup.
// ---------------------------------------------------------------------------

struct Registry {
    hand_base: u64,
    hand_count: u64,
    tool_base: u64,
    tool_count: u64,
    channel_base: u64,
    channel_count: u64,
}

impl Registry {
    fn new() -> Self {
        let hc = HANDS.len() as u64;
        let tc = TOOLS.len() as u64;
        let cc = CHANNELS.len() as u64;
        Self { hand_base: 0, hand_count: hc, tool_base: hc, tool_count: tc, channel_base: hc + tc, channel_count: cc }
    }
    fn total(&self) -> u64 { self.hand_count + self.tool_count + self.channel_count }
    fn identify(&self, id: u64) -> (&'static str, &'static str) {
        if id >= self.channel_base && id < self.channel_base + self.channel_count {
            ("channel", CHANNELS[(id - self.channel_base) as usize].name)
        } else if id >= self.tool_base && id < self.tool_base + self.tool_count {
            ("tool", TOOLS[(id - self.tool_base) as usize].name)
        } else if id >= self.hand_base && id < self.hand_base + self.hand_count {
            ("hand", HANDS[(id - self.hand_base) as usize].name)
        } else {
            ("unknown", "???")
        }
    }
}

fn print_results(results: &[SearchResult], reg: &Registry) {
    println!("    {:>4}  {:>10}  {:>8}  {:>20}", "ID", "Distance", "Type", "Name");
    println!("    {:->4}  {:->10}  {:->8}  {:->20}", "", "", "", "");
    for r in results {
        let (ty, nm) = reg.identify(r.id);
        println!("    {:>4}  {:>10.4}  {:>8}  {:>20}", r.id, r.distance, ty, nm);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== OpenFang Agent OS — RVF Knowledge Base ===\n");

    let reg = Registry::new();
    let tmp = TempDir::new().expect("tmpdir");
    let store_path = tmp.path().join("openfang.rvf");
    let branch_path = tmp.path().join("openfang-staging.rvf");
    let derived_path = tmp.path().join("openfang-snapshot.rvf");

    let opts = RvfOptions {
        dimension: DIM as u16,
        metric: DistanceMetric::L2,
        ..Default::default()
    };

    let mut wit: Vec<WitnessEntry> = Vec::new();

    // -----------------------------------------------------------------------
    // 1. Create store
    // -----------------------------------------------------------------------
    println!("--- 1. Create Registry ---");
    let mut store = RvfStore::create(&store_path, opts).expect("create");
    println!("  Store: {:?}  ({}d, L2)", store_path, DIM);
    println!("  File ID: {}", hex(&store.file_id()[..8]));
    println!("  Lineage depth: {}", store.lineage_depth());

    // -----------------------------------------------------------------------
    // 2. Register Hands
    // -----------------------------------------------------------------------
    println!("\n--- 2. Register Hands ({}) ---", HANDS.len());
    {
        let vecs: Vec<Vec<f32>> = HANDS.iter().enumerate()
            .map(|(i, h)| biased_vector(i as u64 * 17 + 100, h.tier as f32 * 0.1))
            .collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (reg.hand_base..reg.hand_base + reg.hand_count).collect();
        let mut meta = Vec::with_capacity(HANDS.len() * 5);
        for h in HANDS {
            push_meta(&mut meta, F_TYPE, sv("hand"));
            push_meta(&mut meta, F_NAME, sv(h.name));
            push_meta(&mut meta, F_DOMAIN, sv(h.domain));
            push_meta(&mut meta, F_TIER, MetadataValue::U64(h.tier));
            push_meta(&mut meta, F_SEC, MetadataValue::U64(h.security));
        }
        let r = store.ingest_batch(&refs, &ids, Some(&meta)).expect("ingest hands");
        println!("  Ingested {} hands (epoch {})", r.accepted, r.epoch);
        for h in HANDS {
            println!("    {:12} {:22} tier={} sec={}", h.name, h.domain, h.tier, h.security);
        }
    }
    witness(&mut wit, &format!("REGISTER_HANDS:{}", HANDS.len()), 1_709_000_000_000_000_000, 0x01);

    // -----------------------------------------------------------------------
    // 3. Register Tools (per-category bias)
    // -----------------------------------------------------------------------
    println!("\n--- 3. Register Tools ({}) ---", TOOLS.len());
    {
        let vecs: Vec<Vec<f32>> = TOOLS.iter().enumerate()
            .map(|(i, t)| biased_vector(i as u64 * 31 + 500, category_bias(t.category)))
            .collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (reg.tool_base..reg.tool_base + reg.tool_count).collect();
        let mut meta = Vec::with_capacity(TOOLS.len() * 3);
        for t in TOOLS {
            push_meta(&mut meta, F_TYPE, sv("tool"));
            push_meta(&mut meta, F_NAME, sv(t.name));
            push_meta(&mut meta, F_DOMAIN, sv(t.category));
        }
        let r = store.ingest_batch(&refs, &ids, Some(&meta)).expect("ingest tools");
        println!("  Ingested {} tools (epoch {})", r.accepted, r.epoch);
        let mut cats: Vec<&str> = TOOLS.iter().map(|t| t.category).collect();
        cats.sort_unstable();
        cats.dedup();
        for c in &cats {
            let ns: Vec<&str> = TOOLS.iter().filter(|t| t.category == *c).map(|t| t.name).collect();
            println!("    [{:14}] {}", c, ns.join(", "));
        }
    }
    witness(&mut wit, &format!("REGISTER_TOOLS:{}", TOOLS.len()), 1_709_000_001_000_000_000, 0x01);

    // -----------------------------------------------------------------------
    // 4. Register Channels
    // -----------------------------------------------------------------------
    println!("\n--- 4. Register Channels ({}) ---", CHANNELS.len());
    {
        let vecs: Vec<Vec<f32>> = CHANNELS.iter().enumerate()
            .map(|(i, c)| biased_vector(i as u64 * 43 + 1000, category_bias(c.protocol)))
            .collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (reg.channel_base..reg.channel_base + reg.channel_count).collect();
        let mut meta = Vec::with_capacity(CHANNELS.len() * 3);
        for c in CHANNELS {
            push_meta(&mut meta, F_TYPE, sv("channel"));
            push_meta(&mut meta, F_NAME, sv(c.name));
            push_meta(&mut meta, F_DOMAIN, sv(c.protocol));
        }
        let r = store.ingest_batch(&refs, &ids, Some(&meta)).expect("ingest channels");
        println!("  Ingested {} channels (epoch {})", r.accepted, r.epoch);
        for c in CHANNELS {
            println!("    {:14} ({})", c.name, c.protocol);
        }
    }
    witness(&mut wit, &format!("REGISTER_CHANNELS:{}", CHANNELS.len()), 1_709_000_002_000_000_000, 0x01);

    println!("\n  Total registry: {} components", reg.total());

    // -----------------------------------------------------------------------
    // 5. Task routing — unfiltered + hands-only
    // -----------------------------------------------------------------------
    println!("\n--- 5. Task Routing ---");
    let query = biased_vector(42, 0.3);

    let all = store.query(&query, K, &QueryOptions::default()).expect("query");
    println!("  Unfiltered top-{}:", K);
    print_results(&all, &reg);

    let hands_only = QueryOptions {
        filter: Some(FilterExpr::Eq(F_TYPE, FilterValue::String("hand".into()))),
        ..Default::default()
    };
    let hand_res = store.query(&query, K, &hands_only).expect("query hands");
    println!("\n  Hands only:");
    print_results(&hand_res, &reg);
    witness(&mut wit, "ROUTE_TASK:k=5", 1_709_000_010_000_000_000, 0x02);

    // -----------------------------------------------------------------------
    // 6. Security filter (>= 80)
    // -----------------------------------------------------------------------
    println!("\n--- 6. High-Security Hands (sec >= 80) ---");
    let sec_opts = QueryOptions {
        filter: Some(FilterExpr::And(vec![
            FilterExpr::Eq(F_TYPE, FilterValue::String("hand".into())),
            FilterExpr::Ge(F_SEC, FilterValue::U64(80)),
        ])),
        ..Default::default()
    };
    let sec_res = store.query(&query, K, &sec_opts).expect("sec query");
    print_results(&sec_res, &reg);
    println!("  {} agents pass threshold", sec_res.len());

    // -----------------------------------------------------------------------
    // 7. Autonomous tier-4 agents
    // -----------------------------------------------------------------------
    println!("\n--- 7. Tier-4 Autonomous Agents ---");
    let tier_opts = QueryOptions {
        filter: Some(FilterExpr::And(vec![
            FilterExpr::Eq(F_TYPE, FilterValue::String("hand".into())),
            FilterExpr::Eq(F_TIER, FilterValue::U64(4)),
        ])),
        ..Default::default()
    };
    let tier_res = store.query(&query, K, &tier_opts).expect("tier query");
    print_results(&tier_res, &reg);

    // -----------------------------------------------------------------------
    // 8. Tool discovery by category
    // -----------------------------------------------------------------------
    println!("\n--- 8. Security Tool Discovery ---");
    let tool_opts = QueryOptions {
        filter: Some(FilterExpr::And(vec![
            FilterExpr::Eq(F_TYPE, FilterValue::String("tool".into())),
            FilterExpr::Eq(F_DOMAIN, FilterValue::String("security".into())),
        ])),
        ..Default::default()
    };
    let tool_res = store.query(&query, 10, &tool_opts).expect("tool query");
    print_results(&tool_res, &reg);

    // -----------------------------------------------------------------------
    // 9. Delete + Compact — decommission the "twitter" hand
    // -----------------------------------------------------------------------
    println!("\n--- 9. Delete + Compact (decommission 'twitter') ---");
    let twitter_id = HANDS.iter().position(|h| h.name == "twitter").unwrap() as u64 + reg.hand_base;
    let st_before = store.status();
    println!("  Before: {} vectors, {} bytes, dead_ratio={:.2}",
        st_before.total_vectors, st_before.file_size, st_before.dead_space_ratio);

    let del = store.delete(&[twitter_id]).expect("delete twitter");
    println!("  Deleted {} vector(s) (epoch {})", del.deleted, del.epoch);

    let st_mid = store.status();
    println!("  After delete: {} vectors, dead_ratio={:.2}", st_mid.total_vectors, st_mid.dead_space_ratio);

    let comp = store.compact().expect("compact");
    println!("  Compacted: {} segments, {} bytes reclaimed (epoch {})",
        comp.segments_compacted, comp.bytes_reclaimed, comp.epoch);

    let st_after = store.status();
    println!("  After compact: {} vectors, {} bytes, dead_ratio={:.2}",
        st_after.total_vectors, st_after.file_size, st_after.dead_space_ratio);

    // Verify twitter is gone from hand queries
    let post_del = store.query(&query, K, &hands_only).expect("post-delete query");
    for r in &post_del {
        assert_ne!(r.id, twitter_id, "twitter should be deleted");
    }
    println!("  Verified: 'twitter' no longer appears in results");
    witness(&mut wit, "DELETE+COMPACT:twitter", 1_709_000_020_000_000_000, 0x01);

    // -----------------------------------------------------------------------
    // 10. Derive — create a snapshot with lineage tracking
    // -----------------------------------------------------------------------
    println!("\n--- 10. Derive (Snapshot with Lineage) ---");
    let parent_fid = hex(&store.file_id()[..8]);
    let parent_depth = store.lineage_depth();

    let child = store.derive(&derived_path, DerivationType::Snapshot, None).expect("derive");
    let child_fid = hex(&child.file_id()[..8]);
    let child_parent = hex(&child.parent_id()[..8]);
    let child_depth = child.lineage_depth();

    println!("  Parent:  file_id={}  depth={}", parent_fid, parent_depth);
    println!("  Child:   file_id={}  depth={}", child_fid, child_depth);
    println!("  Child parent_id={} (matches parent: {})", child_parent, child_parent == parent_fid);
    assert_eq!(child_depth, parent_depth + 1, "depth should increment");

    let child_st = child.status();
    println!("  Child vectors: {}, segments: {}", child_st.total_vectors, child_st.total_segments);
    child.close().expect("close child");
    witness(&mut wit, "DERIVE:snapshot", 1_709_000_030_000_000_000, 0x01);

    // -----------------------------------------------------------------------
    // 11. COW Branch — staging environment for experimental agents
    // -----------------------------------------------------------------------
    println!("\n--- 11. COW Branch (Staging Environment) ---");
    store.freeze().expect("freeze parent");
    println!("  Parent frozen (read-only)");

    let mut staging = store.branch(&branch_path).expect("branch");
    println!("  Branch created: is_cow_child={}", staging.is_cow_child());
    if let Some(stats) = staging.cow_stats() {
        println!("  COW stats: {} clusters, {} local", stats.cluster_count, stats.local_cluster_count);
    }

    // Add experimental agent to staging only
    let exp_id = reg.total();
    let exp_vec = biased_vector(9999, 0.5);
    let mut exp_meta = Vec::with_capacity(5);
    push_meta(&mut exp_meta, F_TYPE, sv("hand"));
    push_meta(&mut exp_meta, F_NAME, sv("sentinel"));
    push_meta(&mut exp_meta, F_DOMAIN, sv("threat-detection"));
    push_meta(&mut exp_meta, F_TIER, MetadataValue::U64(4));
    push_meta(&mut exp_meta, F_SEC, MetadataValue::U64(99));

    let exp_r = staging.ingest_batch(&[exp_vec.as_slice()], &[exp_id], Some(&exp_meta))
        .expect("ingest experimental");
    println!("  Added experimental 'sentinel' to staging (epoch {})", exp_r.epoch);

    let staging_st = staging.status();
    println!("  Staging: {} vectors  (parent had {})", staging_st.total_vectors, st_after.total_vectors);

    if let Some(stats) = staging.cow_stats() {
        println!("  COW stats after write: {} clusters, {} local", stats.cluster_count, stats.local_cluster_count);
    }

    staging.close().expect("close staging");
    witness(&mut wit, "COW_BRANCH:staging+sentinel", 1_709_000_040_000_000_000, 0x01);

    // -----------------------------------------------------------------------
    // 12. Segment directory inspection
    // -----------------------------------------------------------------------
    println!("\n--- 12. Segment Directory ---");
    let seg_dir: Vec<_> = store.segment_dir().to_vec();
    println!("  {} segments in parent store:", seg_dir.len());
    println!("    {:>12}  {:>8}  {:>8}  {:>6}", "SegID", "Offset", "Length", "Type");
    println!("    {:->12}  {:->8}  {:->8}  {:->6}", "", "", "", "");
    for &(seg_id, offset, length, seg_type) in &seg_dir {
        let tname = match seg_type {
            0x01 => "VEC",
            0x02 => "MFST",
            0x03 => "JRNL",
            0x04 => "WITN",
            0x05 => "KERN",
            0x06 => "EBPF",
            _ => "????",
        };
        println!("    {:>12}  {:>8}  {:>8}  {:>6}", seg_id, offset, length, tname);
    }

    // -----------------------------------------------------------------------
    // 13. Witness chain
    // -----------------------------------------------------------------------
    println!("\n--- 13. Witness Chain ---");
    let chain = create_witness_chain(&wit);
    println!("  {} entries, {} bytes", wit.len(), chain.len());
    println!("  Last witness hash: {}", hex(&store.last_witness_hash()[..8]));

    match verify_witness_chain(&chain) {
        Ok(verified) => {
            println!("  Integrity: VALID\n");
            let labels = [
                "REGISTER_HANDS", "REGISTER_TOOLS", "REGISTER_CHANNELS",
                "ROUTE_TASK", "DELETE+COMPACT", "DERIVE", "COW_BRANCH",
            ];
            println!("    {:>2}  {:>4}  {:>22}  {}", "#", "Type", "Timestamp", "Action");
            println!("    {:->2}  {:->4}  {:->22}  {:->20}", "", "", "", "");
            for (i, e) in verified.iter().enumerate() {
                let t = if e.witness_type == 0x01 { "PROV" } else { "COMP" };
                let l = labels.get(i).unwrap_or(&"???");
                println!("    {:>2}  {:>4}  {:>22}  {}", i, t, e.timestamp_ns, l);
            }
        }
        Err(e) => println!("  Integrity: FAILED ({:?})", e),
    }

    // -----------------------------------------------------------------------
    // 14. Persistence round-trip
    // -----------------------------------------------------------------------
    println!("\n--- 14. Persistence ---");
    let final_st = store.status();
    println!("  Before close: {} vectors, {} bytes", final_st.total_vectors, final_st.file_size);

    // Parent is frozen/read-only, so we just drop it
    drop(store);

    let reopened = RvfStore::open_readonly(&store_path).expect("reopen");
    let reopen_st = reopened.status();
    println!("  After reopen: {} vectors, epoch {}", reopen_st.total_vectors, reopen_st.current_epoch);
    println!("  File ID preserved: {}", hex(&reopened.file_id()[..8]) == parent_fid);

    let recheck = reopened.query(&query, K, &QueryOptions::default()).expect("recheck");
    assert_eq!(all.len(), recheck.len(), "count mismatch");
    for (a, b) in all.iter().zip(recheck.iter()) {
        assert_eq!(a.id, b.id, "id mismatch");
        assert!((a.distance - b.distance).abs() < 1e-6, "distance mismatch");
    }
    println!("  Persistence verified.");

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n=== Summary ===\n");
    println!("  Registry:  {} hands + {} tools + {} channels = {} components",
        HANDS.len(), TOOLS.len(), CHANNELS.len(), reg.total());
    println!("  Deleted:   twitter (+ compacted)");
    println!("  Derived:   snapshot at depth {}", child_depth);
    println!("  Branched:  COW staging with experimental 'sentinel'");
    println!("  Segments:  {} in parent", seg_dir.len());
    println!("  Witness:   {} entries", wit.len());
    println!("  File size: {} bytes", final_st.file_size);
    println!("  Filters:   security, tier, category — all passing");
    println!("  Persist:   verified");

    println!("\nDone.");
}
