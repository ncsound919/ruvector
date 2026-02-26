# OpenFang Agent OS — RVF Example

A deep RVF integration example that models the [OpenFang](https://github.com/RightNow-AI/openfang) Agent Operating System as a searchable component registry, exercising the full RVF capability surface.

## Run

```bash
cd examples/rvf
cargo run --example openfang
```

## What It Does

Creates a single RVF file (~34 KB) containing an entire agent OS registry, then exercises 14 distinct RVF capabilities against it.

### Registry Contents

| Component | Count | Description |
|-----------|------:|-------------|
| **Hands** | 7 | Autonomous agents (Clip, Lead, Collector, Predictor, Researcher, Twitter, Browser) |
| **Tools** | 38 | Built-in capabilities across 13 categories |
| **Channels** | 20 | Messaging adapters (Telegram, Discord, Slack, WhatsApp, etc.) |
| **Total** | 65 | All searchable in one vector space |

### RVF Capabilities Exercised

| # | Capability | RVF API | What It Shows |
|---|-----------|---------|---------------|
| 1 | **Store creation** | `RvfStore::create` | 128-dim L2 store with file identity |
| 2-4 | **Batch ingestion** | `ingest_batch` | Multi-type metadata (String, U64) with per-category vector biasing |
| 5 | **Nearest-neighbor search** | `query` | Unfiltered + type-filtered task routing |
| 6 | **Combined filters** | `FilterExpr::And` + `Ge` | Security threshold filtering (>= 80) |
| 7 | **Equality filter** | `FilterExpr::Eq` | Tier-4 autonomous agent isolation |
| 8 | **Category filter** | `FilterExpr::And` | Tool discovery by category |
| 9 | **Delete + compact** | `delete` + `compact` | Decommission 'twitter' hand, reclaim 512 bytes |
| 10 | **Derive (lineage)** | `derive` | Snapshot with parent-child provenance, depth tracking |
| 11 | **COW branching** | `freeze` + `branch` | Staging environment with experimental 'sentinel' agent |
| 12 | **Segment inspection** | `segment_dir` | Raw segment directory (VEC, MANIFEST, JOURNAL, etc.) |
| 13 | **Witness chain** | `create_witness_chain` + `verify` | 7-entry cryptographic audit trail |
| 14 | **Persistence** | `close` + `open_readonly` | Round-trip verification with file ID preservation |

## Metadata Schema

| Field ID | Constant | Name | Type | Applies To |
|:--------:|----------|------|------|------------|
| 0 | `F_TYPE` | component_type | String | All (`"hand"`, `"tool"`, `"channel"`) |
| 1 | `F_NAME` | name | String | All |
| 2 | `F_DOMAIN` | domain / category / protocol | String | All |
| 3 | `F_TIER` | tier | U64 (1-4) | Hands only |
| 4 | `F_SEC` | security_level | U64 (0-100) | Hands only |

## Hands

| Hand | Domain | Tier | Security |
|------|--------|:----:|:--------:|
| clip | video-processing | 3 | 60 |
| lead | sales-automation | 2 | 70 |
| collector | osint-intelligence | 4 | 90 |
| predictor | forecasting | 3 | 80 |
| researcher | fact-checking | 3 | 75 |
| twitter | social-media | 2 | 65 |
| browser | web-automation | 4 | 95 |

## Tool Categories (13)

`browser`, `communication`, `database`, `document`, `filesystem`, `inference`, `integration`, `memory`, `network`, `scheduling`, `security`, `system`, `transform`

## Channel Adapters (20)

Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Email (SMTP/IMAP), Teams, Google Chat, LinkedIn, Twitter/X, Mastodon, Bluesky, Reddit, IRC, XMPP, Webhooks (in/out), gRPC

## Architecture Notes

### Vector Biasing

Tools and channels use `category_bias()` — a hash-based offset applied to the first 16 dimensions — so items sharing a category cluster in vector space. Hands use tier-proportional bias (`tier * 0.1`) to create performance-tier clusters.

### Delete + Compact Lifecycle

Step 9 demonstrates the full decommission workflow:
1. `delete(&[twitter_id])` — soft-delete, marks vector as tombstoned
2. `compact()` — rewrites the store, reclaiming dead space
3. Post-delete queries confirm the vector is gone

### COW Branching

Step 11 shows a staging/production pattern:
1. `freeze()` — makes the parent read-only (immutable baseline)
2. `branch()` — creates a COW child inheriting all parent vectors
3. New vectors added to the child don't affect the parent
4. `cow_stats()` reports cluster-level copy-on-write telemetry

### Lineage Tracking

Step 10 derives a snapshot child and verifies:
- Child `parent_id` matches parent `file_id`
- Lineage depth increments (0 -> 1)
- Provenance chain is cryptographically verifiable

## About OpenFang

[OpenFang](https://openfang.sh) by RightNow AI is a Rust-based Agent Operating System — 137K lines of code across 14 crates, compiling to a single ~32 MB binary. It runs autonomous agents 24/7 with 16 security systems, 27 LLM providers, and 40 channel adapters.

- GitHub: [RightNow-AI/openfang](https://github.com/RightNow-AI/openfang)
- License: MIT / Apache 2.0
