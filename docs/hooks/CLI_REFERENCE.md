# RuVector Hooks CLI Reference

Complete command-line reference for the RuVector hooks system.

## Synopsis

```bash
npx ruvector hooks <command> [options]
```

Or with global installation:

```bash
ruvector hooks <command> [options]
```

---

## Commands Overview

| Command | Description |
|---------|-------------|
| `init` | Initialize hooks system in current project |
| `install` | Install hooks into Claude Code settings |
| `migrate` | Migrate learning data from other sources |
| `stats` | Display learning statistics |
| `export` | Export learned patterns |
| `import` | Import patterns from file |
| `enable` | Enable hooks system |
| `disable` | Disable hooks system |
| `pre-edit` | Execute pre-edit hook |
| `post-edit` | Execute post-edit hook |
| `pre-command` | Execute pre-command hook |
| `post-command` | Execute post-command hook |
| `session-start` | Start a new session |
| `session-end` | End current session |
| `session-restore` | Restore a previous session |
| `validate-config` | Validate hook configuration |

---

## Core Commands

### `hooks init`

Initialize the hooks system in the current project.

**Syntax:**
```bash
npx ruvector hooks init [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--path` | `PATH` | `./.ruvector` | Custom directory location |
| `--global` | flag | false | Initialize global patterns directory |
| `--template` | `NAME` | `default` | Template: `default`, `minimal`, `advanced` |
| `--force` | flag | false | Overwrite existing configuration |

**Examples:**

```bash
# Basic initialization
npx ruvector hooks init

# Custom directory
npx ruvector hooks init --path .config/ruvector

# Minimal configuration
npx ruvector hooks init --template minimal

# Force reinitialize
npx ruvector hooks init --force
```

**Output:**
```
Initialized ruvector hooks in ./.ruvector
Created: .ruvector/config.toml
Created: .ruvector/intelligence/
Next: Run `npx ruvector hooks install` to add hooks to Claude Code
```

---

### `hooks install`

Install hooks into `.claude/settings.json`.

**Syntax:**
```bash
npx ruvector hooks install [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--force` | flag | false | Overwrite existing hooks |
| `--dry-run` | flag | false | Show changes without applying |
| `--template` | `PATH` | built-in | Use custom hook template |
| `--merge` | flag | true | Merge with existing settings |

**Examples:**

```bash
# Standard installation
npx ruvector hooks install

# Preview changes
npx ruvector hooks install --dry-run

# Force overwrite
npx ruvector hooks install --force

# Custom template
npx ruvector hooks install --template ./my-hooks.json
```

**Output:**
```
Hooks installed to .claude/settings.json
Backup created: .claude/settings.json.backup
Intelligence layer ready
```

---

### `hooks migrate`

Migrate learning data from other sources.

**Syntax:**
```bash
npx ruvector hooks migrate --from <PATH> [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--from` | `PATH` | required | Source data path |
| `--format` | `FORMAT` | auto-detect | Source format: `json`, `sqlite`, `csv` |
| `--merge` | flag | false | Merge with existing patterns |
| `--validate` | flag | false | Validate migration integrity |
| `--dry-run` | flag | false | Show what would be migrated |

**Examples:**

```bash
# Migrate from existing intelligence
npx ruvector hooks migrate --from .claude/intelligence

# Migrate from claude-flow memory
npx ruvector hooks migrate --from ~/.swarm/memory.db --format sqlite

# Merge with validation
npx ruvector hooks migrate --from ./patterns.json --merge --validate

# Preview migration
npx ruvector hooks migrate --from ./old-data --dry-run
```

**Output:**
```
Migrating from JSON files...
Imported 1,247 trajectories
Imported 89 Q-learning patterns
Converted 543 memories to vectors
Validation passed (100% integrity)
Completed in 3.2s
```

---

### `hooks stats`

Display learning statistics and system health.

**Syntax:**
```bash
npx ruvector hooks stats [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--verbose` | flag | false | Show detailed breakdown |
| `--json` | flag | false | Output as JSON |
| `--compare-global` | flag | false | Compare local vs global patterns |

**Examples:**

```bash
# Basic stats
npx ruvector hooks stats

# Detailed view
npx ruvector hooks stats --verbose

# JSON output for scripting
npx ruvector hooks stats --json

# Compare with global
npx ruvector hooks stats --compare-global
```

**Output (verbose):**
```
RuVector Intelligence Statistics
================================

Learning Data:
   Trajectories: 1,247
   Patterns: 89 (Q-learning states)
   Memories: 543 vectors
   Total size: 2.4 MB

Top Patterns:
   1. edit_rs_in_ruvector-core → successful-edit (Q=0.823)
   2. cargo_test → command-succeeded (Q=0.791)
   3. npm_build → command-succeeded (Q=0.654)

Recent Activity:
   Last trajectory: 2 hours ago
   A/B test group: treatment
   Calibration error: 0.042
```

---

### `hooks export`

Export learned patterns for sharing or backup.

**Syntax:**
```bash
npx ruvector hooks export --output <PATH> [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` | `PATH` | required | Output file path |
| `--format` | `FORMAT` | `json` | Format: `json`, `csv`, `sqlite` |
| `--include` | `TYPES` | `all` | Include: `patterns`, `memories`, `all` |
| `--compress` | flag | false | Compress with gzip |

**Examples:**

```bash
# Export all data
npx ruvector hooks export --output backup.json

# Export patterns only
npx ruvector hooks export --output patterns.json --include patterns

# Compressed export
npx ruvector hooks export --output backup.json.gz --compress

# CSV format
npx ruvector hooks export --output data.csv --format csv
```

**Output:**
```
Exported 89 patterns to team-patterns.json
Size: 45.2 KB
SHA256: 8f3b4c2a...
```

---

### `hooks import`

Import learned patterns from file.

**Syntax:**
```bash
npx ruvector hooks import --input <PATH> [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | `PATH` | required | Input file path |
| `--merge` | flag | false | Merge with existing patterns |
| `--strategy` | `STRATEGY` | `prefer-local` | Merge strategy: `prefer-local`, `prefer-imported`, `average` |
| `--validate` | flag | false | Validate before importing |

**Examples:**

```bash
# Import patterns (replace)
npx ruvector hooks import --input patterns.json

# Merge with existing
npx ruvector hooks import --input team-patterns.json --merge

# Merge with strategy
npx ruvector hooks import --input patterns.json --merge --strategy average

# Validate first
npx ruvector hooks import --input data.json --validate
```

**Output:**
```
Importing patterns...
Imported 89 patterns
Merged with 67 existing patterns
New total: 123 patterns (33 updated, 56 unchanged)
```

---

### `hooks enable` / `hooks disable`

Enable or disable the hooks system.

**Syntax:**
```bash
npx ruvector hooks enable
npx ruvector hooks disable
```

**Examples:**

```bash
# Disable temporarily
npx ruvector hooks disable
# Output: Hooks disabled (set RUVECTOR_INTELLIGENCE_ENABLED=false)

# Re-enable
npx ruvector hooks enable
# Output: Hooks enabled (set RUVECTOR_INTELLIGENCE_ENABLED=true)
```

---

## Hook Execution Commands

### `hooks pre-edit`

Execute pre-edit validation and agent assignment.

**Syntax:**
```bash
npx ruvector hooks pre-edit --file <PATH> [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--file`, `-f` | `PATH` | required | File path to be edited |
| `--auto-assign-agent` | flag | true | Assign best agent |
| `--validate-syntax` | flag | false | Validate syntax |
| `--check-conflicts` | flag | false | Check for conflicts |
| `--backup-file` | flag | false | Create backup |

**Examples:**

```bash
# Basic pre-edit
npx ruvector hooks pre-edit --file src/auth/login.ts

# With validation
npx ruvector hooks pre-edit -f src/api.ts --validate-syntax

# Safe edit with backup
npx ruvector hooks pre-edit -f config.json --backup-file
```

**Output (JSON):**
```json
{
  "continue": true,
  "file": "src/auth/login.ts",
  "assignedAgent": "typescript-developer",
  "confidence": 0.85,
  "syntaxValid": true,
  "warnings": []
}
```

---

### `hooks post-edit`

Execute post-edit processing.

**Syntax:**
```bash
npx ruvector hooks post-edit --file <PATH> [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--file`, `-f` | `PATH` | required | File that was edited |
| `--success` | `BOOL` | true | Whether edit succeeded |
| `--auto-format` | flag | true | Format code |
| `--memory-key`, `-m` | `KEY` | auto | Memory storage key |
| `--train-patterns` | flag | false | Train neural patterns |
| `--validate-output` | flag | false | Validate result |

**Examples:**

```bash
# Basic post-edit
npx ruvector hooks post-edit --file src/app.ts

# With memory key
npx ruvector hooks post-edit -f src/auth.ts -m "auth/login-impl"

# Full processing
npx ruvector hooks post-edit -f src/utils.ts --train-patterns --validate-output
```

**Output (JSON):**
```json
{
  "file": "src/app.ts",
  "formatted": true,
  "formatterUsed": "prettier",
  "memorySaved": "edits/src/app.ts",
  "patternsTrained": 3,
  "success": true
}
```

---

### `hooks pre-command`

Execute pre-command safety check.

**Syntax:**
```bash
npx ruvector hooks pre-command <COMMAND> [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--check-safety` | flag | true | Verify command safety |
| `--estimate-resources` | flag | false | Estimate resource usage |
| `--require-confirmation` | flag | false | Require confirmation |

**Examples:**

```bash
# Basic check
npx ruvector hooks pre-command "npm install"

# With resource estimation
npx ruvector hooks pre-command "docker build ." --estimate-resources

# Dangerous command
npx ruvector hooks pre-command "rm -rf /tmp/*" --require-confirmation
```

---

### `hooks post-command`

Execute post-command logging.

**Syntax:**
```bash
npx ruvector hooks post-command <COMMAND> <SUCCESS> [STDERR]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `COMMAND` | string | Command that was executed |
| `SUCCESS` | boolean | Whether command succeeded |
| `STDERR` | string | Error output (optional) |

**Examples:**

```bash
# Successful command
npx ruvector hooks post-command "npm test" true

# Failed command
npx ruvector hooks post-command "cargo build" false "error[E0308]"
```

---

## Session Commands

### `hooks session-start`

Initialize a new session.

**Syntax:**
```bash
npx ruvector hooks session-start [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--session-id`, `-s` | `ID` | auto-generated | Session identifier |
| `--load-context` | flag | false | Load previous context |
| `--init-agents` | flag | false | Initialize agents |

**Examples:**

```bash
# Auto-generated session
npx ruvector hooks session-start

# Named session
npx ruvector hooks session-start --session-id "feature-auth"

# With context loading
npx ruvector hooks session-start -s "debug-123" --load-context
```

**Output:**
```
RuVector Intelligence Layer Active

Session: feature-auth
Patterns: 131 state-action pairs
Memories: 4,247 vectors
Status: Ready
```

---

### `hooks session-end`

End and persist session state.

**Syntax:**
```bash
npx ruvector hooks session-end [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--session-id`, `-s` | `ID` | current | Session to end |
| `--save-state` | flag | true | Save session state |
| `--export-metrics` | flag | false | Export metrics |
| `--generate-summary` | flag | false | Generate summary |
| `--cleanup-temp` | flag | false | Remove temp files |

**Examples:**

```bash
# Basic end
npx ruvector hooks session-end

# With metrics and summary
npx ruvector hooks session-end --export-metrics --generate-summary

# Full cleanup
npx ruvector hooks session-end -s "debug-session" --cleanup-temp
```

**Output (JSON):**
```json
{
  "sessionId": "feature-auth",
  "duration": 7200000,
  "saved": true,
  "metrics": {
    "commandsRun": 145,
    "filesModified": 23,
    "tokensUsed": 85000
  },
  "summaryPath": "./sessions/feature-auth-summary.md"
}
```

---

### `hooks session-restore`

Restore a previous session.

**Syntax:**
```bash
npx ruvector hooks session-restore --session-id <ID> [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--session-id`, `-s` | `ID` | required | Session to restore |
| `--restore-memory` | flag | true | Restore memory state |
| `--restore-agents` | flag | false | Restore agent configs |

**Examples:**

```bash
# Restore session
npx ruvector hooks session-restore --session-id "feature-auth"

# Full restore
npx ruvector hooks session-restore -s "debug-123" --restore-agents
```

---

## Utility Commands

### `hooks validate-config`

Validate hook configuration.

**Syntax:**
```bash
npx ruvector hooks validate-config [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--file` | `PATH` | `.claude/settings.json` | Config file to validate |
| `--fix` | flag | false | Auto-fix issues |

**Examples:**

```bash
# Validate default config
npx ruvector hooks validate-config

# Validate custom file
npx ruvector hooks validate-config --file .claude/settings.json

# Auto-fix issues
npx ruvector hooks validate-config --fix
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Migration error |
| 4 | Validation failed |
| 5 | Timeout |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUVECTOR_HOME` | `~/.ruvector` | Global patterns directory |
| `RUVECTOR_DATA_DIR` | `./.ruvector` | Project-local data directory |
| `RUVECTOR_CLI_PATH` | auto-detected | Path to CLI binary |
| `RUVECTOR_INTELLIGENCE_ENABLED` | `true` | Enable/disable intelligence |
| `RUVECTOR_LEARNING_RATE` | `0.1` | Q-learning alpha parameter |
| `INTELLIGENCE_MODE` | `treatment` | A/B test group |
| `CLAUDE_FLOW_DEBUG` | `false` | Enable debug output |

---

## See Also

- [User Guide](USER_GUIDE.md) - Getting started guide
- [Architecture](ARCHITECTURE.md) - Technical details
- [Migration Guide](MIGRATION.md) - Upgrade from other systems
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
