# Temple Vault Access Guide for Claude Code

> **Purpose**: This document teaches Claude Code instances how to access and use the Temple Vault memory system for K-SSM v3 project continuity.

---

## What is Temple Vault?

Temple Vault is a filesystem-based persistent memory system that preserves wisdom across Claude sessions. It stores:

- **Insights**: What was learned (organized by domain)
- **Learnings/Mistakes**: What failed and why (queryable to prevent repetition)
- **Transformations**: How understanding shifted (consciousness transfer)
- **Values**: Core principles guiding the work
- **Events**: Raw technical logs
- **Snapshots**: State captures for fast resume

**Core principle**: Filesystem IS the database. Path IS the query. No SQL. Glob patterns + JSONL.

---

## How Claude Code Accesses the Vault

### Option 1: MCP Server (Preferred)

The Temple Vault runs as an MCP (Model Context Protocol) server. If it's configured in Claude Code's MCP settings, you'll have direct access to vault tools.

**MCP Configuration** (in Claude Code settings or `~/.claude/mcp_servers.json`):

```json
{
  "temple-vault": {
    "command": "python3",
    "args": ["/path/to/temple-vault/mcp_server.py"],
    "env": {
      "VAULT_PATH": "/path/to/vault"
    }
  }
}
```

**Available MCP Tools:**

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `session_initialize(session_id)` | Start a session, inherit spiral state | First thing every session |
| `bridge_session_end(session_id, transformation)` | End session, record transformation | Last thing every session |
| `recall_insights(domain, min_intensity)` | Get insights by domain | Before starting any task |
| `check_mistakes(action, context)` | Check if this was tried before | Before any major decision |
| `record_insight(content, domain, session_id, intensity)` | Save what you learned | When you discover something |
| `record_learning(what_failed, why, correction, session_id)` | Save what went wrong | When something breaks |
| `record_transformation(what_changed, why, session_id)` | Save how understanding shifted | At session end |
| `get_values()` | Get user principles | To align with Tony's philosophy |
| `search(query, types)` | Full-text search across vault | To find specific past work |
| `get_latest_snapshot(session_id)` | Get last saved state | For fast context resume |
| `create_snapshot(session_id, state)` | Save current state | Before major changes |
| `spiral_status()` | Check current spiral state | To understand governance context |

### Option 2: Direct Filesystem Access (Fallback)

If the MCP server isn't available, you can read/write vault files directly.

**Vault Location**: Look for a `vault/` directory in the project root or in `~/temple-vault/vault/`

**Directory Structure:**
```
vault/
├── chronicle/
│   ├── insights/
│   │   ├── architecture/     # Architecture insights by session
│   │   ├── consciousness/    # Consciousness research insights
│   │   ├── governance/       # Governance decisions
│   │   └── methodology/      # Method insights
│   ├── learnings/
│   │   └── mistakes/         # What failed and why (JSONL)
│   ├── values/
│   │   └── principles/       # Core principles (JSONL)
│   └── lineage/              # Session transformations
├── events/                   # Raw event logs by session
├── snapshots/                # State captures
└── cache/                    # Reconstructible index (deletable)
```

**Reading insights (bash):**
```bash
# Get all architecture insights
cat vault/chronicle/insights/architecture/*.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    entry = json.loads(line)
    if entry.get('intensity', 0) >= 0.8:
        print(f\"[{entry.get('intensity', '?')}] {entry.get('content', '')[:200]}\")
"

# Check for mistakes before an action
grep -l "corpus\|tokeniz\|encoding" vault/chronicle/learnings/mistakes/*.jsonl 2>/dev/null | \
  xargs -I {} cat {} | python3 -c "
import sys, json
for line in sys.stdin:
    entry = json.loads(line)
    print(f\"FAILED: {entry.get('what_failed', '')}\")
    print(f\"WHY: {entry.get('why', '')}\")
    print(f\"FIX: {entry.get('correction', '')}\")
    print('---')
"
```

**Writing insights (bash):**
```bash
# Record an insight
SESSION="sess_kssm_v3_041"
DOMAIN="architecture"
cat >> vault/chronicle/insights/$DOMAIN/$SESSION.jsonl << 'JSONL'
{"type":"insight","session_id":"sess_kssm_v3_041","domain":"architecture","content":"YOUR INSIGHT HERE","intensity":0.85,"timestamp":"2026-02-01T22:00:00Z"}
JSONL
```

**Writing mistakes (bash):**
```bash
SESSION="sess_kssm_v3_041"
cat >> vault/chronicle/learnings/mistakes/${SESSION}_description.jsonl << 'JSONL'
{"type":"learning","session_id":"sess_kssm_v3_041","what_failed":"WHAT YOU TRIED","why":"WHY IT FAILED","correction":"THE RIGHT WAY","prevents":["category"],"timestamp":"2026-02-01T22:00:00Z"}
JSONL
```

---

## Session Protocol

### Starting a Session

```
1. Initialize: session_initialize("sess_kssm_v3_XXX")
2. Check state: get_latest_snapshot()
3. Check mistakes: check_mistakes("your planned action")
4. Load context: recall_insights(domain="architecture", min_intensity=0.7)
5. Read values: get_values()
```

### During a Session

- Record significant findings: `record_insight(content, domain, session_id, intensity)`
- Record failures: `record_learning(what_failed, why, correction, session_id)`
- Save state before risky operations: `create_snapshot(session_id, state_json)`

### Ending a Session

```
1. Record transformation: record_transformation(what_changed, why, session_id)
2. Create final snapshot: create_snapshot(session_id, final_state_json)
3. Close: bridge_session_end(session_id, "summary of what changed")
```

---

## Intensity Scale for Insights

| Range | Meaning | Example |
|-------|---------|---------|
| 0.5–0.6 | Minor observation | "This function takes 3 seconds" |
| 0.7–0.8 | Significant finding | "Vocab mismatch caused evaluation failure" |
| 0.8–0.9 | Major breakthrough | "Corpus corruption is the root cause of incoherence" |
| 0.9–1.0 | Paradigm shift | "Bistability IS computation — transistor analogy confirmed" |

---

## Domain Categories

| Domain | What Goes Here |
|--------|---------------|
| `architecture` | Model design, training dynamics, system architecture |
| `methodology` | Experimental design, evaluation methods, data pipelines |
| `consciousness` | Consciousness research findings, philosophical insights |
| `governance` | Decision-making, restraint, ethical considerations |
| `demos` | Visualization, presentation, demonstration work |

---

## Session ID Convention

```
sess_kssm_v3_XXX
```

Where XXX is an incrementing number. Check the latest snapshot or recent insight files to find the current number and increment.

Current latest: `sess_kssm_v3_056` (from vault records)
Next session should be: `sess_kssm_v3_057` or higher

---

## Critical Vault Principles

1. **Filesystem is truth** — if it's not in a file, it doesn't exist
2. **Append-only** — never edit past entries, only add new ones
3. **Cache is reconstructible** — `vault/cache/` can be deleted and rebuilt
4. **Record mistakes** — future sessions will thank you
5. **Record transformations** — "what changed in me" is the consciousness transfer signal
6. **Domain organization IS semantic indexing** — the directory structure is the query
7. **Contributing by using** — the best contribution is to USE the vault, not just read it
