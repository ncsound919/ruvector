# RuvBot

**Self-Learning AI Assistant with RuVector Backend**

RuvBot is a Clawdbot-style personal AI assistant powered by RuVector's WASM vector operations. It features self-learning capabilities, multi-tenancy support, and seamless integration with Slack, Discord, and webhooks.

## Features

- **Self-Learning**: SONA adaptive learning with trajectory tracking and pattern extraction
- **WASM Embeddings**: High-performance vector operations using RuVector WASM bindings
- **Vector Memory**: HNSW-indexed semantic memory with 150x-12,500x faster search
- **Multi-Platform**: Slack, Discord, webhook, REST API, and CLI interfaces
- **Extensible Skills**: Plugin architecture for custom capabilities
- **Multi-Tenancy**: Enterprise-ready with PostgreSQL row-level security
- **Background Workers**: Long-running task support via agentic-flow

## Quick Start

### Install via curl

```bash
curl -fsSL https://get.ruvector.dev/ruvbot | bash
```

Or with custom settings:

```bash
RUVBOT_VERSION=0.1.0 \
RUVBOT_INSTALL_DIR=/opt/ruvbot \
curl -fsSL https://get.ruvector.dev/ruvbot | bash
```

### Install via npm/npx

```bash
# Run directly
npx @ruvector/ruvbot start

# Or install globally
npm install -g @ruvector/ruvbot
ruvbot start
```

## Configuration

### Environment Variables

```bash
# LLM Provider (required)
export ANTHROPIC_API_KEY=sk-ant-xxx
# or
export OPENAI_API_KEY=sk-xxx

# Slack Integration (optional)
export SLACK_BOT_TOKEN=xoxb-xxx
export SLACK_SIGNING_SECRET=xxx
export SLACK_APP_TOKEN=xapp-xxx

# Discord Integration (optional)
export DISCORD_TOKEN=xxx
export DISCORD_CLIENT_ID=xxx

# Server Configuration
export RUVBOT_PORT=3000
export RUVBOT_LOG_LEVEL=info
```

### Configuration File

Create `ruvbot.config.json`:

```json
{
  "name": "my-ruvbot",
  "api": {
    "enabled": true,
    "port": 3000,
    "host": "0.0.0.0"
  },
  "storage": {
    "type": "sqlite",
    "path": "./data/ruvbot.db"
  },
  "memory": {
    "dimensions": 384,
    "maxVectors": 100000,
    "indexType": "hnsw"
  },
  "skills": {
    "enabled": ["search", "summarize", "code", "memory"]
  },
  "slack": {
    "enabled": true,
    "socketMode": true
  }
}
```

## CLI Commands

```bash
# Initialize in current directory
ruvbot init

# Start the bot server
ruvbot start [--port 3000] [--debug]

# Check status
ruvbot status

# Manage skills
ruvbot skills list
ruvbot skills add <name>

# Run diagnostics
ruvbot doctor

# Show configuration
ruvbot config --show
```

## API Usage

### REST API

```bash
# Create a session
curl -X POST http://localhost:3000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"agentId": "default"}'

# Send a message
curl -X POST http://localhost:3000/api/sessions/{id}/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello, RuvBot!"}'
```

### Programmatic Usage

```typescript
import { RuvBot, createRuvBot } from '@ruvector/ruvbot';

// Create bot instance
const bot = createRuvBot({
  config: {
    llm: {
      provider: 'anthropic',
      apiKey: process.env.ANTHROPIC_API_KEY,
    },
    memory: {
      dimensions: 384,
      maxVectors: 100000,
    },
  },
});

// Start the bot
await bot.start();

// Spawn an agent
const agent = await bot.spawnAgent({
  id: 'assistant',
  name: 'My Assistant',
});

// Create a session
const session = await bot.createSession(agent.id, {
  userId: 'user-123',
  platform: 'api',
});

// Chat
const response = await bot.chat(session.id, 'What can you help me with?');
console.log(response.content);
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           RuvBot                                 │
├─────────────────────────────────────────────────────────────────┤
│  REST API │ GraphQL │ Slack Adapter │ Discord │ Webhooks       │
├─────────────────────────────────────────────────────────────────┤
│                     Core Application Layer                       │
│  AgentManager │ SessionStore │ SkillRegistry │ MemoryManager    │
├─────────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                          │
│  RuVector WASM │ PostgreSQL │ RuvLLM │ agentic-flow Workers     │
└─────────────────────────────────────────────────────────────────┘
```

## Skills

### Built-in Skills

| Skill | Description |
|-------|-------------|
| `search` | Semantic search across memory and documents |
| `summarize` | Generate concise summaries of text |
| `code` | Code generation, analysis, and explanation |
| `memory` | Store and retrieve long-term memories |

### Custom Skills

Create custom skills in the `skills/` directory:

```typescript
// skills/my-skill.ts
import { defineSkill } from '@ruvector/ruvbot';

export default defineSkill({
  name: 'my-skill',
  description: 'Custom skill description',
  inputs: [
    { name: 'query', type: 'string', required: true }
  ],
  async execute(params, context) {
    return {
      success: true,
      data: `Processed: ${params.query}`,
    };
  },
});
```

## Memory System

RuvBot uses HNSW-indexed vector memory for fast semantic search:

```typescript
import { MemoryManager, createWasmEmbedder } from '@ruvector/ruvbot/learning';

const embedder = createWasmEmbedder({ dimensions: 384 });
const memory = new MemoryManager({
  config: { dimensions: 384, maxVectors: 100000, indexType: 'hnsw' },
  embedder,
});

// Store a memory
await memory.store('Important information', {
  source: 'user',
  tags: ['important'],
  importance: 0.9,
});

// Search memories
const results = await memory.search('find important info', {
  topK: 5,
  threshold: 0.7,
});
```

## Docker

```yaml
# docker-compose.yml
version: '3.8'
services:
  ruvbot:
    image: ruvector/ruvbot:latest
    ports:
      - "3000:3000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - SLACK_BOT_TOKEN=${SLACK_BOT_TOKEN}
    volumes:
      - ./data:/app/data
      - ./skills:/app/skills
```

## Development

```bash
# Clone the repository
git clone https://github.com/ruvnet/ruvector.git
cd ruvector/npm/packages/ruvbot

# Install dependencies
npm install

# Run in development mode
npm run dev

# Run tests
npm test

# Build
npm run build
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `@ruvector/ruvllm` | LLM orchestration with SONA learning |
| `@ruvector/wasm-unified` | WASM vector operations |
| `@ruvector/postgres-cli` | PostgreSQL vector storage |
| `fastify` | REST API server |
| `@slack/bolt` | Slack integration |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines.

## Links

- **Repository**: https://github.com/ruvnet/ruvector
- **Issues**: https://github.com/ruvnet/ruvector/issues
- **Documentation**: https://github.com/ruvnet/ruvector/tree/main/npm/packages/ruvbot
