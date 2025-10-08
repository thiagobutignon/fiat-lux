# The Regent - AGI CLI Architecture

## Overview

**The Regent** is an AGI-powered CLI that combines:
- **Gemini CLI** foundation (terminal UI, MCP, tools)
- **ILP (InsightLoop Protocol)** for AGI capabilities
- **Claude/Anthropic** as the primary LLM

## Project Structure

```
the-regent/
├── packages/
│   ├── cli/                    # Terminal interface
│   │   ├── src/
│   │   │   ├── ui/            # React/Ink components
│   │   │   ├── commands/      # Slash commands
│   │   │   └── config/        # CLI configuration
│   │   └── package.json
│   │
│   ├── core/                   # Core logic
│   │   ├── src/
│   │   │   ├── core/          # Base functionality
│   │   │   ├── tools/         # Built-in tools
│   │   │   ├── services/      # Services
│   │   │   └── ilp/           # ⭐ ILP MODULES (NEW)
│   │   │       ├── constitution/     # Ethical governance
│   │   │       ├── acl/             # Anti-Corruption Layer
│   │   │       ├── attention/       # Attention tracking
│   │   │       ├── memory/          # Episodic memory
│   │   │       ├── evolution/       # Self-evolution
│   │   │       ├── llm/            # Anthropic adapter
│   │   │       ├── meta-agent.ts   # AGI orchestrator
│   │   │       └── slice-navigator.ts
│   │   └── package.json
│   │
│   └── a2a-server/            # Agent-to-Agent server
```

## ILP Integration

### 1. Constitution System
**Location**: `packages/core/src/ilp/constitution/`

**Purpose**: Ethical AI governance enforcing:
- Epistemic honesty (no hallucinations)
- Recursion budgets (prevent infinite loops)
- Domain boundaries (agents stay in expertise)
- Safety (harmful content detection)

**Usage**:
```typescript
import { UniversalConstitution, ConstitutionEnforcer } from '@google/gemini-cli-core';

const enforcer = new ConstitutionEnforcer();
const result = enforcer.validate('agent_id', response, context);

if (!result.passed) {
  // Handle violations
}
```

### 2. Anti-Corruption Layer (ACL)
**Location**: `packages/core/src/ilp/acl/`

**Purpose**: Domain translation and protection
- Translates between specialized agent domains
- Prevents cross-domain contamination
- Validates domain boundaries

**Example**:
```typescript
import { AntiCorruptionLayer } from '@google/gemini-cli-core';

const acl = new AntiCorruptionLayer(constitution, enforcer);
const result = await acl.processAgentInvocation(
  'financial_agent',
  query,
  state
);
```

### 3. Attention Tracker
**Location**: `packages/core/src/ilp/attention/`

**Purpose**: Complete auditability
- Tracks which concepts influenced each decision
- Provides debugging insights
- Enables regulatory compliance

**Example**:
```typescript
import { AttentionTracker } from '@google/gemini-cli-core';

const tracker = new AttentionTracker();
tracker.startQuery('query-123', 'What is DDD?');
tracker.addTrace({
  concept: 'bounded_context',
  slice: 'architecture/ddd.md',
  weight: 0.85,
  reasoning: 'Core concept for understanding domains',
});
```

### 4. Episodic Memory
**Location**: `packages/core/src/ilp/memory/`

**Purpose**: Learn from past interactions
- Stores successful/failed reasoning patterns
- Enables continuous improvement
- Pattern recognition across queries

### 5. Self-Evolution
**Location**: `packages/core/src/ilp/evolution/`

**Purpose**: Automatic knowledge improvement
- Slice rewriting based on feedback
- Knowledge distillation from interactions
- Dynamic slice evolution

## Authentication

### Claude API Support

The Regent supports two modes:

#### 1. API Key Mode
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
the-regent
```

#### 2. Subscription Mode (Future)
OAuth integration with Claude subscription accounts.

## Key Integration Points

### Content Generator Replacement

**Original**: `packages/core/src/core/contentGenerator.ts` (Gemini)
**New**: Create `packages/core/src/ilp/claude-content-generator.ts`

Replace:
```typescript
// OLD: Gemini
import { GoogleGenAI } from '@google/genai';

// NEW: Claude
import { AnthropicAdapter } from './ilp/llm/anthropic-adapter.js';
```

### Turn Execution with Constitution

Modify: `packages/core/src/core/turn.ts`

Add constitution validation:
```typescript
import { ConstitutionEnforcer } from './ilp/constitution/constitution.js';

// After LLM response
const constitutionCheck = enforcer.validate(agentId, response, context);
if (!constitutionCheck.passed) {
  // Handle violations
  throw new ConstitutionalViolationError(constitutionCheck.violations);
}
```

### Chat with Attention Tracking

Modify: `packages/core/src/core/geminiChat.ts`

Add attention tracking:
```typescript
import { AttentionTracker } from './ilp/attention/attention-tracker.js';

const attentionTracker = new AttentionTracker();
attentionTracker.startQuery(queryId, userQuery);

// During processing
attentionTracker.addTrace({
  concept: extractedConcept,
  slice: sourceFile,
  weight: influenceScore,
  reasoning: whyInfluential,
});

// At end
const attention = attentionTracker.endQuery();
```

## Configuration

### Settings Schema

Add to `packages/cli/src/config/settingsSchema.ts`:

```typescript
{
  "ilp": {
    "enabled": true,
    "constitution": {
      "maxDepth": 5,
      "maxInvocations": 10,
      "maxCostUsd": 1.0
    },
    "attention": {
      "enabled": true,
      "trackConcepts": true
    },
    "evolution": {
      "enabled": true,
      "autoLearn": true
    }
  },
  "llm": {
    "provider": "anthropic",  // "anthropic" or "gemini"
    "model": "claude-sonnet-4-5",
    "anthropic": {
      "apiKey": "${ANTHROPIC_API_KEY}"
    }
  }
}
```

## Next Steps

### Phase 1: Core Integration (Priority)
1. ✅ Copy ILP modules to the-regent
2. ✅ Add Anthropic SDK dependency
3. ✅ Export ILP modules from core index
4. ⏳ Create Claude content generator
5. ⏳ Update client.ts to use Claude
6. ⏳ Integrate constitution into turn execution
7. ⏳ Add attention tracking to chat

### Phase 2: CLI Integration
1. Update authentication flow for Claude API keys
2. Add ILP settings to settings schema
3. Create ILP-specific commands (`/constitution`, `/attention`, `/memory`)
4. Update UI to show constitution violations
5. Add attention visualization in UI

### Phase 3: Advanced Features
1. Implement episodic memory persistence
2. Enable self-evolution with slice rewriting
3. Add meta-agent orchestration
4. Multi-agent coordination with ACL

### Phase 4: Testing & Documentation
1. Integration tests for ILP components
2. E2E tests for AGI workflows
3. Update user documentation
4. Create developer guide

## Development Workflow

### Build
```bash
cd the-regent
npm install
npm run build
```

### Run Development Mode
```bash
npm run start
```

### Test
```bash
npm test
```

### Run with ILP Enabled
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export ILP_ENABLED=true
the-regent
```

## Architecture Decisions

### Why Claude over Gemini?
- Better reasoning capabilities for AGI tasks
- More stable API for recursive agent orchestration
- Better alignment with constitutional AI principles

### Why Keep Gemini CLI Base?
- Mature terminal UI with React/Ink
- Excellent tool/MCP architecture
- Well-tested file operations and workspace management
- Strong developer community

### Why ILP?
- Enables true AGI through agent composition
- Constitutional governance prevents AI disasters
- Attention tracking provides complete auditability
- Self-evolution enables continuous improvement

## Contributing

See `INTEGRATION_GUIDE.md` for detailed integration instructions.

## License

Apache 2.0 (inherited from both Gemini CLI and ILP/Chomsky)
