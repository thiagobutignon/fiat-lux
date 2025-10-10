# Integration Layer

This directory contains the integration layer for connecting the Chomsky DevTools Dashboard (AMARELO) with the 5 core nodes of the Chomsky AGI project.

## 🎯 Overview

The integration layer provides a clean, type-safe interface for communicating with all 5 nodes:

- **🟣 ROXO** (`glass.ts`) - Core .glass organisms & GlassRuntime
- **🟢 VERDE** (`gvcs.ts`) - Genetic Version Control System
- **🔴 VERMELHO** (`security.ts`) - Security & Behavioral Analysis
- **🩶 CINZA** (`cognitive.ts`) - Cognitive OS & Manipulation Detection
- **🟠 LARANJA** (`sqlo.ts`) - O(1) Database

## 📁 Structure

```
/lib/integrations/
├── index.ts          # Central export point
├── glass.ts          # ROXO integration (GlassRuntime, query execution, code emergence)
├── gvcs.ts           # VERDE integration (versions, canary deployment, fitness)
├── security.ts       # VERMELHO integration (duress, behavioral profiling)
├── cognitive.ts      # CINZA integration (manipulation detection, Dark Tetrad)
├── sqlo.ts           # LARANJA integration (O(1) database, episodic memory, RBAC)
└── README.md         # This file
```

## 🚀 Usage

### Import from index

```typescript
import {
  GlassIntegration,
  GVCSIntegration,
  SecurityIntegration,
  CognitiveIntegration,
  SQLOIntegration,
  checkAllNodesHealth,
  getIntegrationStatus
} from '@/lib/integrations';
```

### ROXO - Execute Query

```typescript
import { executeQuery } from '@/lib/integrations';

const result = await executeQuery(organismId, "What is the treatment efficacy?");
console.log(result.answer);
console.log(result.functions_used);
console.log(result.confidence);
```

### VERDE - Get Version History

```typescript
import { getVersionHistory, getCanaryStatus } from '@/lib/integrations';

const versions = await getVersionHistory(organismId);
const canary = await getCanaryStatus(organismId);

console.log(`Active version: ${versions.find(v => v.status === 'active')?.version}`);
console.log(`Canary traffic: ${canary.canary_traffic}%`);
```

### VERMELHO - Analyze Duress

```typescript
import { analyzeDuress, getBehavioralProfile } from '@/lib/integrations';

const duressAnalysis = await analyzeDuress(query, userId);
if (duressAnalysis.is_duress) {
  console.log(`⚠️ Duress detected: ${duressAnalysis.severity}`);
  console.log(`Recommended action: ${duressAnalysis.recommended_action}`);
}

const profile = await getBehavioralProfile(userId);
console.log(`Avg WPM: ${profile.typing_patterns.avg_wpm}`);
```

### CINZA - Detect Manipulation

```typescript
import { detectManipulation, getDarkTetradProfile } from '@/lib/integrations';

const manipulation = await detectManipulation(text);
if (manipulation.detected) {
  console.log(`⚠️ Manipulation detected: ${manipulation.techniques.map(t => t.name).join(', ')}`);
}

const darkTetrad = await getDarkTetradProfile(text);
console.log(`Dark Tetrad risk: ${darkTetrad.risk_level}`);
console.log(`Machiavellianism: ${darkTetrad.machiavellianism}`);
```

### LARANJA - Store & Retrieve Data

```typescript
import { storeEpisodicMemory, getEpisodicMemory, checkPermission } from '@/lib/integrations';

// Store query result
await storeEpisodicMemory({
  organism_id: organismId,
  query: "What is the treatment efficacy?",
  result: queryResult,
  user_id: userId,
  timestamp: new Date().toISOString(),
  session_id: sessionId,
});

// Retrieve history
const history = await getEpisodicMemory(organismId, 10);
console.log(`Last 10 queries:`, history);

// Check permissions
const canQuery = await checkPermission(userId, 'query');
if (!canQuery) {
  throw new Error('User does not have query permission');
}
```

## 🔧 Configuration

Each integration has an `*_ENABLED` flag and `*_API_URL` environment variable:

```bash
# .env.local
ROXO_API_URL=http://localhost:3001
VERDE_API_URL=http://localhost:3002
VERMELHO_API_URL=http://localhost:3003
CINZA_API_URL=http://localhost:3004
LARANJA_API_URL=http://localhost:3005
```

To enable an integration, set the flag in the respective file:

```typescript
// glass.ts
const ROXO_ENABLED = true; // Change from false to true
```

## 📊 Health Checks

Check if all nodes are available:

```typescript
import { checkAllNodesHealth, getIntegrationStatus } from '@/lib/integrations';

// Health check
const health = await checkAllNodesHealth();
console.log('ROXO:', health.roxo.status, health.roxo.version);
console.log('VERDE:', health.verde.status, health.verde.version);
console.log('VERMELHO:', health.vermelho.status, health.vermelho.version);
console.log('CINZA:', health.cinza.status, health.cinza.version);
console.log('LARANJA:', health.laranja.status, health.laranja.performance_us, 'μs');

// Integration status
const status = getIntegrationStatus();
console.log(`Integration progress: ${status.progress_percent}%`);
console.log(`Nodes available: ${status.available_count}/${status.total_count}`);
status.nodes.forEach(node => {
  console.log(`${node.color} ${node.name}: ${node.available ? 'READY' : 'STUB'}`);
});
```

## 🎨 Design Principles

### 1. Stub First, Integrate Later

All integrations start as stubs that:
- Log function calls to console
- Return mock data matching the expected structure
- Throw errors if called when disabled
- Allow development to proceed without blocking on other nodes

### 2. Type Safety

All integrations are fully typed with TypeScript:
- Parameters match expected node APIs
- Return types match dashboard expectations
- Types exported for reuse

### 3. Performance Targets

Each integration documents performance targets:
- LARANJA queries: <1ms (O(1) database)
- ROXO pattern detection: <0.5ms
- VERDE version lookup: <100ms
- VERMELHO duress analysis: <500ms
- CINZA manipulation detection: <500ms

### 4. Error Handling

Integrations throw descriptive errors when:
- Node is disabled but called
- API call fails
- Invalid parameters provided

### 5. Progressive Enhancement

Enable nodes one at a time:
1. LARANJA first (database foundation)
2. ROXO second (core query execution)
3. VERDE third (versioning)
4. VERMELHO + CINZA fourth (security layer)

## 📝 Implementation Checklist

To implement a real integration:

1. **Enable the node**
   - Set `*_ENABLED = true` in the integration file
   - Configure `*_API_URL` environment variable

2. **Replace stubs with real API calls**
   - Find all `// TODO: Real implementation` comments
   - Replace with actual API calls to the node
   - Test with real data

3. **Update API routes**
   - Modify `/app/api/*` routes to use integration layer
   - Remove mock data generation
   - Add error handling

4. **Test integration**
   - Unit tests for integration functions
   - E2E tests for full query flow
   - Performance tests (check targets)

5. **Update documentation**
   - Mark node as ✅ READY in this README
   - Update integration examples with real behavior
   - Document any API changes

## ✅ Integration Status

| Node | Status | Progress | Notes |
|------|--------|----------|-------|
| 🟣 ROXO | STUB | 0% | Awaiting GlassRuntime API |
| 🟢 VERDE | STUB | 0% | Awaiting GVCS API |
| 🔴 VERMELHO | STUB | 0% | Awaiting Security API |
| 🩶 CINZA | STUB | 0% | Awaiting Cognitive API |
| 🟠 LARANJA | STUB | 0% | Awaiting .sqlo API |

**Overall Integration**: 0% (0/5 nodes ready)

## 🔗 API Contracts

### ROXO Expected API

```typescript
// Expected from ROXO
class GlassRuntime {
  constructor(organism: GlassOrganism);
  async query({ query: string }): Promise<QueryResult>;
  async getPatterns(): Promise<Pattern[]>;
  async detectPatterns(): Promise<Pattern[]>;
  async getEmergedFunctions(): Promise<EmergedFunction[]>;
  async synthesizeCode(patterns: Pattern[]): Promise<EmergedFunction[]>;
  async ingest(documents: any[]): Promise<void>;
  async getKnowledgeGraph(): Promise<any>;
  async validateQuery(query: string): Promise<{ status: 'pass' | 'fail'; details: string }>;
}
```

### VERDE Expected API

```typescript
// Expected from VERDE
const gvcsClient = {
  async getVersions(organismId: string): Promise<VersionInfo[]>,
  async getCurrentVersion(organismId: string): Promise<VersionInfo>,
  async getEvolutionData(organismId: string): Promise<EvolutionData>,
  async getCanaryStatus(organismId: string): Promise<CanaryStatus>,
  async deployCanary(organismId: string, version: string, trafficPercent: number): Promise<void>,
  async promoteCanary(organismId: string): Promise<void>,
  async rollbackCanary(organismId: string): Promise<void>,
  async rollback(organismId: string, version: string): Promise<void>,
  async getOldButGold(organismId: string): Promise<VersionInfo[]>,
  async markOldButGold(organismId: string, version: string): Promise<void>,
  async recordFitness(organismId: string, fitness: number): Promise<void>,
  async getFitnessTrajectory(organismId: string): Promise<FitnessPoint[]>,
  async autoCommit(organismId: string, message: string): Promise<string>,
};
```

### VERMELHO Expected API

```typescript
// Expected from VERMELHO
const securityClient = {
  async analyzeDuress({ text: string, userId: string }): Promise<DuressAnalysis>,
  async analyzeQueryDuress({ query: string, userId: string, organismId: string }): Promise<DuressAnalysis>,
  async getProfile(userId: string): Promise<BehavioralProfile>,
  async updateProfile(userId: string, data: any): Promise<void>,
  async analyzeLinguisticFingerprint({ text: string, userId: string }): Promise<{ match: boolean; confidence: number; deviations: string[] }>,
  async analyzeTypingPatterns({ patterns: TypingPattern[], userId: string }): Promise<{ match: boolean; confidence: number }>,
  async analyzeEmotion(text: string): Promise<EmotionalState>,
  async compareEmotionalState(userId: string, emotionalState: EmotionalState): Promise<{ deviation: number; alert: boolean }>,
  async analyzeTemporalPattern(userId: string, timestamp: number): Promise<{ anomaly: boolean; confidence: number }>,
  async comprehensiveAnalysis(params: any): Promise<SecurityAnalysis>,
};
```

### CINZA Expected API

```typescript
// Expected from CINZA
const cognitiveClient = {
  async detectManipulation({ text: string }): Promise<ManipulationDetection>,
  async detectQueryManipulation({ query: string, userId: string, organismId: string }): Promise<ManipulationDetection>,
  async getManipulationTechniques(): Promise<ManipulationTechnique[]>,
  async getDarkTetradProfile({ text: string }): Promise<DarkTetradProfile>,
  async getUserDarkTetrad(userId: string): Promise<DarkTetradProfile>,
  async detectCognitiveBiases({ text: string }): Promise<CognitiveBias[]>,
  async processStream(stream: ReadableStream, onDetection: Function): Promise<void>,
  async validateConstitutional({ text: string, principles: string[] }): Promise<{ status: 'pass' | 'fail'; violations: string[] }>,
  async triggerSelfSurgery(organismId: string): Promise<{ optimizations: string[]; applied: boolean }>,
  async getOptimizationSuggestions(organismId: string): Promise<string[]>,
  async detectManipulationI18n({ text: string, language: string }): Promise<ManipulationDetection>,
  async comprehensiveAnalysis(params: any): Promise<ComprehensiveAnalysis>,
};
```

### LARANJA Expected API

```typescript
// Expected from LARANJA
const sqloClient = {
  async query(table: string, filters: any, options?: any): Promise<any[]>,
  async insert(table: string, data: any): Promise<void>,
  async update(table: string, filters: any, updates: any): Promise<void>,
  async delete(table: string, filters: any): Promise<void>,
  async checkPermission(userId: string, permission: string): Promise<boolean>,
  async assignRole(userId: string, roleId: string): Promise<void>,
  async consolidate(): Promise<{ optimized: number; duration_ms: number }>,
  async getConsolidationStatus(): Promise<{ last_run: string; next_run: string; status: string }>,
  async getMetrics(): Promise<{ avg_query_time_us: number; total_queries: number; cache_hit_rate: number }>,
};
```

## 📚 Examples

See `/app/api/organisms/[id]/query/route.ts` for a full example of how to integrate multiple nodes in a query flow:

1. Validate query (ROXO constitutional check)
2. Check duress (VERMELHO security)
3. Detect manipulation (CINZA cognitive)
4. Execute query (ROXO runtime)
5. Record fitness (VERDE evolution)
6. Store in episodic memory (LARANJA database)
7. Log constitutional checks (LARANJA logs)

## 🎊 Ready for Integration!

All 5 integration stubs are complete and ready to receive real implementations from their respective nodes. The DevTools Dashboard is waiting! 🟡✨

---

**Last Updated**: 2025-10-10
**Node**: 🟡 AMARELO
**Status**: Integration Layer Ready ✅
