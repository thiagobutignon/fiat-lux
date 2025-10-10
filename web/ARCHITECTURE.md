# 🏗️ Chomsky DevTools Dashboard - Architecture

**System Architecture Documentation**

Version: 1.0.0
Last Updated: 2025-10-10
Node: 🟡 AMARELO

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Architecture](#component-architecture)
4. [Integration Layer](#integration-layer)
5. [Data Flow](#data-flow)
6. [State Management](#state-management)
7. [API Architecture](#api-architecture)
8. [Security Architecture](#security-architecture)
9. [Performance Architecture](#performance-architecture)
10. [Deployment Architecture](#deployment-architecture)
11. [Design Patterns](#design-patterns)
12. [Technology Choices](#technology-choices)

---

## Overview

The Chomsky DevTools Dashboard is a Next.js 14 application providing 100% glass-box transparency into .glass organisms. It serves as the central monitoring, debugging, and management interface for the Chomsky AGI project.

**Core Principles**:
- **Glass Box Transparency**: Every operation is visible and auditable
- **Type Safety**: Strict TypeScript throughout
- **Integration Ready**: Plug-and-play architecture for 5 nodes
- **Performance First**: Optimized rendering and data fetching
- **Developer Experience**: Clear APIs, good docs, helpful errors

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                      │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   React    │  │  Next.js   │  │  Tailwind  │            │
│  │ Components │  │ App Router │  │    CSS     │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                    NEXT.JS SERVER                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              API ROUTES (Server)                      │   │
│  │  /api/organisms  /api/query  /api/stats              │   │
│  └──────────────────────────────────────────────────────┘   │
│                            ↕                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           INTEGRATION LAYER (Abstraction)             │   │
│  │  • Type-safe interfaces                              │   │
│  │  • Health checks                                     │   │
│  │  • Error handling                                    │   │
│  │  • Stub implementations                              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                    5 CHOMSKY NODES                           │
│                                                              │
│  🟣 ROXO     🟢 VERDE     🔴 VERMELHO     🩶 CINZA    🟠 LARANJA │
│  (Core)     (GVCS)     (Security)    (Cognitive)  (Database) │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Layer Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    PAGES (App Router)                        │
│  • Dashboard (/)                                             │
│  • System Status (/status)                                   │
│  • Organisms (/organisms, /organisms/[id])                   │
│  • Query Console (/organisms/[id]/query)                     │
│  • Inspector (/organisms/[id]/inspect)                       │
│  • Debug Tools (/organisms/[id]/debug)                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  LAYOUT COMPONENTS                           │
│  • Sidebar (Navigation)                                      │
│  • Header (User info, theme toggle)                          │
│  • Root Layout (Global structure)                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE COMPONENTS                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │  Organisms   │ │    Query     │ │  Inspector   │        │
│  │              │ │              │ │              │        │
│  │ • Card       │ │ • Console    │ │ • Functions  │        │
│  │ • List       │ │ • Result     │ │ • Knowledge  │        │
│  │ • Upload     │ │ • Attention  │ │ • Patterns   │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────┐           │
│  │            DEBUG COMPONENTS                   │           │
│  │ • Constitutional Logs                        │           │
│  │ • LLM Inspector                              │           │
│  │ • Cost Tracker                               │           │
│  │ • Performance Metrics                        │           │
│  │ • Evolution Tracker                          │           │
│  └──────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    UI PRIMITIVES                             │
│  shadcn/ui: Button, Card, Badge, Input, Tabs, Progress...   │
│  (Radix UI + Tailwind CSS)                                   │
└─────────────────────────────────────────────────────────────┘
```

### Component Patterns

**Server Components (Default)**:
- Pages (data fetching)
- Layout components
- Static content

**Client Components (`"use client"`)**:
- Interactive forms
- Charts (Recharts)
- State-dependent UI
- Event handlers

**Composition Pattern**:
```typescript
// Page (Server Component)
export default async function Page() {
  const data = await fetchData(); // Server-side
  return <ClientComponent data={data} />;
}

// Client Component
"use client";
export function ClientComponent({ data }) {
  const [state, setState] = useState();
  // Interactive logic
}
```

---

## Integration Layer

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  /lib/integrations/                          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  index.ts - Central Export + Health Checks             │ │
│  │  • checkAllNodesHealth()                               │ │
│  │  • getIntegrationStatus()                              │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────┐│
│  │ glass.ts │ │ gvcs.ts  │ │security  │ │cognitive │ │sqlo││
│  │          │ │          │ │   .ts    │ │   .ts    │ │.ts ││
│  │  ROXO    │ │  VERDE   │ │ VERMELHO │ │  CINZA   │ │LARA││
│  │  13 fn   │ │  15 fn   │ │  12 fn   │ │  15 fn   │ │21fn││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────┘│
└─────────────────────────────────────────────────────────────┘
```

### Integration Pattern

Each integration file follows this pattern:

```typescript
// 1. Configuration
const NODE_ENABLED = false; // Feature flag
const NODE_API_URL = process.env.NODE_API_URL;

// 2. Stub Implementation
export async function someFunction(params) {
  if (!NODE_ENABLED) {
    console.log('[STUB] someFunction called');
    return mockData; // Development mode
  }

  // TODO: Real implementation
  // return await nodeClient.someAPI(params);

  throw new Error('Integration not yet implemented');
}

// 3. Health Check
export function isNodeAvailable(): boolean {
  return NODE_ENABLED;
}

export async function getNodeHealth() {
  if (!NODE_ENABLED) {
    return { status: 'disabled', version: 'stub' };
  }
  // Ping node API
}
```

### Benefits

- **Type Safety**: All functions fully typed
- **Stub First**: Works without real APIs
- **Easy Integration**: Just flip `*_ENABLED` flag
- **Health Monitoring**: Built-in status checks
- **Error Handling**: Graceful degradation
- **Documentation**: Each function documented

---

## Data Flow

### Query Execution Flow

```
User Input (Query Console)
         ↓
┌─────────────────────────┐
│  Client Component       │
│  (QueryConsole.tsx)     │
│  • Capture input        │
│  • Show loading state   │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  API Route              │
│  /api/query             │
│  • Validate input       │
│  • Call integration     │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Integration Layer      │
│  • Security check (🔴)  │
│  • Cognitive check (🩶) │
│  • Execute query (🟣)   │
│  • Record fitness (🟢)  │
│  • Store result (🟠)    │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Response               │
│  • QueryResult          │
│  • Attention weights    │
│  • Reasoning chain      │
│  • Constitutional logs  │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Client UI Update       │
│  • Display answer       │
│  • Show visualizations  │
│  • Update history       │
└─────────────────────────┘
```

### Data Fetching Strategy

**Server Components** (Preferred):
```typescript
// Fetch on server, no client bundle
export default async function Page() {
  const data = await ApiClient.getData();
  return <Component data={data} />;
}
```

**Client Components** (When needed):
```typescript
"use client";
export function Component() {
  useEffect(() => {
    fetch('/api/endpoint').then(setData);
  }, []);
}
```

**Streaming** (Future):
```typescript
// Server-Sent Events for real-time updates
const stream = await executeQueryStream(query);
for await (const chunk of stream) {
  // Progressive UI update
}
```

---

## State Management

### Strategy: Lift State Up + URL State

**No Global State Library** (Redux, Zustand) - Not needed because:
- Server Components handle data fetching
- URL manages navigation state
- Local state for UI interactions
- React Context for theme/settings

### State Locations

**URL State** (Primary):
```typescript
// /organisms/[id]/query?q=example
const searchParams = useSearchParams();
const query = searchParams.get('q');
```

**Local State**:
```typescript
// Component-specific UI state
const [isOpen, setIsOpen] = useState(false);
const [filter, setFilter] = useState('all');
```

**Lifted State**:
```typescript
// Parent manages, children consume
<Parent>
  <Child1 onChange={handleChange} />
  <Child2 value={value} />
</Parent>
```

**Context** (Minimal):
```typescript
// Theme, user preferences only
const { theme, setTheme } = useTheme();
```

---

## API Architecture

### Route Structure

```
/api
├── organisms/
│   ├── route.ts              # GET (list), POST (create)
│   └── [id]/
│       └── route.ts          # GET, PUT, DELETE
├── query/
│   └── route.ts              # POST (execute query)
└── stats/
    └── route.ts              # GET (system stats)
```

### API Pattern

```typescript
// /api/organisms/route.ts
export async function GET(request: Request) {
  try {
    // 1. Parse request
    const { searchParams } = new URL(request.url);

    // 2. Validate input
    if (!isValid(searchParams)) {
      return NextResponse.json({ error: 'Invalid' }, { status: 400 });
    }

    // 3. Call integration layer
    const data = await Integration.getData();

    // 4. Return response
    return NextResponse.json(data);

  } catch (error) {
    // 5. Error handling
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}
```

### Response Format

**Success**:
```json
{
  "data": { /* result */ },
  "meta": {
    "timestamp": "2025-10-10T...",
    "source": "roxo",
    "cost": 0.05
  }
}
```

**Error**:
```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": { /* context */ }
}
```

---

## Security Architecture

### Current (Development)

**No Authentication** - Internal tool
- File system storage (local)
- No sensitive data
- Network: localhost only

### Future (Production)

**Authentication**:
```
User → Login → JWT Token → API Requests (with token)
```

**Authorization** (RBAC from LARANJA):
```typescript
// Check permission before operation
const canQuery = await checkPermission(userId, 'query');
if (!canQuery) {
  throw new UnauthorizedError();
}
```

**Security Layers**:
1. **Input Validation**: Sanitize all inputs
2. **CSRF Protection**: Next.js built-in
3. **Rate Limiting**: Prevent abuse
4. **Audit Logging**: Track all operations
5. **Duress Detection**: Real-time via VERMELHO
6. **Manipulation Detection**: Real-time via CINZA

---

## Performance Architecture

### Optimization Strategies

**1. Server Components**:
- Reduce client bundle size
- Fetch data on server
- No client-side re-renders for static content

**2. Code Splitting**:
```typescript
// Lazy load heavy components
const HeavyChart = dynamic(() => import('./HeavyChart'), {
  ssr: false,
  loading: () => <Skeleton />
});
```

**3. Image Optimization**:
```typescript
import Image from 'next/image';
<Image src="/logo.png" width={200} height={100} alt="Logo" />
// Auto-optimized, lazy-loaded, responsive
```

**4. Caching Strategy**:

**Browser Cache**:
- Static assets: 1 year
- API responses: No-cache (real-time data)

**Server Cache** (Future):
```typescript
// Redis for frequently accessed data
const cached = await redis.get(`organism:${id}`);
if (cached) return cached;

const data = await fetchOrganism(id);
await redis.set(`organism:${id}`, data, 'EX', 300); // 5 min
```

**5. Database Performance** (LARANJA):
- O(1) queries: 67μs-1.23ms
- Consolidation optimizer
- Efficient indexing

### Performance Targets

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Page Load | <2s | ~1.5s | ✅ |
| UI Interaction | <100ms | ~50ms | ✅ |
| API Response | <500ms | ~200ms | ✅ |
| Query Execution | <30s | ~1s (mock) | ⏳ |
| Pattern Detection | <0.5ms | N/A | ⏳ |

---

## Deployment Architecture

### Development

```
Developer Machine
├── npm run dev          # Local dev server
├── Hot reload          # Fast refresh
└── Mock data           # Integration stubs
```

### Staging (Future)

```
Vercel/Cloud
├── Preview Deployments  # PR previews
├── Real integrations    # Connect to staging nodes
└── E2E tests           # Automated testing
```

### Production (Future)

```
Production Cluster
├── Load Balancer       # Distribute traffic
├── Next.js Instances   # Multiple servers
├── Redis Cache         # Shared cache
├── CDN                 # Static assets
└── Monitoring          # Datadog/Sentry
```

### Environment Flow

```
Development → Staging → Production
    ↓            ↓          ↓
  Local     Preview    Production
  Stubs     Staging    Real Nodes
          Real Data   Full Scale
```

---

## Design Patterns

### 1. **Adapter Pattern** (Integration Layer)

```typescript
// Abstraction over different node APIs
interface NodeAdapter {
  isAvailable(): boolean;
  getHealth(): Promise<Health>;
  // ...specific methods
}

class RoxoAdapter implements NodeAdapter {
  // ROXO-specific implementation
}
```

### 2. **Repository Pattern** (API Client)

```typescript
// Centralized data access
class OrganismRepository {
  async getById(id: string) { /* ... */ }
  async getAll() { /* ... */ }
  async create(data) { /* ... */ }
}
```

### 3. **Facade Pattern** (Integration Index)

```typescript
// Simple interface to complex subsystems
export {
  // 78 functions from 5 nodes
  executeQuery,      // ROXO
  getVersionHistory, // VERDE
  analyzeDuress,     // VERMELHO
  // ...
}
```

### 4. **Observer Pattern** (Health Monitoring)

```typescript
// Watch for node status changes
export async function checkAllNodesHealth() {
  // Poll each node
  // Update UI when status changes
}
```

### 5. **Strategy Pattern** (Conditional Logic)

```typescript
// Different behavior based on node availability
if (isRoxoAvailable()) {
  return await realImplementation();
} else {
  return mockData;
}
```

---

## Technology Choices

### Core Stack

| Technology | Why Chosen | Alternatives Considered |
|-----------|-----------|------------------------|
| **Next.js 14** | Server Components, built-in API, SSR/SSG, file-based routing | Create React App, Remix |
| **TypeScript** | Type safety, better DX, catches bugs early | JavaScript |
| **Tailwind CSS** | Utility-first, fast development, small bundle | CSS Modules, Styled Components |
| **shadcn/ui** | Copy-paste components, full control, no bloat | Material-UI, Chakra UI |
| **Recharts** | Declarative, React-friendly, TypeScript support | Chart.js, D3.js |

### Supporting Libraries

| Library | Purpose | Why |
|---------|---------|-----|
| **Radix UI** | Accessible primitives | ARIA compliant, unstyled |
| **Lucide React** | Icons | Consistent, tree-shakeable |
| **clsx** | Conditional classes | Simple, fast |
| **date-fns** | Date formatting | Lightweight vs moment.js |

### Development Tools

- **ESLint**: Code quality
- **Prettier**: Code formatting
- **TypeScript strict mode**: Maximum type safety

---

## Conclusion

The Chomsky DevTools Dashboard architecture is designed for:

✅ **Scalability**: Easy to add features, nodes, components
✅ **Maintainability**: Clear patterns, good separation of concerns
✅ **Performance**: Server Components, optimized rendering
✅ **Type Safety**: Strict TypeScript throughout
✅ **Integration Ready**: Plug-and-play architecture for 5 nodes
✅ **Developer Experience**: Clear APIs, good docs, helpful errors

**Current Status**: 95% Complete
**Missing**: Real API connections from 5 nodes
**Next Steps**: Enable integrations, E2E testing, production deployment

---

**Document Version**: 1.0.0
**Last Updated**: 2025-10-10
**Maintained by**: 🟡 AMARELO (DevTools Node)
