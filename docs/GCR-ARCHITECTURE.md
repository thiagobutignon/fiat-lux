# GCR - Grammar Container Runtime

**Status**: 🔄 Planning Phase
**Owner**: 🟣 ROXO
**Timeline**: 3-4 dias
**Started**: 2025-10-10

---

## 🎯 Mission

Build and run containers in **O(1) time** - deterministic, fast, glass-box container runtime.

**Why GCR?**
- Docker/Podman are O(n): image layers, dependency resolution, filesystem overlay
- GCR is O(1): content-addressable, hash-based, deterministic builds
- 100% transparent: every layer visible, auditable, explainable
- Zero black box magic

---

## 🏗️ Architecture

### Core Components

```
src/grammar-lang/tools/gcr/
├── types.ts           # Container specs, image metadata
├── builder.ts         # Build containers from .gcr specs
├── runtime.ts         # Execute containers with isolation
├── image.ts           # Image management (save/load/list)
├── isolation.ts       # Namespace/cgroup-like isolation
├── network.ts         # Container networking (port mapping)
├── storage.ts         # Volume management (content-addressable)
├── cli.ts             # gcr CLI (build/run/push/pull/ps/stop)
├── registry.ts        # Container registry (optional)
└── __tests__/         # Test suite
```

---

## 📋 Container Spec Format (.gcr)

**File**: `myapp.gcr`

```yaml
# Grammar Container Runtime Spec v1.0
format: gcr-v1.0
name: myapp
version: 1.0.0

# Base image (content-addressable hash)
base: sha256:abc123...  # OR 'scratch' for empty

# Build instructions (deterministic)
build:
  # Copy files (content-addressable)
  copy:
    - src: ./app/
      dest: /app/
      hash: sha256:def456...

  # Install dependencies (O(1) via GLM)
  dependencies:
    - name: "express"
      version: "4.18.0"
      hash: sha256:789abc...

  # Run build commands (deterministic)
  commands:
    - gsx build.gl  # Grammar script executor
    - glm install   # Grammar package manager

  # Environment variables
  env:
    NODE_ENV: production
    PORT: 3000

# Runtime configuration
runtime:
  # Entry point
  entrypoint: ["gsx", "server.gl"]

  # Working directory
  workdir: /app

  # User (security)
  user: appuser
  uid: 1000
  gid: 1000

  # Resource limits
  resources:
    memory: 512MB
    cpu: 1.0
    storage: 1GB

  # Exposed ports
  ports:
    - 3000/tcp
    - 8080/tcp

  # Volumes (persistent storage)
  volumes:
    - /app/data
    - /app/logs

  # Health check
  healthcheck:
    command: ["gsx", "health.gl"]
    interval: 30s
    timeout: 5s
    retries: 3

# Metadata
metadata:
  author: "user@example.com"
  description: "My Grammar Language App"
  tags:
    - web
    - api
    - production
  created: 2025-10-10T00:00:00Z
```

---

## 🚀 CLI Interface

### Build Container
```bash
# Build from .gcr spec
$ gcr build myapp.gcr

Building: myapp:1.0.0
├── Base image: scratch
├── Copying files: 247 files (hash verified)
├── Installing deps: 12 packages (O(1) via GLM)
├── Running commands: 2 commands
└── ✅ Built: sha256:xyz789... (150MB)

Image: myapp:1.0.0 (sha256:xyz789...)
```

### Run Container
```bash
# Run container
$ gcr run myapp:1.0.0 --name myapp-prod --port 3000:3000

Starting: myapp-prod (sha256:xyz789...)
├── Isolation: namespace created
├── Network: port 3000 → 3000
├── Storage: /app/data mounted
└── ✅ Running (PID: 12345)

Container: myapp-prod
URL: http://localhost:3000
```

### List Containers
```bash
# List running containers
$ gcr ps

CONTAINER ID  IMAGE         NAME         STATUS      PORTS
abc123...     myapp:1.0.0   myapp-prod   running     3000→3000
def456...     db:2.0.0      mydb         running     5432→5432
```

### Stop Container
```bash
# Stop container
$ gcr stop myapp-prod

Stopping: myapp-prod
├── Sending SIGTERM
├── Waiting 10s for graceful shutdown
└── ✅ Stopped

Container: myapp-prod (stopped)
```

### Image Management
```bash
# List images
$ gcr images

REPOSITORY  TAG     IMAGE ID      SIZE    CREATED
myapp       1.0.0   xyz789...     150MB   2 hours ago
db          2.0.0   abc123...     300MB   1 day ago

# Remove image
$ gcr rmi myapp:1.0.0

# Pull from registry
$ gcr pull registry.example.com/myapp:1.0.0

# Push to registry
$ gcr push myapp:1.0.0 registry.example.com/myapp:1.0.0
```

---

## 🔐 Isolation & Security

### Namespace Isolation
```typescript
interface ContainerIsolation {
  // Process isolation
  pid_namespace: boolean;      // Separate process tree

  // Network isolation
  net_namespace: boolean;      // Separate network stack

  // Filesystem isolation
  mount_namespace: boolean;    // Separate filesystem view

  // User isolation
  user_namespace: boolean;     // Separate user/group IDs

  // IPC isolation
  ipc_namespace: boolean;      // Separate IPC mechanisms

  // Resource limits (cgroup-like)
  resource_limits: {
    memory: string;            // e.g., "512MB"
    cpu: number;               // e.g., 1.0 (1 core)
    storage: string;           // e.g., "1GB"
  };

  // Capabilities (privileges)
  capabilities: string[];      // e.g., ["NET_BIND_SERVICE"]
}
```

**Implementation**:
- Use Node.js `child_process` with custom isolation
- Leverage OS-level namespaces (Linux) or equivalent
- Fallback to chroot-like mechanisms on other OSes
- Constitutional AI enforcement for security policies

---

## 💾 Image Format

### Content-Addressable Layers

**Directory Structure**:
```
.gcr/
├── images/
│   └── sha256:<hash>/
│       ├── manifest.json    # Image metadata
│       ├── config.json      # Container config
│       └── layers/
│           ├── layer1.tar.gz
│           ├── layer2.tar.gz
│           └── layer3.tar.gz
├── containers/
│   └── <container_id>/
│       ├── config.json      # Runtime config
│       ├── state.json       # Container state
│       └── rootfs/          # Container filesystem
└── volumes/
    └── <volume_id>/         # Persistent volumes
```

**Manifest Format**:
```json
{
  "format": "gcr-v1.0",
  "name": "myapp",
  "version": "1.0.0",
  "hash": "sha256:xyz789...",
  "size": 157286400,
  "layers": [
    {
      "hash": "sha256:layer1...",
      "size": 52428800,
      "type": "base"
    },
    {
      "hash": "sha256:layer2...",
      "size": 104857600,
      "type": "app"
    }
  ],
  "config": {
    "entrypoint": ["gsx", "server.gl"],
    "env": { "PORT": "3000" },
    "ports": [3000],
    "volumes": ["/app/data"]
  },
  "metadata": {
    "author": "user@example.com",
    "created": "2025-10-10T00:00:00Z",
    "tags": ["web", "api"]
  }
}
```

---

## ⚡ O(1) Guarantees

### Build System
- ✅ **Hash-based layer lookup**: O(1)
- ✅ **Dependency resolution**: O(1) via GLM
- ✅ **File copying**: O(k) where k = file count (parallelizable)
- ✅ **Layer caching**: O(1) hash lookup

### Runtime
- ✅ **Container lookup**: O(1) via hash table
- ✅ **Process spawn**: O(1) (single fork)
- ✅ **Network setup**: O(1) (pre-allocated ports)
- ✅ **Volume mount**: O(1) (direct mount)

### Image Management
- ✅ **Image lookup**: O(1) hash-based
- ✅ **Image list**: O(n) where n = images (but indexed)
- ✅ **Image pull/push**: O(k) where k = layer count

---

## 🧪 Testing Strategy

### Unit Tests
```typescript
// builder.test.ts
describe('GCRBuilder', () => {
  it('should build container from .gcr spec', async () => {
    const builder = new GCRBuilder();
    const image = await builder.build('test.gcr');
    expect(image.hash).toMatch(/^sha256:/);
    expect(image.layers.length).toBeGreaterThan(0);
  });

  it('should cache layers (O(1) rebuild)', async () => {
    const builder = new GCRBuilder();
    const image1 = await builder.build('test.gcr');
    const image2 = await builder.build('test.gcr');
    expect(image1.hash).toBe(image2.hash); // Same hash = cached
  });
});

// runtime.test.ts
describe('GCRRuntime', () => {
  it('should run container with isolation', async () => {
    const runtime = new GCRRuntime();
    const container = await runtime.run('test-image', {
      name: 'test-container',
      ports: { 3000: 3000 }
    });
    expect(container.status).toBe('running');
    expect(container.pid).toBeGreaterThan(0);
    await runtime.stop(container.id);
  });
});
```

### Integration Tests
```typescript
// e2e.test.ts
describe('GCR E2E', () => {
  it('should build → run → stop workflow', async () => {
    // Build
    const image = await gcr.build('myapp.gcr');

    // Run
    const container = await gcr.run(image.hash, {
      name: 'myapp-test',
      ports: { 3000: 3000 }
    });

    // Verify
    const response = await fetch('http://localhost:3000/health');
    expect(response.ok).toBe(true);

    // Stop
    await gcr.stop(container.id);
    expect(container.status).toBe('stopped');
  });
});
```

---

## 📊 Performance Targets

| Operation | Target | Docker | GCR |
|-----------|--------|--------|-----|
| Build (cached) | <100ms | ~5s | ✅ <100ms |
| Build (cold) | <5s | ~30s | ✅ <5s |
| Container start | <50ms | ~500ms | ✅ <50ms |
| Container stop | <100ms | ~2s | ✅ <100ms |
| Image lookup | <1ms | ~10ms | ✅ <1ms |
| Image list | <10ms | ~100ms | ✅ <10ms |

---

## 🎯 Deliverables (3-4 dias)

### DIA 1: Container Spec + Types (Foundation)
**Files**: types.ts, spec-parser.ts, cli.ts (skeleton)
**LOC**: ~600 linhas

**Deliverables**:
- ✅ .gcr format definition
- ✅ TypeScript types (ContainerSpec, Image, Container, etc.)
- ✅ Spec parser (.gcr → typed objects)
- ✅ Validation (schema checking)
- ✅ CLI skeleton (gcr command structure)

**Tests**:
- Parse valid .gcr files
- Validate required fields
- Reject invalid specs

---

### DIA 2: Build System (O(1) Build)
**Files**: builder.ts, layers.ts, cache.ts
**LOC**: ~800 linhas

**Deliverables**:
- ✅ GCRBuilder class
- ✅ Layer creation (content-addressable)
- ✅ Dependency resolution (via GLM)
- ✅ Build cache (hash-based, O(1))
- ✅ Image manifest generation
- ✅ CLI: `gcr build`

**Tests**:
- Build from scratch
- Build with base image
- Cache hit/miss
- Layer deduplication

---

### DIA 3: Runtime Engine + Isolation
**Files**: runtime.ts, isolation.ts, process.ts
**LOC**: ~900 linhas

**Deliverables**:
- ✅ GCRRuntime class
- ✅ Container execution (spawn process)
- ✅ Namespace isolation (pid, net, mount, user, ipc)
- ✅ Resource limits (memory, cpu, storage)
- ✅ Container lifecycle (start/stop/restart)
- ✅ CLI: `gcr run`, `gcr ps`, `gcr stop`

**Tests**:
- Run container
- Isolation verification
- Resource limits
- Graceful shutdown

---

### DIA 4: Image Management + Networking + Demo
**Files**: image.ts, network.ts, storage.ts, demo.ts
**LOC**: ~700 linhas

**Deliverables**:
- ✅ Image storage (save/load/remove)
- ✅ Image listing (with metadata)
- ✅ Container networking (port mapping)
- ✅ Volume management (persistent storage)
- ✅ CLI: `gcr images`, `gcr rmi`, `gcr pull`, `gcr push`
- ✅ Complete E2E demo
- ✅ Documentation (README, examples)

**Tests**:
- Image CRUD operations
- Network connectivity
- Volume persistence
- Full workflow

---

## 🏁 Success Criteria

- ✅ Build containers in <5s (cold) / <100ms (cached)
- ✅ Run containers in <50ms
- ✅ O(1) image lookup
- ✅ 100% deterministic builds (same input = same hash)
- ✅ Full isolation (processes, network, filesystem)
- ✅ Constitutional AI integration
- ✅ Glass box (every layer visible and auditable)
- ✅ CLI matches Docker UX (familiar commands)
- ✅ E2E demo working (build → run → stop)

---

## 🔗 Integration Points

### GLM (Grammar Language Manager)
- Dependency installation: `glm install`
- Package resolution: O(1) hash-based

### GSX (Grammar Script eXecutor)
- Entrypoint execution: `gsx server.gl`
- Build commands: `gsx build.gl`

### Constitutional AI
- Build-time validation (malicious code detection)
- Runtime enforcement (resource limits, capabilities)
- Audit trail (who built what, when)

### .sqlo Database
- Image metadata storage
- Container state persistence
- Audit logs

---

## 📝 Notes

**Why not just use Docker?**
- Docker is O(n): layers, overlayfs, dependency resolution
- Docker is opaque: complex internals, hard to audit
- Docker is non-deterministic: timestamps, randomness
- GCR is O(1): hash-based, deterministic, glass-box

**Platform Support**:
- Linux: Full namespace support (pid, net, mount, user, ipc)
- macOS: Limited (fallback to chroot-like isolation)
- Windows: WSL2 or native Windows containers
- Target: Cross-platform with graceful degradation

**Registry Support** (Optional - DIA 4+):
- Content-addressable registry
- O(1) push/pull (hash-based)
- Deduplication (shared layers)
- RBAC via .sqlo

---

**Status**: 🔄 Planning Complete - Ready to Implement!
**Next**: DIA 1 - Container Spec + Types
**Owner**: 🟣 ROXO
**Timeline**: Starting now

---

_Created: 2025-10-10_
_Author: ROXO (Purple Node)_
_Version: 1.0_
