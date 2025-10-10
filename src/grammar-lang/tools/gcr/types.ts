/**
 * GCR (Grammar Container Runtime) - Type Definitions
 *
 * O(1) container runtime with glass-box transparency.
 * Content-addressable, deterministic, auditable.
 */

// ============================================================================
// Container Spec (.gcr file format)
// ============================================================================

export interface ContainerSpec {
  // Format version
  format: 'gcr-v1.0';

  // Container identity
  name: string;
  version: string;

  // Base image (content-addressable hash or 'scratch')
  base: string; // e.g., 'sha256:abc123...' or 'scratch'

  // Build configuration
  build: BuildConfig;

  // Runtime configuration
  runtime: RuntimeConfig;

  // Metadata
  metadata: ContainerMetadata;
}

// ============================================================================
// Build Configuration
// ============================================================================

export interface BuildConfig {
  // Copy files (content-addressable)
  copy?: CopyInstruction[];

  // Dependencies (via GLM)
  dependencies?: Dependency[];

  // Build commands (deterministic)
  commands?: string[];

  // Environment variables (build-time)
  env?: Record<string, string>;
}

export interface CopyInstruction {
  src: string;  // Source path
  dest: string; // Destination path in container
  hash?: string; // Content hash (for verification)
}

export interface Dependency {
  name: string;
  version: string;
  hash: string; // Package content hash
}

// ============================================================================
// Runtime Configuration
// ============================================================================

export interface RuntimeConfig {
  // Entry point
  entrypoint: string[]; // e.g., ["gsx", "server.gl"]

  // Working directory
  workdir?: string; // e.g., "/app"

  // User (security)
  user?: string; // e.g., "appuser"
  uid?: number;  // e.g., 1000
  gid?: number;  // e.g., 1000

  // Resource limits
  resources?: ResourceLimits;

  // Exposed ports
  ports?: PortSpec[];

  // Volumes (persistent storage)
  volumes?: string[];

  // Health check
  healthcheck?: HealthCheck;

  // Environment variables (runtime)
  env?: Record<string, string>;
}

export interface ResourceLimits {
  memory?: string;  // e.g., "512MB", "1GB"
  cpu?: number;     // e.g., 1.0 (1 core)
  storage?: string; // e.g., "1GB", "10GB"
  gpu?: number | number[]; // e.g., 0 (single GPU) or [0, 1] (multiple GPUs)
}

export type PortSpec = `${number}/${'tcp' | 'udp'}`;

export interface HealthCheck {
  command: string[];   // e.g., ["gsx", "health.gl"]
  interval?: string;   // e.g., "30s"
  timeout?: string;    // e.g., "5s"
  retries?: number;    // e.g., 3
  startPeriod?: string; // e.g., "60s"
}

// ============================================================================
// Container Metadata
// ============================================================================

export interface ContainerMetadata {
  author?: string;
  description?: string;
  tags?: string[];
  created?: string; // ISO timestamp
  license?: string;
  homepage?: string;
  repository?: string;
}

// ============================================================================
// Image (Built Container)
// ============================================================================

export interface ContainerImage {
  // Image identity
  hash: string; // Content-addressable hash (sha256:...)
  name: string;
  version: string;

  // Image size
  size: number; // bytes

  // Layers (content-addressable)
  layers: ImageLayer[];

  // Configuration (from spec)
  config: RuntimeConfig;

  // Metadata
  metadata: ContainerMetadata & {
    buildTime: string; // ISO timestamp
    builder: string;   // Who built it
  };

  // Manifest (for storage)
  manifest: ImageManifest;
}

export interface ImageLayer {
  hash: string;     // Layer content hash
  size: number;     // Layer size in bytes
  type: LayerType;  // Layer purpose
  created: string;  // ISO timestamp
}

export type LayerType = 'base' | 'dependencies' | 'app' | 'config' | 'metadata';

export interface ImageManifest {
  format: 'gcr-v1.0';
  name: string;
  version: string;
  hash: string;
  size: number;
  layers: ImageLayer[];
  config: RuntimeConfig;
  metadata: ContainerMetadata;
}

// ============================================================================
// Container (Running Instance)
// ============================================================================

export interface Container {
  // Container identity
  id: string;       // Container ID (short hash)
  name: string;     // User-provided name
  image: string;    // Image name:version (e.g., "webserver:1.0.0")
  imageHash: string; // Image this container runs

  // Runtime state
  status: ContainerStatus;
  pid?: number;     // Process ID (if running)
  exitCode?: number; // Exit code (if stopped/exited)

  // Creation time
  created: string;  // ISO timestamp
  started?: string; // ISO timestamp
  finished?: string; // ISO timestamp (when container stopped/exited)

  // Runtime configuration (from image + run options)
  config: RuntimeConfig;

  // Isolation
  isolation: ContainerIsolation;

  // Networking
  network: ContainerNetwork;

  // Storage
  storage: ContainerStorage;

  // Logs
  logs: {
    stdout: string; // Path to stdout log file
    stderr: string; // Path to stderr log file
  };

  // Stats
  stats?: ContainerStats;
}

export type ContainerStatus =
  | 'created'   // Container created but not started
  | 'starting'  // Container is starting
  | 'running'   // Container is running
  | 'stopping'  // Container is stopping
  | 'stopped'   // Container stopped gracefully
  | 'exited'    // Container exited (may have failed)
  | 'paused'    // Container paused
  | 'error';    // Container in error state

export interface ContainerRuntimeConfig {
  // Port mappings
  ports?: PortMapping[];

  // Volume mounts
  volumes?: VolumeMount[];

  // Environment variables (overrides)
  env?: Record<string, string>;

  // Resource limits (overrides)
  resources?: ResourceLimits;

  // Restart policy
  restart?: RestartPolicy;
}

export interface PortMapping {
  host: number;      // Host port
  container: number; // Container port
  protocol: 'tcp' | 'udp';
}

export interface VolumeMount {
  name: string;      // Volume name or path
  mountPath: string; // Path in container
  readonly?: boolean;
}

export type RestartPolicy = 'no' | 'always' | 'on-failure' | 'unless-stopped';

// ============================================================================
// Container Isolation
// ============================================================================

export interface ContainerIsolation {
  // Namespace isolation
  pid_namespace: boolean;   // Separate process tree
  net_namespace: boolean;   // Separate network stack
  mount_namespace: boolean; // Separate filesystem view
  user_namespace: boolean;  // Separate user/group IDs
  ipc_namespace: boolean;   // Separate IPC mechanisms

  // Resource limits (cgroup-like)
  resource_limits: ResourceLimits;

  // Capabilities (privileges)
  capabilities?: string[]; // e.g., ["NET_BIND_SERVICE"]

  // Security options
  securityOpt?: string[];

  // Read-only root filesystem
  readonlyRootfs?: boolean;
}

// ============================================================================
// Container Networking
// ============================================================================

export interface ContainerNetwork {
  // Network mode
  mode: NetworkMode;

  // IP address
  ipAddress?: string;

  // Port mappings
  ports: PortMapping[];

  // Hostname
  hostname?: string;

  // DNS
  dns?: string[];
}

export type NetworkMode = 'bridge' | 'host' | 'none' | 'container';

// ============================================================================
// Container Storage
// ============================================================================

export interface ContainerStorage {
  // Root filesystem
  rootfs: string; // Path to rootfs

  // Volumes
  volumes: VolumeMount[];

  // Storage driver
  driver: StorageDriver;

  // Storage size
  size?: number; // bytes
}

export type StorageDriver = 'overlay2' | 'vfs' | 'btrfs' | 'zfs' | 'content-addressable';

// ============================================================================
// Container Stats
// ============================================================================

export interface ContainerStats {
  // CPU
  cpu: {
    usage: number;      // Percentage (0-100)
    throttling: number; // Times throttled
  };

  // Memory
  memory: {
    usage: number;      // Bytes used
    limit: number;      // Bytes limit
    percentage: number; // Percentage (0-100)
  };

  // Network
  network: {
    rx_bytes: number;   // Bytes received
    tx_bytes: number;   // Bytes transmitted
  };

  // Disk I/O
  disk: {
    read_bytes: number;
    write_bytes: number;
  };

  // Timestamp
  timestamp: string; // ISO timestamp
}

// ============================================================================
// Build Context
// ============================================================================

export interface BuildContext {
  // Build directory
  contextPath: string;

  // .gcr spec file
  specPath: string;

  // Parsed spec
  spec: ContainerSpec;

  // Build options
  options: BuildOptions;

  // Build state
  state: BuildState;
}

export interface BuildOptions {
  // Cache control
  noCache?: boolean;     // Disable layer cache
  pull?: boolean;        // Always pull base image

  // Output control
  quiet?: boolean;       // Suppress build output
  verbose?: boolean;     // Show detailed output

  // Tagging
  tags?: string[];       // Additional tags

  // Build-time args
  buildArgs?: Record<string, string>;

  // Platform
  platform?: string;     // e.g., "linux/amd64"
}

export interface BuildState {
  // Current step
  currentStep: number;
  totalSteps: number;

  // Current layer
  currentLayer?: string;

  // Progress
  progress: number; // 0-100

  // Status
  status: BuildStatus;

  // Errors
  errors: BuildError[];
}

export type BuildStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface BuildError {
  step: number;
  message: string;
  timestamp: string;
}

// ============================================================================
// Registry
// ============================================================================

export interface ContainerRegistry {
  // Registry URL
  url: string;

  // Authentication
  auth?: RegistryAuth;

  // Images in registry
  images: RegistryImage[];
}

export interface RegistryAuth {
  username: string;
  password: string;
  email?: string;
}

export interface RegistryImage {
  name: string;
  tags: string[];
  hash: string;
  size: number;
  pushed: string; // ISO timestamp
}

// ============================================================================
// CLI Command Types
// ============================================================================

export interface GCRCommand {
  command: 'build' | 'run' | 'ps' | 'stop' | 'images' | 'rmi' | 'pull' | 'push' | 'exec' | 'logs';
  args: string[];
  options: Record<string, any>;
}

export interface BuildCommand extends GCRCommand {
  command: 'build';
  specPath: string;
  options: BuildOptions;
}

export interface RunCommand extends GCRCommand {
  command: 'run';
  image: string; // Image hash or name:tag
  options: {
    name?: string;
    ports?: PortMapping[];
    volumes?: VolumeMount[];
    env?: Record<string, string>;
    detach?: boolean;
    rm?: boolean; // Remove on exit
    restart?: RestartPolicy;
  };
}

export interface StopCommand extends GCRCommand {
  command: 'stop';
  container: string; // Container ID or name
  options: {
    time?: number; // Seconds to wait before killing
  };
}

export interface ExecCommand extends GCRCommand {
  command: 'exec';
  container: string; // Container ID or name
  cmd: string[];     // Command to execute
  options: {
    interactive?: boolean;
    tty?: boolean;
    user?: string;
    env?: Record<string, string>;
  };
}

// ============================================================================
// Events
// ============================================================================

export interface ContainerEvent {
  type: ContainerEventType;
  container: string; // Container ID
  image?: string;    // Image hash
  timestamp: string; // ISO timestamp
  metadata?: Record<string, any>;
}

export type ContainerEventType =
  | 'create'
  | 'start'
  | 'stop'
  | 'pause'
  | 'unpause'
  | 'die'
  | 'kill'
  | 'restart'
  | 'health_status'
  | 'exec_create'
  | 'exec_start'
  | 'exec_die';

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Type guards
 */
export function isContainerRunning(container: Container): boolean {
  return container.status === 'running';
}

export function isContainerStopped(container: Container): boolean {
  return container.status === 'stopped' || container.status === 'exited';
}

export function isImageHash(str: string): boolean {
  return /^sha256:[a-f0-9]{64}$/i.test(str);
}

export function isValidContainerName(name: string): boolean {
  return /^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/.test(name);
}

/**
 * Utility types
 */
export type PartialContainerSpec = Partial<ContainerSpec>;
export type ContainerFilter = Partial<{
  status: ContainerStatus;
  name: string;
  image: string;
}>;
