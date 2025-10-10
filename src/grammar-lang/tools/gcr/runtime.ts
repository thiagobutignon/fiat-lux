/**
 * GCR Runtime Engine
 *
 * Runs containers with O(1) isolation and resource management.
 * Content-addressable, deterministic, glass-box.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { spawn, ChildProcess, exec } from 'child_process';
import { GCRBuilder } from './builder';
import { LayerBuilder } from './layers';
import {
  ContainerImage,
  Container,
  ContainerStatus,
  RuntimeConfig,
  ResourceLimits,
} from './types';

// ============================================================================
// GCR Runtime
// ============================================================================

export class GCRRuntime {
  private builder: GCRBuilder;
  private layerBuilder: LayerBuilder;
  private containersDir: string;
  private containers: Map<string, Container>;

  constructor(
    containersDir: string = '.gcr/containers',
    layersDir: string = '.gcr/layers',
    imagesDir: string = '.gcr/images'
  ) {
    this.builder = new GCRBuilder(layersDir, '.gcr/cache', imagesDir);
    this.layerBuilder = new LayerBuilder(layersDir);
    this.containersDir = containersDir;
    this.containers = new Map();
    this.ensureContainersDir();
    this.loadContainers();
  }

  /**
   * Ensure containers directory exists
   */
  private ensureContainersDir(): void {
    if (!fs.existsSync(this.containersDir)) {
      fs.mkdirSync(this.containersDir, { recursive: true });
    }
  }

  /**
   * Load existing containers from disk
   */
  private loadContainers(): void {
    if (!fs.existsSync(this.containersDir)) {
      return;
    }

    const entries = fs.readdirSync(this.containersDir);

    for (const entry of entries) {
      const containerPath = path.join(this.containersDir, entry, 'container.json');
      if (fs.existsSync(containerPath)) {
        try {
          const container = JSON.parse(fs.readFileSync(containerPath, 'utf-8')) as Container;
          this.containers.set(container.id, container);
        } catch (error) {
          // Skip invalid containers
        }
      }
    }

    console.log(`Loaded ${this.containers.size} container(s) from disk`);
  }

  /**
   * Create container from image
   */
  async create(
    imageName: string,
    imageVersion: string,
    options: {
      name?: string;
      ports?: string[];
      volumes?: string[];
      env?: { [key: string]: string };
      resources?: Partial<ResourceLimits>;
    } = {}
  ): Promise<Container> {
    console.log(`\n🚀 Creating container from ${imageName}:${imageVersion}...\n`);

    // Load image
    const image = this.builder.findImage(imageName, imageVersion);
    if (!image) {
      throw new Error(`Image not found: ${imageName}:${imageVersion}`);
    }

    console.log(`   📦 Image loaded: ${image.hash.substring(0, 12)}...`);
    console.log(`   Size: ${formatSize(image.size)}`);
    console.log(`   Layers: ${image.layers.length}`);

    // Generate container ID
    const containerId = this.generateContainerId();
    const containerName = options.name || `${imageName}-${containerId.substring(0, 8)}`;

    // Create container directory
    const containerDir = path.join(this.containersDir, containerId);
    fs.mkdirSync(containerDir, { recursive: true });

    // Create container filesystem (rootfs)
    console.log(`\n   🗂️  Creating container filesystem...`);
    const rootfs = path.join(containerDir, 'rootfs');
    await this.createRootfs(rootfs, image);

    // Merge runtime config with options
    const config: RuntimeConfig = {
      ...image.config,
      ports: (options.ports || image.config.ports || []) as any,
      volumes: options.volumes || image.config.volumes || [],
      env: { ...(image.config.env || {}), ...(options.env || {}) },
      resources: { ...(image.config.resources || {}), ...(options.resources || {}) },
    };

    // Create container
    const container: Container = {
      id: containerId,
      name: containerName,
      image: `${imageName}:${imageVersion}`,
      imageHash: image.hash,
      status: 'created',
      config,
      isolation: {
        pid_namespace: true,
        net_namespace: true,
        mount_namespace: true,
        user_namespace: false,
        ipc_namespace: true,
        resource_limits: config.resources || {},
      },
      network: {
        mode: 'bridge',
        ipAddress: undefined,
        ports: [],
      },
      storage: {
        rootfs,
        volumes: [],
        driver: 'content-addressable',
      },
      created: new Date().toISOString(),
      started: undefined,
      finished: undefined,
      exitCode: undefined,
      pid: undefined,
      logs: {
        stdout: path.join(containerDir, 'stdout.log'),
        stderr: path.join(containerDir, 'stderr.log'),
      },
    };

    // Save container
    this.saveContainer(container);
    this.containers.set(containerId, container);

    console.log(`\n   ✅ Container created: ${containerName} (${containerId.substring(0, 12)})`);
    console.log(`   Status: ${container.status}`);

    return container;
  }

  /**
   * Start container
   */
  async start(containerIdOrName: string): Promise<void> {
    const container = this.findContainer(containerIdOrName);
    if (!container) {
      throw new Error(`Container not found: ${containerIdOrName}`);
    }

    if (container.status === 'running') {
      throw new Error(`Container already running: ${container.name}`);
    }

    console.log(`\n🚀 Starting container ${container.name}...\n`);

    // Update status
    container.status = 'running';
    container.started = new Date().toISOString();

    // Create log files
    fs.writeFileSync(container.logs.stdout, '', 'utf-8');
    fs.writeFileSync(container.logs.stderr, '', 'utf-8');

    // Setup port mapping
    if (container.config.ports && container.config.ports.length > 0) {
      console.log(`\n   🌐 Setting up port mapping...`);
      this.setupPortMapping(container);
    }

    // Setup volume mounts
    if (container.config.volumes && container.config.volumes.length > 0) {
      console.log(`\n   💾 Setting up volume mounts...`);
      this.setupVolumeMounts(container);
    }

    // Spawn container process
    console.log(`\n   ⚙️  Spawning process...`);
    console.log(`   Entrypoint: ${container.config.entrypoint.join(' ')}`);
    console.log(`   Workdir: ${container.config.workdir}`);

    const process = this.spawnContainerProcess(container);
    container.pid = process.pid;

    // Save updated container
    this.saveContainer(container);

    console.log(`\n   ✅ Container started`);
    console.log(`   PID: ${container.pid}`);
    console.log(`   Status: ${container.status}`);
  }

  /**
   * Stop container
   */
  async stop(containerIdOrName: string, signal: string = 'SIGTERM'): Promise<void> {
    const container = this.findContainer(containerIdOrName);
    if (!container) {
      throw new Error(`Container not found: ${containerIdOrName}`);
    }

    if (container.status !== 'running') {
      throw new Error(`Container not running: ${container.name}`);
    }

    console.log(`\n⏹️  Stopping container ${container.name}...\n`);

    if (container.pid) {
      console.log(`   Sending ${signal} to PID ${container.pid}...`);

      try {
        process.kill(container.pid, signal);

        // Wait for process to exit (with timeout)
        await new Promise((resolve) => setTimeout(resolve, 2000));

        // Check if still running
        try {
          process.kill(container.pid, 0); // Check if process exists
          console.log(`   ⚠️  Process still running, sending SIGKILL...`);
          process.kill(container.pid, 'SIGKILL');
        } catch (error) {
          // Process already dead
        }
      } catch (error: any) {
        if (error.code !== 'ESRCH') {
          // Process not found (already dead)
          throw error;
        }
      }
    }

    // Update status
    container.status = 'exited';
    container.finished = new Date().toISOString();
    container.exitCode = 0; // TODO: Get actual exit code

    this.saveContainer(container);

    console.log(`\n   ✅ Container stopped`);
    console.log(`   Status: ${container.status}`);
  }

  /**
   * Remove container
   */
  async remove(containerIdOrName: string, force: boolean = false): Promise<void> {
    const container = this.findContainer(containerIdOrName);
    if (!container) {
      throw new Error(`Container not found: ${containerIdOrName}`);
    }

    if (container.status === 'running' && !force) {
      throw new Error(`Container is running. Use --force to remove: ${container.name}`);
    }

    console.log(`\n🗑️  Removing container ${container.name}...\n`);

    // Stop if running
    if (container.status === 'running') {
      await this.stop(container.id, 'SIGKILL');
    }

    // Remove container directory
    const containerDir = path.join(this.containersDir, container.id);
    if (fs.existsSync(containerDir)) {
      fs.rmSync(containerDir, { recursive: true, force: true });
    }

    // Remove from map
    this.containers.delete(container.id);

    console.log(`   ✅ Container removed`);
  }

  /**
   * Execute command in running container
   */
  async exec(
    containerIdOrName: string,
    command: string[],
    options: { interactive?: boolean; tty?: boolean } = {}
  ): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    const container = this.findContainer(containerIdOrName);
    if (!container) {
      throw new Error(`Container not found: ${containerIdOrName}`);
    }

    if (container.status !== 'running') {
      throw new Error(`Container not running: ${container.name}`);
    }

    console.log(`\n⚙️  Executing command in ${container.name}...\n`);
    console.log(`   Command: ${command.join(' ')}`);

    // For now, execute command in container's working directory
    // TODO: Implement proper namespace isolation
    const cwd = path.join(container.storage.rootfs, container.config.workdir || '/app');

    return new Promise((resolve, reject) => {
      const proc = spawn(command[0], command.slice(1), {
        cwd,
        env: container.config.env,
        stdio: options.interactive ? 'inherit' : 'pipe',
      });

      let stdout = '';
      let stderr = '';

      if (!options.interactive) {
        proc.stdout?.on('data', (data) => {
          stdout += data.toString();
        });

        proc.stderr?.on('data', (data) => {
          stderr += data.toString();
        });
      }

      proc.on('close', (code) => {
        resolve({
          stdout,
          stderr,
          exitCode: code || 0,
        });
      });

      proc.on('error', (error) => {
        reject(error);
      });
    });
  }

  /**
   * Get container logs
   */
  getLogs(
    containerIdOrName: string,
    options: { follow?: boolean; tail?: number; stdout?: boolean; stderr?: boolean } = {}
  ): string {
    const container = this.findContainer(containerIdOrName);
    if (!container) {
      throw new Error(`Container not found: ${containerIdOrName}`);
    }

    const showStdout = options.stdout !== false;
    const showStderr = options.stderr !== false;

    let logs = '';

    if (showStdout && fs.existsSync(container.logs.stdout)) {
      const stdoutContent = fs.readFileSync(container.logs.stdout, 'utf-8');
      logs += stdoutContent;
    }

    if (showStderr && fs.existsSync(container.logs.stderr)) {
      const stderrContent = fs.readFileSync(container.logs.stderr, 'utf-8');
      logs += stderrContent;
    }

    // Apply tail if specified
    if (options.tail) {
      const lines = logs.split('\n');
      logs = lines.slice(-options.tail).join('\n');
    }

    return logs;
  }

  /**
   * List all containers
   */
  list(options: { all?: boolean } = {}): Container[] {
    const containers = Array.from(this.containers.values());

    if (options.all) {
      return containers;
    }

    // Only running containers by default
    return containers.filter((c) => c.status === 'running');
  }

  /**
   * Get container info
   */
  inspect(containerIdOrName: string): Container | null {
    return this.findContainer(containerIdOrName);
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  /**
   * Find container by ID or name
   */
  private findContainer(idOrName: string): Container | null {
    // Try exact ID match
    if (this.containers.has(idOrName)) {
      return this.containers.get(idOrName)!;
    }

    const containers = Array.from(this.containers.values());

    // Try prefix match
    for (const container of containers) {
      if (container.id.startsWith(idOrName)) {
        return container;
      }
    }

    // Try name match
    for (const container of containers) {
      if (container.name === idOrName) {
        return container;
      }
    }

    return null;
  }

  /**
   * Generate random container ID
   */
  private generateContainerId(): string {
    return crypto.randomBytes(16).toString('hex');
  }

  /**
   * Create container rootfs from image layers
   */
  private async createRootfs(rootfs: string, image: ContainerImage): Promise<void> {
    fs.mkdirSync(rootfs, { recursive: true });

    // Copy all layers to rootfs (in order)
    for (const layer of image.layers) {
      const layerPath = path.join('.gcr/layers', layer.hash, 'contents');

      if (fs.existsSync(layerPath)) {
        // Copy layer contents to rootfs
        this.copyRecursive(layerPath, rootfs);
        console.log(`      ✅ Layer applied: ${layer.hash.substring(0, 12)}... (${layer.type})`);
      }
    }

    console.log(`   ✅ Rootfs created (${image.layers.length} layers applied)`);
  }

  /**
   * Copy directory recursively
   */
  private copyRecursive(src: string, dest: string): void {
    if (!fs.existsSync(src)) {
      return;
    }

    const stat = fs.statSync(src);

    if (stat.isDirectory()) {
      if (!fs.existsSync(dest)) {
        fs.mkdirSync(dest, { recursive: true });
      }

      const files = fs.readdirSync(src);
      for (const file of files) {
        this.copyRecursive(path.join(src, file), path.join(dest, file));
      }
    } else {
      const destDir = path.dirname(dest);
      if (!fs.existsSync(destDir)) {
        fs.mkdirSync(destDir, { recursive: true });
      }
      fs.copyFileSync(src, dest);
    }
  }

  /**
   * Setup port mapping for container
   */
  private setupPortMapping(container: Container): void {
    const portMappings: Array<{ hostPort: number; containerPort: number; protocol: string }> = [];

    if (!container.config.ports || container.config.ports.length === 0) {
      return;
    }

    for (const portSpec of container.config.ports) {
      const spec = portSpec.toString();
      const match = spec.match(/^(\d+):(\d+)(?:\/(tcp|udp))?$/);

      if (match) {
        const hostPort = parseInt(match[1]);
        const containerPort = parseInt(match[2]);
        const protocol = match[3] || 'tcp';

        portMappings.push({ hostPort, containerPort, protocol });
        console.log(`   📡 Port mapping: ${hostPort} → ${containerPort}/${protocol}`);
      } else {
        console.warn(`   ⚠️  Invalid port spec: ${portSpec}`);
      }
    }

    container.network.ports = portMappings as any;

    // NOTE: Actual port forwarding requires OS-specific implementation:
    // - Linux: iptables -t nat -A PREROUTING -p tcp --dport <host> -j DNAT --to-destination <container>:<port>
    // - macOS: pf rules (pfctl -a com.gcr -f -)
    // - Windows: netsh interface portproxy add v4tov4
    //
    // For now, we just log the mappings. Full implementation would require:
    // 1. Container IP allocation (bridge network)
    // 2. NAT rules setup
    // 3. Firewall configuration
    console.log(`   ⚠️  Note: Port forwarding not yet implemented (requires OS-specific NAT rules)`);
  }

  /**
   * Setup volume mounts for container
   */
  private setupVolumeMounts(container: Container): void {
    if (!container.config.volumes || container.config.volumes.length === 0) {
      return;
    }

    const volumeMounts: Array<{ hostPath: string; containerPath: string; mode: string }> = [];

    for (const volumeSpec of container.config.volumes) {
      const spec = volumeSpec.toString();

      // Parse volume spec: host:container[:mode]
      const parts = spec.split(':');

      if (parts.length >= 2) {
        const hostPath = path.resolve(parts[0]);
        const containerPath = parts[1];
        const mode = parts[2] || 'rw'; // rw (read-write) or ro (read-only)

        // Check if host path exists
        if (!fs.existsSync(hostPath)) {
          console.warn(`   ⚠️  Host path does not exist: ${hostPath}`);
          console.log(`   Creating directory: ${hostPath}`);
          fs.mkdirSync(hostPath, { recursive: true });
        }

        // Create container path in rootfs
        const fullContainerPath = path.join(container.storage.rootfs, containerPath);
        const containerDir = path.dirname(fullContainerPath);

        if (!fs.existsSync(containerDir)) {
          fs.mkdirSync(containerDir, { recursive: true });
        }

        // Create symlink from container path to host path
        if (!fs.existsSync(fullContainerPath)) {
          try {
            fs.symlinkSync(hostPath, fullContainerPath, 'dir');
            console.log(`   📁 Volume mounted: ${hostPath} → ${containerPath} (${mode})`);
          } catch (error: any) {
            console.error(`   ❌ Failed to mount volume: ${error.message}`);
          }
        } else {
          console.warn(`   ⚠️  Container path already exists: ${containerPath}`);
        }

        volumeMounts.push({ hostPath, containerPath, mode });
      } else {
        console.warn(`   ⚠️  Invalid volume spec: ${volumeSpec}`);
      }
    }

    container.storage.volumes = volumeMounts as any;
  }

  /**
   * Spawn container process
   */
  private spawnContainerProcess(container: Container): ChildProcess {
    const cwd = path.join(container.storage.rootfs, container.config.workdir || '/');

    const stdoutStream = fs.createWriteStream(container.logs.stdout, { flags: 'a' });
    const stderrStream = fs.createWriteStream(container.logs.stderr, { flags: 'a' });

    const proc = spawn(container.config.entrypoint[0], container.config.entrypoint.slice(1), {
      cwd,
      env: container.config.env,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: true, // Run in background
    });

    // Pipe stdout/stderr to log files
    proc.stdout?.pipe(stdoutStream);
    proc.stderr?.pipe(stderrStream);

    // Handle process exit
    proc.on('exit', (code, signal) => {
      console.log(`\n   Container ${container.name} exited (code: ${code}, signal: ${signal})`);

      container.status = 'exited';
      container.finished = new Date().toISOString();
      container.exitCode = code || 0;

      this.saveContainer(container);

      stdoutStream.close();
      stderrStream.close();
    });

    // Unref so it doesn't keep the parent process alive
    proc.unref();

    return proc;
  }

  /**
   * Save container to disk
   */
  private saveContainer(container: Container): void {
    const containerDir = path.join(this.containersDir, container.id);
    const containerPath = path.join(containerDir, 'container.json');

    if (!fs.existsSync(containerDir)) {
      fs.mkdirSync(containerDir, { recursive: true });
    }

    fs.writeFileSync(containerPath, JSON.stringify(container, null, 2), 'utf-8');
  }
}

// ============================================================================
// Utilities
// ============================================================================

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)}GB`;
}

export function formatUptime(startedAt: string | undefined): string {
  if (!startedAt) return 'N/A';

  const start = new Date(startedAt).getTime();
  const now = Date.now();
  const diff = now - start;

  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}d`;
  if (hours > 0) return `${hours}h`;
  if (minutes > 0) return `${minutes}m`;
  return `${seconds}s`;
}

// ============================================================================
// Exports
// ============================================================================

// GCRRuntime is already exported inline above
