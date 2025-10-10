#!/usr/bin/env node
/**
 * GCR CLI - Grammar Container Runtime Command Line Interface
 *
 * Commands:
 * - gcr build <spec>         Build container from .gcr spec
 * - gcr run <image>          Run container
 * - gcr ps                   List running containers
 * - gcr stop <container>     Stop container
 * - gcr images               List images
 * - gcr rmi <image>          Remove image
 * - gcr pull <image>         Pull image from registry
 * - gcr push <image>         Push image to registry
 * - gcr exec <container>     Execute command in container
 * - gcr logs <container>     Show container logs
 */

import * as fs from 'fs';
import * as path from 'path';
import { parseGCRFile, validateGCRFile } from './spec-parser';
import { ContainerSpec, BuildOptions } from './types';
import { GCRBuilder } from './builder';
import { GCRRuntime, formatUptime } from './runtime';

// ============================================================================
// CLI Entry Point
// ============================================================================

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    showHelp();
    process.exit(0);
  }

  const command = args[0];
  const commandArgs = args.slice(1);

  try {
    switch (command) {
      case 'build':
        await cmdBuild(commandArgs);
        break;

      case 'run':
        await cmdRun(commandArgs);
        break;

      case 'ps':
        await cmdPs(commandArgs);
        break;

      case 'stop':
        await cmdStop(commandArgs);
        break;

      case 'images':
        await cmdImages(commandArgs);
        break;

      case 'rmi':
        await cmdRmi(commandArgs);
        break;

      case 'pull':
        await cmdPull(commandArgs);
        break;

      case 'push':
        await cmdPush(commandArgs);
        break;

      case 'exec':
        await cmdExec(commandArgs);
        break;

      case 'logs':
        await cmdLogs(commandArgs);
        break;

      case 'help':
      case '--help':
      case '-h':
        showHelp();
        break;

      case 'version':
      case '--version':
      case '-v':
        showVersion();
        break;

      default:
        console.error(`Unknown command: ${command}`);
        console.error('Run "gcr help" for usage information.');
        process.exit(1);
    }
  } catch (error: any) {
    console.error(`Error: ${error.message}`);
    if (process.env.DEBUG) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

// ============================================================================
// Commands (Stubs - will be implemented in later days)
// ============================================================================

/**
 * Build container from .gcr spec
 */
async function cmdBuild(args: string[]) {
  if (args.length === 0) {
    console.error('Error: Missing .gcr spec file');
    console.error('Usage: gcr build <spec.gcr> [options]');
    process.exit(1);
  }

  const specPath = args[0];

  // Check if file exists
  if (!fs.existsSync(specPath)) {
    console.error(`Error: File not found: ${specPath}`);
    process.exit(1);
  }

  // Validate spec
  console.log(`Validating ${specPath}...`);
  const errors = validateGCRFile(specPath);
  if (errors.length > 0) {
    console.error('Validation errors:');
    errors.forEach(err => console.error(`  - ${err}`));
    process.exit(1);
  }

  console.log(`✅ Spec valid`);

  // Parse build options from args
  const options: BuildOptions = {
    noCache: args.includes('--no-cache'),
    pull: args.includes('--pull'),
    quiet: args.includes('--quiet'),
    verbose: args.includes('--verbose'),
  };

  // Create builder
  const builder = new GCRBuilder();

  // Build image
  try {
    const image = await builder.build(specPath, options);

    console.log(`\n✅ Successfully built: ${image.name}:${image.version}`);
    console.log(`   Image ID: ${image.hash.substring(0, 12)}...`);
    console.log(`   Size: ${formatSize(image.size)}`);
    console.log(`   Layers: ${image.layers.length}`);
  } catch (error: any) {
    console.error(`\n❌ Build failed: ${error.message}`);
    if (options.verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

/**
 * Format size helper
 */
function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)}GB`;
}

/**
 * Run container
 */
async function cmdRun(args: string[]) {
  if (args.length === 0) {
    console.error('Error: Missing image name');
    console.error('Usage: gcr run <image> [options]');
    process.exit(1);
  }

  const imageSpec = args[0];

  // Parse image name:version
  const [imageName, imageVersion = 'latest'] = imageSpec.split(':');

  // Parse options
  const name = getOption(args, '--name');
  const ports = getOptionArray(args, '--port');
  const volumes = getOptionArray(args, '--volume') || getOptionArray(args, '-v');
  const envArgs = getOptionArray(args, '--env') || getOptionArray(args, '-e');
  const gpuArg = getOption(args, '--gpu');

  // Parse GPU
  let gpu: number | number[] | undefined;
  if (gpuArg) {
    if (gpuArg.includes(',')) {
      // Multiple GPUs: --gpu 0,1,2
      gpu = gpuArg.split(',').map(s => parseInt(s.trim()));
    } else {
      // Single GPU: --gpu 0
      gpu = parseInt(gpuArg);
    }
  }

  // Parse env vars
  const env: { [key: string]: string } = {};
  if (envArgs) {
    for (const envVar of envArgs) {
      const [key, value] = envVar.split('=');
      if (key && value) {
        env[key] = value;
      }
    }
  }

  // Create runtime
  const runtime = new GCRRuntime();

  try {
    // Create container
    const container = await runtime.create(imageName, imageVersion, {
      name,
      ports,
      volumes,
      env,
      gpu,
    });

    // Start container
    await runtime.start(container.id);

    console.log(`\n✅ Container started successfully`);
    console.log(`   Container: ${container.name} (${container.id.substring(0, 12)})`);
    console.log(`   Image: ${container.image}`);
    console.log(`   Status: ${container.status}`);
    console.log(`   PID: ${container.pid}`);

    if (container.config.ports && container.config.ports.length > 0) {
      console.log(`   Ports: ${container.config.ports.join(', ')}`);
    }

    if (gpu !== undefined) {
      const gpuStr = Array.isArray(gpu) ? gpu.join(', ') : gpu.toString();
      console.log(`   GPU: ${gpuStr}`);
    }
  } catch (error: any) {
    console.error(`\n❌ Failed to run container: ${error.message}`);
    process.exit(1);
  }
}

/**
 * List running containers
 */
async function cmdPs(args: string[]) {
  const all = args.includes('--all') || args.includes('-a');

  const runtime = new GCRRuntime();
  const containers = runtime.list({ all });

  if (containers.length === 0) {
    console.log('CONTAINER ID  IMAGE              NAME              STATUS    UPTIME');
    console.log('(no containers)');
    return;
  }

  console.log('CONTAINER ID  IMAGE              NAME              STATUS    UPTIME');

  for (const container of containers) {
    const id = container.id.substring(0, 12);
    const image = container.image.padEnd(18);
    const name = container.name.padEnd(17);
    const status = container.status.padEnd(9);
    const uptime = formatUptime(container.started);

    console.log(`${id}  ${image} ${name} ${status} ${uptime}`);
  }
}

/**
 * Stop container
 */
async function cmdStop(args: string[]) {
  if (args.length === 0) {
    console.error('Error: Missing container ID or name');
    console.error('Usage: gcr stop <container>');
    process.exit(1);
  }

  const containerIdOrName = args[0];

  const runtime = new GCRRuntime();

  try {
    await runtime.stop(containerIdOrName);
  } catch (error: any) {
    console.error(`\n❌ Failed to stop container: ${error.message}`);
    process.exit(1);
  }
}

/**
 * List images
 */
async function cmdImages(args: string[]) {
  const builder = new GCRBuilder();
  const images = builder.listImages();

  if (images.length === 0) {
    console.log('REPOSITORY           TAG        IMAGE ID      SIZE       CREATED');
    console.log('(no images available)');
    return;
  }

  console.log('REPOSITORY           TAG        IMAGE ID      SIZE       CREATED');

  for (const image of images) {
    const repository = image.name.padEnd(20);
    const tag = image.version.padEnd(10);
    const imageId = image.hash.substring(7, 19); // Remove 'sha256:' and take 12 chars
    const size = formatSize(image.size).padEnd(10);
    const created = formatTimeAgo(image.metadata.buildTime);

    console.log(`${repository} ${tag} ${imageId} ${size} ${created}`);
  }
}

/**
 * Format time ago helper
 */
function formatTimeAgo(timestamp: string): string {
  const now = Date.now();
  const then = new Date(timestamp).getTime();
  const diffMs = now - then;

  const seconds = Math.floor(diffMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
  if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
  if (minutes > 0) return `${minutes} min${minutes > 1 ? 's' : ''} ago`;
  return 'just now';
}

/**
 * Remove image
 */
async function cmdRmi(args: string[]) {
  if (args.length === 0) {
    console.error('Error: Missing image name');
    console.error('Usage: gcr rmi <image>');
    process.exit(1);
  }

  const imageSpec = args[0];
  const force = args.includes('-f') || args.includes('--force');

  const builder = new GCRBuilder();
  const runtime = new GCRRuntime();

  try {
    // Parse image spec
    let image: any;
    let imageHash: string;

    if (imageSpec.startsWith('sha256:')) {
      // Direct hash
      imageHash = imageSpec;
      if (!builder.hasImage(imageHash)) {
        console.error(`Error: Image not found: ${imageHash}`);
        process.exit(1);
      }
      image = builder.loadImage(imageHash);
    } else {
      // name:version format
      const [imageName, imageVersion = 'latest'] = imageSpec.split(':');
      image = builder.findImage(imageName, imageVersion);

      if (!image) {
        console.error(`Error: Image not found: ${imageName}:${imageVersion}`);
        process.exit(1);
      }

      imageHash = image.hash;
    }

    // Check if any containers are using this image
    const containers = runtime.list({ all: true });
    const usingContainers = containers.filter(c => c.imageHash === imageHash);

    if (usingContainers.length > 0 && !force) {
      console.error(`Error: Image is in use by ${usingContainers.length} container(s):`);
      for (const container of usingContainers) {
        console.error(`  - ${container.name} (${container.id.substring(0, 12)})`);
      }
      console.error('\nUse --force to remove the image anyway');
      process.exit(1);
    }

    // Delete the image
    builder.deleteImage(imageHash);

    // Delete tag symlink if it exists
    const tagPath = path.join('.gcr/images', `${image.name}_${image.version}`);
    if (fs.existsSync(tagPath)) {
      fs.unlinkSync(tagPath);
    }

    console.log(`✅ Image removed: ${image.name}:${image.version}`);
    console.log(`   Hash: ${imageHash.substring(0, 19)}...`);

    if (usingContainers.length > 0 && force) {
      console.log(`   ⚠️  ${usingContainers.length} container(s) were using this image`);
    }
  } catch (error: any) {
    console.error(`\n❌ Failed to remove image: ${error.message}`);
    process.exit(1);
  }
}

/**
 * Pull image from registry
 */
async function cmdPull(args: string[]) {
  if (args.length === 0) {
    console.error('Error: Missing image name');
    console.error('Usage: gcr pull <image>');
    process.exit(1);
  }

  const image = args[0];

  // TODO: Actual pull implementation (DIA 4)
  console.log(`🚧 Pull functionality will be implemented in DIA 4`);
  console.log(`   Would pull: ${image}`);
}

/**
 * Push image to registry
 */
async function cmdPush(args: string[]) {
  if (args.length === 0) {
    console.error('Error: Missing image name');
    console.error('Usage: gcr push <image>');
    process.exit(1);
  }

  const image = args[0];

  // TODO: Actual push implementation (DIA 4)
  console.log(`🚧 Push functionality will be implemented in DIA 4`);
  console.log(`   Would push: ${image}`);
}

/**
 * Execute command in container
 */
async function cmdExec(args: string[]) {
  if (args.length < 2) {
    console.error('Error: Missing container and command');
    console.error('Usage: gcr exec <container> <command...>');
    process.exit(1);
  }

  const containerIdOrName = args[0];
  const command = args.slice(1);

  const interactive = args.includes('-it') || args.includes('-i');

  const runtime = new GCRRuntime();

  try {
    const result = await runtime.exec(containerIdOrName, command, { interactive });

    if (!interactive) {
      if (result.stdout) {
        console.log(result.stdout);
      }
      if (result.stderr) {
        console.error(result.stderr);
      }
      process.exit(result.exitCode);
    }
  } catch (error: any) {
    console.error(`\n❌ Failed to execute command: ${error.message}`);
    process.exit(1);
  }
}

/**
 * Show container logs
 */
async function cmdLogs(args: string[]) {
  if (args.length === 0) {
    console.error('Error: Missing container ID or name');
    console.error('Usage: gcr logs <container>');
    process.exit(1);
  }

  const containerIdOrName = args[0];

  const follow = args.includes('-f') || args.includes('--follow');
  const tail = parseInt(getOption(args, '--tail') || '0') || undefined;

  const runtime = new GCRRuntime();

  try {
    const logs = runtime.getLogs(containerIdOrName, { follow, tail });
    console.log(logs);

    // TODO: Implement follow mode (tail -f)
    if (follow) {
      console.log('\n(Follow mode not yet implemented - showing current logs only)');
    }
  } catch (error: any) {
    console.error(`\n❌ Failed to get logs: ${error.message}`);
    process.exit(1);
  }
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Get option value from args
 */
function getOption(args: string[], option: string): string | undefined {
  const index = args.indexOf(option);
  if (index !== -1 && index + 1 < args.length) {
    return args[index + 1];
  }
  return undefined;
}

/**
 * Get array of option values from args
 */
function getOptionArray(args: string[], option: string): string[] | undefined {
  const values: string[] = [];

  for (let i = 0; i < args.length; i++) {
    if (args[i] === option && i + 1 < args.length) {
      values.push(args[i + 1]);
    }
  }

  return values.length > 0 ? values : undefined;
}

// ============================================================================
// Help & Version
// ============================================================================

function showHelp() {
  console.log(`
GCR - Grammar Container Runtime v1.0.0

Usage: gcr <command> [options]

Container Management:
  build <spec>         Build container from .gcr spec file
  run <image>          Run a container from an image
  ps                   List running containers
  stop <container>     Stop a running container
  exec <container>     Execute command in a running container
  logs <container>     Show container logs

Image Management:
  images               List local images
  rmi <image>          Remove an image
  pull <image>         Pull image from registry
  push <image>         Push image to registry

Other:
  help                 Show this help message
  version              Show version information

Examples:
  # Build container from spec
  gcr build myapp.gcr

  # Run container
  gcr run myapp:1.0.0 --name myapp-prod --port 3000:3000

  # List running containers
  gcr ps

  # Stop container
  gcr stop myapp-prod

  # List images
  gcr images

For more information, visit: https://github.com/chomsky/gcr
`);
}

function showVersion() {
  console.log('GCR (Grammar Container Runtime) v1.0.0');
  console.log('O(1) container runtime - deterministic, fast, glass-box');
}

// ============================================================================
// Run CLI
// ============================================================================

if (require.main === module) {
  main().catch(error => {
    console.error('Fatal error:', error.message);
    process.exit(1);
  });
}

export { main };
