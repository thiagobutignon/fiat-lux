#!/usr/bin/env node
/**
 * GCUDA CLI - Grammar CUDA Command Line Interface
 *
 * Commands:
 * - gcuda devices          List GPU devices
 * - gcuda info <device>    Show device information
 * - gcuda stats <device>   Show device statistics
 * - gcuda compile <file>   Compile CUDA kernel
 * - gcuda run <kernel>     Run compiled kernel
 */

import { DeviceManager, formatMemory, formatUtilization } from './device-manager';
import { KernelCompiler } from './compiler';

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
      case 'devices':
        await cmdDevices(commandArgs);
        break;

      case 'info':
        await cmdInfo(commandArgs);
        break;

      case 'stats':
        await cmdStats(commandArgs);
        break;

      case 'compile':
        await cmdCompile(commandArgs);
        break;

      case 'run':
        await cmdRun(commandArgs);
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
        console.error('Run "gcuda help" for usage information.');
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
// Commands
// ============================================================================

/**
 * List all GPU devices
 */
async function cmdDevices(args: string[]) {
  const manager = new DeviceManager();

  console.log('🔍 Scanning for GPU devices...\n');

  const devices = await manager.listDevices();

  if (devices.length === 0) {
    console.log('No GPU devices found.');
    console.log('\nMake sure you have:');
    console.log('  - NVIDIA GPUs: nvidia-smi installed');
    console.log('  - AMD GPUs: rocm-smi installed');
    return;
  }

  console.log(`Found ${devices.length} GPU device(s):\n`);

  console.log('ID  NAME                      VENDOR   COMPUTE  MEMORY      CORES');
  console.log('─'.repeat(75));

  for (const device of devices) {
    const id = device.id.toString().padEnd(3);
    const name = device.name.padEnd(25);
    const vendor = device.vendor.padEnd(8);
    const compute = device.compute.padEnd(8);
    const memory = formatMemory(device.memory).padEnd(11);
    const cores = device.cores.toString().padStart(5);

    console.log(`${id} ${name} ${vendor} ${compute} ${memory} ${cores}`);
  }

  console.log();
}

/**
 * Show detailed device information
 */
async function cmdInfo(args: string[]) {
  if (args.length === 0) {
    console.error('Error: Missing device ID');
    console.error('Usage: gcuda info <device-id>');
    process.exit(1);
  }

  const deviceId = parseInt(args[0]);
  const manager = new DeviceManager();

  // Ensure devices are loaded
  await manager.listDevices();

  const device = manager.getDevice(deviceId);

  if (!device) {
    console.error(`Error: Device ${deviceId} not found`);
    process.exit(1);
  }

  console.log(`\n📊 Device ${device.id} Information\n`);
  console.log(`Name:              ${device.name}`);
  console.log(`Vendor:            ${device.vendor}`);
  console.log(`Compute:           ${device.compute}`);
  console.log(`Memory Total:      ${formatMemory(device.memory)}`);
  console.log(`Memory Free:       ${formatMemory(device.memoryFree)}`);
  console.log(`Cores:             ${device.cores}`);
  console.log(`Clock Speed:       ${device.clockSpeed} MHz`);
  console.log(`PCIe Bus:          ${device.pcieBus}`);
  if (device.uuid) {
    console.log(`UUID:              ${device.uuid}`);
  }
  console.log();
}

/**
 * Show device statistics
 */
async function cmdStats(args: string[]) {
  if (args.length === 0) {
    console.error('Error: Missing device ID');
    console.error('Usage: gcuda stats <device-id>');
    process.exit(1);
  }

  const deviceId = parseInt(args[0]);
  const manager = new DeviceManager();

  // Ensure devices are loaded
  await manager.listDevices();

  const device = manager.getDevice(deviceId);

  if (!device) {
    console.error(`Error: Device ${deviceId} not found`);
    process.exit(1);
  }

  console.log(`\n📈 Device ${device.id} Statistics\n`);

  try {
    const stats = await manager.getDeviceStats(deviceId);

    if (!stats) {
      console.log('Statistics not available for this device.');
      return;
    }

    console.log(`Utilization:       ${formatUtilization(stats.utilization)}`);
    console.log(`Memory Used:       ${formatMemory(stats.memoryUsed)} / ${formatMemory(stats.memoryTotal)}`);
    console.log(`Temperature:       ${stats.temperature}°C`);
    console.log(`Power Usage:       ${stats.powerUsage.toFixed(1)}W / ${stats.powerLimit.toFixed(1)}W`);
    console.log();
  } catch (error: any) {
    console.error(`Failed to get stats: ${error.message}`);
    process.exit(1);
  }
}

/**
 * Compile CUDA kernel
 */
async function cmdCompile(args: string[]) {
  if (args.length === 0) {
    console.error('Error: Missing kernel file');
    console.error('Usage: gcuda compile <kernel.cu> [options]');
    process.exit(1);
  }

  const kernelFile = args[0];

  // Parse options
  const flags: string[] = [];
  const arch: string[] = [];
  let optimization: 'O0' | 'O1' | 'O2' | 'O3' = 'O3';
  let verbose = false;

  for (let i = 1; i < args.length; i++) {
    if (args[i] === '--arch' && i + 1 < args.length) {
      arch.push(args[i + 1]);
      i++;
    } else if (args[i] === '--flag' && i + 1 < args.length) {
      flags.push(args[i + 1]);
      i++;
    } else if (args[i] === '-O0') {
      optimization = 'O0';
    } else if (args[i] === '-O1') {
      optimization = 'O1';
    } else if (args[i] === '-O2') {
      optimization = 'O2';
    } else if (args[i] === '-O3') {
      optimization = 'O3';
    } else if (args[i] === '--verbose' || args[i] === '-v') {
      verbose = true;
    }
  }

  const compiler = new KernelCompiler();

  try {
    const kernel = await compiler.compileFromFile(kernelFile, {
      flags,
      arch: arch.length > 0 ? arch : undefined,
      optimization,
      verbose,
    });

    console.log(`\n✅ Kernel compiled successfully`);
    console.log(`   Hash: ${kernel.hash}`);
    console.log(`   Entry Point: ${kernel.entryPoint}`);
    console.log(`   Language: ${kernel.lang}`);
    console.log(`   Compiler: ${kernel.metadata.compiler}`);
    if (kernel.metadata.arch && kernel.metadata.arch.length > 0) {
      console.log(`   Architectures: ${kernel.metadata.arch.join(', ')}`);
    }
    console.log();
  } catch (error: any) {
    console.error(`\n❌ Compilation failed: ${error.message}`);
    if (verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

/**
 * Run compiled kernel
 */
async function cmdRun(args: string[]) {
  if (args.length === 0) {
    console.error('Error: Missing kernel name');
    console.error('Usage: gcuda run <kernel> [options]');
    process.exit(1);
  }

  const kernelName = args[0];

  console.log(`🚀 Running kernel ${kernelName}...\n`);
  console.log('⚠️  Kernel execution not yet implemented (GCUDA DIA 2)');
  console.log();
}

// ============================================================================
// Help & Version
// ============================================================================

function showHelp() {
  console.log(`
GCUDA - Grammar CUDA v1.0.0

Usage: gcuda <command> [options]

Device Management:
  devices              List all GPU devices
  info <device>        Show detailed device information
  stats <device>       Show device statistics (utilization, memory, etc.)

Kernel Management:
  compile <file>       Compile CUDA kernel (coming in DIA 2)
  run <kernel>         Run compiled kernel (coming in DIA 2)

Other:
  help                 Show this help message
  version              Show version information

Examples:
  # List all GPUs
  gcuda devices

  # Show info for GPU 0
  gcuda info 0

  # Show stats for GPU 0
  gcuda stats 0

  # Compile kernel (DIA 2)
  gcuda compile matmul.cu --arch sm_80

  # Run kernel (DIA 2)
  gcuda run matmul --grid 128,128 --block 16,16

For more information, visit: https://github.com/chomsky/gcuda
`);
}

function showVersion() {
  console.log('GCUDA (Grammar CUDA) v1.0.0');
  console.log('O(1) GPU acceleration - deterministic, fast, glass-box');
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
