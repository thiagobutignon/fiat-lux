/**
 * GCUDA Device Manager
 *
 * Detects and manages GPU devices with O(1) lookups.
 * Glass-box: uses system commands (nvidia-smi, rocm-smi, etc.)
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import * as os from 'os';
import {
  GPUDevice,
  GPUVendor,
  DeviceStats,
  GPURequirements,
  DeviceError,
} from './types';

const execAsync = promisify(exec);

// ============================================================================
// Device Manager
// ============================================================================

export class DeviceManager {
  private devices: Map<number, GPUDevice>;
  private cacheExpiry: number = 60000; // 60 seconds
  private lastScan: number = 0;

  constructor() {
    this.devices = new Map();
  }

  /**
   * List all available GPU devices
   * Scans system if cache is expired
   */
  async listDevices(): Promise<GPUDevice[]> {
    const now = Date.now();

    if (now - this.lastScan > this.cacheExpiry || this.devices.size === 0) {
      await this.scanDevices();
      this.lastScan = now;
    }

    return Array.from(this.devices.values());
  }

  /**
   * Get device by ID
   * O(1) lookup
   */
  getDevice(id: number): GPUDevice | null {
    return this.devices.get(id) || null;
  }

  /**
   * Select best device matching requirements
   */
  async selectBestDevice(requirements: GPURequirements): Promise<GPUDevice | null> {
    const devices = await this.listDevices();

    // Filter by requirements
    const matching = devices.filter(device => {
      if (requirements.vendor && requirements.vendor !== 'any') {
        if (device.vendor !== requirements.vendor) return false;
      }

      if (requirements.compute) {
        if (parseFloat(device.compute) < parseFloat(requirements.compute)) {
          return false;
        }
      }

      if (requirements.memory) {
        if (device.memory < requirements.memory) {
          return false;
        }
      }

      if (requirements.cores) {
        if (device.cores < requirements.cores) {
          return false;
        }
      }

      return true;
    });

    if (matching.length === 0) {
      return null;
    }

    // Select device with most free memory
    matching.sort((a, b) => b.memoryFree - a.memoryFree);
    return matching[0];
  }

  /**
   * Get device utilization stats
   */
  async getDeviceStats(id: number): Promise<DeviceStats | null> {
    const device = this.getDevice(id);
    if (!device) {
      throw new DeviceError(`Device ${id} not found`);
    }

    switch (device.vendor) {
      case 'nvidia':
        return await this.getNvidiaStats(id);
      case 'amd':
        return await this.getAMDStats(id);
      default:
        return null;
    }
  }

  // ==========================================================================
  // Private: Device Scanning
  // ==========================================================================

  /**
   * Scan for all GPU devices on the system
   */
  private async scanDevices(): Promise<void> {
    this.devices.clear();

    // Try NVIDIA GPUs
    const nvidiaDevices = await this.scanNvidiaDevices();
    for (const device of nvidiaDevices) {
      this.devices.set(device.id, device);
    }

    // Try AMD GPUs
    const amdDevices = await this.scanAMDDevices();
    for (const device of amdDevices) {
      this.devices.set(this.devices.size, device);
    }

    // Try Apple GPUs (macOS)
    if (os.platform() === 'darwin') {
      const appleDevices = await this.scanAppleDevices();
      for (const device of appleDevices) {
        this.devices.set(this.devices.size, device);
      }
    }
  }

  /**
   * Scan for NVIDIA GPUs using nvidia-smi
   */
  private async scanNvidiaDevices(): Promise<GPUDevice[]> {
    try {
      const { stdout } = await execAsync(
        'nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.free,pcie.bus_id,uuid --format=csv,noheader,nounits'
      );

      const devices: GPUDevice[] = [];
      const lines = stdout.trim().split('\n');

      for (const line of lines) {
        const parts = line.split(',').map(s => s.trim());

        if (parts.length >= 6) {
          const [idStr, name, compute, memoryTotal, memoryFree, pcieBus, uuid] = parts;

          // Get core count (rough estimate based on GPU name)
          const cores = this.estimateNvidiaCores(name);

          devices.push({
            id: parseInt(idStr),
            name,
            vendor: 'nvidia',
            compute,
            memory: parseInt(memoryTotal) * 1024 * 1024, // MB to bytes
            memoryFree: parseInt(memoryFree) * 1024 * 1024,
            cores,
            clockSpeed: 0, // Would need additional query
            pcieBus,
            uuid: uuid || undefined,
          });
        }
      }

      return devices;
    } catch (error: any) {
      // nvidia-smi not available or no NVIDIA GPUs
      return [];
    }
  }

  /**
   * Scan for AMD GPUs using rocm-smi
   */
  private async scanAMDDevices(): Promise<GPUDevice[]> {
    try {
      const { stdout } = await execAsync('rocm-smi --showid --showmeminfo vram');

      // Parse rocm-smi output
      // This is a simplified parser - real implementation would be more robust
      const devices: GPUDevice[] = [];

      // TODO: Implement proper rocm-smi parsing
      // For now, return empty array
      return devices;
    } catch (error: any) {
      // rocm-smi not available or no AMD GPUs
      return [];
    }
  }

  /**
   * Scan for Apple GPUs using system_profiler
   */
  private async scanAppleDevices(): Promise<GPUDevice[]> {
    try {
      const { stdout } = await execAsync(
        'system_profiler SPDisplaysDataType -json'
      );

      const data = JSON.parse(stdout);
      const devices: GPUDevice[] = [];

      // Parse Apple GPU info
      if (data.SPDisplaysDataType) {
        for (const display of data.SPDisplaysDataType) {
          if (display.sppci_model) {
            devices.push({
              id: devices.length,
              name: display.sppci_model,
              vendor: 'apple',
              compute: '0.0', // Apple doesn't expose compute capability
              memory: 0,      // Shared memory, hard to determine
              memoryFree: 0,
              cores: 0,       // Not exposed
              clockSpeed: 0,
              pcieBus: display.sppci_bus || '',
            });
          }
        }
      }

      return devices;
    } catch (error: any) {
      return [];
    }
  }

  /**
   * Get NVIDIA device stats
   */
  private async getNvidiaStats(id: number): Promise<DeviceStats> {
    try {
      const { stdout } = await execAsync(
        `nvidia-smi -i ${id} --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits`
      );

      const parts = stdout.trim().split(',').map(s => s.trim());

      return {
        utilization: parseFloat(parts[0]) || 0,
        memoryUsed: parseInt(parts[1]) * 1024 * 1024, // MB to bytes
        memoryTotal: parseInt(parts[2]) * 1024 * 1024,
        temperature: parseFloat(parts[3]) || 0,
        powerUsage: parseFloat(parts[4]) || 0,
        powerLimit: parseFloat(parts[5]) || 0,
      };
    } catch (error: any) {
      throw new DeviceError(`Failed to get stats for NVIDIA device ${id}: ${error.message}`);
    }
  }

  /**
   * Get AMD device stats
   */
  private async getAMDStats(id: number): Promise<DeviceStats> {
    // TODO: Implement AMD stats using rocm-smi
    return {
      utilization: 0,
      memoryUsed: 0,
      memoryTotal: 0,
      temperature: 0,
      powerUsage: 0,
      powerLimit: 0,
    };
  }

  /**
   * Estimate NVIDIA core count from GPU name
   * This is a rough approximation based on known architectures
   */
  private estimateNvidiaCores(name: string): number {
    const nameLower = name.toLowerCase();

    // RTX 40 series (Ada Lovelace)
    if (nameLower.includes('4090')) return 16384;
    if (nameLower.includes('4080')) return 9728;
    if (nameLower.includes('4070 ti')) return 7680;
    if (nameLower.includes('4070')) return 5888;
    if (nameLower.includes('4060 ti')) return 4352;
    if (nameLower.includes('4060')) return 3072;

    // RTX 30 series (Ampere)
    if (nameLower.includes('3090 ti')) return 10752;
    if (nameLower.includes('3090')) return 10496;
    if (nameLower.includes('3080 ti')) return 10240;
    if (nameLower.includes('3080')) return 8704;
    if (nameLower.includes('3070 ti')) return 6144;
    if (nameLower.includes('3070')) return 5888;
    if (nameLower.includes('3060 ti')) return 4864;
    if (nameLower.includes('3060')) return 3584;

    // RTX 20 series (Turing)
    if (nameLower.includes('2080 ti')) return 4352;
    if (nameLower.includes('2080')) return 2944;
    if (nameLower.includes('2070')) return 2304;
    if (nameLower.includes('2060')) return 1920;

    // GTX 10 series (Pascal)
    if (nameLower.includes('1080 ti')) return 3584;
    if (nameLower.includes('1080')) return 2560;
    if (nameLower.includes('1070')) return 1920;
    if (nameLower.includes('1060')) return 1280;

    // A/H series (Data center)
    if (nameLower.includes('a100')) return 6912;
    if (nameLower.includes('a40')) return 10752;
    if (nameLower.includes('h100')) return 14592;

    // Default fallback
    return 2048;
  }
}

// ============================================================================
// Utilities
// ============================================================================

export function formatMemory(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)}GB`;
}

export function formatUtilization(percent: number): string {
  return `${percent.toFixed(1)}%`;
}
