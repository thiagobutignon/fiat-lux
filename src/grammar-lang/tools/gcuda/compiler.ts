/**
 * GCUDA Kernel Compiler
 *
 * Compiles GPU kernels with O(1) content-addressable caching.
 * Supports CUDA, OpenCL, Metal.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { exec } from 'child_process';
import { promisify } from 'util';
import {
  GCUDAKernel,
  KernelLang,
  CompileOptions,
  KernelMetadata,
  CompilationError,
} from './types';

const execAsync = promisify(exec);

// ============================================================================
// Kernel Compiler
// ============================================================================

export class KernelCompiler {
  private kernelsDir: string;
  private kernels: Map<string, GCUDAKernel>;

  constructor(kernelsDir: string = '.gcuda/kernels') {
    this.kernelsDir = kernelsDir;
    this.kernels = new Map();
    this.ensureKernelsDir();
    this.loadKernels();
  }

  /**
   * Compile kernel from source code
   */
  async compile(
    source: string,
    lang: KernelLang,
    entryPoint: string,
    options: CompileOptions = {}
  ): Promise<GCUDAKernel> {
    console.log(`\n🔨 Compiling ${lang} kernel...\n`);

    // Calculate hash for content-addressable storage
    const hash = this.calculateHash(source, options);

    // Check if already compiled
    if (this.kernels.has(hash)) {
      console.log(`   ✅ Kernel already compiled (cached)`);
      console.log(`   Hash: ${hash.substring(0, 19)}...`);
      return this.kernels.get(hash)!;
    }

    // Compile based on language
    let compiled: Buffer;
    let metadata: KernelMetadata;

    switch (lang) {
      case 'cuda':
        ({ compiled, metadata } = await this.compileCUDA(source, options));
        break;

      case 'opencl':
        ({ compiled, metadata } = await this.compileOpenCL(source, options));
        break;

      case 'metal':
        ({ compiled, metadata } = await this.compileMetal(source, options));
        break;

      default:
        throw new CompilationError(`Unsupported language: ${lang}`);
    }

    // Create kernel object
    const kernel: GCUDAKernel = {
      hash,
      name: entryPoint,
      version: '1.0.0',
      lang,
      source,
      sourcePath: '',
      compiled,
      entryPoint,
      metadata,
    };

    // Save kernel
    this.saveKernel(kernel);
    this.kernels.set(hash, kernel);

    console.log(`\n   ✅ Compilation successful`);
    console.log(`   Hash: ${hash.substring(0, 19)}...`);
    console.log(`   Size: ${formatSize(compiled.length)}`);

    return kernel;
  }

  /**
   * Compile kernel from file
   */
  async compileFromFile(
    filePath: string,
    options: CompileOptions = {}
  ): Promise<GCUDAKernel> {
    if (!fs.existsSync(filePath)) {
      throw new CompilationError(`File not found: ${filePath}`);
    }

    const source = fs.readFileSync(filePath, 'utf-8');
    const ext = path.extname(filePath);

    // Detect language from extension
    let lang: KernelLang;
    if (ext === '.cu' || ext === '.cuh') {
      lang = 'cuda';
    } else if (ext === '.cl') {
      lang = 'opencl';
    } else if (ext === '.metal') {
      lang = 'metal';
    } else {
      throw new CompilationError(`Unknown file extension: ${ext}`);
    }

    // Extract entry point from filename (e.g., "matmul.cu" -> "matmul_kernel")
    const basename = path.basename(filePath, ext);
    const entryPoint = `${basename}_kernel`;

    const kernel = await this.compile(source, lang, entryPoint, options);
    kernel.sourcePath = filePath;

    return kernel;
  }

  /**
   * Get kernel by hash
   * O(1) lookup
   */
  getKernel(hash: string): GCUDAKernel | null {
    return this.kernels.get(hash) || null;
  }

  /**
   * List all compiled kernels
   */
  listKernels(): GCUDAKernel[] {
    return Array.from(this.kernels.values());
  }

  /**
   * Delete kernel
   */
  deleteKernel(hash: string): void {
    const kernelPath = path.join(this.kernelsDir, hash);
    if (fs.existsSync(kernelPath)) {
      fs.rmSync(kernelPath, { recursive: true, force: true });
    }
    this.kernels.delete(hash);
  }

  // ==========================================================================
  // Private: Compilation
  // ==========================================================================

  /**
   * Compile CUDA kernel using nvcc
   */
  private async compileCUDA(
    source: string,
    options: CompileOptions
  ): Promise<{ compiled: Buffer; metadata: KernelMetadata }> {
    // Check if nvcc is available
    const hasNvcc = await this.checkNvccAvailable();

    if (!hasNvcc) {
      console.log(`   ⚠️  nvcc not available - storing source only (runtime compilation)`);

      // Store source as PTX (will be JIT compiled at runtime)
      const compiled = Buffer.from(source, 'utf-8');

      const metadata: KernelMetadata = {
        compileTime: new Date().toISOString(),
        compiler: 'cuda-runtime',
        flags: options.flags || [],
        arch: options.arch || [],
        size: compiled.length,
      };

      return { compiled, metadata };
    }

    const tmpDir = path.join(this.kernelsDir, '.tmp');
    fs.mkdirSync(tmpDir, { recursive: true });

    const tmpSource = path.join(tmpDir, 'kernel.cu');
    const tmpOutput = path.join(tmpDir, 'kernel.ptx');

    try {
      // Write source to temp file
      fs.writeFileSync(tmpSource, source, 'utf-8');

      // Build nvcc command
      const flags = options.flags || [];
      const arch = options.arch || ['sm_70']; // Default to Volta
      const optimization = options.optimization || 'O3';

      const cmd = [
        'nvcc',
        `-${optimization}`,
        '--ptx',
        ...flags,
        ...arch.map(a => `--gpu-architecture=${a}`),
        tmpSource,
        '-o',
        tmpOutput,
      ].join(' ');

      console.log(`   Running: ${cmd}`);

      // Execute nvcc
      const { stdout, stderr } = await execAsync(cmd);

      if (options.verbose && stdout) {
        console.log(`   ${stdout}`);
      }

      if (stderr) {
        console.warn(`   Warning: ${stderr}`);
      }

      // Read compiled PTX
      const compiled = fs.readFileSync(tmpOutput);

      const metadata: KernelMetadata = {
        compileTime: new Date().toISOString(),
        compiler: 'nvcc',
        flags,
        arch,
        size: compiled.length,
      };

      return { compiled, metadata };
    } catch (error: any) {
      throw new CompilationError(`CUDA compilation failed: ${error.message}`);
    } finally {
      // Cleanup
      if (fs.existsSync(tmpSource)) fs.unlinkSync(tmpSource);
      if (fs.existsSync(tmpOutput)) fs.unlinkSync(tmpOutput);
    }
  }

  /**
   * Check if nvcc is available
   */
  private async checkNvccAvailable(): Promise<boolean> {
    try {
      await execAsync('which nvcc');
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Compile OpenCL kernel
   */
  private async compileOpenCL(
    source: string,
    options: CompileOptions
  ): Promise<{ compiled: Buffer; metadata: KernelMetadata }> {
    // OpenCL kernels are compiled at runtime by the driver
    // We just store the source as-is
    const compiled = Buffer.from(source, 'utf-8');

    const metadata: KernelMetadata = {
      compileTime: new Date().toISOString(),
      compiler: 'opencl-runtime',
      flags: options.flags || [],
      arch: options.arch || [],
      size: compiled.length,
    };

    return { compiled, metadata };
  }

  /**
   * Compile Metal kernel
   */
  private async compileMetal(
    source: string,
    options: CompileOptions
  ): Promise<{ compiled: Buffer; metadata: KernelMetadata }> {
    // Metal kernels need to be compiled with xcrun metal
    // For now, store source (compile at runtime)
    const compiled = Buffer.from(source, 'utf-8');

    const metadata: KernelMetadata = {
      compileTime: new Date().toISOString(),
      compiler: 'metal-runtime',
      flags: options.flags || [],
      arch: options.arch || [],
      size: compiled.length,
    };

    return { compiled, metadata };
  }

  // ==========================================================================
  // Private: Storage
  // ==========================================================================

  /**
   * Calculate content-addressable hash
   */
  private calculateHash(source: string, options: CompileOptions): string {
    const hash = crypto.createHash('sha256');
    hash.update(source);
    hash.update(JSON.stringify(options.flags || []));
    hash.update(JSON.stringify(options.arch || []));
    hash.update(options.optimization || 'O3');
    return `sha256:${hash.digest('hex')}`;
  }

  /**
   * Save kernel to disk
   */
  private saveKernel(kernel: GCUDAKernel): void {
    const kernelPath = path.join(this.kernelsDir, kernel.hash);
    fs.mkdirSync(kernelPath, { recursive: true });

    // Save source
    const sourcePath = path.join(kernelPath, 'source.txt');
    fs.writeFileSync(sourcePath, kernel.source, 'utf-8');

    // Save compiled binary
    if (kernel.compiled) {
      const compiledPath = path.join(kernelPath, 'compiled.bin');
      fs.writeFileSync(compiledPath, kernel.compiled);
    }

    // Save metadata
    const metadataPath = path.join(kernelPath, 'metadata.json');
    const metadata = {
      hash: kernel.hash,
      name: kernel.name,
      version: kernel.version,
      lang: kernel.lang,
      entryPoint: kernel.entryPoint,
      sourcePath: kernel.sourcePath,
      metadata: kernel.metadata,
    };
    fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2), 'utf-8');
  }

  /**
   * Load all kernels from disk
   */
  private loadKernels(): void {
    if (!fs.existsSync(this.kernelsDir)) {
      return;
    }

    const entries = fs.readdirSync(this.kernelsDir);

    for (const entry of entries) {
      if (entry.startsWith('sha256:')) {
        try {
          const kernelPath = path.join(this.kernelsDir, entry);
          const metadataPath = path.join(kernelPath, 'metadata.json');

          if (fs.existsSync(metadataPath)) {
            const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
            const sourcePath = path.join(kernelPath, 'source.txt');
            const compiledPath = path.join(kernelPath, 'compiled.bin');

            const kernel: GCUDAKernel = {
              hash: metadata.hash,
              name: metadata.name,
              version: metadata.version,
              lang: metadata.lang,
              source: fs.readFileSync(sourcePath, 'utf-8'),
              sourcePath: metadata.sourcePath || '',
              compiled: fs.existsSync(compiledPath) ? fs.readFileSync(compiledPath) : undefined,
              entryPoint: metadata.entryPoint,
              metadata: metadata.metadata,
            };

            this.kernels.set(kernel.hash, kernel);
          }
        } catch (error) {
          // Skip invalid kernels
        }
      }
    }

    if (this.kernels.size > 0) {
      console.log(`Loaded ${this.kernels.size} compiled kernel(s) from cache`);
    }
  }

  /**
   * Ensure kernels directory exists
   */
  private ensureKernelsDir(): void {
    if (!fs.existsSync(this.kernelsDir)) {
      fs.mkdirSync(this.kernelsDir, { recursive: true });
    }
  }
}

// ============================================================================
// Utilities
// ============================================================================

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}
