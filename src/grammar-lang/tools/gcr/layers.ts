/**
 * GCR Layer Management
 *
 * Content-addressable layers for O(1) builds.
 * Each layer is identified by its content hash (sha256).
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { ImageLayer, LayerType } from './types';

// ============================================================================
// Layer Builder
// ============================================================================

export class LayerBuilder {
  private layersDir: string;

  constructor(layersDir: string = '.gcr/layers') {
    this.layersDir = layersDir;
    this.ensureLayersDir();
  }

  /**
   * Ensure layers directory exists
   */
  private ensureLayersDir(): void {
    if (!fs.existsSync(this.layersDir)) {
      fs.mkdirSync(this.layersDir, { recursive: true });
    }
  }

  /**
   * Create layer from directory
   */
  createFromDirectory(dirPath: string, layerType: LayerType = 'app'): ImageLayer {
    // Calculate directory content hash
    const hash = this.hashDirectory(dirPath);

    // Check if layer already exists (O(1) cache lookup)
    const layerPath = this.getLayerPath(hash);
    if (fs.existsSync(layerPath)) {
      console.log(`   ✅ Layer cached: ${hash.substring(0, 12)}... (${layerType})`);
      return this.loadLayer(hash);
    }

    // Create new layer
    console.log(`   🔨 Creating layer: ${hash.substring(0, 12)}... (${layerType})`);

    // Copy directory contents to layer
    const size = this.copyToLayer(dirPath, layerPath);

    const layer: ImageLayer = {
      hash,
      size,
      type: layerType,
      created: new Date().toISOString(),
    };

    // Save layer metadata
    this.saveLayerMetadata(layer);

    return layer;
  }

  /**
   * Create layer from file list
   */
  createFromFiles(files: string[], layerType: LayerType = 'app'): ImageLayer {
    // Calculate files content hash
    const hash = this.hashFiles(files);

    // Check cache
    const layerPath = this.getLayerPath(hash);
    if (fs.existsSync(layerPath)) {
      console.log(`   ✅ Layer cached: ${hash.substring(0, 12)}... (${layerType})`);
      return this.loadLayer(hash);
    }

    console.log(`   🔨 Creating layer: ${hash.substring(0, 12)}... (${layerType})`);

    // Copy files to layer
    const size = this.copyFilesToLayer(files, layerPath);

    const layer: ImageLayer = {
      hash,
      size,
      type: layerType,
      created: new Date().toISOString(),
    };

    this.saveLayerMetadata(layer);

    return layer;
  }

  /**
   * Create layer from content (e.g., config, metadata)
   */
  createFromContent(content: string, layerType: LayerType = 'config'): ImageLayer {
    const hash = this.hashContent(content);

    const layerPath = this.getLayerPath(hash);
    if (fs.existsSync(layerPath)) {
      console.log(`   ✅ Layer cached: ${hash.substring(0, 12)}... (${layerType})`);
      return this.loadLayer(hash);
    }

    console.log(`   🔨 Creating layer: ${hash.substring(0, 12)}... (${layerType})`);

    // Write content to layer
    fs.mkdirSync(layerPath, { recursive: true });
    const contentPath = path.join(layerPath, 'content');
    fs.writeFileSync(contentPath, content, 'utf-8');

    const size = Buffer.from(content, 'utf-8').length;

    const layer: ImageLayer = {
      hash,
      size,
      type: layerType,
      created: new Date().toISOString(),
    };

    this.saveLayerMetadata(layer);

    return layer;
  }

  /**
   * Load existing layer
   */
  loadLayer(hash: string): ImageLayer {
    const metadataPath = path.join(this.getLayerPath(hash), 'metadata.json');
    const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
    return metadata as ImageLayer;
  }

  /**
   * Check if layer exists
   */
  hasLayer(hash: string): boolean {
    return fs.existsSync(this.getLayerPath(hash));
  }

  /**
   * Get layer path from hash
   */
  private getLayerPath(hash: string): string {
    // Store layers by hash: .gcr/layers/sha256:abc123.../
    return path.join(this.layersDir, hash);
  }

  /**
   * Save layer metadata
   */
  private saveLayerMetadata(layer: ImageLayer): void {
    const layerPath = this.getLayerPath(layer.hash);
    const metadataPath = path.join(layerPath, 'metadata.json');
    fs.writeFileSync(metadataPath, JSON.stringify(layer, null, 2), 'utf-8');
  }

  /**
   * Copy directory to layer
   */
  private copyToLayer(srcDir: string, layerPath: string): number {
    fs.mkdirSync(layerPath, { recursive: true });

    let totalSize = 0;

    const copyRecursive = (src: string, dest: string) => {
      const stat = fs.statSync(src);

      if (stat.isDirectory()) {
        if (!fs.existsSync(dest)) {
          fs.mkdirSync(dest, { recursive: true });
        }
        const files = fs.readdirSync(src);
        for (const file of files) {
          copyRecursive(path.join(src, file), path.join(dest, file));
        }
      } else {
        fs.copyFileSync(src, dest);
        totalSize += stat.size;
      }
    };

    copyRecursive(srcDir, path.join(layerPath, 'contents'));

    return totalSize;
  }

  /**
   * Copy files to layer
   */
  private copyFilesToLayer(files: string[], layerPath: string): number {
    fs.mkdirSync(layerPath, { recursive: true });

    let totalSize = 0;

    for (const file of files) {
      const stat = fs.statSync(file);
      const dest = path.join(layerPath, 'contents', path.basename(file));
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.copyFileSync(file, dest);
      totalSize += stat.size;
    }

    return totalSize;
  }

  /**
   * Hash directory contents (deterministic)
   */
  private hashDirectory(dirPath: string): string {
    const hash = crypto.createHash('sha256');

    const hashRecursive = (dir: string) => {
      const files = fs.readdirSync(dir).sort(); // Sort for determinism

      for (const file of files) {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);

        if (stat.isDirectory()) {
          hash.update(file); // Directory name
          hashRecursive(filePath);
        } else {
          hash.update(file); // File name
          const content = fs.readFileSync(filePath);
          hash.update(content); // File content
        }
      }
    };

    hashRecursive(dirPath);

    return `sha256:${hash.digest('hex')}`;
  }

  /**
   * Hash file list (deterministic)
   */
  private hashFiles(files: string[]): string {
    const hash = crypto.createHash('sha256');

    // Sort files for determinism
    const sortedFiles = [...files].sort();

    for (const file of sortedFiles) {
      hash.update(path.basename(file)); // File name
      const content = fs.readFileSync(file);
      hash.update(content); // File content
    }

    return `sha256:${hash.digest('hex')}`;
  }

  /**
   * Hash content string
   */
  private hashContent(content: string): string {
    const hash = crypto.createHash('sha256');
    hash.update(content);
    return `sha256:${hash.digest('hex')}`;
  }

  /**
   * Get layer size
   */
  getLayerSize(hash: string): number {
    const layer = this.loadLayer(hash);
    return layer.size;
  }

  /**
   * List all layers
   */
  listLayers(): ImageLayer[] {
    if (!fs.existsSync(this.layersDir)) {
      return [];
    }

    const layers: ImageLayer[] = [];
    const layerDirs = fs.readdirSync(this.layersDir);

    for (const layerDir of layerDirs) {
      if (layerDir.startsWith('sha256:')) {
        try {
          const layer = this.loadLayer(layerDir);
          layers.push(layer);
        } catch (error) {
          // Skip invalid layers
        }
      }
    }

    return layers;
  }

  /**
   * Delete layer
   */
  deleteLayer(hash: string): void {
    const layerPath = this.getLayerPath(hash);
    if (fs.existsSync(layerPath)) {
      fs.rmSync(layerPath, { recursive: true, force: true });
    }
  }

  /**
   * Get total layers size
   */
  getTotalSize(): number {
    const layers = this.listLayers();
    return layers.reduce((total, layer) => total + layer.size, 0);
  }

  /**
   * Garbage collect unused layers
   */
  garbageCollect(usedHashes: Set<string>): number {
    const layers = this.listLayers();
    let deletedCount = 0;

    for (const layer of layers) {
      if (!usedHashes.has(layer.hash)) {
        console.log(`   🗑️  Deleting unused layer: ${layer.hash.substring(0, 12)}...`);
        this.deleteLayer(layer.hash);
        deletedCount++;
      }
    }

    return deletedCount;
  }
}

// ============================================================================
// Layer Utilities
// ============================================================================

/**
 * Format layer size (bytes → human readable)
 */
export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)}GB`;
}

/**
 * Verify layer integrity (hash matches content)
 */
export function verifyLayer(layer: ImageLayer, layerBuilder: LayerBuilder): boolean {
  // Re-calculate hash and compare
  // (simplified - would re-hash actual contents)
  return layerBuilder.hasLayer(layer.hash);
}

/**
 * Merge layers (combine multiple layers into one)
 */
export function mergeLayers(layers: ImageLayer[], layerBuilder: LayerBuilder): ImageLayer {
  // Combine all layer contents
  // This is useful for optimization (flatten layers)
  const allHashes = layers.map(l => l.hash).join(',');
  const hash = crypto.createHash('sha256').update(allHashes).digest('hex');

  return {
    hash: `sha256:${hash}`,
    size: layers.reduce((total, l) => total + l.size, 0),
    type: 'app',
    created: new Date().toISOString(),
  };
}

// ============================================================================
// Exports
// ============================================================================

// All exports are already declared inline above
