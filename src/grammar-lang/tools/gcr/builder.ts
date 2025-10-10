/**
 * GCR Builder
 *
 * Builds container images from .gcr specs with O(1) caching.
 * Content-addressable, deterministic, glass-box.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { parseGCRFile } from './spec-parser';
import { LayerBuilder, formatSize } from './layers';
import { BuildCache, hashSpec, isCacheValid, formatDuration } from './cache';
import {
  ContainerSpec,
  ContainerImage,
  ImageLayer,
  ImageManifest,
  BuildOptions,
  BuildContext,
  BuildState,
} from './types';

// ============================================================================
// GCR Builder
// ============================================================================

export class GCRBuilder {
  private layerBuilder: LayerBuilder;
  private buildCache: BuildCache;
  private imagesDir: string;

  constructor(
    layersDir: string = '.gcr/layers',
    cacheDir: string = '.gcr/cache',
    imagesDir: string = '.gcr/images'
  ) {
    this.layerBuilder = new LayerBuilder(layersDir);
    this.buildCache = new BuildCache(cacheDir);
    this.imagesDir = imagesDir;
    this.ensureImagesDir();
  }

  /**
   * Ensure images directory exists
   */
  private ensureImagesDir(): void {
    if (!fs.existsSync(this.imagesDir)) {
      fs.mkdirSync(this.imagesDir, { recursive: true });
    }
  }

  /**
   * Build container image from .gcr spec
   */
  async build(specPath: string, options: BuildOptions = {}): Promise<ContainerImage> {
    console.log(`\n🔨 Building container from ${specPath}...\n`);

    const startTime = Date.now();

    // Parse spec
    console.log(`📋 Parsing spec...`);
    const spec = parseGCRFile(specPath);
    const specHash = hashSpec(specPath);

    console.log(`   Name: ${spec.name}:${spec.version}`);
    console.log(`   Base: ${spec.base}`);

    // Check cache (unless --no-cache)
    if (!options.noCache) {
      console.log(`\n💾 Checking build cache...`);
      const cacheKey = this.buildCache.getCacheKey({
        specHash,
        baseImage: spec.base,
        buildArgs: options.buildArgs,
        platform: options.platform,
      });

      const cached = this.buildCache.get(cacheKey);
      if (cached && isCacheValid(cached, this.layerBuilder)) {
        console.log(`   ✅ Cache HIT! (built ${formatDuration(Date.now() - new Date(cached.buildTime).getTime())} ago)`);
        console.log(`   📦 Image: ${cached.imageHash.substring(0, 12)}...`);

        // Load cached image
        const image = this.loadImage(cached.imageHash);
        const duration = Date.now() - startTime;
        console.log(`\n✅ Build complete in ${formatDuration(duration)} (cached)`);
        return image;
      }

      console.log(`   ⚠️  Cache MISS - building from scratch`);
    }

    // Build context
    const context: BuildContext = {
      contextPath: path.dirname(specPath),
      specPath,
      spec,
      options,
      state: {
        currentStep: 0,
        totalSteps: this.calculateTotalSteps(spec),
        progress: 0,
        status: 'running',
        errors: [],
      },
    };

    // Build layers
    console.log(`\n🏗️  Building layers...`);
    const layers: ImageLayer[] = [];

    // Step 1: Base layer
    if (spec.base !== 'scratch') {
      console.log(`\n📦 Step 1: Pull base image ${spec.base}`);
      const baseLayer = await this.buildBaseLayer(spec.base);
      layers.push(baseLayer);
    } else {
      console.log(`\n📦 Step 1: Using scratch (empty base)`);
    }

    // Step 2: Copy files
    if (spec.build.copy && spec.build.copy.length > 0) {
      console.log(`\n📁 Step 2: Copy files (${spec.build.copy.length} instructions)`);
      for (const copyInst of spec.build.copy) {
        const srcPath = path.join(context.contextPath, copyInst.src);
        if (fs.existsSync(srcPath)) {
          const layer = this.layerBuilder.createFromDirectory(srcPath, 'app');
          layers.push(layer);
          console.log(`      Copied: ${copyInst.src} → ${copyInst.dest} (${formatSize(layer.size)})`);
        } else {
          console.warn(`      ⚠️  Source not found: ${srcPath}`);
        }
      }
    }

    // Step 3: Install dependencies
    if (spec.build.dependencies && spec.build.dependencies.length > 0) {
      console.log(`\n📦 Step 3: Install dependencies (${spec.build.dependencies.length} packages)`);
      const depsLayer = await this.buildDependenciesLayer(spec.build.dependencies);
      layers.push(depsLayer);
    }

    // Step 4: Run build commands
    if (spec.build.commands && spec.build.commands.length > 0) {
      console.log(`\n⚙️  Step 4: Run build commands (${spec.build.commands.length} commands)`);
      for (const command of spec.build.commands) {
        console.log(`      Running: ${command}`);
        // TODO: Actually execute commands (DIA 3)
        console.log(`      ⚠️  Command execution not yet implemented (DIA 3)`);
      }
    }

    // Step 5: Create config layer
    console.log(`\n⚙️  Step 5: Create configuration`);
    const configLayer = this.buildConfigLayer(spec);
    layers.push(configLayer);

    // Step 6: Create metadata layer
    console.log(`\n📋 Step 6: Create metadata`);
    const metadataLayer = this.buildMetadataLayer(spec);
    layers.push(metadataLayer);

    // Calculate image hash (from all layer hashes)
    const imageHash = this.calculateImageHash(layers);

    console.log(`\n📊 Image statistics:`);
    console.log(`   Layers: ${layers.length}`);
    console.log(`   Total size: ${formatSize(layers.reduce((sum, l) => sum + l.size, 0))}`);
    console.log(`   Image hash: ${imageHash.substring(0, 12)}...`);

    // Create image manifest
    const manifest: ImageManifest = {
      format: 'gcr-v1.0',
      name: spec.name,
      version: spec.version,
      hash: imageHash,
      size: layers.reduce((sum, l) => sum + l.size, 0),
      layers,
      config: spec.runtime,
      metadata: spec.metadata,
    };

    // Create image
    const image: ContainerImage = {
      hash: imageHash,
      name: spec.name,
      version: spec.version,
      size: manifest.size,
      layers,
      config: spec.runtime,
      metadata: {
        ...spec.metadata,
        buildTime: new Date().toISOString(),
        builder: 'gcr-v1.0',
      },
      manifest,
    };

    // Save image
    this.saveImage(image);

    // Update cache
    const buildDuration = Date.now() - startTime;
    this.buildCache.set(this.buildCache.getCacheKey({
      specHash,
      baseImage: spec.base,
      buildArgs: options.buildArgs,
      platform: options.platform,
      layerHashes: layers.map(l => l.hash),
    }), {
      cacheKey: '',
      imageHash,
      imageName: spec.name,
      imageVersion: spec.version,
      layers,
      buildTime: image.metadata.buildTime,
      buildDuration,
      metadata: {
        specHash,
        baseImage: spec.base,
        platform: options.platform,
      },
    });

    console.log(`\n✅ Build complete in ${formatDuration(buildDuration)}`);
    console.log(`📦 Image: ${spec.name}:${spec.version} (${imageHash.substring(0, 12)}...)`);

    return image;
  }

  /**
   * Build base layer from base image
   */
  private async buildBaseLayer(baseImage: string): Promise<ImageLayer> {
    // TODO: Actually pull base image from registry (DIA 4)
    // For now, create empty base layer
    console.log(`   ⚠️  Base image pull not yet implemented (DIA 4)`);
    console.log(`   Creating empty base layer...`);

    return this.layerBuilder.createFromContent('', 'base');
  }

  /**
   * Build dependencies layer
   */
  private async buildDependenciesLayer(dependencies: any[]): Promise<ImageLayer> {
    // TODO: Use GLM to install dependencies (integration)
    // For now, create layer from dependency metadata
    const depsContent = JSON.stringify(dependencies, null, 2);

    for (const dep of dependencies) {
      console.log(`      Installing: ${dep.name}@${dep.version}`);
    }

    console.log(`   ⚠️  GLM integration not yet implemented`);

    return this.layerBuilder.createFromContent(depsContent, 'dependencies');
  }

  /**
   * Build config layer
   */
  private buildConfigLayer(spec: ContainerSpec): ImageLayer {
    const config = {
      entrypoint: spec.runtime.entrypoint,
      workdir: spec.runtime.workdir,
      user: spec.runtime.user,
      env: spec.runtime.env,
      ports: spec.runtime.ports,
      volumes: spec.runtime.volumes,
    };

    const configContent = JSON.stringify(config, null, 2);
    return this.layerBuilder.createFromContent(configContent, 'config');
  }

  /**
   * Build metadata layer
   */
  private buildMetadataLayer(spec: ContainerSpec): ImageLayer {
    const metadata = {
      name: spec.name,
      version: spec.version,
      ...spec.metadata,
    };

    const metadataContent = JSON.stringify(metadata, null, 2);
    return this.layerBuilder.createFromContent(metadataContent, 'metadata');
  }

  /**
   * Calculate total build steps
   */
  private calculateTotalSteps(spec: ContainerSpec): number {
    let steps = 1; // Base layer

    if (spec.build.copy) steps += spec.build.copy.length;
    if (spec.build.dependencies) steps += 1;
    if (spec.build.commands) steps += spec.build.commands.length;

    steps += 2; // Config + metadata

    return steps;
  }

  /**
   * Calculate image hash from layers
   */
  private calculateImageHash(layers: ImageLayer[]): string {
    const hash = crypto.createHash('sha256');

    // Hash all layer hashes (order matters)
    for (const layer of layers) {
      hash.update(layer.hash);
    }

    return `sha256:${hash.digest('hex')}`;
  }

  /**
   * Save image to storage
   */
  private saveImage(image: ContainerImage): void {
    const imagePath = path.join(this.imagesDir, image.hash);
    fs.mkdirSync(imagePath, { recursive: true });

    // Save manifest
    const manifestPath = path.join(imagePath, 'manifest.json');
    fs.writeFileSync(manifestPath, JSON.stringify(image.manifest, null, 2), 'utf-8');

    // Save full image metadata
    const imagePath2 = path.join(imagePath, 'image.json');
    fs.writeFileSync(imagePath2, JSON.stringify(image, null, 2), 'utf-8');

    // Create tag symlink (name:version → hash)
    const tagPath = path.join(this.imagesDir, `${image.name}_${image.version}`);
    if (fs.existsSync(tagPath)) {
      fs.unlinkSync(tagPath);
    }
    fs.symlinkSync(image.hash, tagPath, 'dir');
  }

  /**
   * Load image from storage
   */
  loadImage(imageHash: string): ContainerImage {
    const imagePath = path.join(this.imagesDir, imageHash, 'image.json');
    const image = JSON.parse(fs.readFileSync(imagePath, 'utf-8'));
    return image as ContainerImage;
  }

  /**
   * Check if image exists
   */
  hasImage(imageHash: string): boolean {
    return fs.existsSync(path.join(this.imagesDir, imageHash));
  }

  /**
   * Find image by name:version
   */
  findImage(name: string, version: string): ContainerImage | null {
    const tagPath = path.join(this.imagesDir, `${name}_${version}`);

    if (!fs.existsSync(tagPath)) {
      return null;
    }

    // Resolve symlink
    const imageHash = fs.readlinkSync(tagPath);
    return this.loadImage(imageHash);
  }

  /**
   * List all images
   */
  listImages(): ContainerImage[] {
    if (!fs.existsSync(this.imagesDir)) {
      return [];
    }

    const images: ContainerImage[] = [];
    const entries = fs.readdirSync(this.imagesDir);

    for (const entry of entries) {
      if (entry.startsWith('sha256:')) {
        try {
          const image = this.loadImage(entry);
          images.push(image);
        } catch (error) {
          // Skip invalid images
        }
      }
    }

    return images;
  }

  /**
   * Delete image
   */
  deleteImage(imageHash: string): void {
    const imagePath = path.join(this.imagesDir, imageHash);
    if (fs.existsSync(imagePath)) {
      fs.rmSync(imagePath, { recursive: true, force: true });
    }
  }
}

// ============================================================================
// Exports
// ============================================================================

// GCRBuilder is already exported inline above
