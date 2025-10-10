/**
 * GCR Spec Parser
 *
 * Parses .gcr files (YAML format) into ContainerSpec objects.
 * Validates schema and performs type checking.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';
import {
  ContainerSpec,
  BuildConfig,
  RuntimeConfig,
  ContainerMetadata,
  isValidContainerName,
} from './types';

// ============================================================================
// Parser
// ============================================================================

export class GCRSpecParser {
  /**
   * Parse .gcr file from path
   */
  parseFile(filePath: string): ContainerSpec {
    // Read file
    const content = fs.readFileSync(filePath, 'utf-8');

    // Parse YAML
    const rawSpec = yaml.load(content) as any;

    // Validate and convert to ContainerSpec
    return this.validateSpec(rawSpec, filePath);
  }

  /**
   * Parse .gcr from string content
   */
  parseString(content: string): ContainerSpec {
    const rawSpec = yaml.load(content) as any;
    return this.validateSpec(rawSpec, '<string>');
  }

  /**
   * Validate raw spec and convert to ContainerSpec
   */
  private validateSpec(rawSpec: any, source: string): ContainerSpec {
    const errors: string[] = [];

    // Check format
    if (rawSpec.format !== 'gcr-v1.0') {
      errors.push(`Invalid format: expected 'gcr-v1.0', got '${rawSpec.format}'`);
    }

    // Check required fields
    if (!rawSpec.name) {
      errors.push('Missing required field: name');
    } else if (!isValidContainerName(rawSpec.name)) {
      errors.push(`Invalid name: '${rawSpec.name}' (must match /^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/)`);
    }

    if (!rawSpec.version) {
      errors.push('Missing required field: version');
    }

    if (!rawSpec.base) {
      errors.push('Missing required field: base');
    }

    if (!rawSpec.runtime) {
      errors.push('Missing required field: runtime');
    } else {
      // Validate runtime.entrypoint
      if (!rawSpec.runtime.entrypoint) {
        errors.push('Missing required field: runtime.entrypoint');
      } else if (!Array.isArray(rawSpec.runtime.entrypoint)) {
        errors.push('Invalid runtime.entrypoint: must be array');
      }
    }

    // Throw if errors
    if (errors.length > 0) {
      throw new GCRSpecError(
        `Invalid .gcr spec in ${source}:\n${errors.map(e => `  - ${e}`).join('\n')}`,
        errors
      );
    }

    // Build ContainerSpec
    const spec: ContainerSpec = {
      format: 'gcr-v1.0',
      name: rawSpec.name,
      version: rawSpec.version,
      base: rawSpec.base,
      build: this.parseBuildConfig(rawSpec.build || {}),
      runtime: this.parseRuntimeConfig(rawSpec.runtime),
      metadata: this.parseMetadata(rawSpec.metadata || {}),
    };

    return spec;
  }

  /**
   * Parse build config
   */
  private parseBuildConfig(rawBuild: any): BuildConfig {
    const build: BuildConfig = {};

    // Copy instructions
    if (rawBuild.copy) {
      if (!Array.isArray(rawBuild.copy)) {
        throw new GCRSpecError('build.copy must be an array', ['build.copy']);
      }
      build.copy = rawBuild.copy.map((c: any) => ({
        src: c.src,
        dest: c.dest,
        hash: c.hash,
      }));
    }

    // Dependencies
    if (rawBuild.dependencies) {
      if (!Array.isArray(rawBuild.dependencies)) {
        throw new GCRSpecError('build.dependencies must be an array', ['build.dependencies']);
      }
      build.dependencies = rawBuild.dependencies.map((d: any) => ({
        name: d.name,
        version: d.version,
        hash: d.hash,
      }));
    }

    // Commands
    if (rawBuild.commands) {
      if (!Array.isArray(rawBuild.commands)) {
        throw new GCRSpecError('build.commands must be an array', ['build.commands']);
      }
      build.commands = rawBuild.commands;
    }

    // Environment variables
    if (rawBuild.env) {
      build.env = rawBuild.env;
    }

    return build;
  }

  /**
   * Parse runtime config
   */
  private parseRuntimeConfig(rawRuntime: any): RuntimeConfig {
    const runtime: RuntimeConfig = {
      entrypoint: rawRuntime.entrypoint,
    };

    // Optional fields
    if (rawRuntime.workdir) runtime.workdir = rawRuntime.workdir;
    if (rawRuntime.user) runtime.user = rawRuntime.user;
    if (rawRuntime.uid !== undefined) runtime.uid = rawRuntime.uid;
    if (rawRuntime.gid !== undefined) runtime.gid = rawRuntime.gid;

    // Resources
    if (rawRuntime.resources) {
      runtime.resources = {
        memory: rawRuntime.resources.memory,
        cpu: rawRuntime.resources.cpu,
        storage: rawRuntime.resources.storage,
      };
    }

    // Ports
    if (rawRuntime.ports) {
      if (!Array.isArray(rawRuntime.ports)) {
        throw new GCRSpecError('runtime.ports must be an array', ['runtime.ports']);
      }
      runtime.ports = rawRuntime.ports;
    }

    // Volumes
    if (rawRuntime.volumes) {
      if (!Array.isArray(rawRuntime.volumes)) {
        throw new GCRSpecError('runtime.volumes must be an array', ['runtime.volumes']);
      }
      runtime.volumes = rawRuntime.volumes;
    }

    // Health check
    if (rawRuntime.healthcheck) {
      runtime.healthcheck = {
        command: rawRuntime.healthcheck.command,
        interval: rawRuntime.healthcheck.interval,
        timeout: rawRuntime.healthcheck.timeout,
        retries: rawRuntime.healthcheck.retries,
        startPeriod: rawRuntime.healthcheck.startPeriod,
      };
    }

    // Environment variables
    if (rawRuntime.env) {
      runtime.env = rawRuntime.env;
    }

    return runtime;
  }

  /**
   * Parse metadata
   */
  private parseMetadata(rawMetadata: any): ContainerMetadata {
    const metadata: ContainerMetadata = {};

    if (rawMetadata.author) metadata.author = rawMetadata.author;
    if (rawMetadata.description) metadata.description = rawMetadata.description;
    if (rawMetadata.tags) {
      if (!Array.isArray(rawMetadata.tags)) {
        throw new GCRSpecError('metadata.tags must be an array', ['metadata.tags']);
      }
      metadata.tags = rawMetadata.tags;
    }
    if (rawMetadata.created) metadata.created = rawMetadata.created;
    if (rawMetadata.license) metadata.license = rawMetadata.license;
    if (rawMetadata.homepage) metadata.homepage = rawMetadata.homepage;
    if (rawMetadata.repository) metadata.repository = rawMetadata.repository;

    return metadata;
  }

  /**
   * Write ContainerSpec to .gcr file
   */
  writeFile(spec: ContainerSpec, filePath: string): void {
    const yamlContent = yaml.dump(spec, {
      indent: 2,
      lineWidth: 100,
      noRefs: true,
    });

    fs.writeFileSync(filePath, yamlContent, 'utf-8');
  }

  /**
   * Convert ContainerSpec to YAML string
   */
  toString(spec: ContainerSpec): string {
    return yaml.dump(spec, {
      indent: 2,
      lineWidth: 100,
      noRefs: true,
    });
  }
}

// ============================================================================
// Error Class
// ============================================================================

export class GCRSpecError extends Error {
  public readonly validationErrors: string[];

  constructor(message: string, validationErrors: string[] = []) {
    super(message);
    this.name = 'GCRSpecError';
    this.validationErrors = validationErrors;
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Parse .gcr file (convenience function)
 */
export function parseGCRFile(filePath: string): ContainerSpec {
  const parser = new GCRSpecParser();
  return parser.parseFile(filePath);
}

/**
 * Parse .gcr string (convenience function)
 */
export function parseGCRString(content: string): ContainerSpec {
  const parser = new GCRSpecParser();
  return parser.parseString(content);
}

/**
 * Validate .gcr file without parsing (returns validation errors)
 */
export function validateGCRFile(filePath: string): string[] {
  try {
    parseGCRFile(filePath);
    return []; // No errors
  } catch (error: any) {
    if (error instanceof GCRSpecError) {
      return error.validationErrors;
    }
    return [error.message || 'Unknown error'];
  }
}

/**
 * Check if file is a valid .gcr file
 */
export function isValidGCRFile(filePath: string): boolean {
  return validateGCRFile(filePath).length === 0;
}

/**
 * Get .gcr file extension
 */
export function getGCRExtension(filePath: string): string {
  return path.extname(filePath);
}

/**
 * Check if path is a .gcr file
 */
export function isGCRFile(filePath: string): boolean {
  return getGCRExtension(filePath) === '.gcr';
}
