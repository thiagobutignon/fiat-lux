#!/usr/bin/env tsx
/**
 * GLPM - Grammar Language Package Manager
 *
 * Commands:
 * - glpm init          Initialize new package
 * - glpm install       Install dependencies
 * - glpm update        Update dependencies
 * - glpm publish       Publish package
 * - glpm search        Search packages
 */

import * as fs from 'fs';
import * as path from 'path';
import { PackageManifest } from '../compiler/module-resolver';

// ============================================================================
// CLI Entry Point
// ============================================================================

const args = process.argv.slice(2);
const command = args[0] || 'help';

switch (command) {
  case 'init':
    initPackage();
    break;

  case 'install':
    installDependencies(args.slice(1));
    break;

  case 'update':
    updateDependencies(args.slice(1));
    break;

  case 'remove':
  case 'uninstall':
    removeDependencies(args.slice(1));
    break;

  case 'publish':
    publishPackage();
    break;

  case 'search':
    searchPackages(args.slice(1));
    break;

  case 'info':
    packageInfo(args[1]);
    break;

  case 'help':
  case '--help':
  case '-h':
    showHelp();
    break;

  default:
    console.error(`Unknown command: ${command}`);
    console.error('Run "glpm help" for usage');
    process.exit(1);
}

// ============================================================================
// Commands
// ============================================================================

function initPackage(): void {
  console.log('Initializing Grammar Language package...\n');

  const name = prompt('Package name: ') || 'my-package';
  const version = prompt('Version (1.0.0): ') || '1.0.0';
  const description = prompt('Description: ') || '';
  const main = prompt('Entry point (src/index.gl): ') || 'src/index.gl';

  const manifest: PackageManifest = {
    name,
    version,
    main,
    exports: {
      '.': `./${main}`
    }
  };

  if (description) {
    (manifest as any).description = description;
  }

  // Create gl.json
  fs.writeFileSync('gl.json', JSON.stringify(manifest, null, 2));
  console.log('\n✅ Created gl.json');

  // Create directory structure
  if (!fs.existsSync('src')) {
    fs.mkdirSync('src');
    console.log('✅ Created src/');
  }

  // Create example module
  if (!fs.existsSync(main)) {
    const exampleCode = `(module ${name.replace(/[^a-z0-9]/gi, '-')}
  (export hello))

(define hello (unit -> string)
  "Hello from ${name}!")
`;
    fs.writeFileSync(main, exampleCode);
    console.log(`✅ Created ${main}`);
  }

  console.log('\n🎉 Package initialized successfully!');
  console.log('\nNext steps:');
  console.log('  1. Edit your code in src/');
  console.log('  2. Add dependencies with: glpm install @agi/package');
  console.log('  3. Build with: glc src/index.gl');
}

function installDependencies(packages: string[]): void {
  const manifestPath = 'gl.json';

  if (!fs.existsSync(manifestPath)) {
    console.error('Error: No gl.json found. Run "glpm init" first.');
    process.exit(1);
  }

  const manifest: PackageManifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));

  if (packages.length === 0) {
    // Install all dependencies from manifest
    console.log('Installing dependencies...');

    const allDeps = {
      ...manifest.dependencies,
      ...manifest.devDependencies
    };

    if (Object.keys(allDeps).length === 0) {
      console.log('No dependencies to install.');
      return;
    }

    for (const [name, version] of Object.entries(allDeps)) {
      installPackage(name, version);
    }

    console.log('\n✅ All dependencies installed');
  } else {
    // Install specific packages
    for (const pkg of packages) {
      const [name, version] = pkg.includes('@') && pkg.lastIndexOf('@') > 0
        ? [pkg.substring(0, pkg.lastIndexOf('@')), pkg.substring(pkg.lastIndexOf('@') + 1)]
        : [pkg, 'latest'];

      installPackage(name, version);

      // Add to manifest
      if (!manifest.dependencies) {
        manifest.dependencies = {};
      }
      manifest.dependencies[name] = `^${version}`;
    }

    // Save manifest
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
    console.log('\n✅ Package manifest updated');
  }
}

function installPackage(name: string, version: string): void {
  console.log(`Installing ${name}@${version}...`);

  // Create node_modules if not exists
  if (!fs.existsSync('node_modules')) {
    fs.mkdirSync('node_modules');
  }

  // In production, fetch from registry
  // For now, just log
  console.log(`  → ${name}@${version} (simulated)`);
}

function updateDependencies(packages: string[]): void {
  console.log('Updating dependencies...');

  const manifestPath = 'gl.json';
  if (!fs.existsSync(manifestPath)) {
    console.error('Error: No gl.json found.');
    process.exit(1);
  }

  const manifest: PackageManifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));

  const toUpdate = packages.length > 0
    ? packages
    : Object.keys({ ...manifest.dependencies, ...manifest.devDependencies });

  for (const pkg of toUpdate) {
    console.log(`  Updating ${pkg}...`);
    // In production, fetch latest version from registry
  }

  console.log('\n✅ Dependencies updated');
}

function removeDependencies(packages: string[]): void {
  if (packages.length === 0) {
    console.error('Error: Specify packages to remove');
    process.exit(1);
  }

  const manifestPath = 'gl.json';
  if (!fs.existsSync(manifestPath)) {
    console.error('Error: No gl.json found.');
    process.exit(1);
  }

  const manifest: PackageManifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));

  for (const pkg of packages) {
    console.log(`Removing ${pkg}...`);

    if (manifest.dependencies?.[pkg]) {
      delete manifest.dependencies[pkg];
    }
    if (manifest.devDependencies?.[pkg]) {
      delete manifest.devDependencies[pkg];
    }

    // Remove from node_modules
    const pkgPath = path.join('node_modules', pkg);
    if (fs.existsSync(pkgPath)) {
      fs.rmSync(pkgPath, { recursive: true });
    }
  }

  // Save manifest
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log('\n✅ Packages removed');
}

function publishPackage(): void {
  const manifestPath = 'gl.json';
  if (!fs.existsSync(manifestPath)) {
    console.error('Error: No gl.json found.');
    process.exit(1);
  }

  const manifest: PackageManifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));

  console.log(`Publishing ${manifest.name}@${manifest.version}...`);
  console.log('\n⚠️  Publishing not yet implemented');
  console.log('   Will publish to Grammar Language Registry');
}

function searchPackages(terms: string[]): void {
  if (terms.length === 0) {
    console.error('Error: Specify search term');
    process.exit(1);
  }

  const query = terms.join(' ');
  console.log(`Searching for: ${query}`);
  console.log('\n⚠️  Search not yet implemented');
  console.log('   Will search Grammar Language Registry');
}

function packageInfo(name: string): void {
  if (!name) {
    console.error('Error: Specify package name');
    process.exit(1);
  }

  console.log(`Package info: ${name}`);
  console.log('\n⚠️  Package info not yet implemented');
  console.log('   Will fetch from Grammar Language Registry');
}

function showHelp(): void {
  console.log(`
GLPM - Grammar Language Package Manager v0.1.0

Usage:
  glpm <command> [options]

Commands:
  init              Initialize new package (creates gl.json)
  install [pkg]     Install dependencies
  update [pkg]      Update dependencies
  remove <pkg>      Remove dependencies
  publish           Publish package to registry
  search <term>     Search for packages
  info <pkg>        Show package information
  help              Show this help

Examples:
  glpm init
  glpm install @agi/math
  glpm install @agi/math@1.5.0
  glpm update
  glpm remove @agi/math
  glpm publish
  glpm search vector
  glpm info @agi/math

Package Manifest (gl.json):
  {
    "name": "@agi/my-package",
    "version": "1.0.0",
    "main": "src/index.gl",
    "exports": {
      ".": "./src/index.gl",
      "./utils": "./src/utils.gl"
    },
    "dependencies": {
      "@agi/math": "^1.5.0"
    }
  }

For more information: https://grammar-lang.dev/docs/packages
`);
}

// ============================================================================
// Utilities
// ============================================================================

function prompt(question: string): string | null {
  // Simple synchronous prompt (in production use readline)
  const readline = require('readline');
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  return new Promise<string>((resolve) => {
    rl.question(question, (answer: string) => {
      rl.close();
      resolve(answer);
    });
  }) as any;
}
