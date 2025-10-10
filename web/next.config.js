/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config, { isServer }) => {
    // Externalize native modules for server-side
    if (isServer) {
      // Mark native modules as external to prevent webpack bundling
      config.externals.push('onnxruntime-node', '@xenova/transformers');
    }

    // Ignore .node binary files - don't try to bundle them
    config.module.rules.push({
      test: /\.node$/,
      use: 'ignore-loader',
    });

    // Suppress warnings
    config.infrastructureLogging = {
      level: 'error',
    };

    return config;
  },

  // Explicitly tell Next.js not to bundle these in server components
  experimental: {
    serverComponentsExternalPackages: [
      'onnxruntime-node',
      '@xenova/transformers',
    ],
  },

  // Disable strict mode for development
  reactStrictMode: false,
};

module.exports = nextConfig;
