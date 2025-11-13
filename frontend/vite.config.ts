import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { visualizer } from 'rollup-plugin-visualizer'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const isProduction = mode === 'production'
  const isAnalyze = process.env.ANALYZE === 'true'

  return {
    plugins: [
      react({
        // Enable React Fast Refresh optimizations
        fastRefresh: true,
        // Remove React DevTools in production
        babel: {
          plugins: isProduction ? [] : [],
        },
      }),
      // Bundle analyzer (only when ANALYZE=true)
      isAnalyze && visualizer({
        open: true,
        filename: 'dist/stats.html',
        gzipSize: true,
        brotliSize: true,
      }),
    ].filter(Boolean),

    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },

    // Build optimizations
    build: {
      // Output directory
      outDir: 'dist',
      
      // Generate source maps for production (optional - set to false for smaller builds)
      sourcemap: isProduction ? false : true,
      
      // Minification
      minify: isProduction ? 'esbuild' : false, // esbuild is faster than terser
      
      // Chunk size warning limit (in KB)
      chunkSizeWarningLimit: 1000,
      
      // Rollup options for advanced optimizations
      rollupOptions: {
        output: {
          // Manual chunk splitting for better caching
          manualChunks: (id) => {
            // Vendor chunks
            if (id.includes('node_modules')) {
              // React and React DOM in separate chunk
              if (id.includes('react') || id.includes('react-dom')) {
                return 'vendor-react'
              }
              // React Router
              if (id.includes('react-router')) {
                return 'vendor-router'
              }
              // Chart library (recharts)
              if (id.includes('recharts')) {
                return 'vendor-charts'
              }
              // Query library (tanstack)
              if (id.includes('@tanstack')) {
                return 'vendor-query'
              }
              // Other vendor libraries
              return 'vendor'
            }
            
            // Route-based code splitting
            if (id.includes('/pages/')) {
              const pageName = id.split('/pages/')[1].split('/')[0]
              return `page-${pageName}`
            }
          },
          
          // Optimize chunk file names
          chunkFileNames: isProduction
            ? 'assets/js/[name]-[hash].js'
            : 'assets/js/[name].js',
          entryFileNames: isProduction
            ? 'assets/js/[name]-[hash].js'
            : 'assets/js/[name].js',
          assetFileNames: (assetInfo) => {
            const info = assetInfo.name.split('.')
            const ext = info[info.length - 1]
            if (/\.(gif|jpe?g|png|svg|webp|avif)$/.test(assetInfo.name)) {
              return `assets/images/[name]-[hash][extname]`
            }
            if (/\.(woff2?|eot|ttf|otf)$/.test(assetInfo.name)) {
              return `assets/fonts/[name]-[hash][extname]`
            }
            if (/\.css$/.test(assetInfo.name)) {
              return `assets/css/[name]-[hash][extname]`
            }
            return `assets/[name]-[hash][extname]`
          },
        },
      },
      
      // CSS code splitting
      cssCodeSplit: true,
      
      // Target modern browsers for smaller bundle
      target: ['es2020', 'edge88', 'firefox78', 'chrome87', 'safari14'],
      
      // Build assets inline threshold (4KB)
      assetsInlineLimit: 4096,
      
      // Report compressed size
      reportCompressedSize: true,
    },

    // ESBuild options for production
    esbuild: isProduction ? {
      drop: ['console', 'debugger'], // Remove console.log and debugger in production
      legalComments: 'none', // Remove legal comments
    } : {},

    // Optimize dependencies
    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        'react-router-dom',
        '@tanstack/react-query',
        'axios',
        'recharts',
        'lucide-react',
        'zustand',
      ],
      exclude: [],
    },

    // Server configuration (development only)
    server: {
      port: 3001,
      host: '0.0.0.0', // Listen on all network interfaces
      proxy: {
        '/api': {
          target: 'http://localhost:8001',
          changeOrigin: true,
        },
      },
      // Enable HMR
      hmr: {
        overlay: true,
      },
    },

    // Preview server (for testing production build)
    preview: {
      port: 3001,
      host: '0.0.0.0',
    },

    // Test configuration
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: './vitest.setup.ts',
    },

    // Define global constants
    define: {
      __APP_VERSION__: JSON.stringify(process.env.npm_package_version || '1.0.0'),
    },
  }
})

