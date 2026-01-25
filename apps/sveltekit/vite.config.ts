import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		port: 5173,
		host: '0.0.0.0',
		strictPort: true,
		watch: {
			usePolling: true,
			interval: 1000
		},
		hmr: {
			clientPort: 5173
		},
		// Pre-transform critical files for faster first load
		warmup: {
			clientFiles: [
				'./src/routes/+layout.svelte',
				'./src/routes/+page.svelte',
				'./src/lib/components/shared/*.svelte',
				'./src/app.css'
			]
		}
	},
	preview: {
		port: 5173,
		host: '0.0.0.0'
	},
	optimizeDeps: {
		// Pre-bundle ALL heavy dependencies including sub-modules
		include: [
			// Charting - very heavy
			'plotly.js-dist-min',
			'svelte-plotly.js',
			// Icons - many individual modules
			'lucide-svelte',
			'lucide-svelte/icons/*',
			// UI components
			'bits-ui',
			'bits-ui/**',
			// Utilities
			'clsx',
			'tailwind-merge',
			'tailwind-variants'
		],
		// Exclude svelte internals (they're handled specially)
		exclude: ['svelte', '@sveltejs/kit'],
		// Don't re-scan for new deps during dev
		noDiscovery: true,
		// Hold until initial crawl completes
		holdUntilCrawlEnd: true
	},
	// Fast transpilation
	esbuild: {
		target: 'esnext'
	},
	// Reduce CSS processing on first load
	css: {
		devSourcemap: false
	}
});
