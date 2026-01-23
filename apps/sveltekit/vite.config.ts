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
			interval: 100 // Fast polling for container environments
		},
		hmr: {
			clientPort: 5173
		}
	},
	preview: {
		port: 5173,
		host: '0.0.0.0'
	},
	optimizeDeps: {
		// Pre-bundle heavy dependencies to avoid re-optimization
		include: [
			'plotly.js-dist-min',
			'svelte-plotly.js',
			'lucide-svelte',
			'bits-ui',
			'clsx',
			'tailwind-merge',
			'tailwind-variants'
		],
		// Don't re-scan for new deps on file changes
		noDiscovery: true
	}
});
