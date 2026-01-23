<script lang="ts">
	import '../app.css';
	import { Navbar, Toast, MetricInfoDialog } from '$components/shared';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();
</script>

<div class="relative min-h-screen bg-background">
	<!-- Subtle background pattern -->
	<div class="pointer-events-none fixed inset-0 -z-10">
		<div class="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,hsl(var(--primary)/0.08),transparent)]"></div>
		<div class="absolute inset-0 opacity-[0.015]" style="background-image: url(&quot;data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23888' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E&quot;);"></div>
	</div>

	<Navbar />
	<main class="relative">
		{@render children()}
	</main>
	<Toast />
	<MetricInfoDialog />
</div>
