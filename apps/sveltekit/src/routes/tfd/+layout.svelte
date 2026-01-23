<script lang="ts">
	import { PageSubNav } from '$components/shared';
	import { setCurrentProject } from '$stores';
	import { onMount } from 'svelte';
	import { Shield } from 'lucide-svelte';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();

	onMount(() => {
		setCurrentProject('Transaction Fraud Detection', 'tfd');
	});
</script>

<svelte:head>
	<title>Transaction Fraud Detection - COELHO RealTime</title>
</svelte:head>

<!-- Project Header Banner -->
<div class="border-b border-border bg-gradient-to-r from-red-500/5 via-transparent to-rose-500/5">
	<div class="mx-auto max-w-6xl px-3 py-3 sm:px-5 lg:px-6">
		<div class="flex items-center gap-3">
			<div class="flex h-10 w-10 items-center justify-center rounded-md bg-red-500/10">
				<Shield class="h-5 w-5 text-red-500" />
			</div>
			<div>
				<h1 class="text-lg font-bold text-foreground">Transaction Fraud Detection</h1>
				<p class="text-xs text-muted-foreground">Real-time fraud detection using streaming ML</p>
			</div>
		</div>
	</div>
</div>

<PageSubNav projectKey="tfd" />

<div class="mx-auto max-w-6xl px-3 py-5 sm:px-5 lg:px-6">
	{@render children()}
</div>
