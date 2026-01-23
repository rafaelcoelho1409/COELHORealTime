<script lang="ts">
	import { PageSubNav } from '$components/shared';
	import { setCurrentProject } from '$stores';
	import { onMount } from 'svelte';
	import { ShoppingCart } from 'lucide-svelte';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();

	onMount(() => {
		setCurrentProject('E-Commerce Customer Interactions', 'ecci');
	});
</script>

<svelte:head>
	<title>E-Commerce Customer Interactions - COELHO RealTime</title>
</svelte:head>

<!-- Project Header Banner -->
<div class="border-b border-border bg-gradient-to-r from-green-500/5 via-transparent to-emerald-500/5">
	<div class="mx-auto max-w-6xl px-3 py-3 sm:px-5 lg:px-6">
		<div class="flex items-center gap-3">
			<div class="flex h-10 w-10 items-center justify-center rounded-md bg-green-500/10">
				<ShoppingCart class="h-5 w-5 text-green-500" />
			</div>
			<div>
				<h1 class="text-lg font-bold text-foreground">E-Commerce Customer Interactions</h1>
				<p class="text-xs text-muted-foreground">Customer segmentation and behavior analysis</p>
			</div>
		</div>
	</div>
</div>

<PageSubNav projectKey="ecci" />

<div class="mx-auto max-w-6xl px-3 py-5 sm:px-5 lg:px-6">
	{@render children()}
</div>
