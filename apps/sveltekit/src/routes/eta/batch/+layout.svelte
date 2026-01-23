<script lang="ts">
	import { BatchSubNav } from '$components/shared';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();
</script>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<h1 class="text-2xl font-bold text-foreground">Batch ML - Estimated Time of Arrival</h1>
		<BatchSubNav projectKey="eta" />
	</div>

	{@render children()}
</div>
