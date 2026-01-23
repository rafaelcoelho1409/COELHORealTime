<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		class?: string;
	}

	let { children, class: className = '' }: Props = $props();
</script>

<div
	class="inline-flex h-11 items-center justify-center gap-1 rounded-lg bg-muted/50 p-1.5 text-muted-foreground shadow-sm {className}"
	role="tablist"
>
	{@render children()}
</div>
