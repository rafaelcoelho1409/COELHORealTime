<script lang="ts">
	import type { Snippet } from 'svelte';
	import { getContext } from 'svelte';
	import { cn } from '$lib/utils';
	import { TABS_CONTEXT_KEY, type TabsContext } from './Tabs.svelte';

	interface Props {
		value: string;
		children: Snippet;
		class?: string;
	}

	let { value, children, class: className = '' }: Props = $props();

	const ctx = getContext<TabsContext>(TABS_CONTEXT_KEY);
	const isActive = $derived(ctx?.value === value);
</script>

{#if isActive}
	<div
		role="tabpanel"
		data-state={isActive ? 'active' : 'inactive'}
		class={cn(
			'mt-4 animate-fade-in ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
			className
		)}
	>
		{@render children()}
	</div>
{/if}
