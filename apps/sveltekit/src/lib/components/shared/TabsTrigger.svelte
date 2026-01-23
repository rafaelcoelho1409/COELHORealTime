<script lang="ts">
	import type { Snippet } from 'svelte';
	import { getContext } from 'svelte';
	import { cn } from '$lib/utils';
	import { TABS_CONTEXT_KEY, type TabsContext } from './Tabs.svelte';

	interface Props {
		value: string;
		children: Snippet;
		class?: string;
		disabled?: boolean;
	}

	let { value, children, class: className = '', disabled = false }: Props = $props();

	const ctx = getContext<TabsContext>(TABS_CONTEXT_KEY);
	const isActive = $derived(ctx?.value === value);

	function handleClick() {
		if (disabled || !ctx) return;
		ctx.setValue(value);
	}
</script>

<button
	type="button"
	role="tab"
	aria-selected={isActive}
	data-state={isActive ? 'active' : 'inactive'}
	{disabled}
	onclick={handleClick}
	class={cn(
		'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md px-4 py-2 text-sm font-medium transition-all duration-200',
		'ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
		'disabled:pointer-events-none disabled:opacity-50',
		isActive
			? 'bg-blue-600/10 text-blue-700 shadow-md'
			: 'text-muted-foreground hover:bg-blue-600/5 hover:text-foreground',
		className
	)}
>
	{@render children()}
</button>
