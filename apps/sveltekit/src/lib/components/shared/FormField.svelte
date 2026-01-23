<script lang="ts">
	import { cn } from '$lib/utils';
	import { AlertCircle } from 'lucide-svelte';
	import type { Snippet } from 'svelte';

	interface Props {
		label: string;
		id?: string;
		required?: boolean;
		error?: string;
		hint?: string;
		class?: string;
		children: Snippet;
	}

	let { label, id, required = false, error, hint, class: className, children }: Props = $props();
</script>

<div class={cn('space-y-2', className)}>
	<label
		for={id}
		class="flex items-center gap-1 text-sm font-medium leading-none text-foreground peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
	>
		{label}
		{#if required}
			<span class="text-destructive">*</span>
		{/if}
	</label>

	{@render children()}

	{#if error}
		<p class="flex items-center gap-1.5 text-sm text-destructive">
			<AlertCircle class="h-3.5 w-3.5" />
			{error}
		</p>
	{:else if hint}
		<p class="text-xs text-muted-foreground">{hint}</p>
	{/if}
</div>
