<script lang="ts">
	import { cn } from '$lib/utils';
	import { ChevronDown } from 'lucide-svelte';

	interface Props {
		value?: string;
		options: string[] | { value: string; label: string }[];
		placeholder?: string;
		disabled?: boolean;
		required?: boolean;
		id?: string;
		name?: string;
		class?: string;
		onchange?: (e: Event & { currentTarget: HTMLSelectElement }) => void;
	}

	let {
		value = '',
		options,
		placeholder,
		disabled = false,
		required = false,
		id,
		name,
		class: className,
		onchange
	}: Props = $props();

	const normalizedOptions = $derived(
		options.map((opt) => (typeof opt === 'string' ? { value: opt, label: opt } : opt))
	);
</script>

<div class="group relative">
	<select
		{value}
		{disabled}
		{required}
		{id}
		{name}
		class={cn(
			'flex h-10 w-full cursor-pointer appearance-none rounded-md border border-input bg-background pl-4 pr-10 py-2 text-sm transition-all duration-200',
			'shadow-sm hover:shadow',
			'ring-offset-background',
			'hover:border-muted-foreground/50 hover:bg-background/80',
			'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:border-primary focus:shadow-md',
			'disabled:cursor-not-allowed disabled:opacity-50 disabled:bg-muted disabled:hover:shadow-none',
			className
		)}
		{onchange}
	>
		{#if placeholder}
			<option value="" disabled class="text-muted-foreground">{placeholder}</option>
		{/if}
		{#each normalizedOptions as option}
			<option value={option.value}>{option.label}</option>
		{/each}
	</select>
	<ChevronDown
		class="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground transition-transform duration-200 group-focus-within:rotate-180"
	/>
</div>
