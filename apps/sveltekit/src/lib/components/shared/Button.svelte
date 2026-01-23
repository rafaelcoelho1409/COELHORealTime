<script lang="ts">
	import { cn } from '$lib/utils';
	import { Loader2 } from 'lucide-svelte';
	import type { Snippet } from 'svelte';

	interface Props {
		variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link' | 'success';
		size?: 'default' | 'sm' | 'lg' | 'icon';
		disabled?: boolean;
		loading?: boolean;
		type?: 'button' | 'submit' | 'reset';
		class?: string;
		onclick?: () => void;
		children: Snippet;
	}

	let {
		variant = 'default',
		size = 'default',
		disabled = false,
		loading = false,
		type = 'button',
		class: className,
		onclick,
		children
	}: Props = $props();

	const variants = {
		default:
			'bg-blue-600 text-white shadow-sm hover:bg-blue-700 hover:shadow-md active:shadow-sm',
		destructive:
			'bg-blue-800 text-white shadow-sm hover:bg-blue-900 hover:shadow-md active:shadow-sm',
		outline:
			'border-2 border-input bg-transparent hover:bg-accent hover:text-accent-foreground hover:border-primary/50',
		secondary:
			'bg-secondary text-secondary-foreground shadow-sm hover:bg-secondary/80 hover:shadow-md',
		ghost: 'hover:bg-accent hover:text-accent-foreground',
		link: 'text-primary underline-offset-4 hover:underline',
		success: 'bg-blue-500 text-white shadow-sm hover:bg-blue-600 hover:shadow-md active:shadow-sm'
	};

	const sizes = {
		default: 'h-8 px-3 py-1.5 text-sm',
		sm: 'h-7 rounded-md px-2.5 text-xs',
		lg: 'h-10 rounded-lg px-6 text-sm',
		icon: 'h-8 w-8'
	};
</script>

<button
	{type}
	class={cn(
		'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md font-medium ring-offset-background transition-all duration-200',
		'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
		'disabled:pointer-events-none disabled:opacity-50',
		'active:scale-[0.98] active:transition-transform',
		variants[variant],
		sizes[size],
		className
	)}
	disabled={disabled || loading}
	{onclick}
>
	{#if loading}
		<Loader2 class="h-4 w-4 animate-spin" />
	{/if}
	{@render children()}
</button>
