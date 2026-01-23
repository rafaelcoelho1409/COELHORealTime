<script lang="ts">
	import { cn } from '$lib/utils';
	import type { Snippet } from 'svelte';

	interface Props {
		variant?: 'default' | 'elevated' | 'outline' | 'ghost';
		hover?: boolean;
		class?: string;
		children: Snippet;
	}

	let { variant = 'default', hover = false, class: className, children }: Props = $props();

	const variants = {
		default: 'border border-border bg-card shadow-sm',
		elevated: 'border border-border bg-card shadow-lg',
		outline: 'border-2 border-border bg-transparent',
		ghost: 'border-0 bg-transparent'
	};

	const hoverStyles = hover
		? 'transition-all duration-200 hover:shadow-lg hover:border-primary/20 hover:-translate-y-0.5'
		: '';
</script>

<div
	class={cn(
		'rounded-lg text-card-foreground',
		variants[variant],
		hoverStyles,
		className
	)}
>
	{@render children()}
</div>
