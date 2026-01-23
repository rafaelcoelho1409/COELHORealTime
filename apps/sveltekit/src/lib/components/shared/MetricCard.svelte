<script lang="ts">
	import { cn, formatNumber } from '$lib/utils';
	import { Info, TrendingUp, TrendingDown, Minus } from 'lucide-svelte';
	import type { MetricInfo } from '$types';

	interface Props {
		name: string;
		value: number | undefined;
		decimals?: number;
		info?: MetricInfo;
		trend?: 'up' | 'down' | 'neutral';
		trendValue?: string;
		onInfoClick?: () => void;
		class?: string;
	}

	let {
		name,
		value,
		decimals = 4,
		info,
		trend,
		trendValue,
		onInfoClick,
		class: className
	}: Props = $props();

	const trendIcons = {
		up: TrendingUp,
		down: TrendingDown,
		neutral: Minus
	};

	const trendColors = {
		up: 'text-green-500',
		down: 'text-red-500',
		neutral: 'text-muted-foreground'
	};
</script>

<div
	class={cn(
		'group relative overflow-hidden rounded-md border border-border bg-card p-2.5 transition-all duration-200 hover:shadow-md hover:border-primary/20',
		className
	)}
>
	<!-- Subtle gradient overlay on hover -->
	<div
		class="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 transition-opacity group-hover:opacity-100"
	></div>

	<div class="relative">
		<div class="flex items-start justify-between gap-1">
			<div class="flex-1 min-w-0">
				<p class="text-xs font-medium text-muted-foreground truncate">{name}</p>
				<div class="mt-1 flex items-baseline gap-1.5">
					<p class="text-xl font-bold tracking-tight text-foreground">
						{formatNumber(value, decimals)}
					</p>
					{#if trend && trendValue}
						{@const TrendIcon = trendIcons[trend]}
						<span class={cn('flex items-center gap-0.5 text-[10px] font-medium', trendColors[trend])}>
							<TrendIcon class="h-2.5 w-2.5" />
							{trendValue}
						</span>
					{/if}
				</div>
			</div>

			{#if onInfoClick}
				<button
					type="button"
					class="flex-shrink-0 rounded-full p-1 text-muted-foreground transition-all hover:bg-blue-500/10 hover:text-blue-600"
					onclick={onInfoClick}
					title="More info"
				>
					<Info class="h-3 w-3" />
				</button>
			{/if}
		</div>

		{#if info}
			<p class="mt-2 text-[10px] leading-relaxed text-muted-foreground line-clamp-2">
				{info.context}
			</p>
		{/if}
	</div>
</div>
