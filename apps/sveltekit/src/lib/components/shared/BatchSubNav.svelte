<script lang="ts">
	import { page } from '$app/stores';
	import { cn } from '$lib/utils';
	import { Target, BarChart2 } from 'lucide-svelte';
	import type { ProjectKey } from '$types';

	interface Props {
		projectKey: ProjectKey;
	}

	let { projectKey }: Props = $props();

	const subTabs = $derived([
		{
			href: `/${projectKey}/batch/prediction`,
			label: 'Prediction',
			icon: Target
		},
		{
			href: `/${projectKey}/batch/metrics`,
			label: 'Metrics',
			icon: BarChart2
		}
	]);

	// Project-specific colors
	const projectColors: Record<ProjectKey, string> = {
		tfd: 'data-[active=true]:bg-blue-600 data-[active=true]:text-white',
		eta: 'data-[active=true]:bg-blue-600 data-[active=true]:text-white',
		ecci: 'data-[active=true]:bg-blue-600 data-[active=true]:text-white'
	};
</script>

<div class="flex items-center gap-1 rounded-lg bg-muted/50 p-1.5 shadow-sm">
	{#each subTabs as tab}
		{@const Icon = tab.icon}
		{@const active = $page.url.pathname === tab.href}
		<a
			href={tab.href}
			class={cn(
				'flex flex-1 items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all duration-200',
				'text-muted-foreground hover:text-foreground',
				active ? 'shadow-md' : 'hover:bg-muted',
				projectColors[projectKey]
			)}
			data-active={active}
		>
			<Icon class="h-4 w-4" />
			{tab.label}
		</a>
	{/each}
</div>
