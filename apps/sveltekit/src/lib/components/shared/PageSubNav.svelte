<script lang="ts">
	import { page } from '$app/stores';
	import { cn } from '$lib/utils';
	import { Activity, Layers, Database } from 'lucide-svelte';
	import type { ProjectKey } from '$types';

	interface Props {
		projectKey: ProjectKey;
	}

	let { projectKey }: Props = $props();

	const tabs = $derived([
		{
			href: `/${projectKey}/incremental`,
			label: 'Incremental ML',
			description: 'Real-time streaming ML with River',
			icon: Activity
		},
		{
			href: `/${projectKey}/batch/prediction`,
			label: 'Batch ML',
			description: 'Batch training with Scikit-Learn/CatBoost',
			icon: Layers,
			prefixes: [`/${projectKey}/batch`]
		},
		{
			href: `/${projectKey}/sql`,
			label: 'Delta Lake SQL',
			description: 'Query data with SQL',
			icon: Database
		}
	]);

	function isActive(tab: (typeof tabs)[0], pathname: string): boolean {
		if (tab.prefixes) {
			return tab.prefixes.some((prefix) => pathname.startsWith(prefix));
		}
		return pathname === tab.href;
	}

	// Project-specific accent colors
	const projectColors: Record<ProjectKey, string> = {
		tfd: 'data-[active=true]:border-blue-600 data-[active=true]:text-blue-600 dark:data-[active=true]:text-blue-400',
		eta: 'data-[active=true]:border-blue-600 data-[active=true]:text-blue-600 dark:data-[active=true]:text-blue-400',
		ecci: 'data-[active=true]:border-blue-600 data-[active=true]:text-blue-600 dark:data-[active=true]:text-blue-400'
	};
</script>

<div class="border-b border-border bg-card/50 backdrop-blur-sm">
	<div class="mx-auto max-w-6xl px-3 sm:px-5 lg:px-6">
		<nav class="-mb-px flex gap-1" aria-label="Tabs">
			{#each tabs as tab}
				{@const Icon = tab.icon}
				{@const active = isActive(tab, $page.url.pathname)}
				<a
					href={tab.href}
					class={cn(
						'group relative flex items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium transition-all duration-200',
						'border-transparent text-muted-foreground',
						'hover:text-foreground hover:bg-muted/50',
						projectColors[projectKey]
					)}
					data-active={active}
					title={tab.description}
				>
					<Icon
						class={cn(
							'h-4 w-4 transition-colors',
							active ? 'opacity-100' : 'opacity-60 group-hover:opacity-100'
						)}
					/>
					<span>{tab.label}</span>

					{#if active}
						<span
							class="absolute bottom-0 left-0 right-0 h-0.5 bg-current rounded-full"
						></span>
					{/if}
				</a>
			{/each}
		</nav>
	</div>
</div>
