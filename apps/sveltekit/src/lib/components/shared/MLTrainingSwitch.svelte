<script lang="ts">
	import { cn } from '$lib/utils';
	import { Activity, Brain, Loader2 } from 'lucide-svelte';
	import Button from './Button.svelte';

	interface Props {
		enabled: boolean;
		loading?: boolean;
		modelName: string;
		mlflowUrl?: string;
		onToggle: (enabled: boolean) => void;
	}

	let { enabled, loading = false, modelName, mlflowUrl = '', onToggle }: Props = $props();
</script>

<div
	class={cn('rounded-md border bg-card p-3 transition-all duration-300', enabled
		? 'border-blue-500/40 shadow-md shadow-blue-500/10'
		: 'border-border')}
>
	<div class="flex items-center justify-between gap-3">
		<div class="space-y-0.5">
			<div class="flex items-center gap-1.5">
				<Activity class={cn('h-4 w-4', enabled ? 'text-blue-500' : 'text-muted-foreground')} />
				<h3 class="text-sm font-medium text-foreground">Real-time ML Training</h3>
			</div>
			<p class="text-[10px] text-muted-foreground">
				{enabled ? 'Processing live Kafka stream data' : 'Toggle to start processing live data'}
			</p>
		</div>

		<button
			type="button"
			class={cn(
			'relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-md border-2 border-transparent transition-all duration-200',
				'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
				enabled ? 'bg-blue-500' : 'bg-muted',
				loading && 'cursor-not-allowed opacity-50'
			)}
			role="switch"
			aria-checked={enabled}
			disabled={loading}
			onclick={() => onToggle(!enabled)}
		>
			<span class="sr-only">Toggle ML training</span>
			<span
				class={cn(
				'pointer-events-none inline-flex h-5 w-5 transform items-center justify-center rounded-sm bg-white shadow-lg ring-0 transition-all duration-200',
					enabled ? 'translate-x-5' : 'translate-x-0'
				)}
			>
				{#if loading}
					<Loader2 class="h-3 w-3 animate-spin text-muted-foreground" />
				{/if}
			</span>
		</button>
	</div>

	<hr class="my-2.5 border-border" />

	<div class="flex items-center gap-1.5">
		<span class="inline-flex items-center gap-1 rounded-md bg-blue-500/10 px-2 py-0.5 text-[10px] font-medium text-blue-600">
			<Brain class="h-3 w-3" />
			{modelName}
		</span>
		{#if mlflowUrl}
			<a class="flex-1" href={mlflowUrl} target="_blank" rel="noreferrer">
				<Button variant="secondary" size="sm" class="w-full">
					<img
						src="https://cdn.simpleicons.org/mlflow/0194E2"
						alt="MLflow"
						class="h-3 w-3"
					/>
					MLflow
				</Button>
			</a>
		{:else}
			<Button variant="secondary" size="sm" class="w-full opacity-60" disabled>
				<img
					src="https://cdn.simpleicons.org/mlflow/0194E2"
					alt="MLflow"
					class="h-3 w-3 opacity-60"
				/>
				MLflow
			</Button>
		{/if}
	</div>
</div>
