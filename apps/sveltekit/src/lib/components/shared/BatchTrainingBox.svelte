<script lang="ts">
	import { cn } from '$lib/utils';
	import { Loader2, Play, Square, Settings, Brain, Database } from 'lucide-svelte';

	interface Props {
		modelName: string;
		isTraining: boolean;
		progress: number;
		stage: string;
		status: string;
		mode: 'percentage' | 'max_rows';
		percentage: number;
		maxRows: number;
		totalRows: number;
		onModeChange: (mode: 'percentage' | 'max_rows') => void;
		onPercentageChange: (value: number) => void;
		onMaxRowsChange: (value: number) => void;
		onStartTraining: () => void;
		onStopTraining: () => void;
	}

	let {
		modelName,
		isTraining,
		progress,
		stage,
		status,
		mode,
		percentage,
		maxRows,
		totalRows,
		onModeChange,
		onPercentageChange,
		onMaxRowsChange,
		onStartTraining,
		onStopTraining
	}: Props = $props();
</script>

<div
	class={cn(
		'relative overflow-hidden rounded-md border bg-card transition-all duration-300',
		isTraining ? 'border-blue-500/40 shadow-md shadow-blue-500/10' : 'border-border'
	)}
>
	<!-- Header -->
	<div class="flex items-center justify-between border-b border-border bg-muted/30 px-3 py-2.5">
		<div class="flex items-center gap-2">
			<div
				class={cn(
					'flex h-7 w-7 items-center justify-center rounded-md',
					isTraining ? 'bg-blue-500/10 text-blue-600' : 'bg-muted text-muted-foreground'
				)}
			>
				<Brain class="h-4 w-4" />
			</div>
			<div>
				<h3 class="text-xs font-semibold text-foreground">Batch ML Training</h3>
				<p class="text-[10px] text-muted-foreground">{modelName}</p>
			</div>
		</div>

		{#if isTraining}
			<button
				type="button"
			class="inline-flex items-center gap-1.5 rounded-md bg-blue-800 px-2.5 py-1.5 text-xs font-medium text-white shadow-sm transition-all hover:bg-blue-900 hover:shadow-md"
				onclick={onStopTraining}
			>
				<Square class="h-3 w-3" />
				Stop
			</button>
		{:else}
			<button
				type="button"
			class="inline-flex items-center gap-1.5 rounded-md bg-blue-600 px-2.5 py-1.5 text-xs font-medium text-white shadow-sm transition-all hover:bg-blue-700 hover:shadow-md"
				onclick={onStartTraining}
			>
				<Play class="h-3 w-3" />
				Train
			</button>
		{/if}
	</div>

	<div class="p-3">
		{#if !isTraining}
			<!-- Training Configuration -->
			<div class="space-y-3">
				<!-- Mode Selection -->
				<div>
					<label class="mb-1.5 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
						<Settings class="h-3 w-3" />
						Training Mode
					</label>
					<div class="flex gap-1.5">
						<button
							type="button"
							class={cn(
							'flex-1 rounded-md border-2 px-2.5 py-1.5 text-xs font-medium transition-all',
								mode === 'max_rows'
						? 'border-blue-500 bg-blue-500/5 text-blue-700'
									: 'border-border text-muted-foreground hover:border-muted-foreground/50 hover:text-foreground'
							)}
							onclick={() => onModeChange('max_rows')}
						>
							Max Rows
						</button>
						<button
							type="button"
							class={cn(
							'flex-1 rounded-md border-2 px-2.5 py-1.5 text-xs font-medium transition-all',
								mode === 'percentage'
						? 'border-blue-500 bg-blue-500/5 text-blue-700'
									: 'border-border text-muted-foreground hover:border-muted-foreground/50 hover:text-foreground'
							)}
							onclick={() => onModeChange('percentage')}
						>
							Percentage
						</button>
					</div>
				</div>

				{#if mode === 'max_rows'}
					<div>
						<label
							for="max-rows"
							class="mb-1.5 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground"
						>
							<Database class="h-3 w-3" />
							Max Training Rows
						</label>
						<input
							id="max-rows"
							type="number"
							min="1000"
							max={totalRows || 10000000}
							value={maxRows}
							oninput={(e) => onMaxRowsChange(parseInt(e.currentTarget.value) || 10000)}
						class="w-full rounded-md border border-input bg-background px-2.5 py-1.5 text-xs shadow-sm transition-all hover:border-muted-foreground/50 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:ring-offset-2"
						/>
						{#if totalRows > 0}
							<p class="mt-1.5 flex items-center gap-1 text-[10px] text-muted-foreground">
							<span class="inline-block h-1 w-1 rounded-full bg-blue-500"></span>
								Available: {totalRows.toLocaleString()} rows
							</p>
						{/if}
					</div>
				{:else}
					<div>
						<label
							for="percentage"
							class="mb-1.5 flex items-center justify-between text-[10px] font-medium uppercase tracking-wider text-muted-foreground"
						>
							<span>Training Data</span>
							<span class="rounded-full bg-blue-500/10 px-1.5 py-0.5 text-[10px] font-bold text-blue-700">
								{percentage}%
							</span>
						</label>
						<input
							id="percentage"
							type="range"
							min="1"
							max="100"
							value={percentage}
							oninput={(e) => onPercentageChange(parseInt(e.currentTarget.value))}
							class="mt-1.5 w-full"
						/>
						<div class="mt-0.5 flex justify-between text-[10px] text-muted-foreground">
							<span>1%</span>
							<span>50%</span>
							<span>100%</span>
						</div>
					</div>
				{/if}
			</div>
		{:else}
			<!-- Training Progress -->
			<div class="space-y-2.5">
				<div class="flex items-center gap-2">
					<div class="flex h-6 w-6 items-center justify-center rounded-md bg-blue-500/10">
						<Loader2 class="h-3 w-3 animate-spin text-blue-600" />
					</div>
					<div>
						<span class="text-xs font-semibold text-foreground">{stage || 'Training...'}</span>
						<p class="text-[10px] text-muted-foreground">{status}</p>
					</div>
				</div>

				<div class="space-y-1">
					<div class="flex justify-between text-[10px]">
						<span class="font-medium text-muted-foreground">Progress</span>
						<span class="font-bold text-blue-700">{progress}%</span>
					</div>
					<div class="h-2 w-full overflow-hidden rounded-full bg-muted">
						<div
						class="progress-bar h-full rounded-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-300"
							style="width: {progress}%"
						></div>
					</div>
				</div>
			</div>
		{/if}
	</div>
</div>
