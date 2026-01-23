<script lang="ts">
	import { onMount } from 'svelte';
	import { Card, CardContent, Button } from '$components/shared';
	import {
		sqlQueryInput,
		sqlQueryResults,
		sqlLoading,
		sqlError,
		sqlExecutionTime,
		SQL_QUERY_TEMPLATES,
		updateSqlQuery,
		updateSqlResults,
		setSqlLoading,
		setSqlError,
		setSqlExecutionTime
	} from '$stores/sql';
	import { toast } from '$stores/ui';
	import * as sqlApi from '$api/sql';
	import { Play, Eraser, Terminal, Table2, Bookmark, Database, Clock, RefreshCw, ChevronDown } from 'lucide-svelte';
	import type { ProjectName } from '$types';

	const PROJECT: ProjectName = 'Estimated Time of Arrival';

	let tableSearch = $state('');
	let tableRowCount = $state(0);
	let tableColumns = $state<{ name: string; type: string }[]>([]);
	let columnsExpanded = $state(false);

	onMount(async () => {
		await fetchTableSchema();
	});

	async function fetchTableSchema() {
		const result = await sqlApi.getTableSchema(PROJECT);
		if (result.data) {
			tableRowCount = result.data.row_count || 0;
			tableColumns = result.data.columns || [];
		}
	}

	async function executeQuery() {
		const query = $sqlQueryInput[PROJECT];
		if (!query.trim()) {
			toast.error('Please enter a query');
			return;
		}

		setSqlLoading(PROJECT, true);
		setSqlError(PROJECT, '');

		const startTime = performance.now();
		const result = await sqlApi.executeQuery(PROJECT, query);
		const endTime = performance.now();

		setSqlExecutionTime(PROJECT, Math.round(endTime - startTime));

		if (result.error) {
			setSqlError(PROJECT, result.error);
			toast.error('Query failed');
		} else if (result.data) {
			updateSqlResults(PROJECT, result.data);
			toast.success(`Query returned ${result.data.row_count} rows`);
		}

		setSqlLoading(PROJECT, false);
	}

	function clearQuery() {
		updateSqlQuery(PROJECT, '');
	}

	function applyTemplate(query: string) {
		updateSqlQuery(PROJECT, query);
	}

	const currentQuery = $derived($sqlQueryInput[PROJECT]);
	const currentResults = $derived($sqlQueryResults[PROJECT]);
	const isLoading = $derived($sqlLoading[PROJECT]);
	const currentError = $derived($sqlError[PROJECT]);
	const executionTime = $derived($sqlExecutionTime[PROJECT]);
	const templates = SQL_QUERY_TEMPLATES[PROJECT];

	const filteredData = $derived(
		tableSearch
			? currentResults.data.filter((row) =>
					Object.values(row).some((val) =>
						String(val).toLowerCase().includes(tableSearch.toLowerCase())
					)
				)
			: currentResults.data
	);

	const hasResults = $derived(currentResults.columns.length > 0);
</script>

<!-- 40%/60% Layout -->
<div class="flex gap-6">
	<!-- Left Column - Query Editor, Table Info, Templates (40%) -->
	<div class="w-[40%] space-y-3">
		<!-- SQL Editor Card -->
		<Card>
			<CardContent class="space-y-3 pt-4">
				<div class="flex items-center justify-between">
					<div class="flex items-center gap-2">
						<Terminal class="h-4 w-4 text-primary" />
						<span class="text-sm font-bold">SQL Editor</span>
					</div>
					<span class="flex items-center gap-1 rounded-full bg-orange-100 px-2 py-0.5 text-xs text-orange-700 dark:bg-orange-900 dark:text-orange-300">
						<Database class="h-3 w-3" />
						DuckDB
					</span>
				</div>

				<hr class="border-border" />

				<textarea
					value={currentQuery}
					oninput={(e) => updateSqlQuery(PROJECT, e.currentTarget.value)}
					class="min-h-[180px] w-full rounded-md border border-input bg-muted/30 p-3 font-mono text-xs text-foreground placeholder-muted-foreground focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring"
					placeholder="SELECT * FROM data LIMIT 100"
					onkeydown={(e) => {
						if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
							e.preventDefault();
							executeQuery();
						}
					}}
				></textarea>

				<div class="flex gap-2">
					<Button size="sm" class="flex-1" onclick={executeQuery} loading={isLoading}>
						{#if !isLoading}
							<Play class="mr-1 h-3.5 w-3.5" />
						{/if}
						Run
					</Button>
					<Button variant="outline" size="sm" onclick={clearQuery}>
						<Eraser class="h-3.5 w-3.5" />
					</Button>
				</div>
			</CardContent>
		</Card>

		<!-- Table Info Card -->
		<Card>
			<CardContent class="space-y-2 py-3">
				<div class="flex items-center justify-between">
					<div class="flex items-center gap-2">
						<Table2 class="h-3.5 w-3.5 text-muted-foreground" />
						<span class="text-xs text-muted-foreground">Table:</span>
						<code class="rounded bg-muted px-1 text-xs">data</code>
					</div>
					{#if tableRowCount > 0}
						<span class="text-xs text-muted-foreground">~{tableRowCount.toLocaleString()} rows</span>
					{:else}
						<button type="button" onclick={fetchTableSchema} class="text-muted-foreground hover:text-foreground">
							<RefreshCw class="h-3 w-3" />
						</button>
					{/if}
				</div>

				{#if tableColumns.length > 0}
					<button
						type="button"
						class="flex w-full items-center justify-between rounded px-2 py-1 text-xs text-muted-foreground hover:bg-muted"
						onclick={() => (columnsExpanded = !columnsExpanded)}
					>
						<span>Columns ({tableColumns.length})</span>
						<ChevronDown class="h-3 w-3 transition-transform {columnsExpanded ? 'rotate-180' : ''}" />
					</button>
					{#if columnsExpanded}
						<div class="max-h-[150px] space-y-1 overflow-y-auto px-2">
							{#each tableColumns as col}
								<div class="flex items-center gap-2 text-xs">
									<code class="text-foreground">{col.name}</code>
									<span class="text-muted-foreground">{col.type}</span>
								</div>
							{/each}
						</div>
					{/if}
				{/if}
			</CardContent>
		</Card>

		<!-- Query Templates Card -->
		<Card>
			<CardContent class="space-y-2 py-3">
				<div class="flex items-center gap-2">
					<Bookmark class="h-3.5 w-3.5 text-muted-foreground" />
					<span class="text-xs font-medium text-muted-foreground">Templates</span>
				</div>
				<div class="space-y-0.5">
					{#each templates as template}
						<button
							type="button"
							class="w-full rounded px-2 py-1 text-left text-xs transition-colors hover:bg-muted"
							onclick={() => applyTemplate(template.query)}
						>
							{template.name}
						</button>
					{/each}
				</div>
			</CardContent>
		</Card>
	</div>

	<!-- Right Column - Query Results (60%) -->
	<div class="w-[60%]">
		<Card class="h-full">
			<CardContent class="flex h-full flex-col space-y-3 pt-4">
				<!-- Header -->
				<div class="flex items-center justify-between">
					<div class="flex items-center gap-2">
						<Database class="h-5 w-5 text-primary" />
						<span class="text-base font-bold">Query Results</span>
					</div>
					{#if hasResults}
						<div class="flex items-center gap-2">
							<input
								type="search"
								placeholder="Filter results..."
								bind:value={tableSearch}
								class="h-7 w-40 rounded-md border border-input bg-muted/30 px-2 text-xs"
							/>
							<span class="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
								{filteredData.length}/{currentResults.row_count}
							</span>
							{#if executionTime > 0}
								<span class="flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-xs text-green-700 dark:bg-green-900 dark:text-green-300">
									<Clock class="h-3 w-3" />
									{executionTime}ms
								</span>
							{/if}
						</div>
					{/if}
				</div>

				<hr class="border-border" />

				<!-- Error Display -->
				{#if currentError}
					<div class="flex items-start gap-2 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
						<span>{currentError}</span>
					</div>
				{/if}

				<!-- Loading State -->
				{#if isLoading}
					<div class="flex flex-1 flex-col items-center justify-center py-12">
						<div class="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
						<p class="mt-2 text-sm text-muted-foreground">Executing query...</p>
					</div>
				{:else if hasResults}
					<!-- Results Table -->
					<div class="flex-1 overflow-auto rounded-md border border-border">
						<table class="w-full text-sm">
							<thead class="sticky top-0 bg-muted">
								<tr>
									{#each currentResults.columns as col}
										<th class="whitespace-nowrap border-b border-border px-4 py-2 text-left text-xs font-medium">
											{col}
										</th>
									{/each}
								</tr>
							</thead>
							<tbody>
								{#each filteredData as row, i}
									<tr class={i % 2 === 0 ? 'bg-background' : 'bg-muted/30'}>
										{#each currentResults.columns as col}
											<td class="whitespace-nowrap border-b border-border px-4 py-2 text-xs">
												{row[col] ?? '-'}
											</td>
										{/each}
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{:else}
					<!-- Empty State -->
					<div class="flex flex-1 flex-col items-center justify-center py-12 text-center">
						<Database class="h-12 w-12 text-muted-foreground/50" />
						<p class="mt-4 text-sm text-muted-foreground">Execute a query to see results</p>
						<p class="mt-1 text-xs text-muted-foreground">Press Ctrl+Enter to run</p>
					</div>
				{/if}
			</CardContent>
		</Card>
	</div>
</div>
