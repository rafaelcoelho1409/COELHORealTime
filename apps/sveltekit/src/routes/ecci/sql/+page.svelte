<script lang="ts">
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
	import {
		Play,
		Eraser,
		Terminal,
		Bookmark,
		Database,
		Clock,
		ChevronDown,
		ChevronUp,
		ChevronsUpDown,
		Square,
		ChevronLeft,
		ChevronRight
	} from 'lucide-svelte';
	import type { ProjectName } from '$types';
	import { onMount } from 'svelte';

	const PROJECT: ProjectName = 'E-Commerce Customer Interactions';
	const PAGE_SIZE = 50;

	let tableSearch = $state('');
	let sortColumn = $state('');
	let sortDirection = $state<'asc' | 'desc'>('asc');
	let abortController: AbortController | null = $state(null);
	let currentPage = $state(1);

	// Cleanup: Stop query on page refresh, tab change, or navigation
	function handleBeforeUnload() {
		if (abortController) {
			abortController.abort();
		}
	}

	onMount(() => {
		// Add beforeunload listener for page refresh
		window.addEventListener('beforeunload', handleBeforeUnload);

		// Cleanup function runs on component destroy (tab change, navigation)
		return () => {
			window.removeEventListener('beforeunload', handleBeforeUnload);
			if (abortController) {
				abortController.abort();
				abortController = null;
				setSqlLoading(PROJECT, false);
			}
		};
	});

	async function executeQuery() {
		const query = $sqlQueryInput[PROJECT];
		if (!query.trim()) {
			toast.warning('Please enter a SQL query to execute');
			return;
		}

		// Create new abort controller for this query
		abortController = new AbortController();

		setSqlLoading(PROJECT, true);
		setSqlError(PROJECT, '');
		setSqlExecutionTime(PROJECT, 0);
		currentPage = 1; // Reset to first page on new query

		const result = await sqlApi.executeQuery(PROJECT, query, 10000, abortController.signal);

		if (result.error) {
			if (result.error === 'Query cancelled by user') {
				toast.info('Query cancelled');
			} else {
				setSqlError(PROJECT, result.error);
				updateSqlResults(PROJECT, { columns: [], data: [], row_count: 0 });
				toast.error(result.error);
			}
		} else if (result.data) {
			updateSqlResults(PROJECT, result.data);
			setSqlExecutionTime(PROJECT, Math.round(result.data.execution_time_ms || 0));
			toast.success(
				`Query executed (DuckDB): ${result.data.row_count} rows in ${Math.round(result.data.execution_time_ms || 0)}ms`
			);
		}

		abortController = null;
		setSqlLoading(PROJECT, false);
	}

	function stopQuery() {
		if (abortController) {
			abortController.abort();
			abortController = null;
		}
	}

	function clearQuery() {
		updateSqlQuery(PROJECT, 'SELECT * FROM data LIMIT 100');
		updateSqlResults(PROJECT, { columns: [], data: [], row_count: 0 });
		setSqlError(PROJECT, '');
		setSqlExecutionTime(PROJECT, 0);
		currentPage = 1;
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

	const filteredData = $derived.by(() => {
		const search = tableSearch.trim().toLowerCase();
		const data = currentResults.data as Record<string, unknown>[];
		if (!search) {
			return data;
		}
		return data.filter((row) =>
			Object.values(row).some((val) => String(val).toLowerCase().includes(search))
		);
	});

	const sortedData = $derived.by(() => {
		const data = [...filteredData];
		if (!sortColumn) {
			return data;
		}
		return data.sort((a, b) => {
			const aVal = a[sortColumn];
			const bVal = b[sortColumn];
			const aNum = Number(aVal);
			const bNum = Number(bVal);
			let comparison = 0;
			if (!Number.isNaN(aNum) && !Number.isNaN(bNum)) {
				comparison = aNum - bNum;
			} else {
				comparison = String(aVal ?? '').localeCompare(String(bVal ?? ''));
			}
			return sortDirection === 'asc' ? comparison : -comparison;
		});
	});

	// Pagination
	const totalPages = $derived(Math.ceil(sortedData.length / PAGE_SIZE));
	const paginatedData = $derived(
		sortedData.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE)
	);

	function toggleSort(column: string) {
		if (sortColumn !== column) {
			sortColumn = column;
			sortDirection = 'asc';
			return;
		}
		if (sortDirection === 'asc') {
			sortDirection = 'desc';
			return;
		}
		sortColumn = '';
		sortDirection = 'asc';
	}

	function goToPage(page: number) {
		if (page >= 1 && page <= totalPages) {
			currentPage = page;
		}
	}

	const hasResults = $derived(currentResults.columns.length > 0);
</script>

<!-- 35%/65% Layout with matched heights -->
<div class="flex gap-6" style="height: calc(100vh - 180px);">
	<!-- Left Column - SQL Editor, Templates -->
	<div class="flex w-[35%] flex-col gap-3">
		<!-- SQL Editor Card -->
		<Card class="flex-1">
			<CardContent class="flex h-full flex-col space-y-3 pt-4">
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
					class="min-h-[180px] flex-1 w-full rounded-md border border-input bg-muted/30 p-3 font-mono text-xs text-foreground placeholder-muted-foreground focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring resize-none"
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

	<!-- Right Column - Query Results (65%) -->
	<div class="w-[65%] flex flex-col">
		<Card class="flex-1 flex flex-col min-h-0">
			<CardContent class="flex flex-1 flex-col space-y-3 pt-4 min-h-0">
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
								{sortedData.length}/{currentResults.row_count}
							</span>
							{#if executionTime > 0}
								<span class="flex items-center gap-1 rounded-full bg-blue-500/10 px-2 py-0.5 text-xs text-blue-600">
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
						<button
							type="button"
							onclick={stopQuery}
							class="mt-3 inline-flex items-center gap-1 rounded-md bg-red-100 px-3 py-1.5 text-xs font-medium text-red-700 transition-colors hover:bg-red-200 dark:bg-red-900 dark:text-red-300 dark:hover:bg-red-800"
						>
							<Square class="h-3 w-3" />
							Stop
						</button>
					</div>
				{:else if hasResults}
					<!-- Results Table with horizontal and vertical scroll -->
					<div class="flex-1 min-h-0 overflow-auto rounded-md border border-border">
						<table class="w-full text-sm min-w-max">
							<thead class="sticky top-0 z-10 bg-muted">
								<tr>
									{#each currentResults.columns as col}
										<th class="whitespace-nowrap border-b border-border px-3 py-1.5 text-left text-[11px] font-semibold uppercase tracking-wide">
											<button
												type="button"
												class="flex items-center gap-1 text-muted-foreground hover:text-foreground"
												onclick={() => toggleSort(col)}
											>
												<span>{col}</span>
												{#if sortColumn === col}
													{#if sortDirection === 'asc'}
														<ChevronUp class="h-3 w-3 text-blue-600" />
													{:else}
														<ChevronDown class="h-3 w-3 text-blue-600" />
													{/if}
												{:else}
													<ChevronsUpDown class="h-3 w-3 text-muted-foreground" />
												{/if}
											</button>
										</th>
									{/each}
								</tr>
							</thead>
							<tbody>
								{#each paginatedData as row, i}
									<tr class={i % 2 === 0 ? 'bg-background' : 'bg-muted/30'}>
										{#each currentResults.columns as col}
											<td class="whitespace-nowrap border-b border-border px-3 py-1 text-xs">
												{row[col] ?? '-'}
											</td>
										{/each}
									</tr>
								{/each}
							</tbody>
						</table>
					</div>

					<!-- Pagination Controls -->
					{#if totalPages > 1}
						<div class="flex items-center justify-between border-t border-border pt-3">
							<span class="text-xs text-muted-foreground">
								Showing {(currentPage - 1) * PAGE_SIZE + 1} - {Math.min(currentPage * PAGE_SIZE, sortedData.length)} of {sortedData.length}
							</span>
							<div class="flex items-center gap-1">
								<button
									type="button"
									onclick={() => goToPage(1)}
									disabled={currentPage === 1}
									class="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-50 disabled:cursor-not-allowed"
									title="First page"
								>
									<ChevronLeft class="h-4 w-4" />
									<ChevronLeft class="h-4 w-4 -ml-2" />
								</button>
								<button
									type="button"
									onclick={() => goToPage(currentPage - 1)}
									disabled={currentPage === 1}
									class="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-50 disabled:cursor-not-allowed"
									title="Previous page"
								>
									<ChevronLeft class="h-4 w-4" />
								</button>
								<span class="px-2 text-xs text-muted-foreground">
									Page {currentPage} of {totalPages}
								</span>
								<button
									type="button"
									onclick={() => goToPage(currentPage + 1)}
									disabled={currentPage === totalPages}
									class="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-50 disabled:cursor-not-allowed"
									title="Next page"
								>
									<ChevronRight class="h-4 w-4" />
								</button>
								<button
									type="button"
									onclick={() => goToPage(totalPages)}
									disabled={currentPage === totalPages}
									class="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-50 disabled:cursor-not-allowed"
									title="Last page"
								>
									<ChevronRight class="h-4 w-4" />
									<ChevronRight class="h-4 w-4 -ml-2" />
								</button>
							</div>
						</div>
					{/if}
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
