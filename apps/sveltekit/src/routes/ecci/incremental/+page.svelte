<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { beforeNavigate } from '$app/navigation';
	import {
		Card,
		CardContent,
		Button,
		Input,
		Select,
		MLTrainingSwitch,
		MetricCard,
		Tabs,
		TabsList,
		TabsTrigger,
		TabsContent
	} from '$components/shared';
	import {
		formData,
		predictionResults,
		predictionLoading,
		mlflowMetrics,
		mlflowExperimentUrl,
		incrementalMlEnabled,
		updateFormField,
		updateProjectStore,
		mlTrainingEnabled
	} from '$stores';
	import { toast } from '$stores/ui';
	import { metricInfoDialogOpen, metricInfoDialogContent } from '$stores';
	import * as incrementalApi from '$api/incremental';
	import { randomizeECCIForm } from '$lib/utils/randomize';
	import {
		Shuffle,
		ShoppingCart,
		Target,
		BarChart3,
		Layers,
		MapPin,
		FlaskConical,
		RefreshCw
	} from 'lucide-svelte';
	import type { DropdownOptions, ProjectName } from '$types';

	const PROJECT: ProjectName = 'E-Commerce Customer Interactions';
	const MODEL_NAME = 'DBSTREAM Clustering (River)';
	const ECCI_FEATURE_OPTIONS = [
		'event_type',
		'product_category',
		'referrer_url',
		'quantity',
		'time_on_page_seconds',
		'session_event_sequence',
		'device_type',
		'browser',
		'os'
	];

	// Transform MLflow metrics from API format (metrics.Silhouette) to simple format (silhouette)
	function transformMetrics(rawMetrics: Record<string, unknown>): Record<string, number> {
		return {
			silhouette: Number(rawMetrics['metrics.Silhouette'] ?? rawMetrics['metrics.silhouette'] ?? rawMetrics['metrics.silhouette_score'] ?? 0),
			silhouette_score: Number(rawMetrics['metrics.SilhouetteScore'] ?? rawMetrics['metrics.silhouette_score'] ?? 0),
			rolling_silhouette: Number(rawMetrics['metrics.RollingSilhouette'] ?? rawMetrics['metrics.rolling_silhouette'] ?? 0),
			time_rolling_silhouette: Number(rawMetrics['metrics.TimeRollingSilhouette'] ?? rawMetrics['metrics.time_rolling_silhouette'] ?? 0),
			n_clusters: Number(rawMetrics['metrics.NClusters'] ?? rawMetrics['metrics.n_clusters'] ?? 0),
			n_micro_clusters: Number(rawMetrics['metrics.NMicroClusters'] ?? rawMetrics['metrics.n_micro_clusters'] ?? 0)
		};
	}

	let trainingLoading = $state(false);
	let sampleLoading = $state(false);
	let metricsInterval: ReturnType<typeof setInterval> | null = null;
	let activeTab = $state('prediction');
	let dropdownOptions = $state<DropdownOptions>({});
	let selectedFeature = $state(ECCI_FEATURE_OPTIONS[0]);
	let clusterFeatureCounts = $state<Record<string, Record<string, number>>>({});

	// MLflow run info
	let mlflowRunInfo = $state<{
		run_id: string;
		run_id_full?: string;
		status: string;
		start_time: string;
		is_live: boolean;
	}>({ run_id: '', status: '', start_time: '', is_live: false });

	function formatRunStartTime(value?: string | number): string {
		if (!value) return '';
		const timestamp = typeof value === 'number' ? value : Date.parse(value);
		if (!Number.isNaN(timestamp)) {
			return new Date(timestamp).toISOString().replace('T', ' ').slice(0, 19);
		}
		return String(value);
	}

	function normalizeRunInfo(info: {
		run_id?: string;
		status?: string;
		start_time?: string | number;
		is_live?: boolean;
	}) {
		const runId = info.run_id ?? '';
		return {
			run_id: runId ? runId.slice(0, 8) : '',
			run_id_full: runId,
			status: info.status ?? '',
			start_time: formatRunStartTime(info.start_time),
			is_live: Boolean(info.is_live)
		};
	}

	onMount(async () => {
		try {
			const response = await fetch('/data/dropdown_options_ecci.json');
			dropdownOptions = await response.json();
		} catch (e) {
			console.error('Failed to load dropdown options:', e);
		}

		if (!Object.keys($formData[PROJECT] || {}).length) {
			updateProjectStore(formData, PROJECT, randomizeECCIForm(dropdownOptions));
		}

		const availability = await incrementalApi.checkModelAvailable(PROJECT);
		if (availability.data?.experiment_url) {
			updateProjectStore(mlflowExperimentUrl, PROJECT, availability.data.experiment_url);
		}
		if (availability.data?.run_id && !mlflowRunInfo.run_id) {
			mlflowRunInfo = normalizeRunInfo({
				run_id: availability.data.run_id,
				status: 'FINISHED',
				is_live: false,
				start_time: availability.data.trained_at || availability.data.last_trained
			});
		}

		await checkTrainingStatus();
		await fetchMetrics();

		if ($incrementalMlEnabled[PROJECT]) {
			startMetricsPolling();
		}
	});

	onDestroy(() => {
		stopMetricsPolling();
	});

	// Stop training when navigating away to prevent multiple training scripts
	beforeNavigate(async ({ to }) => {
		if ($incrementalMlEnabled[PROJECT]) {
			// Stop training before navigating away
			await incrementalApi.stopTraining();
			updateProjectStore(incrementalMlEnabled, PROJECT, false);
			mlTrainingEnabled.set(false);
			mlflowRunInfo = { ...mlflowRunInfo, is_live: false, status: 'FINISHED' };
			stopMetricsPolling();
			toast.info('Training stopped - navigated away from page');
		}
	});

	async function checkTrainingStatus() {
		const result = await incrementalApi.getProjectTrainingStatus(PROJECT);
		if (result.data) {
			updateProjectStore(incrementalMlEnabled, PROJECT, result.data.is_active);
			if (result.data.is_active) {
				mlTrainingEnabled.set(true);
				mlflowRunInfo = { ...mlflowRunInfo, is_live: true, status: 'RUNNING' };
			}
		}
	}

	function startMetricsPolling() {
		if (metricsInterval) return;
		metricsInterval = setInterval(fetchMetrics, 5000);
		fetchMetrics();
	}

	function stopMetricsPolling() {
		if (metricsInterval) {
			clearInterval(metricsInterval);
			metricsInterval = null;
		}
	}

	async function fetchMetrics(forceRefresh = false) {
		const result = await incrementalApi.getMLflowMetrics(PROJECT, forceRefresh);
		if (result.data) {
			// Transform metrics from API format (metrics.Silhouette) to simple format (silhouette)
			const transformed = transformMetrics(result.data);
			updateProjectStore(mlflowMetrics, PROJECT, transformed);
		}
		const runInfoSource = result.data?.run_info ?? result.data;
		if (runInfoSource?.run_id || runInfoSource?.status) {
			mlflowRunInfo = normalizeRunInfo(runInfoSource);
		}
		if (forceRefresh) {
			toast.success('Metrics refreshed');
		}
	}

	async function toggleTraining(enabled: boolean) {
		trainingLoading = true;
		try {
			if (enabled) {
				const result = await incrementalApi.startTraining(PROJECT);
				if (result.error) {
					toast.error(result.error);
				} else {
					toast.success('Training started');
					updateProjectStore(incrementalMlEnabled, PROJECT, true);
					mlTrainingEnabled.set(true);
					mlflowRunInfo = { ...mlflowRunInfo, is_live: true, status: 'RUNNING' };
					startMetricsPolling();
				}
			} else {
				const result = await incrementalApi.stopTraining();
				if (result.error) {
					toast.error(result.error);
				} else {
					toast.success('Training stopped');
					updateProjectStore(incrementalMlEnabled, PROJECT, false);
					mlTrainingEnabled.set(false);
					mlflowRunInfo = { ...mlflowRunInfo, is_live: false, status: 'FINISHED' };
					stopMetricsPolling();
				}
			}
		} finally {
			trainingLoading = false;
		}
	}

	function generateRandomCoordinates() {
		// Houston area coordinates
		const lat = 29.5 + Math.random() * 0.6;
		const lon = -95.8 + Math.random() * 0.8;
		updateFormField(PROJECT, 'lat', Number(lat.toFixed(3)));
		updateFormField(PROJECT, 'lon', Number(lon.toFixed(3)));
	}

	function loadRandomSample() {
		sampleLoading = true;
		try {
			const randomData = randomizeECCIForm(dropdownOptions);
			updateProjectStore(formData, PROJECT, randomData);
			updateProjectStore(predictionResults, PROJECT, {});
			toast.success('Form randomized');
		} finally {
			sampleLoading = false;
		}
	}

	async function predict() {
		updateProjectStore(predictionLoading, PROJECT, true);
		try {
			const form = $formData[PROJECT];
			const result = await incrementalApi.predict(PROJECT, form);
			if (result.error) {
				toast.error(result.error);
			} else if (result.data) {
				updateProjectStore(predictionResults, PROJECT, result.data);
				const cluster = result.data.cluster ?? result.data.cluster_id;
				const sourceLabel = result.data.model_source
					? result.data.model_source.toUpperCase()
					: 'MLFLOW';
				toast.success(`Assigned to Cluster ${cluster} (${sourceLabel})`);
				await fetchClusterFeatureCounts(selectedFeature);
				await fetchMetrics();
			}
		} finally {
			updateProjectStore(predictionLoading, PROJECT, false);
		}
	}

	async function fetchClusterFeatureCounts(feature: string) {
		const result = await incrementalApi.getClusterFeatureCounts(feature);
		if (result.data) {
			clusterFeatureCounts = result.data;
		}
	}

	function handleFeatureChange(event: Event) {
		selectedFeature = (event.currentTarget as HTMLSelectElement).value;
		if (hasPrediction) {
			fetchClusterFeatureCounts(selectedFeature);
		}
	}

	function openMetricInfo(metricKey: string) {
		fetch('/data/metric_info_ecci.json')
			.then((r) => r.json())
			.then((data) => {
				const info = data.metrics[metricKey];
				if (info) {
					metricInfoDialogContent.set({
						name: info.name,
						formula: info.formula,
						explanation: info.explanation,
						context: info.context,
						range: info.range,
						optimal: info.optimal,
						docsUrl: info.docs_url?.incremental || ''
					});
					metricInfoDialogOpen.set(true);
				}
			});
	}

	const currentForm = $derived($formData[PROJECT]);
	const currentPrediction = $derived($predictionResults[PROJECT]);
	const currentMetrics = $derived($mlflowMetrics[PROJECT]);
	const isLoading = $derived($predictionLoading[PROJECT]);
	const isTrainingEnabled = $derived($incrementalMlEnabled[PROJECT]);
	const currentMlflowUrl = $derived($mlflowExperimentUrl[PROJECT]);

	// Derived values for prediction display
	const predictedCluster = $derived(
		currentPrediction?.cluster ?? currentPrediction?.cluster_id ?? 0
	);
	const hasPrediction = $derived(
		currentPrediction && Object.keys(currentPrediction).length > 0
	);
	const selectedClusterFeatureCounts = $derived(
		clusterFeatureCounts[String(predictedCluster)] ?? {}
	);
	const topClusterFeatureEntries = $derived(
		Object.entries(selectedClusterFeatureCounts)
			.sort((a, b) => b[1] - a[1])
			.slice(0, 10)
	);
	const maxClusterFeatureCount = $derived(
		topClusterFeatureEntries.length
			? Math.max(...topClusterFeatureEntries.map(([, count]) => count))
			: 1
	);
</script>

<div class="flex gap-6">
	<!-- LEFT COLUMN (40%) - Training Switch + Form -->
	<div class="w-[40%] space-y-4">
		<MLTrainingSwitch
			enabled={isTrainingEnabled}
			loading={trainingLoading}
			modelName={MODEL_NAME}
			mlflowUrl={currentMlflowUrl}
			onToggle={toggleTraining}
		/>

		<Card>
			<CardContent class="space-y-3 pt-4">
				<!-- Form Legend -->
				<div class="flex items-center gap-2">
					<ShoppingCart class="h-5 w-5 text-primary" />
					<h3 class="text-base font-bold">Customer Interaction</h3>
				</div>

				<!-- Predict + Randomize Buttons -->
				<div class="flex gap-2">
					<Button class="flex-1" onclick={predict} loading={isLoading}>
						Predict
					</Button>
					<Button
						variant="secondary"
						class="flex-1 bg-blue-500/10 text-blue-600 hover:bg-blue-500/20"
						onclick={loadRandomSample}
						loading={sampleLoading}
					>
						<Shuffle class="mr-2 h-4 w-4" />
						Randomize
					</Button>
				</div>

				<hr class="border-border" />

				<!-- Form fields in 3-column grid -->
				<div class="grid grid-cols-3 gap-2">
					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Browser</p>
						<Select
							value={(currentForm.browser as string) ?? 'Chrome'}
							options={dropdownOptions.browser || ['Chrome']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'browser', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Device</p>
						<Select
							value={(currentForm.device_type as string) ?? 'Desktop'}
							options={dropdownOptions.device_type || ['Desktop']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'device_type', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">OS</p>
						<Select
							value={(currentForm.os as string) ?? 'Windows'}
							options={dropdownOptions.os || ['Windows']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'os', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Event Type</p>
						<Select
							value={(currentForm.event_type as string) ?? 'page_view'}
							options={dropdownOptions.event_type || ['page_view']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'event_type', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Category</p>
						<Select
							value={(currentForm.product_category as string) ?? 'Electronics'}
							options={dropdownOptions.product_category || ['Electronics']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'product_category', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Price</p>
						<Input
							type="number"
							value={currentForm.price ?? ''}
							min="0"
							step="0.01"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'price', parseFloat(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Date</p>
						<Input
							type="date"
							value={(currentForm.timestamp_date as string) ?? ''}
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'timestamp_date', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Time</p>
						<Input
							type="time"
							value={(currentForm.timestamp_time as string) ?? ''}
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'timestamp_time', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Product ID</p>
						<Input
							value={(currentForm.product_id as string) ?? ''}
							placeholder="prod_1050"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'product_id', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Latitude</p>
						<Input
							type="number"
							value={currentForm.lat ?? ''}
							min="29.5"
							max="30.1"
							step="0.001"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'lat', parseFloat(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Longitude</p>
						<Input
							type="number"
							value={currentForm.lon ?? ''}
							min="-95.8"
							max="-95.0"
							step="0.001"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'lon', parseFloat(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Coords</p>
						<Button
							variant="outline"
							size="sm"
							class="h-8 w-full text-xs"
							onclick={generateRandomCoordinates}
						>
							<Shuffle class="mr-1 h-3 w-3" />
							Random
						</Button>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Quantity</p>
						<Input
							type="number"
							value={currentForm.quantity ?? ''}
							min="1"
							step="1"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'quantity', parseInt(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Time (s)</p>
						<Input
							type="number"
							value={currentForm.time_on_page_seconds ?? ''}
							min="0"
							step="1"
							class="h-8 text-sm"
							oninput={(e) =>
								updateFormField(PROJECT, 'time_on_page_seconds', parseInt(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Sequence</p>
						<Input
							type="number"
							value={currentForm.session_event_sequence ?? ''}
							min="1"
							step="1"
							class="h-8 text-sm"
							oninput={(e) =>
								updateFormField(PROJECT, 'session_event_sequence', parseInt(e.currentTarget.value))}
						/>
					</div>

					<div class="col-span-3 space-y-1">
						<p class="text-xs text-muted-foreground">Referrer</p>
						<Input
							value={currentForm.referrer_url ?? ''}
							placeholder="google.com"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'referrer_url', e.currentTarget.value)}
						/>
					</div>
				</div>

				<!-- Display fields (read-only info) -->
				<div class="space-y-1 text-xs text-muted-foreground">
					<p>Customer ID: {currentForm.customer_id || '-'}</p>
					<p>Event ID: {currentForm.event_id || '-'}</p>
					<p>Page URL: {currentForm.page_url || '-'}</p>
					<p>Search Query: {currentForm.search_query || '-'}</p>
					<p>Session ID: {currentForm.session_id || '-'}</p>
				</div>
			</CardContent>
		</Card>
	</div>

	<!-- RIGHT COLUMN (60%) - Tabs for Prediction, Metrics, Analytics -->
	<div class="w-[60%]">
		<Tabs value={activeTab} onValueChange={(v) => (activeTab = v)}>
			<TabsList class="w-full">
				<TabsTrigger value="prediction" class="flex-1">
					<Target class="mr-2 h-4 w-4" />
					Prediction
				</TabsTrigger>
				<TabsTrigger value="metrics" class="flex-1">
					<BarChart3 class="mr-2 h-4 w-4" />
					Metrics
				</TabsTrigger>
				<TabsTrigger value="analytics" class="flex-1">
					<Layers class="mr-2 h-4 w-4" />
					Analytics
				</TabsTrigger>
			</TabsList>

			<!-- Prediction Tab -->
			<TabsContent value="prediction" class="mt-4">
				<div class="space-y-4 pt-4">
					<div class="flex flex-wrap items-center gap-2 rounded-md bg-muted/50 p-2">
						<span
							class="inline-flex items-center gap-1 rounded-md bg-purple-500/10 px-2 py-1 text-xs font-medium text-purple-600"
						>
							<FlaskConical class="h-3 w-3" />
							MLflow
						</span>
						{#if mlflowRunInfo.is_live}
							<span
								class="inline-flex items-center gap-1 rounded-md bg-green-500/10 px-2 py-1 text-xs font-medium text-green-600"
							>
								<span class="h-2 w-2 animate-pulse rounded-full bg-green-500"></span>
								LIVE
							</span>
						{:else if mlflowRunInfo.status}
							<span
								class="inline-flex items-center gap-1 rounded-md bg-blue-500/10 px-2 py-1 text-xs font-medium text-blue-600"
							>
								{mlflowRunInfo.status}
							</span>
						{/if}
						{#if mlflowRunInfo.run_id}
							<span class="text-xs text-muted-foreground">
								Run:
								<code class="rounded bg-muted px-1" title={mlflowRunInfo.run_id_full}>
									{mlflowRunInfo.run_id}
								</code>
							</span>
						{/if}
						{#if mlflowRunInfo.start_time}
							<span class="text-xs text-muted-foreground">Started: {mlflowRunInfo.start_time}</span>
						{/if}
					</div>
					<Card>
						<CardContent class="p-4">
							<div class="mb-2 flex items-center gap-2">
								<MapPin class="h-4 w-4 text-primary" />
								<span class="text-sm font-bold">Customer Location</span>
							</div>
							<div
								class="flex h-[200px] items-center justify-center rounded-md bg-gradient-to-br from-green-100 to-green-50 dark:from-green-900/20 dark:to-green-800/10"
							>
								<div class="space-y-2 text-center text-sm text-muted-foreground">
									<MapPin class="mx-auto h-8 w-8" />
									<p>Map visualization</p>
									<p class="text-xs">
										Location: ({currentForm.lat || '-'}, {currentForm.lon || '-'})
									</p>
								</div>
							</div>
						</CardContent>
					</Card>

					<div class="flex gap-3">
						<Card class="h-[320px] w-1/2">
							<CardContent class="h-full p-4">
								<div class="flex h-full flex-col">
									<div class="mb-2 flex items-center gap-2">
										<Target class="h-4 w-4 text-primary" />
										<span class="text-sm font-bold">Predicted Cluster</span>
									</div>
									<div class="flex flex-1 items-center justify-center">
										{#if hasPrediction}
											<div class="text-center">
												<div
													class="mx-auto flex h-32 w-32 items-center justify-center rounded-full border-8 border-green-500 bg-green-50 dark:bg-green-900/20"
												>
													<span class="text-4xl font-bold text-green-600">{predictedCluster}</span>
												</div>
												<p class="mt-4 text-sm text-muted-foreground">Customer Segment</p>
											</div>
										{:else}
											<div
												class="rounded-lg border border-blue-200 bg-blue-50 p-4 text-center text-sm text-blue-600 dark:border-blue-800 dark:bg-blue-900/20"
											>
												Click <strong>Predict</strong> to identify the customer segment.
											</div>
										{/if}
									</div>
								</div>
							</CardContent>
						</Card>

						<Card class="h-[320px] w-1/2">
							<CardContent class="h-full p-4">
								<div class="flex h-full flex-col">
									<div class="mb-2 flex items-center gap-2">
										<BarChart3 class="h-4 w-4 text-primary" />
										<span class="text-sm font-bold">Cluster Behavior</span>
									</div>
									{#if hasPrediction}
										<div class="space-y-3">
											<Select
												value={selectedFeature}
												options={ECCI_FEATURE_OPTIONS}
												class="h-8 text-sm"
												onchange={handleFeatureChange}
											/>
											<div class="space-y-2">
												{#if topClusterFeatureEntries.length > 0}
													{#each topClusterFeatureEntries as [label, count]}
														<div class="flex items-center gap-2 text-xs">
															<span class="w-24 truncate text-muted-foreground">{label}</span>
															<div class="h-2 flex-1 rounded bg-muted">
																<div
																	class="h-2 rounded bg-blue-500"
																	style={`width: ${(count / maxClusterFeatureCount) * 100}%`}
																></div>
															</div>
															<span class="w-10 text-right text-muted-foreground">{count}</span>
														</div>
													{/each}
												{:else}
													<p class="text-xs text-muted-foreground">No feature data available.</p>
												{/if}
											</div>
										</div>
									{:else}
										<div
											class="flex flex-1 items-center justify-center rounded-lg border border-blue-200 bg-blue-50 p-4 text-center text-sm text-blue-600 dark:border-blue-800 dark:bg-blue-900/20"
										>
											Cluster behavior shown after prediction.
										</div>
									{/if}
								</div>
							</CardContent>
						</Card>
					</div>

					{#if hasPrediction}
						<Card>
							<CardContent class="p-4">
								<div class="flex items-start gap-2 text-sm">
									<span class="text-blue-500">ⓘ</span>
									<p>
										This customer interaction was assigned to <strong>Cluster {predictedCluster}</strong
										>. Clusters represent groups of similar customer behaviors based on their browsing
										patterns, device usage, and purchase activities.
									</p>
								</div>
							</CardContent>
						</Card>
					{/if}
				</div>
			</TabsContent>

			<!-- Metrics Tab -->
			<TabsContent value="metrics" class="mt-4 space-y-4">
				<div class="flex items-center justify-between">
					<h2 class="text-lg font-bold">Clustering Metrics</h2>
					<Button variant="ghost" size="sm" onclick={fetchMetrics}>
						<RefreshCw class="h-4 w-4" />
					</Button>
				</div>

				<!-- MLflow Run Info Badge -->
				<div class="flex flex-wrap items-center gap-2 rounded-md bg-muted/50 p-2">
					<span
						class="inline-flex items-center gap-1 rounded-md bg-purple-500/10 px-2 py-1 text-xs font-medium text-purple-600"
					>
						<FlaskConical class="h-3 w-3" />
						MLflow
					</span>
					{#if mlflowRunInfo.is_live}
						<span
							class="inline-flex items-center gap-1 rounded-md bg-green-500/10 px-2 py-1 text-xs font-medium text-green-600"
						>
							<span class="h-2 w-2 animate-pulse rounded-full bg-green-500"></span>
							LIVE
						</span>
					{:else if mlflowRunInfo.status}
						<span
							class="inline-flex items-center gap-1 rounded-md bg-blue-500/10 px-2 py-1 text-xs font-medium text-blue-600"
						>
							{mlflowRunInfo.status}
						</span>
					{/if}
				</div>

				{#if Object.keys(currentMetrics || {}).length > 0}
					<!-- ROW 1: KPI Indicators -->
					<div class="grid grid-cols-4 gap-2">
						<MetricCard
							name="Silhouette"
							value={currentMetrics.silhouette}
							onInfoClick={() => openMetricInfo('silhouette')}
						/>
						<MetricCard
							name="Rolling Silhouette"
							value={currentMetrics.rolling_silhouette}
							onInfoClick={() => openMetricInfo('rolling_silhouette')}
						/>
						<MetricCard
							name="N Clusters"
							value={currentMetrics.n_clusters}
							decimals={0}
							onInfoClick={() => openMetricInfo('n_clusters')}
						/>
						<MetricCard
							name="N Micro-clusters"
							value={currentMetrics.n_micro_clusters}
							decimals={0}
							onInfoClick={() => openMetricInfo('n_micro_clusters')}
						/>
					</div>

					<!-- ROW 2: Additional metrics -->
					<div class="grid grid-cols-3 gap-2">
						<MetricCard
							name="Silhouette Score"
							value={currentMetrics.silhouette_score || currentMetrics.silhouette}
							onInfoClick={() => openMetricInfo('silhouette_score')}
						/>
						<MetricCard
							name="Rolling Silhouette"
							value={currentMetrics.rolling_silhouette}
							onInfoClick={() => openMetricInfo('rolling_silhouette')}
						/>
						<MetricCard
							name="Time Rolling Silhouette"
							value={currentMetrics.time_rolling_silhouette}
							onInfoClick={() => openMetricInfo('time_rolling_silhouette')}
						/>
					</div>

					<!-- ROW 3: Gauges -->
					<div class="grid grid-cols-2 gap-4">
						<Card>
							<CardContent class="p-4">
								<div class="flex items-center justify-between">
									<span class="text-sm font-medium">Silhouette Score</span>
									<button
										class="text-muted-foreground hover:text-foreground"
										onclick={() => openMetricInfo('silhouette')}
									>
										<span class="text-xs">ⓘ</span>
									</button>
								</div>
								<div class="mt-2 flex items-center gap-2">
									<div class="h-2 flex-1 rounded-full bg-muted">
										<div
											class="h-2 rounded-full bg-primary transition-all"
											style="width: {((currentMetrics.silhouette || 0) + 1) * 50}%"
										></div>
									</div>
									<span class="text-sm font-bold"
										>{(currentMetrics.silhouette || 0).toFixed(4)}</span
									>
								</div>
							</CardContent>
						</Card>
						<Card>
							<CardContent class="p-4">
								<div class="flex items-center justify-between">
									<span class="text-sm font-medium">Cluster Statistics</span>
									<button
										class="text-muted-foreground hover:text-foreground"
										onclick={() => openMetricInfo('n_clusters')}
									>
										<span class="text-xs">ⓘ</span>
									</button>
								</div>
								<div class="mt-2 flex justify-around text-center">
									<div>
										<p class="text-2xl font-bold">{currentMetrics.n_clusters || 0}</p>
										<p class="text-xs text-muted-foreground">Clusters</p>
									</div>
									<div>
										<p class="text-2xl font-bold">{currentMetrics.n_micro_clusters || 0}</p>
										<p class="text-xs text-muted-foreground">Micro-clusters</p>
									</div>
								</div>
							</CardContent>
						</Card>
					</div>

				{:else}
					<Card>
						<CardContent class="py-12 text-center">
							<p class="text-muted-foreground">
								{isTrainingEnabled ? 'Loading metrics...' : 'Enable training to see metrics'}
							</p>
						</CardContent>
					</Card>
				{/if}
			</TabsContent>

			<!-- Analytics Tab -->
			<TabsContent value="analytics" class="mt-4 space-y-4">
				<!-- MLflow Run Info Badge -->
				<div class="flex flex-wrap items-center gap-2 rounded-md bg-muted/50 p-2">
					<span
						class="inline-flex items-center gap-1 rounded-md bg-purple-500/10 px-2 py-1 text-xs font-medium text-purple-600"
					>
						<FlaskConical class="h-3 w-3" />
						MLflow
					</span>
					{#if mlflowRunInfo.is_live}
						<span
							class="inline-flex items-center gap-1 rounded-md bg-green-500/10 px-2 py-1 text-xs font-medium text-green-600"
						>
							<span class="h-2 w-2 animate-pulse rounded-full bg-green-500"></span>
							LIVE
						</span>
					{:else if mlflowRunInfo.status}
						<span
							class="inline-flex items-center gap-1 rounded-md bg-blue-500/10 px-2 py-1 text-xs font-medium text-blue-600"
						>
							{mlflowRunInfo.status}
						</span>
					{/if}
				</div>

				<!-- Samples per Cluster -->
				<Card>
					<CardContent class="p-4">
						<div class="mb-4 flex items-center justify-between">
							<div class="flex items-center gap-2">
								<BarChart3 class="h-4 w-4 text-primary" />
								<span class="text-sm font-bold">Samples per Cluster</span>
							</div>
							<Button variant="outline" size="sm">
								<RefreshCw class="mr-2 h-3 w-3" />
								Refresh
							</Button>
						</div>
						<div
							class="flex h-[200px] items-center justify-center rounded-md bg-muted/30 p-4"
						>
							<p class="text-sm text-muted-foreground">Cluster distribution chart</p>
						</div>
					</CardContent>
				</Card>

				<!-- Feature Distribution -->
				<Card>
					<CardContent class="p-4">
						<div class="mb-4 flex items-center justify-between">
							<div class="flex items-center gap-2">
								<BarChart3 class="h-4 w-4 text-primary" />
								<span class="text-sm font-bold">Feature Distribution</span>
							</div>
							<Select
								value="price"
								options={['price', 'quantity', 'time_on_page_seconds', 'session_event_sequence']}
								class="w-[200px]"
							/>
						</div>
						<div
							class="flex h-[200px] items-center justify-center rounded-md bg-muted/30 p-4"
						>
							<p class="text-sm text-muted-foreground">Feature distribution across clusters</p>
						</div>
					</CardContent>
				</Card>

				<!-- Info callout -->
				<Card>
					<CardContent class="p-4">
						<div class="flex items-start gap-2 text-sm">
							<span class="text-blue-500">ⓘ</span>
							<p>
								This tab shows aggregated statistics across all clusters. Use the feature selector
								to explore how different attributes are distributed across customer segments.
							</p>
						</div>
					</CardContent>
				</Card>
			</TabsContent>
		</Tabs>
	</div>
</div>
