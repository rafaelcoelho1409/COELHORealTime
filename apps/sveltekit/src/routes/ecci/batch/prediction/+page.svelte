<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		Card,
		CardContent,
		Button,
		Input,
		Select,
		FormField,
		BatchTrainingBox,
		BatchSubNav
	} from '$components/shared';
	import {
		formData,
		batchPredictionResults,
		batchPredictionLoading,
		batchTrainingLoading,
		batchTrainingProgress,
		batchTrainingStage,
		batchTrainingStatus,
		batchTrainingMode,
		batchTrainingDataPercentage,
		batchTrainingMaxRows,
		batchDeltaLakeTotalRows,
		batchMlflowRuns,
		selectedBatchRun,
		batchModelAvailable,
		updateFormField,
		updateProjectStore
	} from '$stores';
	import { toast } from '$stores/ui';
	import * as batchApi from '$api/batch';
	import { randomizeECCIForm } from '$lib/utils/randomize';
	import { ShoppingCart, Shuffle, MapPin, Users, BarChart3, AlertTriangle, Info } from 'lucide-svelte';
	import type { ProjectName } from '$types';

	const PROJECT: ProjectName = 'E-Commerce Customer Interactions';
	const MODEL_NAME = 'KMeans (Scikit-Learn)';

	let dropdownOptions = $state<Record<string, string[]>>({});
	let sampleLoading = $state(false);
	let statusInterval: ReturnType<typeof setInterval> | null = null;

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

		const countResult = await batchApi.getDeltaLakeRowCount(PROJECT);
		if (countResult.data?.row_count) {
			updateProjectStore(batchDeltaLakeTotalRows, PROJECT, countResult.data.row_count);
		}

		await loadRuns();
	});

	onDestroy(() => {
		stopStatusPolling();
	});

	async function loadRuns() {
		const result = await batchApi.getMLflowRuns(PROJECT);
		if (result.data?.runs) {
			updateProjectStore(batchMlflowRuns, PROJECT, result.data.runs);
			updateProjectStore(batchModelAvailable, PROJECT, result.data.runs.length > 0);
		}
	}

	async function startTraining() {
		updateProjectStore(batchTrainingLoading, PROJECT, true);
		const mode = $batchTrainingMode[PROJECT];
		const result = await batchApi.startTraining(PROJECT, {
			mode,
			percentage: mode === 'percentage' ? $batchTrainingDataPercentage[PROJECT] : undefined,
			maxRows: mode === 'max_rows' ? $batchTrainingMaxRows[PROJECT] : undefined
		});

		if (result.error) {
			toast.error(result.error);
			updateProjectStore(batchTrainingLoading, PROJECT, false);
		} else {
			toast.success('Training started');
			startStatusPolling();
		}
	}

	async function stopTraining() {
		const result = await batchApi.stopTraining(PROJECT);
		if (result.error) {
			toast.error(result.error);
		} else {
			toast.info('Training stopped');
			stopStatusPolling();
			updateProjectStore(batchTrainingLoading, PROJECT, false);
		}
	}

	function startStatusPolling() {
		if (statusInterval) return;
		statusInterval = setInterval(checkTrainingStatus, 2000);
	}

	function stopStatusPolling() {
		if (statusInterval) {
			clearInterval(statusInterval);
			statusInterval = null;
		}
	}

	async function checkTrainingStatus() {
		const result = await batchApi.getTrainingStatus(PROJECT);
		if (result.data) {
			updateProjectStore(batchTrainingStatus, PROJECT, result.data.status);
			updateProjectStore(batchTrainingProgress, PROJECT, result.data.progress);
			updateProjectStore(batchTrainingStage, PROJECT, result.data.stage);

			if (result.data.progress >= 100 || result.data.status === 'completed') {
				stopStatusPolling();
				updateProjectStore(batchTrainingLoading, PROJECT, false);
				toast.success('Training completed');
				await loadRuns();
			}
		}
	}

	function loadRandomSample() {
		sampleLoading = true;
		try {
			const randomData = randomizeECCIForm(dropdownOptions);
			updateProjectStore(formData, PROJECT, randomData);
			updateProjectStore(batchPredictionResults, PROJECT, {});
			toast.success('Form randomized');
		} finally {
			sampleLoading = false;
		}
	}

	function generateRandomCoordinates() {
		// Houston area coordinates
		const lat = 29.5 + Math.random() * 0.6;
		const lon = -95.8 + Math.random() * 0.8;

		formData.update((fd) => ({
			...fd,
			[PROJECT]: {
				...fd[PROJECT],
				lat: parseFloat(lat.toFixed(3)),
				lon: parseFloat(lon.toFixed(3))
			}
		}));
		toast.success('Random coordinates generated');
	}

	async function predict() {
		updateProjectStore(batchPredictionLoading, PROJECT, true);
		try {
			const form = $formData[PROJECT];
			const runId = $selectedBatchRun[PROJECT] || undefined;
			const result = await batchApi.predict(PROJECT, form, runId);
			if (result.error) {
				toast.error(result.error);
			} else if (result.data) {
				updateProjectStore(batchPredictionResults, PROJECT, result.data);
				const cluster = result.data.cluster ?? result.data.cluster_id;
				toast.success(`Assigned to Cluster ${cluster}`);
			}
		} finally {
			updateProjectStore(batchPredictionLoading, PROJECT, false);
		}
	}

	const currentForm = $derived($formData[PROJECT] || {});
	const currentPrediction = $derived($batchPredictionResults[PROJECT]);
	const isLoading = $derived($batchPredictionLoading[PROJECT]);
	const isTraining = $derived($batchTrainingLoading[PROJECT]);
	const modelAvailable = $derived($batchModelAvailable[PROJECT]);

	const cluster = $derived(currentPrediction?.cluster ?? currentPrediction?.cluster_id ?? null);
	const hasPrediction = $derived(cluster !== null && cluster !== undefined);

	// Cluster colors for visual distinction
	const clusterColors = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#ec4899'];
	const clusterColor = $derived(clusterColors[(cluster ?? 0) % clusterColors.length]);
</script>

<!-- 40%/60% Layout -->
<div class="flex gap-6">
	<!-- Left Column - Training Box + Form (40%) -->
	<div class="w-[40%] space-y-4">
		<BatchTrainingBox
			modelName={MODEL_NAME}
			isTraining={isTraining}
			progress={$batchTrainingProgress[PROJECT]}
			stage={$batchTrainingStage[PROJECT]}
			status={$batchTrainingStatus[PROJECT]}
			mode={$batchTrainingMode[PROJECT]}
			percentage={$batchTrainingDataPercentage[PROJECT]}
			maxRows={$batchTrainingMaxRows[PROJECT]}
			totalRows={$batchDeltaLakeTotalRows[PROJECT]}
			onModeChange={(mode) => updateProjectStore(batchTrainingMode, PROJECT, mode)}
			onPercentageChange={(val) => updateProjectStore(batchTrainingDataPercentage, PROJECT, val)}
			onMaxRowsChange={(val) => updateProjectStore(batchTrainingMaxRows, PROJECT, val)}
			onStartTraining={startTraining}
			onStopTraining={stopTraining}
		/>

		<Card>
			<CardContent class="space-y-3 pt-4">
				<div class="flex items-center gap-2">
					<ShoppingCart class="h-5 w-5 text-primary" />
					<h3 class="text-base font-bold">Customer Interaction</h3>
				</div>

				<div class="flex gap-2">
					<Button class="flex-1" onclick={predict} loading={isLoading} disabled={!modelAvailable}>
						Predict
					</Button>
					<Button variant="outline" class="flex-1" onclick={loadRandomSample} loading={sampleLoading}>
						<Shuffle class="mr-2 h-3.5 w-3.5" />
						Randomize
					</Button>
				</div>

				<hr class="border-border" />

				<div class="grid grid-cols-3 gap-2">
					<FormField label="Browser" id="browser" class="space-y-1">
						<Select
							id="browser"
							value={(currentForm.browser as string) ?? 'Chrome'}
							options={dropdownOptions.browser || ['Chrome']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'browser', e.currentTarget.value)}
						/>
					</FormField>

					<FormField label="Device" id="device_type" class="space-y-1">
						<Select
							id="device_type"
							value={(currentForm.device_type as string) ?? 'Desktop'}
							options={dropdownOptions.device_type || ['Desktop']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'device_type', e.currentTarget.value)}
						/>
					</FormField>

					<FormField label="OS" id="os" class="space-y-1">
						<Select
							id="os"
							value={(currentForm.os as string) ?? 'Windows'}
							options={dropdownOptions.os || ['Windows']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'os', e.currentTarget.value)}
						/>
					</FormField>

					<FormField label="Event Type" id="event_type" class="space-y-1">
						<Select
							id="event_type"
							value={(currentForm.event_type as string) ?? 'page_view'}
							options={dropdownOptions.event_type || ['page_view']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'event_type', e.currentTarget.value)}
						/>
					</FormField>

					<FormField label="Category" id="product_category" class="space-y-1">
						<Select
							id="product_category"
							value={(currentForm.product_category as string) ?? 'Electronics'}
							options={dropdownOptions.product_category || ['Electronics']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'product_category', e.currentTarget.value)}
						/>
					</FormField>

					<FormField label="Price" id="price" class="space-y-1">
						<Input
							id="price"
							type="number"
							value={currentForm.price ?? ''}
							step="0.01"
							min="0"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'price', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Date" id="timestamp_date" class="space-y-1">
						<Input
							id="timestamp_date"
							type="date"
							value={(currentForm.timestamp_date as string) ?? ''}
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'timestamp_date', e.currentTarget.value)}
						/>
					</FormField>

					<FormField label="Time" id="timestamp_time" class="space-y-1">
						<Input
							id="timestamp_time"
							type="time"
							value={(currentForm.timestamp_time as string) ?? ''}
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'timestamp_time', e.currentTarget.value)}
						/>
					</FormField>

					<FormField label="Product ID" id="product_id" class="space-y-1">
						<Input
							id="product_id"
							value={(currentForm.product_id as string) ?? ''}
							placeholder="prod_1050"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'product_id', e.currentTarget.value)}
						/>
					</FormField>

					<FormField label="Latitude" id="lat" class="space-y-1">
						<Input
							id="lat"
							type="number"
							value={currentForm.lat ?? ''}
							step="0.001"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'lat', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Longitude" id="lon" class="space-y-1">
						<Input
							id="lon"
							type="number"
							value={currentForm.lon ?? ''}
							step="0.001"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'lon', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Coords" id="random_coords" class="space-y-1">
						<Button variant="outline" size="sm" class="h-8 w-full text-xs" onclick={generateRandomCoordinates}>
							<Shuffle class="mr-1 h-3 w-3" />
							Random
						</Button>
					</FormField>

					<FormField label="Quantity" id="quantity" class="space-y-1">
						<Input
							id="quantity"
							type="number"
							value={currentForm.quantity ?? ''}
							min="1"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'quantity', parseInt(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Time (s)" id="time_on_page_seconds" class="space-y-1">
						<Input
							id="time_on_page_seconds"
							type="number"
							value={currentForm.time_on_page_seconds ?? ''}
							min="0"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'time_on_page_seconds', parseInt(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Sequence" id="session_event_sequence" class="space-y-1">
						<Input
							id="session_event_sequence"
							type="number"
							value={currentForm.session_event_sequence ?? ''}
							min="1"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'session_event_sequence', parseInt(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Referrer" id="referrer_url" class="space-y-1 col-span-3">
						<Input
							id="referrer_url"
							value={(currentForm.referrer_url as string) ?? ''}
							placeholder="google.com"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'referrer_url', e.currentTarget.value)}
						/>
					</FormField>
				</div>

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

	<!-- Right Column - Batch Sub-Nav + Prediction Result (60%) -->
	<div class="w-[60%] space-y-4">
		<BatchSubNav projectKey="ecci" />

		<!-- Customer Location Map -->
		<Card>
			<CardContent class="p-4">
				<div class="mb-2 flex items-center gap-2">
					<MapPin class="h-4 w-4 text-primary" />
					<span class="text-sm font-bold">Customer Location</span>
				</div>
				<div class="flex h-[200px] items-center justify-center rounded-lg bg-muted/30">
					<div class="text-center text-muted-foreground">
						<MapPin class="mx-auto h-12 w-12 opacity-50" />
						<p class="mt-2 text-sm">Map visualization</p>
						<p class="text-xs">{currentForm.lat ?? 0}, {currentForm.lon ?? 0}</p>
					</div>
				</div>
			</CardContent>
		</Card>

		<!-- Prediction Result -->
		{#if hasPrediction}
			<div class="grid grid-cols-2 gap-3">
				<!-- Predicted Cluster -->
				<Card>
					<CardContent class="p-4">
						<div class="mb-3 flex items-center gap-2">
							<Users class="h-4 w-4" style="color: {clusterColor}" />
							<span class="text-sm font-bold">Predicted Cluster</span>
						</div>
						<div class="flex flex-col items-center justify-center py-6">
							<div
								class="flex h-24 w-24 items-center justify-center rounded-full text-white"
								style="background-color: {clusterColor}"
							>
								<span class="text-4xl font-bold">{cluster}</span>
							</div>
							<p class="mt-3 text-sm text-muted-foreground">Customer Segment</p>
						</div>
					</CardContent>
				</Card>

				<!-- Cluster Behavior -->
				<Card>
					<CardContent class="p-4">
						<div class="mb-3 flex items-center gap-2">
							<BarChart3 class="h-4 w-4 text-blue-500" />
							<span class="text-sm font-bold">Cluster Behavior</span>
						</div>
						<p class="mb-2 text-xs text-muted-foreground">
							Select a feature to see its distribution in the predicted cluster:
						</p>
						<Select
							value="event_type"
							options={['event_type', 'product_category', 'device_type', 'browser'].map((f) => ({
								value: f,
								label: f.replace(/_/g, ' ')
							}))}
							class="mb-3"
						/>
						<div class="flex h-[150px] items-center justify-center rounded-lg bg-muted/30">
							<p class="text-sm text-muted-foreground">Feature distribution chart</p>
						</div>
					</CardContent>
				</Card>
			</div>

			<!-- Cluster Interpretation -->
			<div
				class="flex items-start gap-3 rounded-lg border border-purple-200 bg-purple-50 p-4 text-sm text-purple-700 dark:border-purple-900 dark:bg-purple-950 dark:text-purple-300"
			>
				<Info class="mt-0.5 h-4 w-4 flex-shrink-0" />
				<div>
					<p class="font-bold">Cluster Interpretation:</p>
					<p>
						The predicted cluster represents a group of customers with similar interaction patterns.
						Use the 'Cluster Behavior' chart to understand the characteristics of this cluster.
					</p>
				</div>
			</div>
		{:else if modelAvailable}
			<div class="flex items-center gap-2 rounded-lg border border-blue-200 bg-blue-50 p-4 text-sm text-blue-700 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-300">
				<Info class="h-4 w-4" />
				<span>Fill in the customer interaction details and click <strong>Predict</strong> to get the cluster assignment.</span>
			</div>
		{:else}
			<div class="flex items-center gap-2 rounded-lg border border-orange-200 bg-orange-50 p-4 text-sm text-orange-700 dark:border-orange-900 dark:bg-orange-950 dark:text-orange-300">
				<AlertTriangle class="h-4 w-4" />
				<span>No trained model available. Click <strong>Train</strong> to train the batch model first.</span>
			</div>
		{/if}
	</div>
</div>
