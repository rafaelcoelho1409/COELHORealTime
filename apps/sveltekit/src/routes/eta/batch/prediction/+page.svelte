<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { browser } from '$app/environment';
	import {
		Card,
		CardContent,
		Button,
		Input,
		Select,
		BatchSubNav
	} from '$components/shared';
	import Plotly from 'svelte-plotly.js';
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
		batchMlflowExperimentUrl,
		batchLastTrainedRunId,
		updateFormField,
		updateProjectStore
	} from '$stores';
	import { toast } from '$stores/ui';
	import * as batchApi from '$api/batch';
	import { randomizeETAForm, FIELD_CONFIG, clampFieldValue } from '$lib/utils/randomize';
	import {
		Clock,
		Shuffle,
		MapPin,
		AlertTriangle,
		Info,
		GitBranch,
		Database,
		Play,
		Square,
		Loader2,
		CheckCircle,
		Brain,
		BarChart3,
		RefreshCw,
		FlaskConical
	} from 'lucide-svelte';
	import type { ProjectName } from '$types';
	import { cn } from '$lib/utils';

	const PROJECT: ProjectName = 'Estimated Time of Arrival';
	const MODEL_NAME = 'CatBoost Regressor';

	let dropdownOptions = $state<Record<string, string[]>>({});
	let sampleLoading = $state(false);
	let statusInterval: ReturnType<typeof setInterval> | null = null;
	let runsLoading = $state(false);

	onMount(async () => {
		try {
			const response = await fetch('/data/dropdown_options_eta.json');
			dropdownOptions = await response.json();
		} catch (e) {
			console.error('Failed to load dropdown options:', e);
		}

		if (!Object.keys($formData[PROJECT] || {}).length) {
			updateProjectStore(formData, PROJECT, randomizeETAForm(dropdownOptions));
		}

		const countResult = await batchApi.getDeltaLakeRowCount(PROJECT);
		if (countResult.data?.row_count) {
			updateProjectStore(batchDeltaLakeTotalRows, PROJECT, countResult.data.row_count);
		}

		await loadRuns();
		await checkModelAvailable();
	});

	onDestroy(() => {
		stopStatusPolling();
	});

	async function loadRuns() {
		runsLoading = true;
		const result = await batchApi.getMLflowRuns(PROJECT);
		if (result.data?.runs) {
			updateProjectStore(batchMlflowRuns, PROJECT, result.data.runs);
			updateProjectStore(batchModelAvailable, PROJECT, result.data.runs.length > 0);
		}
		runsLoading = false;
	}

	/** Load runs after training completes - stores last trained run ID and auto-selects it */
	async function loadRunsAfterTraining() {
		runsLoading = true;
		const result = await batchApi.getMLflowRuns(PROJECT);
		if (result.data?.runs) {
			updateProjectStore(batchMlflowRuns, PROJECT, result.data.runs);
			updateProjectStore(batchModelAvailable, PROJECT, result.data.runs.length > 0);

			if (result.data.runs.length > 0) {
				// The first run is the newest (just trained) - store and select it
				const newRunId = result.data.runs[0].run_id;
				updateProjectStore(batchLastTrainedRunId, PROJECT, newRunId);
				updateProjectStore(selectedBatchRun, PROJECT, newRunId);
			}
		}
		runsLoading = false;
	}

	async function checkModelAvailable() {
		const result = await batchApi.checkModelAvailable(PROJECT);
		if (result.data) {
			updateProjectStore(batchModelAvailable, PROJECT, result.data.available);
			if (result.data.experiment_url) {
				updateProjectStore(batchMlflowExperimentUrl, PROJECT, result.data.experiment_url);
			}
		}
	}

	async function onRunChange(runId: string) {
		updateProjectStore(selectedBatchRun, PROJECT, runId);
	}

	async function startTraining() {
		// Reset training state
		updateProjectStore(batchTrainingLoading, PROJECT, true);
		updateProjectStore(batchTrainingStatus, PROJECT, 'Starting training...');
		updateProjectStore(batchTrainingProgress, PROJECT, 0);
		updateProjectStore(batchTrainingStage, PROJECT, 'init');

		const mode = $batchTrainingMode[PROJECT];
		const maxRows = $batchTrainingMaxRows[PROJECT];
		const percentage = $batchTrainingDataPercentage[PROJECT];

		// Show toast with training info based on mode
		const dataInfo =
			mode === 'percentage'
				? percentage < 100
					? ` using ${percentage}% of data`
					: ''
				: ` using max ${maxRows.toLocaleString()} rows`;
		toast.info(`Batch ML training started${dataInfo}...`);

		const result = await batchApi.startTraining(PROJECT, {
			mode,
			percentage: mode === 'percentage' ? percentage : undefined,
			maxRows: mode === 'max_rows' ? maxRows : undefined
		});

		if (result.error) {
			toast.error(result.error);
			updateProjectStore(batchTrainingLoading, PROJECT, false);
			updateProjectStore(batchTrainingStatus, PROJECT, `Error: ${result.error}`);
			updateProjectStore(batchTrainingStage, PROJECT, 'error');
		} else if (result.data?.status === 'error') {
			const errorMsg = result.data.message || 'Failed to start training';
			toast.error(errorMsg);
			updateProjectStore(batchTrainingLoading, PROJECT, false);
			updateProjectStore(batchTrainingStatus, PROJECT, `Error: ${errorMsg}`);
			updateProjectStore(batchTrainingStage, PROJECT, 'error');
		} else {
			startStatusPolling();
		}
	}

	async function stopTraining() {
		const result = await batchApi.stopTraining(PROJECT);

		// Reset all training state
		updateProjectStore(batchTrainingLoading, PROJECT, false);
		updateProjectStore(batchTrainingStatus, PROJECT, '');
		updateProjectStore(batchTrainingProgress, PROJECT, 0);
		updateProjectStore(batchTrainingStage, PROJECT, '');
		stopStatusPolling();

		if (result.data?.status === 'stopped') {
			toast.info('Batch ML training stopped');
		} else {
			toast.info('No training was active');
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
			// Update live status from response (matching Reflex field names)
			if (result.data.status_message) {
				updateProjectStore(batchTrainingStatus, PROJECT, result.data.status_message);
			}
			if (result.data.progress_percent !== undefined) {
				updateProjectStore(batchTrainingProgress, PROJECT, result.data.progress_percent);
			}
			if (result.data.current_stage) {
				updateProjectStore(batchTrainingStage, PROJECT, result.data.current_stage);
			}

			// Check for completion
			if (result.data.status === 'completed') {
				stopStatusPolling();
				updateProjectStore(batchTrainingLoading, PROJECT, false);
				updateProjectStore(batchTrainingStatus, PROJECT, 'Training complete!');
				updateProjectStore(batchTrainingProgress, PROJECT, 100);
				updateProjectStore(batchTrainingStage, PROJECT, 'complete');
				toast.success('Batch ML training complete');

				// Refresh runs (stores last trained run ID), check model
				await loadRunsAfterTraining();
				await checkModelAvailable();
			} else if (result.data.status === 'failed') {
				const errorMsg = result.data.error || 'Training failed';
				stopStatusPolling();
				updateProjectStore(batchTrainingLoading, PROJECT, false);
				updateProjectStore(batchTrainingStatus, PROJECT, `Failed: ${errorMsg}`);
				updateProjectStore(batchTrainingStage, PROJECT, 'error');
				toast.error(errorMsg);
			} else if (result.data.status !== 'running') {
				// Unknown status or idle - training may have finished quickly
				stopStatusPolling();
				updateProjectStore(batchTrainingLoading, PROJECT, false);
				updateProjectStore(batchTrainingStatus, PROJECT, '');

				await loadRuns();
				await checkModelAvailable();
			}
		}
	}

	function loadRandomSample() {
		sampleLoading = true;
		try {
			const randomData = randomizeETAForm(dropdownOptions);
			updateProjectStore(formData, PROJECT, randomData);
			updateProjectStore(batchPredictionResults, PROJECT, {});
			toast.success('Form randomized');
		} finally {
			sampleLoading = false;
		}
	}

	function generateRandomCoordinates() {
		// Houston area coordinates
		const originLat = 29.5 + Math.random() * 0.6;
		const originLon = -95.8 + Math.random() * 0.8;
		const destLat = 29.5 + Math.random() * 0.6;
		const destLon = -95.8 + Math.random() * 0.8;

		formData.update((fd) => ({
			...fd,
			[PROJECT]: {
				...fd[PROJECT],
				origin_lat: parseFloat(originLat.toFixed(4)),
				origin_lon: parseFloat(originLon.toFixed(4)),
				destination_lat: parseFloat(destLat.toFixed(4)),
				destination_lon: parseFloat(destLon.toFixed(4))
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
				const eta = result.data['Estimated Time of Arrival'] || result.data.eta_minutes;
				toast.success(`Predicted ETA: ${eta?.toFixed(1)} minutes`);
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
	const runs = $derived($batchMlflowRuns[PROJECT] || []);
	const experimentUrl = $derived($batchMlflowExperimentUrl[PROJECT] || '');
	const lastTrainedRunId = $derived($batchLastTrainedRunId[PROJECT] || '');

	// Get selected run details for MLflow badge
	const selectedRunId = $derived($selectedBatchRun[PROJECT] || '');
	const selectedRun = $derived.by(() => {
		if (!selectedRunId || !runs.length) return null;
		return runs.find((r: { run_id: string }) => r.run_id === selectedRunId) || null;
	});

	function formatRunStartTime(value?: string | number): string {
		if (!value) return '';
		const timestamp = typeof value === 'number' ? value : Date.parse(value);
		if (!Number.isNaN(timestamp)) {
			return new Date(timestamp).toISOString().replace('T', ' ').slice(0, 19);
		}
		return String(value);
	}

	// Helper function to calculate distance (Haversine formula)
	function calcDistanceKm(oLat: number, oLon: number, dLat: number, dLon: number): number {
		const R = 6371; // Earth's radius in km
		const dLatRad = ((dLat - oLat) * Math.PI) / 180;
		const dLonRad = ((dLon - oLon) * Math.PI) / 180;
		const a =
			Math.sin(dLatRad / 2) * Math.sin(dLatRad / 2) +
			Math.cos((oLat * Math.PI) / 180) *
				Math.cos((dLat * Math.PI) / 180) *
				Math.sin(dLonRad / 2) *
				Math.sin(dLonRad / 2);
		const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
		return R * c;
	}

	// Calculate estimated distance using Haversine formula
	const estimatedDistanceKm = $derived.by(() => {
		const form = $formData[PROJECT];
		if (form?.estimated_distance_km) {
			return Number(form.estimated_distance_km);
		}
		const oLat = Number(form?.origin_lat) || 0;
		const oLon = Number(form?.origin_lon) || 0;
		const dLat = Number(form?.destination_lat) || 0;
		const dLon = Number(form?.destination_lon) || 0;
		if (oLat && oLon && dLat && dLon) {
			return Number(calcDistanceKm(oLat, oLon, dLat, dLon).toFixed(2));
		}
		return 0;
	});

	const initialEstimatedTravelTime = $derived.by(() => {
		const form = $formData[PROJECT];
		if (form?.initial_estimated_travel_time_seconds) {
			return Number(form.initial_estimated_travel_time_seconds);
		}
		if (estimatedDistanceKm > 0) {
			return Math.round((estimatedDistanceKm / 40) * 3600);
		}
		return 0;
	});

	// API returns ETA in SECONDS, so we need to divide by 60 to get minutes
	const etaSeconds = $derived(
		currentPrediction?.['Estimated Time of Arrival'] || currentPrediction?.eta_seconds || 0
	);
	const etaMinutes = $derived(etaSeconds > 0 ? etaSeconds / 60 : 0);

	// Plotly prediction chart data (matching Reflex exactly)
	const etaPredictionData = $derived.by(() => {
		return [
			{
				type: 'indicator',
				mode: 'number',
				value: etaSeconds,
				title: { text: '<b>Seconds</b>', font: { size: 18 } },
				number: { font: { size: 48, color: '#3b82f6' } },
				domain: { row: 0, column: 0 }
			},
			{
				type: 'indicator',
				mode: 'number',
				value: etaMinutes,
				title: { text: '<b>Minutes</b>', font: { size: 18 } },
				number: { font: { size: 48, color: '#22c55e' }, valueformat: '.1f' },
				domain: { row: 1, column: 0 }
			}
		];
	});

	const etaPredictionLayout = {
		grid: { rows: 2, columns: 1, pattern: 'independent' },
		height: 250,
		margin: { l: 20, r: 20, t: 40, b: 20 },
		paper_bgcolor: 'transparent',
		plot_bgcolor: 'transparent'
	};

	const plotlyConfig = { displayModeBar: false, responsive: true };

	// Map variables
	let mapContainer: HTMLDivElement | null = null;
	let leafletMap: any = null;
	let originMarker: any = null;
	let destMarker: any = null;
	let routeLine: any = null;
	let L: any = null;

	// Derived coordinates
	const originLat = $derived(Number(currentForm?.origin_lat) || 29.8);
	const originLon = $derived(Number(currentForm?.origin_lon) || -95.4);
	const destLat = $derived(Number(currentForm?.destination_lat) || 29.8);
	const destLon = $derived(Number(currentForm?.destination_lon) || -95.4);

	// Initialize Leaflet map
	async function initMap() {
		if (!browser || !mapContainer) return;

		// Add Leaflet CSS if not already loaded
		if (!document.getElementById('leaflet-css')) {
			const link = document.createElement('link');
			link.id = 'leaflet-css';
			link.rel = 'stylesheet';
			link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
			document.head.appendChild(link);
		}

		// Load Leaflet JS from CDN if not already loaded
		if (!(window as any).L) {
			await new Promise<void>((resolve, reject) => {
				const script = document.createElement('script');
				script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
				script.onload = () => resolve();
				script.onerror = reject;
				document.head.appendChild(script);
			});
		}

		// Wait for CSS to load
		await new Promise((r) => setTimeout(r, 100));

		L = (window as any).L;
		if (!L) return;

		const centerLat = (originLat + destLat) / 2;
		const centerLon = (originLon + destLon) / 2;

		leafletMap = L.map(mapContainer).setView([centerLat, centerLon], 10);

		L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
			attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
		}).addTo(leafletMap);

		// Add markers and line
		updateMapMarkers();
	}

	function updateMapMarkers() {
		if (!L || !leafletMap) return;

		// Remove existing markers and line
		if (originMarker) leafletMap.removeLayer(originMarker);
		if (destMarker) leafletMap.removeLayer(destMarker);
		if (routeLine) leafletMap.removeLayer(routeLine);

		// Create custom icons
		const blueIcon = L.divIcon({
			className: 'custom-marker',
			html: '<div style="background-color: #3b82f6; width: 24px; height: 24px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>',
			iconSize: [24, 24],
			iconAnchor: [12, 12]
		});

		const redIcon = L.divIcon({
			className: 'custom-marker',
			html: '<div style="background-color: #ef4444; width: 24px; height: 24px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>',
			iconSize: [24, 24],
			iconAnchor: [12, 12]
		});

		// Add markers
		originMarker = L.marker([originLat, originLon], { icon: blueIcon })
			.addTo(leafletMap)
			.bindPopup('Origin');

		destMarker = L.marker([destLat, destLon], { icon: redIcon })
			.addTo(leafletMap)
			.bindPopup('Destination');

		// Add bold route line
		routeLine = L.polyline(
			[
				[originLat, originLon],
				[destLat, destLon]
			],
			{
				color: '#333333',
				weight: 4,
				opacity: 0.8
			}
		).addTo(leafletMap);

		// Fit bounds to show both markers
		const bounds = L.latLngBounds([
			[originLat, originLon],
			[destLat, destLon]
		]);
		leafletMap.fitBounds(bounds, { padding: [30, 30] });
	}

	// Watch for coordinate changes
	$effect(() => {
		const _oLat = originLat;
		const _oLon = originLon;
		const _dLat = destLat;
		const _dLon = destLon;

		if (leafletMap && L) {
			updateMapMarkers();
		}
	});

	// Initialize map on mount
	$effect(() => {
		if (browser && mapContainer && !leafletMap) {
			setTimeout(() => {
				initMap();
			}, 50);
		}
	});
</script>

<!-- 40%/60% Layout -->
<div class="flex gap-6">
	<!-- Left Column - Training Box + Form (40%) -->
	<div class="w-[40%] space-y-4">
		<!-- Batch ML Training Box (cloned from Reflex) -->
		<Card>
			<CardContent class="space-y-3 p-3">
				<!-- MLflow Run Section -->
				<div class="flex items-center gap-2">
					<GitBranch class="h-4 w-4 text-blue-600" />
					<span class="text-xs font-medium">MLflow Run</span>
					{#if runsLoading}
						<Loader2 class="h-3 w-3 animate-spin text-muted-foreground" />
					{/if}
					<div class="flex-1"></div>
					<!-- MLflow Button (matching Reflex) -->
					{#if experimentUrl}
						<a
							href={experimentUrl}
							target="_blank"
							rel="noopener noreferrer"
							class="inline-flex items-center gap-1 rounded-md bg-cyan-100 px-2 py-1 text-xs font-medium text-cyan-700 hover:bg-cyan-200 dark:bg-cyan-900 dark:text-cyan-300 dark:hover:bg-cyan-800"
						>
							<img
								src="https://cdn.simpleicons.org/mlflow/0194E2"
								alt="MLflow"
								class="h-3.5 w-3.5"
							/>
							MLflow
						</a>
					{:else}
						<span
							class="inline-flex cursor-not-allowed items-center gap-1 rounded-md bg-gray-100 px-2 py-1 text-xs font-medium text-gray-400 dark:bg-gray-800"
						>
							<img
								src="https://cdn.simpleicons.org/mlflow/0194E2"
								alt="MLflow"
								class="h-3.5 w-3.5 opacity-50"
							/>
							MLflow
						</span>
					{/if}
					<button
						type="button"
						class="rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
						onclick={loadRuns}
						disabled={runsLoading}
						title="Refresh runs"
					>
						<RefreshCw class="h-3 w-3" />
					</button>
				</div>
				{#if runs.length > 0}
					<select
						class="w-full rounded-md border border-input bg-background px-2.5 py-1.5 text-xs shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/30"
						value={$selectedBatchRun[PROJECT] || ''}
						onchange={(e) => onRunChange(e.currentTarget.value)}
					>
						<option value="">Select MLflow run...</option>
						{#each runs as run}
							<option value={run.run_id}>{run.run_id.slice(0, 8)} - {run.status}</option>
						{/each}
					</select>
				{:else}
					<p class="text-xs text-muted-foreground">No runs available. Train a model first.</p>
				{/if}

				<!-- Divider -->
				<hr class="border-border" />

				<!-- Batch ML Training Section -->
				<div class="flex items-center justify-between">
					<div class="flex items-center gap-2">
						<div
							class={cn(
								'flex h-5 w-5 items-center justify-center rounded',
								modelAvailable ? 'text-blue-600' : 'text-muted-foreground'
							)}
						>
							<Database class="h-4 w-4" />
						</div>
						<div>
							<span class="text-sm font-medium">Batch ML Training</span>
							<p class="text-[10px] text-muted-foreground">
								{#if isTraining}
									Training in progress...
								{:else if modelAvailable}
									Model trained and ready
								{:else}
									Click Train to create model
								{/if}
							</p>
						</div>
					</div>
					{#if isTraining}
						<div class="flex items-center gap-2">
							<Loader2 class="h-4 w-4 animate-spin text-blue-600" />
							<button
								type="button"
								class="inline-flex items-center gap-1 rounded-md bg-red-100 px-2 py-1 text-xs font-medium text-red-700 hover:bg-red-200"
								onclick={stopTraining}
							>
								<Square class="h-3 w-3" />
								Stop
							</button>
						</div>
					{:else}
						<button
							type="button"
							class="inline-flex items-center gap-1 rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700"
							onclick={startTraining}
						>
							<Play class="h-3 w-3" />
							Train
						</button>
					{/if}
				</div>

				<!-- Training Progress (shown only during training) -->
				{#if isTraining}
					<div class="space-y-2">
						<div class="h-1.5 w-full overflow-hidden rounded-full bg-muted">
							<div
								class="h-full rounded-full bg-blue-600 transition-all duration-300"
								style="width: {$batchTrainingProgress[PROJECT]}%"
							></div>
						</div>
						<div class="flex items-center gap-2 text-[10px] text-muted-foreground">
							{#if $batchTrainingStage[PROJECT] === 'loading_data'}
								<Database class="h-3 w-3 text-blue-600" />
							{:else if $batchTrainingStage[PROJECT] === 'training'}
								<Brain class="h-3 w-3 text-purple-600" />
							{:else if $batchTrainingStage[PROJECT] === 'evaluating'}
								<BarChart3 class="h-3 w-3 text-green-600" />
							{:else}
								<Loader2 class="h-3 w-3 animate-spin" />
							{/if}
							<span class="italic">{$batchTrainingStatus[PROJECT]}</span>
							<span class="ml-auto font-medium text-blue-600">{$batchTrainingProgress[PROJECT]}%</span>
						</div>
					</div>
				{/if}

				<!-- Last trained run ID display (only shows when a model was trained this session) -->
				{#if lastTrainedRunId}
					<div class="flex items-center gap-1 pt-1">
						<CheckCircle class="h-3 w-3 text-green-600" />
						<span class="text-[10px] text-muted-foreground">Last trained:</span>
						<code class="rounded bg-green-100 px-1 py-0.5 text-[10px] text-green-700 dark:bg-green-900 dark:text-green-300">
							{lastTrainedRunId}
						</code>
					</div>
				{/if}

				<!-- Training Data Options (single row, hidden during training) -->
				{#if !isTraining}
					<div class="flex items-center gap-2">
						<Database class="h-3.5 w-3.5 text-muted-foreground" />
						<span class="text-xs text-muted-foreground">Data</span>
						<select
							class="rounded border border-input bg-background px-2 py-1 text-xs"
							value={$batchTrainingMode[PROJECT]}
							onchange={(e) => updateProjectStore(batchTrainingMode, PROJECT, e.currentTarget.value as 'percentage' | 'max_rows')}
						>
							<option value="percentage">Percentage</option>
							<option value="max_rows">Max Rows</option>
						</select>
						{#if $batchTrainingMode[PROJECT] === 'percentage'}
							<input
								type="number"
								min="1"
								max="100"
								value={$batchTrainingDataPercentage[PROJECT]}
								oninput={(e) => updateProjectStore(batchTrainingDataPercentage, PROJECT, parseInt(e.currentTarget.value) || 10)}
								class="w-14 rounded border border-input bg-background px-2 py-1 text-xs"
							/>
							<span class="text-xs text-muted-foreground">%</span>
						{:else}
							<input
								type="number"
								min="1000"
								max={$batchDeltaLakeTotalRows[PROJECT] || 10000000}
								value={$batchTrainingMaxRows[PROJECT]}
								oninput={(e) => updateProjectStore(batchTrainingMaxRows, PROJECT, parseInt(e.currentTarget.value) || 10000)}
								class="w-20 rounded border border-input bg-background px-2 py-1 text-xs"
							/>
							<span class="text-xs text-muted-foreground">rows</span>
						{/if}
					</div>
				{/if}
			</CardContent>
		</Card>

		<Card>
			<CardContent class="space-y-3 pt-4">
				<div class="flex items-center gap-2">
					<Clock class="h-5 w-5 text-primary" />
					<h3 class="text-base font-bold">Trip Details</h3>
				</div>

				<div class="flex gap-2">
					<Button class="flex-1" onclick={predict} loading={isLoading} disabled={!modelAvailable}>
						Predict
					</Button>
					<Button
						variant="secondary"
						class="flex-1 bg-blue-500/10 text-blue-600 hover:bg-blue-500/20"
						onclick={loadRandomSample}
						loading={sampleLoading}
					>
						<Shuffle class="mr-2 h-3.5 w-3.5" />
						Randomize
					</Button>
				</div>

				<hr class="border-border" />

				<div class="grid grid-cols-3 gap-2">
					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Driver ID ({FIELD_CONFIG.driver_id.min}-{FIELD_CONFIG.driver_id.max})</p>
						<Input
							type="number"
							value={(currentForm.driver_id as number) ?? FIELD_CONFIG.driver_id.min}
							min={FIELD_CONFIG.driver_id.min}
							max={FIELD_CONFIG.driver_id.max}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseInt(e.currentTarget.value) || FIELD_CONFIG.driver_id.min;
								const clamped = clampFieldValue('driver_id', val);
								updateFormField(PROJECT, 'driver_id', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Vehicle ID ({FIELD_CONFIG.vehicle_id.min}-{FIELD_CONFIG.vehicle_id.max})</p>
						<Input
							type="number"
							value={(currentForm.vehicle_id as number) ?? FIELD_CONFIG.vehicle_id.min}
							min={FIELD_CONFIG.vehicle_id.min}
							max={FIELD_CONFIG.vehicle_id.max}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseInt(e.currentTarget.value) || FIELD_CONFIG.vehicle_id.min;
								const clamped = clampFieldValue('vehicle_id', val);
								updateFormField(PROJECT, 'vehicle_id', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Weather</p>
						<Select
							value={(currentForm.weather as string) ?? 'Clear'}
							options={dropdownOptions.weather || ['Clear']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'weather', e.currentTarget.value)}
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
						<p class="text-xs text-muted-foreground">Vehicle Type</p>
						<Select
							value={(currentForm.vehicle_type as string) ?? 'Sedan'}
							options={dropdownOptions.vehicle_type || ['Sedan']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'vehicle_type', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Origin Lat ({FIELD_CONFIG.origin_lat.min}-{FIELD_CONFIG.origin_lat.max})</p>
						<Input
							type="number"
							value={currentForm.origin_lat ?? ''}
							min={FIELD_CONFIG.origin_lat.min}
							max={FIELD_CONFIG.origin_lat.max}
							step={FIELD_CONFIG.origin_lat.step}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.origin_lat.min;
								const clamped = clampFieldValue('origin_lat', val);
								updateFormField(PROJECT, 'origin_lat', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Origin Lon ({FIELD_CONFIG.origin_lon.min}-{FIELD_CONFIG.origin_lon.max})</p>
						<Input
							type="number"
							value={currentForm.origin_lon ?? ''}
							min={FIELD_CONFIG.origin_lon.min}
							max={FIELD_CONFIG.origin_lon.max}
							step={FIELD_CONFIG.origin_lon.step}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.origin_lon.min;
								const clamped = clampFieldValue('origin_lon', val);
								updateFormField(PROJECT, 'origin_lon', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Coords</p>
						<Button
							variant="secondary"
							size="sm"
							class="h-8 w-full bg-blue-500/10 text-xs text-blue-600 hover:bg-blue-500/20"
							onclick={generateRandomCoordinates}
						>
							<Shuffle class="mr-1 h-3 w-3" />
							Random
						</Button>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Dest Lat ({FIELD_CONFIG.destination_lat.min}-{FIELD_CONFIG.destination_lat.max})</p>
						<Input
							type="number"
							value={currentForm.destination_lat ?? ''}
							min={FIELD_CONFIG.destination_lat.min}
							max={FIELD_CONFIG.destination_lat.max}
							step={FIELD_CONFIG.destination_lat.step}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.destination_lat.min;
								const clamped = clampFieldValue('destination_lat', val);
								updateFormField(PROJECT, 'destination_lat', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Dest Lon ({FIELD_CONFIG.destination_lon.min}-{FIELD_CONFIG.destination_lon.max})</p>
						<Input
							type="number"
							value={currentForm.destination_lon ?? ''}
							min={FIELD_CONFIG.destination_lon.min}
							max={FIELD_CONFIG.destination_lon.max}
							step={FIELD_CONFIG.destination_lon.step}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.destination_lon.min;
								const clamped = clampFieldValue('destination_lon', val);
								updateFormField(PROJECT, 'destination_lon', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Hour ({FIELD_CONFIG.hour_of_day.min}-{FIELD_CONFIG.hour_of_day.max})</p>
						<Input
							type="number"
							value={currentForm.hour_of_day ?? ''}
							min={FIELD_CONFIG.hour_of_day.min}
							max={FIELD_CONFIG.hour_of_day.max}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseInt(e.currentTarget.value) || FIELD_CONFIG.hour_of_day.min;
								const clamped = clampFieldValue('hour_of_day', val);
								updateFormField(PROJECT, 'hour_of_day', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Rating ({FIELD_CONFIG.driver_rating.min}-{FIELD_CONFIG.driver_rating.max})</p>
						<Input
							type="number"
							value={currentForm.driver_rating ?? ''}
							min={FIELD_CONFIG.driver_rating.min}
							max={FIELD_CONFIG.driver_rating.max}
							step={FIELD_CONFIG.driver_rating.step}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.driver_rating.min;
								const clamped = clampFieldValue('driver_rating', val);
								updateFormField(PROJECT, 'driver_rating', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Temp C ({FIELD_CONFIG.temperature_celsius.min}-{FIELD_CONFIG.temperature_celsius.max})</p>
						<Input
							type="number"
							value={currentForm.temperature_celsius ?? ''}
							min={FIELD_CONFIG.temperature_celsius.min}
							max={FIELD_CONFIG.temperature_celsius.max}
							step={FIELD_CONFIG.temperature_celsius.step}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.temperature_celsius.min;
								const clamped = clampFieldValue('temperature_celsius', val);
								updateFormField(PROJECT, 'temperature_celsius', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Traffic Factor ({FIELD_CONFIG.debug_traffic_factor.min}-{FIELD_CONFIG.debug_traffic_factor.max})</p>
						<Input
							type="number"
							value={currentForm.debug_traffic_factor ?? ''}
							min={FIELD_CONFIG.debug_traffic_factor.min}
							max={FIELD_CONFIG.debug_traffic_factor.max}
							step={FIELD_CONFIG.debug_traffic_factor.step}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.debug_traffic_factor.min;
								const clamped = clampFieldValue('debug_traffic_factor', val);
								updateFormField(PROJECT, 'debug_traffic_factor', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Weather Factor ({FIELD_CONFIG.debug_weather_factor.min}-{FIELD_CONFIG.debug_weather_factor.max})</p>
						<Input
							type="number"
							value={currentForm.debug_weather_factor ?? ''}
							min={FIELD_CONFIG.debug_weather_factor.min}
							max={FIELD_CONFIG.debug_weather_factor.max}
							step={FIELD_CONFIG.debug_weather_factor.step}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.debug_weather_factor.min;
								const clamped = clampFieldValue('debug_weather_factor', val);
								updateFormField(PROJECT, 'debug_weather_factor', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Driver Factor ({FIELD_CONFIG.debug_driver_factor.min}-{FIELD_CONFIG.debug_driver_factor.max})</p>
						<Input
							type="number"
							value={currentForm.debug_driver_factor ?? ''}
							min={FIELD_CONFIG.debug_driver_factor.min}
							max={FIELD_CONFIG.debug_driver_factor.max}
							step={FIELD_CONFIG.debug_driver_factor.step}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.debug_driver_factor.min;
								const clamped = clampFieldValue('debug_driver_factor', val);
								updateFormField(PROJECT, 'debug_driver_factor', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Incident ({FIELD_CONFIG.debug_incident_delay_seconds.min}-{FIELD_CONFIG.debug_incident_delay_seconds.max}s)</p>
						<Input
							type="number"
							value={currentForm.debug_incident_delay_seconds ?? ''}
							min={FIELD_CONFIG.debug_incident_delay_seconds.min}
							max={FIELD_CONFIG.debug_incident_delay_seconds.max}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseInt(e.currentTarget.value) || FIELD_CONFIG.debug_incident_delay_seconds.min;
								const clamped = clampFieldValue('debug_incident_delay_seconds', val);
								updateFormField(PROJECT, 'debug_incident_delay_seconds', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>
				</div>

				<div class="space-y-1 text-xs text-muted-foreground">
					<p>Trip ID: {currentForm.trip_id || '-'}</p>
					<p>Estimated Distance: {estimatedDistanceKm} km</p>
					<p>Initial Estimated Travel Time: {initialEstimatedTravelTime} s</p>
				</div>
			</CardContent>
		</Card>
	</div>

	<!-- Right Column - Batch Sub-Nav + Prediction Result (60%) -->
	<div class="w-[60%] space-y-4">
		<BatchSubNav projectKey="eta" />

		<!-- Prediction Result - Side by Side -->
		<div class="flex gap-3">
			<!-- Origin and Destination Card -->
			<Card class="h-[400px] w-1/2">
				<CardContent class="h-full p-4">
					<div class="flex h-full flex-col">
						<div class="mb-2 flex items-center gap-2">
							<MapPin class="h-4 w-4 text-primary" />
							<span class="text-sm font-bold">Origin and Destination</span>
						</div>
						<div
							bind:this={mapContainer}
							class="flex-1 rounded-md overflow-hidden"
							style="min-height: 280px;"
						></div>
						<div class="mt-2 space-y-1 text-xs text-muted-foreground">
							<p>Estimated Distance: {estimatedDistanceKm} km</p>
							<p>Initial Estimated Travel Time: {initialEstimatedTravelTime} s</p>
						</div>
					</div>
				</CardContent>
			</Card>

			<!-- ETA Prediction Card -->
			<Card class="h-[400px] w-1/2">
				<CardContent class="h-full p-4">
					<div class="flex h-full flex-col">
						<div class="mb-2 flex items-center gap-2">
							<Clock class="h-4 w-4 text-primary" />
							<span class="text-sm font-bold">ETA - Prediction</span>
						</div>

						<!-- MLflow Run Info Badge -->
						<div class="mb-4 flex flex-wrap items-center gap-2 rounded-md bg-muted/50 p-2">
							<span
								class="inline-flex items-center gap-1 rounded-md bg-purple-500/10 px-2 py-1 text-xs font-medium text-purple-600"
							>
								<FlaskConical class="h-3 w-3" />
								MLflow
							</span>
							{#if isTraining}
								<span
									class="inline-flex items-center gap-1 rounded-md bg-amber-500/10 px-2 py-1 text-xs font-medium text-amber-600"
								>
									<Loader2 class="h-3 w-3 animate-spin" />
									TRAINING
								</span>
							{:else if selectedRun?.status}
								<span
									class="inline-flex items-center gap-1 rounded-md bg-blue-500/10 px-2 py-1 text-xs font-medium text-blue-600"
								>
									{selectedRun.status}
								</span>
							{:else if modelAvailable}
								<span
									class="inline-flex items-center gap-1 rounded-md bg-green-500/10 px-2 py-1 text-xs font-medium text-green-600"
								>
									READY
								</span>
							{/if}
							{#if selectedRunId}
								<span class="text-xs text-muted-foreground">
									Run:
									<code
										class="rounded bg-muted px-1"
										title={selectedRunId}
									>
										{selectedRunId.slice(0, 8)}
									</code>
								</span>
							{/if}
							{#if selectedRun?.start_time}
								<span class="text-xs text-muted-foreground">
									Started: {formatRunStartTime(selectedRun.start_time)}
								</span>
							{/if}
							{#if lastTrainedRunId && lastTrainedRunId !== selectedRunId}
								<span class="text-xs text-muted-foreground">
									(Last trained: <code class="rounded bg-green-100 px-1 text-green-700 dark:bg-green-900 dark:text-green-300">{lastTrainedRunId.slice(0, 8)}</code>)
								</span>
							{/if}
						</div>

						<div class="flex flex-1 items-center justify-center">
							{#if etaSeconds > 0}
								<Plotly
									data={etaPredictionData}
									layout={etaPredictionLayout}
									config={plotlyConfig}
								/>
							{:else if modelAvailable}
								<div class="flex items-center gap-2 rounded-lg border border-blue-200 bg-blue-50 p-4 text-sm text-blue-700 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-300">
									<Info class="h-4 w-4" />
									<span>Click <strong>Predict</strong> to get the estimated time of arrival.</span>
								</div>
							{:else}
								<div class="flex items-center gap-2 rounded-lg border border-orange-200 bg-orange-50 p-4 text-sm text-orange-700 dark:border-orange-900 dark:bg-orange-950 dark:text-orange-300">
									<AlertTriangle class="h-4 w-4" />
									<span>No trained model available. Click <strong>Train</strong> to train the batch model first.</span>
								</div>
							{/if}
						</div>
					</div>
				</CardContent>
			</Card>
		</div>
	</div>
</div>
