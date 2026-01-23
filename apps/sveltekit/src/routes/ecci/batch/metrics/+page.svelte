<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { beforeNavigate } from '$app/navigation';
	import { browser } from '$app/environment';
	import {
		Card,
		CardContent,
		Button,
		Input,
		Select,
		MetricCard,
		Tabs,
		TabsList,
		TabsTrigger,
		TabsContent
	} from '$components/shared';
	import Plotly from 'svelte-plotly.js';
	import {
		formData,
		batchPredictionResults,
		batchPredictionLoading,
		batchMlflowMetrics,
		batchMlflowRuns,
		selectedBatchRun,
		yellowBrickImages,
		yellowBrickLoading,
		selectedYellowBrickVisualizer,
		batchTrainingLoading,
		batchTrainingProgress,
		batchTrainingStage,
		batchTrainingStatus,
		batchTrainingMode,
		batchTrainingDataPercentage,
		batchTrainingMaxRows,
		batchDeltaLakeTotalRows,
		batchModelAvailable,
		batchMlflowExperimentUrl,
		batchLastTrainedRunId,
		batchTrainingCatboostLog,
		updateFormField,
		updateProjectStore
	} from '$stores';
	import { toast } from '$stores/ui';
	import { metricInfoDialogOpen, metricInfoDialogContent } from '$stores';
	import * as batchApi from '$api/batch';
	import { randomizeECCIForm } from '$lib/utils/randomize';
	import {
		Shuffle,
		ShoppingCart,
		Target,
		BarChart3,
		LayoutDashboard,
		Boxes,
		ScatterChart,
		Crosshair,
		Settings2,
		FileText,
		Lightbulb,
		Info,
		Square,
		RefreshCw,
		Loader2,
		Database,
		Brain,
		CheckCircle,
		FlaskConical,
		ExternalLink,
		MapPin,
		TrendingUp,
		Clock,
		Star
	} from 'lucide-svelte';
	import type { DropdownOptions, ProjectName } from '$types';

	const PROJECT: ProjectName = 'E-Commerce Customer Interactions';
	const MODEL_NAME = 'KMeans (Scikit-Learn)';

	// Transform MLflow metrics from API format to simple format (clustering metrics)
	function transformMetrics(rawMetrics: Record<string, unknown>): Record<string, number> {
		const getMetric = (name: string): number => {
			const key = `metrics.${name}`;
			const val = rawMetrics[key];
			return val !== undefined && val !== null ? Number(val) : 0;
		};

		return {
			silhouette_score: getMetric('silhouette_score') || getMetric('Silhouette'),
			calinski_harabasz_score: getMetric('calinski_harabasz_score') || getMetric('CalinskiHarabasz'),
			davies_bouldin_score: getMetric('davies_bouldin_score') || getMetric('DaviesBouldin'),
			inertia: getMetric('inertia') || getMetric('Inertia'),
			n_clusters: getMetric('n_clusters') || getMetric('NClusters'),
			preprocessing_time_seconds: getMetric('preprocessing_time_seconds'),
			n_samples: getMetric('n_samples') ||
			           Number(rawMetrics['params.train_samples'] || 0) + Number(rawMetrics['params.test_samples'] || 0)
		};
	}

	// YellowBrick visualizer options organized by category (clustering)
	const YELLOWBRICK_CATEGORIES = {
		Clustering: [
			{ value: '', label: 'Select visualization...' },
			{ value: 'KElbowVisualizer', label: 'K-Elbow' },
			{ value: 'SilhouetteVisualizer', label: 'Silhouette' },
			{ value: 'InterclusterDistance', label: 'Intercluster Distance' }
		],
		'Feature Analysis': [
			{ value: '', label: 'Select visualization...' },
			{ value: 'Rank1D', label: 'Rank 1D' },
			{ value: 'Rank2D', label: 'Rank 2D' },
			{ value: 'PCA', label: 'PCA Decomposition' },
			{ value: 'Manifold', label: 'Manifold' },
			{ value: 'ParallelCoordinates', label: 'Parallel Coordinates' },
			{ value: 'RadViz', label: 'RadViz' },
			{ value: 'JointPlot', label: 'Joint Plot' }
		],
		Target: [
			{ value: '', label: 'Select visualization...' },
			{ value: 'ClassBalance', label: 'Cluster Distribution' },
			{ value: 'FeatureCorrelation', label: 'Feature Correlation (Mutual Info)' },
			{ value: 'FeatureCorrelation_Pearson', label: 'Feature Correlation (Pearson)' },
			{ value: 'BalancedBinningReference', label: 'Balanced Binning Reference' }
		],
		'Model Selection': [
			{ value: '', label: 'Select visualization...' },
			{ value: 'FeatureImportances', label: 'Feature Importances' },
			{ value: 'CVScores', label: 'Cross-Validation Scores' },
			{ value: 'ValidationCurve', label: 'Validation Curve' },
			{ value: 'LearningCurve', label: 'Learning Curve' },
			{ value: 'RFECV', label: 'Recursive Feature Elimination' },
			{ value: 'DroppingCurve', label: 'Dropping Curve' }
		],
		'Text Analysis': [
			{ value: '', label: 'Select visualization...' },
			{ value: 'FreqDistVisualizer', label: 'Frequency Distribution' },
			{ value: 'TSNEVisualizer', label: 't-SNE Visualization' },
			{ value: 'UMAPVisualizer', label: 'UMAP Visualization' },
			{ value: 'DispersionPlot', label: 'Dispersion Plot' },
			{ value: 'WordCorrelationPlot', label: 'Word Correlation' },
			{ value: 'PosTagVisualizer', label: 'POS Tag Distribution' }
		]
	};

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

	let activeTab = $state('prediction');
	let activeMetricsTab = $state('overview');
	let metricsLoading = $state(false);
	let sampleLoading = $state(false);
	let statusInterval: ReturnType<typeof setInterval> | null = null;
	let dropdownOptions = $state<DropdownOptions>({});
	let selectedFeature = $state(ECCI_FEATURE_OPTIONS[0]);
	let clusterFeatureCounts = $state<Record<string, Record<string, number>>>({});

	// YellowBrick cancel flag
	let yellowBrickCancelRequested = $state(false);
	let currentYellowBrickCategory = $state('Clustering');
	let yellowBrickInfoOpen = $state(false);
	let yellowBrickInfoContent = $state<Record<string, unknown>>({});

	onMount(async () => {
		// Load dropdown options
		try {
			const response = await fetch('/data/dropdown_options_ecci.json');
			dropdownOptions = await response.json();
		} catch (e) {
			console.error('Failed to load dropdown options:', e);
		}

		// Initialize form with random data if empty
		if (!Object.keys($formData[PROJECT] || {}).length) {
			updateProjectStore(formData, PROJECT, randomizeECCIForm(dropdownOptions));
		}

		// Load Delta Lake row count
		const countResult = await batchApi.getDeltaLakeRowCount(PROJECT);
		if (countResult.data?.row_count) {
			updateProjectStore(batchDeltaLakeTotalRows, PROJECT, countResult.data.row_count);
		}

		// Initialize batch page with runs and metrics
		await initBatchPage();
	});

	onDestroy(() => {
		stopStatusPolling();
		yellowBrickCancelRequested = true;
	});

	// Stop training when navigating away
	beforeNavigate(async ({ to }) => {
		yellowBrickCancelRequested = true;
		if ($batchTrainingLoading[PROJECT]) {
			await batchApi.stopTraining(PROJECT);
			updateProjectStore(batchTrainingLoading, PROJECT, false);
			updateProjectStore(batchTrainingStatus, PROJECT, '');
			updateProjectStore(batchTrainingProgress, PROJECT, 0);
			stopStatusPolling();
			toast.info('Training stopped - navigated away from page');
		}
	});

	async function initBatchPage() {
		const runId = $selectedBatchRun[PROJECT] || undefined;
		const result = await batchApi.initBatchPage(PROJECT, runId);

		if (result.data) {
			const { runs, model_available, experiment_url, metrics, best_run_id } = result.data;

			updateProjectStore(batchMlflowRuns, PROJECT, runs || []);
			updateProjectStore(batchModelAvailable, PROJECT, model_available);

			if (experiment_url) {
				updateProjectStore(batchMlflowExperimentUrl, PROJECT, experiment_url);
			}

			if (metrics && !metrics._no_runs) {
				const transformed = transformMetrics(metrics);
				updateProjectStore(batchMlflowMetrics, PROJECT, transformed);
			}

			// Auto-select best run if none selected
			if (!$selectedBatchRun[PROJECT] && runs?.length > 0) {
				const bestRun = runs.find((r: { is_best?: boolean }) => r.is_best) || runs[0];
				updateProjectStore(selectedBatchRun, PROJECT, bestRun.run_id);
			}
		}
	}

	async function loadRuns() {
		const result = await batchApi.getMLflowRuns(PROJECT);
		if (result.data?.runs) {
			updateProjectStore(batchMlflowRuns, PROJECT, result.data.runs);
			updateProjectStore(batchModelAvailable, PROJECT, result.data.runs.length > 0);
		}
	}

	async function loadMetrics() {
		metricsLoading = true;
		const runId = $selectedBatchRun[PROJECT] || undefined;
		const result = await batchApi.getMLflowMetrics(PROJECT, runId);
		if (result.data && !result.data._no_runs) {
			const transformed = transformMetrics(result.data);
			updateProjectStore(batchMlflowMetrics, PROJECT, transformed);
		}
		metricsLoading = false;
	}

	async function loadRunsAfterTraining() {
		await new Promise((resolve) => setTimeout(resolve, 1000));
		await loadRuns();
		const runs = $batchMlflowRuns[PROJECT] || [];
		if (runs.length > 0) {
			const latestRun = runs[0];
			updateProjectStore(selectedBatchRun, PROJECT, latestRun.run_id);
			updateProjectStore(batchLastTrainedRunId, PROJECT, latestRun.run_id);
			await loadMetrics();
		}
	}

	async function loadVisualizer(category: string, visualizerName: string) {
		if (!visualizerName) {
			updateProjectStore(selectedYellowBrickVisualizer, PROJECT, '');
			return;
		}

		yellowBrickCancelRequested = false;
		currentYellowBrickCategory = category;
		updateProjectStore(selectedYellowBrickVisualizer, PROJECT, visualizerName);
		updateProjectStore(yellowBrickLoading, PROJECT, { [visualizerName]: true });

		const runId = $selectedBatchRun[PROJECT] || undefined;
		const result = await batchApi.getYellowBrickImage(PROJECT, category, visualizerName, runId);

		if (yellowBrickCancelRequested) return;

		if (result.data?.image_base64) {
			yellowBrickImages.update((imgs) => ({
				...imgs,
				[PROJECT]: {
					...imgs[PROJECT],
					[visualizerName]: result.data!.image_base64
				}
			}));
		} else if (result.error || result.data?.error) {
			toast.error(`Failed to load ${visualizerName}: ${result.error || result.data?.error}`);
		}

		updateProjectStore(yellowBrickLoading, PROJECT, { [visualizerName]: false });
	}

	function cancelYellowBrickLoading() {
		yellowBrickCancelRequested = true;
		updateProjectStore(yellowBrickLoading, PROJECT, {});
		updateProjectStore(selectedYellowBrickVisualizer, PROJECT, '');
		yellowBrickImages.update((imgs) => ({ ...imgs, [PROJECT]: {} }));
		toast.info('Visualization loading cancelled');
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
						docsUrl: info.docs_url?.batch || ''
					});
					metricInfoDialogOpen.set(true);
				}
			});
	}

	function openYellowBrickInfo(visualizerName: string) {
		if (!visualizerName) return;
		fetch('/data/yellowbrick_info_ecci.json')
			.then((r) => r.json())
			.then((data) => {
				const info = data[visualizerName];
				if (info) {
					yellowBrickInfoContent = info;
					yellowBrickInfoOpen = true;
				}
			});
	}

	// Handle outer tab changes (Prediction/Metrics)
	function onOuterTabChange(newTab: string) {
		activeTab = newTab;
	}

	// Reset visualization when switching to a YellowBrick tab
	function onMetricsTabChange(newTab: string) {
		activeMetricsTab = newTab;
		const yellowBrickTabs = ['clustering', 'features', 'target', 'diagnostics', 'text'];
		if (yellowBrickTabs.includes(newTab)) {
			updateProjectStore(selectedYellowBrickVisualizer, PROJECT, '');
			yellowBrickImages.update((imgs) => ({
				...imgs,
				[PROJECT]: {}
			}));
		}
	}

	async function onRunChange(runId: string) {
		updateProjectStore(selectedBatchRun, PROJECT, runId);
		await loadMetrics();
		const yellowBrickTabs = ['clustering', 'features', 'target', 'diagnostics', 'text'];
		if (yellowBrickTabs.includes(activeMetricsTab)) {
			const currentViz = $selectedYellowBrickVisualizer[PROJECT];
			if (currentViz) {
				await loadVisualizer(currentYellowBrickCategory, currentViz);
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
		const lat = 29.5 + Math.random() * 0.6;
		const lon = -95.8 + Math.random() * 0.8;
		updateFormField(PROJECT, 'lat', Number(lat.toFixed(3)));
		updateFormField(PROJECT, 'lon', Number(lon.toFixed(3)));
		toast.success('Random coordinates generated');
	}

	async function predict() {
		updateProjectStore(batchPredictionLoading, PROJECT, true);
		try {
			const form = $formData[PROJECT];
			const runId = $selectedBatchRun[PROJECT] || undefined;
			const result = await batchApi.predict(PROJECT, form, runId);
			if (result.error) {
				toast.error(`Prediction failed: ${result.error}`);
			} else if (result.data) {
				updateProjectStore(batchPredictionResults, PROJECT, result.data);
				const cluster = result.data.cluster ?? result.data.cluster_id ?? result.data.predicted_cluster;
				if (cluster !== undefined) {
					toast.success(`Assigned to Cluster ${cluster}`);
				}
			}
		} finally {
			updateProjectStore(batchPredictionLoading, PROJECT, false);
		}
	}

	// Training functions
	async function startTraining() {
		updateProjectStore(batchTrainingLoading, PROJECT, true);
		updateProjectStore(batchTrainingStatus, PROJECT, 'Starting training...');
		updateProjectStore(batchTrainingProgress, PROJECT, 0);
		updateProjectStore(batchTrainingStage, PROJECT, 'initializing');

		const mode = $batchTrainingMode[PROJECT];
		const result = await batchApi.startTraining(PROJECT, {
			mode,
			percentage: mode === 'percentage' ? $batchTrainingDataPercentage[PROJECT] : undefined,
			maxRows: mode === 'max_rows' ? $batchTrainingMaxRows[PROJECT] : undefined
		});

		if (result.error) {
			toast.error(result.error);
			updateProjectStore(batchTrainingLoading, PROJECT, false);
			updateProjectStore(batchTrainingStatus, PROJECT, '');
			updateProjectStore(batchTrainingStage, PROJECT, '');
		} else {
			toast.success('Training started');
			startStatusPolling();
		}
	}

	async function stopTraining() {
		stopStatusPolling();
		const result = await batchApi.stopTraining(PROJECT);

		updateProjectStore(batchTrainingLoading, PROJECT, false);
		updateProjectStore(batchTrainingStatus, PROJECT, '');
		updateProjectStore(batchTrainingProgress, PROJECT, 0);
		updateProjectStore(batchTrainingStage, PROJECT, '');
		updateProjectStore(batchTrainingCatboostLog, PROJECT, {});

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
			if (result.data.status_message) {
				updateProjectStore(batchTrainingStatus, PROJECT, result.data.status_message);
			}
			if (result.data.progress_percent !== undefined) {
				updateProjectStore(batchTrainingProgress, PROJECT, result.data.progress_percent);
			}
			if (result.data.current_stage) {
				updateProjectStore(batchTrainingStage, PROJECT, result.data.current_stage);
			}

			if (result.data.status === 'completed') {
				stopStatusPolling();
				updateProjectStore(batchTrainingLoading, PROJECT, false);
				updateProjectStore(batchTrainingStatus, PROJECT, 'Training complete!');
				updateProjectStore(batchTrainingProgress, PROJECT, 100);
				updateProjectStore(batchTrainingStage, PROJECT, 'complete');
				toast.success('Batch ML training complete');

				await loadRunsAfterTraining();
				await initBatchPage();
			} else if (result.data.status === 'failed') {
				const errorMsg = result.data.error || 'Training failed';
				stopStatusPolling();
				updateProjectStore(batchTrainingLoading, PROJECT, false);
				updateProjectStore(batchTrainingStatus, PROJECT, `Failed: ${errorMsg}`);
				updateProjectStore(batchTrainingStage, PROJECT, 'error');
				toast.error(errorMsg);
			} else if (result.data.status !== 'running') {
				stopStatusPolling();
				updateProjectStore(batchTrainingLoading, PROJECT, false);
				updateProjectStore(batchTrainingStatus, PROJECT, '');

				await loadRuns();
				await loadMetrics();
			}
		}
	}

	function formatStartTime(timestamp: string | number): string {
		if (!timestamp) return '';
		const ts = typeof timestamp === 'number' ? timestamp : Date.parse(timestamp);
		if (!Number.isNaN(ts)) {
			return new Date(ts).toISOString().replace('T', ' ').slice(0, 19);
		}
		return String(timestamp);
	}

	// Derived values
	const currentForm = $derived($formData[PROJECT] || {});
	const currentPrediction = $derived($batchPredictionResults[PROJECT] || {});
	const currentMetrics = $derived($batchMlflowMetrics[PROJECT] || {});
	const runs = $derived($batchMlflowRuns[PROJECT] || []);
	const selectedRunId = $derived($selectedBatchRun[PROJECT] || '');
	const selectedRun = $derived(runs.find((r) => r.run_id === selectedRunId));
	const currentVisualizer = $derived($selectedYellowBrickVisualizer[PROJECT] || '');
	const currentImage = $derived($yellowBrickImages[PROJECT]?.[currentVisualizer] || '');
	const isImageLoading = $derived($yellowBrickLoading[PROJECT]?.[currentVisualizer] || false);
	const isTraining = $derived($batchTrainingLoading[PROJECT]);
	const modelAvailable = $derived($batchModelAvailable[PROJECT]);
	const isLoading = $derived($batchPredictionLoading[PROJECT]);
	const experimentUrl = $derived($batchMlflowExperimentUrl[PROJECT]);
	const lastTrainedRunId = $derived($batchLastTrainedRunId[PROJECT]);

	// Prediction display
	const predictedCluster = $derived(
		currentPrediction?.cluster ?? currentPrediction?.cluster_id ?? currentPrediction?.predicted_cluster ?? 0
	);
	const hasPrediction = $derived(
		currentPrediction && Object.keys(currentPrediction).length > 0
	);

	// Customer coordinates
	const customerLat = $derived(Number(currentForm?.lat) || 29.8);
	const customerLon = $derived(Number(currentForm?.lon) || -95.4);

	// Plotly data for cluster prediction indicator
	const clusterPredictionData = $derived.by(() => [
		{
			type: 'indicator',
			mode: 'number',
			value: predictedCluster,
			title: { text: '<b>Cluster ID</b>', font: { size: 18 } },
			number: { font: { size: 72, color: '#22c55e' } },
			domain: { x: [0, 1], y: [0, 1] }
		}
	]);

	const clusterPredictionLayout = {
		height: 200,
		margin: { l: 20, r: 20, t: 40, b: 20 },
		paper_bgcolor: 'transparent',
		plot_bgcolor: 'transparent'
	};

	const plotlyConfig = { displayModeBar: false, responsive: true };

	// Map state
	let mapContainer: HTMLDivElement | null = null;
	let leafletMap: any = null;
	let locationMarker: any = null;
	let L: any = null;

	async function initMap() {
		if (!browser || !mapContainer) return;

		if (!document.getElementById('leaflet-css')) {
			const link = document.createElement('link');
			link.id = 'leaflet-css';
			link.rel = 'stylesheet';
			link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
			document.head.appendChild(link);
		}

		if (!(window as any).L) {
			await new Promise<void>((resolve, reject) => {
				const script = document.createElement('script');
				script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
				script.onload = () => resolve();
				script.onerror = reject;
				document.head.appendChild(script);
			});
		}

		await new Promise((r) => setTimeout(r, 100));

		L = (window as any).L;
		if (!L) return;

		leafletMap = L.map(mapContainer).setView([customerLat, customerLon], 12);

		L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
			attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
		}).addTo(leafletMap);

		updateMapMarker();
	}

	function updateMapMarker() {
		if (!L || !leafletMap) return;

		if (locationMarker) leafletMap.removeLayer(locationMarker);

		const greenIcon = L.divIcon({
			className: 'custom-marker',
			html: '<div style="background-color: #22c55e; width: 24px; height: 24px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>',
			iconSize: [24, 24],
			iconAnchor: [12, 12]
		});

		locationMarker = L.marker([customerLat, customerLon], { icon: greenIcon })
			.addTo(leafletMap)
			.bindPopup('Customer Location');

		leafletMap.setView([customerLat, customerLon], 12);
	}

	$effect(() => {
		const _lat = customerLat;
		const _lon = customerLon;

		if (leafletMap && L) {
			updateMapMarker();
		}
	});

	$effect(() => {
		const currentTab = activeTab;

		if (browser && currentTab === 'prediction') {
			setTimeout(() => {
				if (mapContainer) {
					if (leafletMap) {
						try {
							leafletMap.remove();
						} catch (e) {}
						leafletMap = null;
					}
					initMap();
				}
			}, 50);
		}
	});

	onDestroy(() => {
		if (leafletMap) {
			try {
				leafletMap.remove();
			} catch (e) {}
			leafletMap = null;
		}
	});
</script>

<!-- 40%/60% Layout -->
<div class="flex gap-4">
	<!-- Left Column - Training Box + Form (40%) -->
	<div class="w-2/5 min-w-0 space-y-4">
		<!-- Batch ML Training Card -->
		<Card>
			<CardContent class="space-y-3 pt-4">
				<div class="flex items-center justify-between">
					<div class="flex items-center gap-2">
						<Brain class="h-5 w-5 text-primary" />
						<h3 class="text-base font-bold">Batch ML Training</h3>
					</div>
					{#if experimentUrl}
						<a
							href={experimentUrl}
							target="_blank"
							rel="noopener noreferrer"
							class="inline-flex items-center gap-1 text-xs text-purple-600 hover:underline"
						>
							<FlaskConical class="h-3 w-3" />
							MLflow
							<ExternalLink class="h-3 w-3" />
						</a>
					{/if}
				</div>

				<!-- MLflow Run Selector -->
				<div class="flex items-center gap-2">
					<span class="text-xs text-muted-foreground">Run:</span>
					<select
						class="flex-1 rounded border border-input bg-background px-2 py-1 text-xs"
						value={selectedRunId}
						onchange={(e) => onRunChange(e.currentTarget.value)}
					>
						<option value="">Select MLflow run...</option>
						{#each runs as run}
							<option value={run.run_id}>{run.is_best ? '★ ' : ''}{run.run_id}</option>
						{/each}
					</select>
					<button
						type="button"
						class="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
						onclick={loadRuns}
						title="Refresh runs"
					>
						<RefreshCw class="h-3.5 w-3.5" />
					</button>
				</div>

				<!-- Model info badge -->
				<div class="flex items-center gap-2 rounded-md bg-blue-50 px-2 py-1 dark:bg-blue-950">
					<span class="text-xs font-medium text-blue-700 dark:text-blue-300">{MODEL_NAME}</span>
				</div>

				<!-- Training controls -->
				<div class="flex items-center gap-2">
					{#if !isTraining}
						<Button class="flex-1" size="sm" onclick={startTraining}>
							<Brain class="mr-2 h-4 w-4" />
							Train Model
						</Button>
					{:else}
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
					{/if}
				</div>

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
							<span class="ml-auto font-medium text-blue-600"
								>{$batchTrainingProgress[PROJECT]}%</span
							>
						</div>
					</div>
				{/if}

				{#if lastTrainedRunId}
					<div class="flex items-center gap-1 pt-1">
						<CheckCircle class="h-3 w-3 text-green-600" />
						<span class="text-[10px] text-muted-foreground">Last trained:</span>
						<code
							class="rounded bg-green-100 px-1 py-0.5 text-[10px] text-green-700 dark:bg-green-900 dark:text-green-300"
						>
							{lastTrainedRunId}
						</code>
					</div>
				{/if}

				{#if !isTraining}
					<div class="flex items-center gap-2">
						<Database class="h-3.5 w-3.5 text-muted-foreground" />
						<span class="text-xs text-muted-foreground">Data</span>
						<select
							class="rounded border border-input bg-background px-2 py-1 text-xs"
							value={$batchTrainingMode[PROJECT]}
							onchange={(e) =>
								updateProjectStore(
									batchTrainingMode,
									PROJECT,
									e.currentTarget.value as 'percentage' | 'max_rows'
								)}
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
								oninput={(e) =>
									updateProjectStore(
										batchTrainingDataPercentage,
										PROJECT,
										parseInt(e.currentTarget.value) || 10
									)}
								class="w-14 rounded border border-input bg-background px-2 py-1 text-xs"
							/>
							<span class="text-xs text-muted-foreground">%</span>
						{:else}
							<input
								type="number"
								min="1000"
								max={$batchDeltaLakeTotalRows[PROJECT] || 10000000}
								value={$batchTrainingMaxRows[PROJECT]}
								oninput={(e) =>
									updateProjectStore(
										batchTrainingMaxRows,
										PROJECT,
										parseInt(e.currentTarget.value) || 10000
									)}
								class="w-20 rounded border border-input bg-background px-2 py-1 text-xs"
							/>
							<span class="text-xs text-muted-foreground">rows</span>
						{/if}
					</div>
				{/if}
			</CardContent>
		</Card>

		<!-- Customer Interaction Form -->
		<Card>
			<CardContent class="space-y-3 pt-4">
				<div class="flex items-center gap-2">
					<ShoppingCart class="h-5 w-5 text-primary" />
					<h3 class="text-base font-bold">Customer Interaction</h3>
				</div>

				<div class="flex gap-2">
					<Button
						class="flex-1"
						onclick={predict}
						loading={isLoading}
						disabled={!modelAvailable || !selectedRunId}
					>
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
							value={(currentForm.browser as string) || ''}
							options={dropdownOptions.browser || ['Chrome']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'browser', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Device</p>
						<Select
							value={(currentForm.device_type as string) || ''}
							options={dropdownOptions.device_type || ['Desktop']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'device_type', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">OS</p>
						<Select
							value={(currentForm.os as string) || ''}
							options={dropdownOptions.os || ['Windows']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'os', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Event Type</p>
						<Select
							value={(currentForm.event_type as string) || ''}
							options={dropdownOptions.event_type || ['page_view']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'event_type', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Category</p>
						<Select
							value={(currentForm.product_category as string) || ''}
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

				<div class="space-y-1 text-xs text-muted-foreground">
					<p>Customer ID: {currentForm.customer_id || '-'}</p>
					<p>Event ID: {currentForm.event_id || '-'}</p>
					<p>Session ID: {currentForm.session_id || '-'}</p>
				</div>
			</CardContent>
		</Card>
	</div>

	<!-- Right Column - Tabs (60%) -->
	<div class="w-3/5 min-w-0">
		<Tabs value={activeTab} onValueChange={onOuterTabChange}>
			<TabsList class="w-full">
				<TabsTrigger value="prediction" class="flex-1">
					<Target class="mr-2 h-4 w-4" />
					Prediction
				</TabsTrigger>
				<TabsTrigger value="metrics" class="flex-1">
					<BarChart3 class="mr-2 h-4 w-4" />
					Metrics
				</TabsTrigger>
			</TabsList>

			<!-- MLflow Badge -->
			<div class="mt-3 flex flex-wrap items-center gap-2">
				<span
					class="inline-flex items-center gap-1 rounded-md bg-purple-500/10 px-2 py-1 text-xs font-medium text-purple-600"
				>
					<FlaskConical class="h-3 w-3" />
					MLflow
				</span>
				<span
					class="inline-flex items-center gap-1 rounded-md bg-blue-500/10 px-2 py-1 text-xs font-medium text-blue-600"
				>
					{MODEL_NAME}
				</span>
				{#if selectedRunId}
					<span class="text-xs text-muted-foreground">
						Run:
						<code class="rounded bg-muted px-1" title={selectedRunId}>
							{selectedRunId.slice(0, 8)}
						</code>
					</span>
					{#if selectedRun?.start_time}
						<span class="text-xs text-muted-foreground">
							Started: {formatStartTime(selectedRun.start_time)}
						</span>
					{/if}
				{:else}
					<span class="text-xs italic text-muted-foreground">
						No run selected - select a run to enable predictions
					</span>
				{/if}
			</div>

			<!-- Prediction Tab -->
			<TabsContent value="prediction" class="mt-4">
				<div class="space-y-4">
					<!-- Customer Location Map -->
					<Card>
						<CardContent class="p-4">
							<div class="mb-2 flex items-center gap-2">
								<MapPin class="h-4 w-4 text-primary" />
								<span class="text-sm font-bold">Customer Location</span>
							</div>
							<div
								bind:this={mapContainer}
								class="h-[200px] rounded-md overflow-hidden"
							></div>
							<div class="mt-2 text-xs text-muted-foreground">
								Location: ({customerLat.toFixed(4)}, {customerLon.toFixed(4)})
							</div>
						</CardContent>
					</Card>

					<!-- Predicted Cluster -->
					<Card>
						<CardContent class="p-4">
							<div class="mb-2 flex items-center gap-2">
								<Target class="h-4 w-4 text-primary" />
								<span class="text-sm font-bold">Predicted Cluster</span>
							</div>
							{#if hasPrediction}
								<Plotly
									data={clusterPredictionData}
									layout={clusterPredictionLayout}
									config={plotlyConfig}
								/>
							{:else}
								<div
									class="flex h-[200px] items-center justify-center rounded-lg border border-blue-200 bg-blue-50 p-4 text-center text-sm text-blue-600 dark:border-blue-800 dark:bg-blue-900/20"
								>
									Click <strong>Predict</strong> to identify the customer segment.
								</div>
							{/if}
						</CardContent>
					</Card>

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
			<TabsContent value="metrics" class="mt-4">
				<div class="space-y-4">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<LayoutDashboard class="h-4 w-4 text-primary" />
							<h3 class="text-sm font-bold">Clustering Metrics</h3>
						</div>
						<Button variant="ghost" size="sm" onclick={loadMetrics} loading={metricsLoading}>
							<RefreshCw class="h-4 w-4" />
						</Button>
					</div>

					<Tabs value={activeMetricsTab} onValueChange={onMetricsTabChange} class="w-full">
						<TabsList class="grid w-full grid-cols-6">
							<TabsTrigger value="overview" class="px-1 text-[11px]">
								<LayoutDashboard class="mr-1 h-3 w-3" />
								Overview
							</TabsTrigger>
							<TabsTrigger value="clustering" class="px-1 text-[11px]">
								<Boxes class="mr-1 h-3 w-3" />
								Clustering
							</TabsTrigger>
							<TabsTrigger value="features" class="px-1 text-[11px]">
								<ScatterChart class="mr-1 h-3 w-3" />
								Features
							</TabsTrigger>
							<TabsTrigger value="target" class="px-1 text-[11px]">
								<Crosshair class="mr-1 h-3 w-3" />
								Target
							</TabsTrigger>
							<TabsTrigger value="diagnostics" class="px-1 text-[11px]">
								<Settings2 class="mr-1 h-3 w-3" />
								Diagnostics
							</TabsTrigger>
							<TabsTrigger value="text" class="px-1 text-[11px]">
								<FileText class="mr-1 h-3 w-3" />
								Text
							</TabsTrigger>
						</TabsList>

						<!-- Overview Tab -->
						<TabsContent value="overview" class="pt-4">
							<div class="space-y-4">
								{#if Object.keys(currentMetrics).length > 0}
									{#if currentMetrics.n_samples > 0}
										<div
											class="flex items-center gap-2 rounded-md border border-blue-200 bg-blue-50 p-3 dark:border-blue-900 dark:bg-blue-950"
										>
											<Database class="h-4 w-4 text-blue-600" />
											<span class="text-sm font-medium text-blue-700 dark:text-blue-300"
												>Training Data:</span
											>
											<span
												class="rounded bg-blue-600 px-2 py-0.5 text-sm font-bold text-white"
											>
												{(currentMetrics.n_samples as number)?.toLocaleString()} rows
											</span>
										</div>
									{/if}

									<!-- Primary Clustering Metrics -->
									<div class="flex items-center gap-2">
										<Target class="h-4 w-4 text-blue-600" />
										<span class="text-sm font-bold">Primary Metrics</span>
									</div>
									<div class="grid grid-cols-4 gap-2">
										<MetricCard
											name="Silhouette"
											value={currentMetrics.silhouette_score as number}
											onInfoClick={() => openMetricInfo('silhouette_score')}
										/>
										<MetricCard
											name="Calinski-Harabasz"
											value={currentMetrics.calinski_harabasz_score as number}
											decimals={0}
											onInfoClick={() => openMetricInfo('calinski_harabasz_score')}
										/>
										<MetricCard
											name="Davies-Bouldin"
											value={currentMetrics.davies_bouldin_score as number}
											onInfoClick={() => openMetricInfo('davies_bouldin_score')}
										/>
										<MetricCard
											name="N Clusters"
											value={currentMetrics.n_clusters as number}
											decimals={0}
											onInfoClick={() => openMetricInfo('n_clusters')}
										/>
									</div>

									<hr class="border-border" />

									<!-- Secondary Metrics -->
									<div class="flex items-center gap-2">
										<BarChart3 class="h-4 w-4 text-indigo-600" />
										<span class="text-sm font-bold">Secondary Metrics</span>
									</div>
									<div class="grid grid-cols-2 gap-2">
										<MetricCard
											name="Inertia (WCSS)"
											value={currentMetrics.inertia as number}
											decimals={0}
											onInfoClick={() => openMetricInfo('inertia')}
										/>
										<MetricCard
											name="Preprocessing Time"
											value={currentMetrics.preprocessing_time_seconds as number}
											decimals={2}
										/>
									</div>
								{:else}
									<div class="flex flex-col items-center justify-center py-12 text-center">
										<LayoutDashboard class="h-12 w-12 text-muted-foreground/50" />
										<p class="mt-4 text-sm text-muted-foreground">
											{runs.length === 0
												? 'Train a model first to see metrics'
												: 'Loading metrics...'}
										</p>
									</div>
								{/if}
							</div>
						</TabsContent>

						<!-- Clustering Tab -->
						<TabsContent value="clustering" class="pt-4">
							<div class="space-y-4">
								{#if modelAvailable}
									<p class="text-xs text-muted-foreground">
										Select a clustering visualization.
									</p>
									<div class="flex items-center gap-2">
										<select
											class="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
											value={currentVisualizer}
											onchange={(e) => loadVisualizer('Clustering', e.currentTarget.value)}
										>
											{#each YELLOWBRICK_CATEGORIES['Clustering'] as viz}
												<option value={viz.value}>{viz.label}</option>
											{/each}
										</select>
										<button
											type="button"
											class="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
											onclick={() => openYellowBrickInfo(currentVisualizer)}
											title="Learn about this visualization"
											disabled={!currentVisualizer}
										>
											<Info class="h-4 w-4" />
										</button>
									</div>
									{@render visualizationDisplay()}
								{:else}
									{@render noModelWarning()}
								{/if}
							</div>
						</TabsContent>

						<!-- Features Tab -->
						<TabsContent value="features" class="pt-4">
							<div class="space-y-4">
								{#if modelAvailable}
									<p class="text-xs text-muted-foreground">
										Select a feature analysis visualization.
									</p>
									<div class="flex items-center gap-2">
										<select
											class="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
											value={currentVisualizer}
											onchange={(e) => loadVisualizer('Feature Analysis', e.currentTarget.value)}
										>
											{#each YELLOWBRICK_CATEGORIES['Feature Analysis'] as viz}
												<option value={viz.value}>{viz.label}</option>
											{/each}
										</select>
										<button
											type="button"
											class="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
											onclick={() => openYellowBrickInfo(currentVisualizer)}
											title="Learn about this visualization"
											disabled={!currentVisualizer}
										>
											<Info class="h-4 w-4" />
										</button>
									</div>
									{@render visualizationDisplay()}
								{:else}
									{@render noModelWarning()}
								{/if}
							</div>
						</TabsContent>

						<!-- Target Tab -->
						<TabsContent value="target" class="pt-4">
							<div class="space-y-4">
								{#if modelAvailable}
									<p class="text-xs text-muted-foreground">
										Select a target analysis visualization.
									</p>
									<div class="flex items-center gap-2">
										<select
											class="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
											value={currentVisualizer}
											onchange={(e) => loadVisualizer('Target', e.currentTarget.value)}
										>
											{#each YELLOWBRICK_CATEGORIES['Target'] as viz}
												<option value={viz.value}>{viz.label}</option>
											{/each}
										</select>
										<button
											type="button"
											class="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
											onclick={() => openYellowBrickInfo(currentVisualizer)}
											title="Learn about this visualization"
											disabled={!currentVisualizer}
										>
											<Info class="h-4 w-4" />
										</button>
									</div>
									{@render visualizationDisplay()}
								{:else}
									{@render noModelWarning()}
								{/if}
							</div>
						</TabsContent>

						<!-- Diagnostics Tab -->
						<TabsContent value="diagnostics" class="pt-4">
							<div class="space-y-4">
								{#if modelAvailable}
									<p class="text-xs text-muted-foreground">
										Select a model diagnostics visualization.
									</p>
									<div class="flex items-center gap-2">
										<select
											class="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
											value={currentVisualizer}
											onchange={(e) => loadVisualizer('Model Selection', e.currentTarget.value)}
										>
											{#each YELLOWBRICK_CATEGORIES['Model Selection'] as viz}
												<option value={viz.value}>{viz.label}</option>
											{/each}
										</select>
										<button
											type="button"
											class="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
											onclick={() => openYellowBrickInfo(currentVisualizer)}
											title="Learn about this visualization"
											disabled={!currentVisualizer}
										>
											<Info class="h-4 w-4" />
										</button>
									</div>
									{@render visualizationDisplay()}
								{:else}
									{@render noModelWarning()}
								{/if}
							</div>
						</TabsContent>

						<!-- Text Analysis Tab -->
						<TabsContent value="text" class="pt-4">
							<div class="space-y-4">
								{#if modelAvailable}
									<p class="text-xs text-muted-foreground">
										Select a text analysis visualization (for search queries).
									</p>
									<div class="flex items-center gap-2">
										<select
											class="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
											value={currentVisualizer}
											onchange={(e) => loadVisualizer('Text Analysis', e.currentTarget.value)}
										>
											{#each YELLOWBRICK_CATEGORIES['Text Analysis'] as viz}
												<option value={viz.value}>{viz.label}</option>
											{/each}
										</select>
										<button
											type="button"
											class="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
											onclick={() => openYellowBrickInfo(currentVisualizer)}
											title="Learn about this visualization"
											disabled={!currentVisualizer}
										>
											<Info class="h-4 w-4" />
										</button>
									</div>
									{@render visualizationDisplay()}
								{:else}
									{@render noModelWarning()}
								{/if}
							</div>
						</TabsContent>
					</Tabs>
				</div>
			</TabsContent>
		</Tabs>
	</div>
</div>

{#snippet visualizationDisplay()}
	<div
		class="flex min-h-[400px] items-center justify-center rounded-lg border border-border bg-muted/30"
	>
		{#if isImageLoading}
			<div class="flex flex-col items-center justify-center gap-3 p-8">
				<div class="flex items-center gap-2">
					<div class="h-6 w-6 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
					<span class="text-sm text-muted-foreground">Loading visualization...</span>
				</div>
				<button
					type="button"
					class="inline-flex items-center gap-1 rounded-md bg-red-100 px-2 py-1 text-xs font-medium text-red-700 hover:bg-red-200 dark:bg-red-900 dark:text-red-300 dark:hover:bg-red-800"
					onclick={cancelYellowBrickLoading}
				>
					<Square class="h-3 w-3" />
					Stop
				</button>
			</div>
		{:else if currentImage}
			<img
				src="data:image/png;base64,{currentImage}"
				alt={currentVisualizer}
				class="max-h-[500px] max-w-full"
			/>
		{:else}
			<p class="text-sm text-muted-foreground">Select a visualizer</p>
		{/if}
	</div>
{/snippet}

{#snippet noModelWarning()}
	<div
		class="flex items-center gap-2 rounded-lg border border-blue-200 bg-blue-50 p-4 text-sm text-blue-700 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-300"
	>
		<Info class="h-4 w-4" />
		<span>Train a model first to view visualizations.</span>
	</div>
{/snippet}
