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
	import { randomizeETAForm, FIELD_CONFIG, clampFieldValue } from '$lib/utils/randomize';
	import {
		RefreshCw,
		LayoutDashboard,
		TrendingUp,
		ScatterChart,
		Crosshair,
		Settings2,
		Lightbulb,
		Info,
		GitBranch,
		Database,
		Play,
		Square,
		Loader2,
		CheckCircle,
		Brain,
		BarChart3,
		ExternalLink,
		Clock,
		Target,
		MapPin,
		Shuffle,
		AlertTriangle,
		FlaskConical,
		Settings,
		Star
	} from 'lucide-svelte';
	import type { ProjectName } from '$types';
	import { cn } from '$lib/utils';

	const PROJECT: ProjectName = 'Estimated Time of Arrival';
	const MODEL_NAME = 'CatBoost Regressor';

	// Transform MLflow metrics from API format to simple format (matching sklearn regression metrics)
	// MLflow stores metrics with full sklearn names like mean_absolute_error, not abbreviated ones
	function transformMetrics(rawMetrics: Record<string, unknown>): Record<string, number> {
		const getMetric = (name: string): number => {
			const key = `metrics.${name}`;
			const val = rawMetrics[key];
			return val !== undefined && val !== null ? Number(val) : 0;
		};

		return {
			// Primary Error Metrics (Lower is Better)
			mae: getMetric('mean_absolute_error') || getMetric('mae') || getMetric('MAE'),
			rmse: getMetric('root_mean_squared_error') || getMetric('rmse') || getMetric('RMSE'),
			mape: getMetric('mean_absolute_percentage_error') || getMetric('mape') || getMetric('MAPE'),
			smape: getMetric('symmetric_mean_absolute_percentage_error') || getMetric('smape') || getMetric('SMAPE'),
			// Goodness of Fit Metrics (Higher is Better)
			r2_score: getMetric('r2_score') || getMetric('R2'),
			explained_variance: getMetric('explained_variance_score') || getMetric('explained_variance'),
			// Secondary Error Metrics
			mse: getMetric('mean_squared_error') || getMetric('mse') || getMetric('MSE'),
			median_ae: getMetric('median_absolute_error') || getMetric('median_ae'),
			max_error: getMetric('max_error'),
			// D² Metrics (Deviance-based, Higher is Better)
			d2_absolute_error: getMetric('d2_absolute_error_score'),
			d2_pinball: getMetric('d2_pinball_score'),
			d2_tweedie: getMetric('d2_tweedie_score'),
			// Data info (check both metrics and params)
			n_samples: getMetric('n_samples') || getMetric('training_samples') ||
			           Number(rawMetrics['params.train_samples'] || 0) + Number(rawMetrics['params.test_samples'] || 0)
		};
	}

	// YellowBrick visualizer options organized by category (regression)
	const YELLOWBRICK_LABEL_SUFFIX = ' — YellowBrick';
	const SKLEARN_LABEL_SUFFIX = ' — Scikit-Learn';
	const SKLEARN_PREFIX = 'sklearn:';
	const withYellowBrickSuffix = (options: Array<{ value: string; label: string }>) =>
		options.map((option) =>
			option.value
				? { ...option, label: `${option.label}${YELLOWBRICK_LABEL_SUFFIX}` }
				: option
		);
	const YELLOWBRICK_CATEGORIES = {
		Regression: withYellowBrickSuffix([
			{ value: '', label: 'Select visualization...' },
			{ value: 'ResidualsPlot', label: 'Residuals Plot' },
			{ value: 'PredictionError', label: 'Prediction Error' }
		]).concat([
			{ value: `${SKLEARN_PREFIX}PredictionErrorDisplay`, label: `Prediction Error${SKLEARN_LABEL_SUFFIX}` }
		]),
		'Feature Analysis': withYellowBrickSuffix([
			{ value: '', label: 'Select visualization...' },
			{ value: 'Rank1D', label: 'Rank 1D' },
			{ value: 'Rank2D', label: 'Rank 2D' },
			{ value: 'PCA', label: 'PCA Decomposition' },
			{ value: 'Manifold', label: 'Manifold' },
			{ value: 'JointPlot', label: 'Joint Plot' }
		]).concat([
			{ value: `${SKLEARN_PREFIX}PartialDependenceDisplay`, label: `Partial Dependence${SKLEARN_LABEL_SUFFIX}` }
		]),
		Target: withYellowBrickSuffix([
			{ value: '', label: 'Select visualization...' },
			{ value: 'FeatureCorrelation', label: 'Feature Correlation (Mutual Info)' },
			{ value: 'FeatureCorrelation_Pearson', label: 'Feature Correlation (Pearson)' },
			{ value: 'BalancedBinningReference', label: 'Balanced Binning Reference' }
		]),
		'Model Selection': withYellowBrickSuffix([
			{ value: '', label: 'Select visualization...' },
			{ value: 'FeatureImportances', label: 'Feature Importances' },
			{ value: 'CVScores', label: 'Cross-Validation Scores' },
			{ value: 'ValidationCurve', label: 'Validation Curve' },
			{ value: 'LearningCurve', label: 'Learning Curve' },
			{ value: 'RFECV', label: 'Recursive Feature Elimination' },
			{ value: 'DroppingCurve', label: 'Dropping Curve' }
		]).concat([
			{ value: `${SKLEARN_PREFIX}LearningCurveDisplay`, label: `Learning Curve${SKLEARN_LABEL_SUFFIX}` },
			{ value: `${SKLEARN_PREFIX}ValidationCurveDisplay`, label: `Validation Curve${SKLEARN_LABEL_SUFFIX}` }
		])
	};

	let activeTab = $state('prediction');
	let activeMetricsTab = $state('overview');
	let metricsLoading = $state(false);
	let statusInterval: ReturnType<typeof setInterval> | null = null;
	let runsLoading = $state(false);
	let dropdownOptions = $state<Record<string, string[]>>({});
	let sampleLoading = $state(false);

	onMount(async () => {
		try {
			const response = await fetch('/data/dropdown_options_eta.json');
			dropdownOptions = await response.json();
		} catch (e) {
			console.error('Failed to load dropdown options:', e);
		}

		const existingForm = $formData[PROJECT] || {};
		if (!Object.keys(existingForm).length || !existingForm.driver_id) {
			updateProjectStore(formData, PROJECT, randomizeETAForm(dropdownOptions));
		}

		const countResult = await batchApi.getDeltaLakeRowCount(PROJECT);
		if (countResult.data?.row_count) {
			updateProjectStore(batchDeltaLakeTotalRows, PROJECT, countResult.data.row_count);
		}

		await initBatchPage();
	});

	onDestroy(() => {
		stopStatusPolling();
		yellowBrickCancelRequested = true;
		if (leafletMap) {
			try {
				leafletMap.remove();
			} catch (e) {}
			leafletMap = null;
		}
	});

	beforeNavigate(async ({ to }) => {
		yellowBrickCancelRequested = true;
		if (isTraining) {
			await batchApi.stopTraining(PROJECT);
			updateProjectStore(batchTrainingLoading, PROJECT, false);
			updateProjectStore(batchTrainingStatus, PROJECT, '');
			updateProjectStore(batchTrainingProgress, PROJECT, 0);
			stopStatusPolling();
			toast.info('Training stopped - navigated away from page');
		}
	});

	let yellowBrickCancelRequested = $state(false);
	let currentYellowBrickCategory = $state('Regression');

	function cancelYellowBrickLoading() {
		yellowBrickCancelRequested = true;
		updateProjectStore(yellowBrickLoading, PROJECT, {});
		updateProjectStore(selectedYellowBrickVisualizer, PROJECT, '');
		yellowBrickImages.update((imgs) => ({ ...imgs, [PROJECT]: {} }));
		toast.info('Visualization loading cancelled');
	}

	async function initBatchPage() {
		runsLoading = true;
		metricsLoading = true;

		try {
			const runId = $selectedBatchRun[PROJECT] || undefined;
			const result = await batchApi.initBatchPage(PROJECT, runId);
			console.log('[ETA Batch] initBatchPage response:', result);

			if (result.error) {
				console.error('[ETA Batch] initBatchPage error:', result.error);
				toast.error(`Failed to initialize: ${result.error}`);
			} else if (result.data) {
				const data = result.data;
				const runs = data.runs || [];
				console.log('[ETA Batch] Loaded data - runs:', runs.length, 'model_available:', data.model_available);

				updateProjectStore(batchMlflowRuns, PROJECT, runs);
				updateProjectStore(batchModelAvailable, PROJECT, data.model_available);

				if (data.experiment_url) {
					updateProjectStore(batchMlflowExperimentUrl, PROJECT, data.experiment_url);
				}

				if (data.metrics && !data.metrics._no_runs) {
					const transformed = transformMetrics(data.metrics);
					updateProjectStore(batchMlflowMetrics, PROJECT, transformed);
				}

				if (runs.length && !$selectedBatchRun[PROJECT]) {
					const bestRunId = data.best_run_id || runs[0].run_id;
					updateProjectStore(selectedBatchRun, PROJECT, bestRunId);
				}

				if (data.metrics?.run_url) {
					updateProjectStore(batchMlflowExperimentUrl, PROJECT, data.metrics.run_url as string);
				}
			}
		} catch (err) {
			console.error('[ETA Batch] initBatchPage exception:', err);
			toast.error('Failed to initialize batch page');
		} finally {
			runsLoading = false;
			metricsLoading = false;
		}
	}

	async function loadRuns() {
		runsLoading = true;
		try {
			const result = await batchApi.getMLflowRuns(PROJECT);
			console.log('[ETA Batch] loadRuns response:', result);

			if (result.error) {
				console.error('[ETA Batch] loadRuns error:', result.error);
				toast.error(`Failed to load runs: ${result.error}`);
			} else if (result.data) {
				const runs = result.data.runs || [];
				console.log('[ETA Batch] Loaded runs:', runs.length);
				updateProjectStore(batchMlflowRuns, PROJECT, runs);
				updateProjectStore(batchModelAvailable, PROJECT, runs.length > 0);

				if (runs.length === 0) {
					toast.info('No MLflow runs found. Train a model to create runs.');
				}
			}
		} catch (err) {
			console.error('[ETA Batch] loadRuns exception:', err);
			toast.error('Failed to load MLflow runs');
		} finally {
			runsLoading = false;
		}
	}

	async function loadRunsAfterTraining() {
		runsLoading = true;
		try {
			const result = await batchApi.getMLflowRuns(PROJECT);
			console.log('[ETA Batch] loadRunsAfterTraining response:', result);

			if (result.error) {
				console.error('[ETA Batch] loadRunsAfterTraining error:', result.error);
				toast.error(`Failed to load runs after training: ${result.error}`);
			} else if (result.data) {
				const runs = result.data.runs || [];
				console.log('[ETA Batch] Runs after training:', runs.length);
				updateProjectStore(batchMlflowRuns, PROJECT, runs);
				updateProjectStore(batchModelAvailable, PROJECT, runs.length > 0);

				if (runs.length > 0) {
					const newRunId = runs[0].run_id;
					console.log('[ETA Batch] Auto-selecting new run:', newRunId);
					updateProjectStore(batchLastTrainedRunId, PROJECT, newRunId);
					updateProjectStore(selectedBatchRun, PROJECT, newRunId);
					toast.success(`Loaded ${runs.length} MLflow runs`);
				}
			}
		} catch (err) {
			console.error('[ETA Batch] loadRunsAfterTraining exception:', err);
			toast.error('Failed to load MLflow runs after training');
		} finally {
			runsLoading = false;
		}
	}

	async function loadMetrics() {
		metricsLoading = true;
		const runId = $selectedBatchRun[PROJECT] || undefined;
		const result = await batchApi.getMLflowMetrics(PROJECT, runId);
		if (result.data && !result.data._no_runs) {
			const transformed = transformMetrics(result.data);
			updateProjectStore(batchMlflowMetrics, PROJECT, transformed);

			if (result.data.run_url) {
				updateProjectStore(batchMlflowExperimentUrl, PROJECT, result.data.run_url);
			}
		}
		metricsLoading = false;
	}

	async function loadVisualizer(category: string, visualizerName: string) {
		if (!visualizerName) {
			updateProjectStore(selectedYellowBrickVisualizer, PROJECT, '');
			return;
		}

		const isSklearnVisualizer = visualizerName.startsWith(SKLEARN_PREFIX);
		const metricName = isSklearnVisualizer
			? visualizerName.replace(SKLEARN_PREFIX, '')
			: visualizerName;
		const visualizerKey = visualizerName;

		yellowBrickCancelRequested = false;
		currentYellowBrickCategory = category;
		updateProjectStore(selectedYellowBrickVisualizer, PROJECT, visualizerKey);
		updateProjectStore(yellowBrickLoading, PROJECT, { [visualizerKey]: true });

		const runId = $selectedBatchRun[PROJECT] || undefined;
		const result = isSklearnVisualizer
			? await batchApi.getSklearnImage(PROJECT, category, metricName, runId)
			: await batchApi.getYellowBrickImage(PROJECT, category, metricName, runId);

		if (yellowBrickCancelRequested) return;

		if (result.data?.image_base64) {
			yellowBrickImages.update((imgs) => ({
				...imgs,
				[PROJECT]: {
					...imgs[PROJECT],
					[visualizerKey]: result.data!.image_base64
				}
			}));
		} else if (result.error || result.data?.error) {
			toast.error(`Failed to load ${metricName}: ${result.error || result.data?.error}`);
		}

		updateProjectStore(yellowBrickLoading, PROJECT, { [visualizerKey]: false });
	}

	function openMetricInfo(metricKey: string) {
		fetch('/data/metric_info_eta.json')
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

	let yellowBrickInfoOpen = $state(false);
	let yellowBrickInfoContent = $state<{
		name: string;
		category: string;
		description: string;
		interpretation: string;
		context: string;
		whenToUse: string;
		docsUrl: string;
	}>({
		name: '',
		category: '',
		description: '',
		interpretation: '',
		context: '',
		whenToUse: '',
		docsUrl: ''
	});

	function openYellowBrickInfo(visualizerKey: string) {
		fetch('/data/yellowbrick_info_eta.json')
			.then((r) => r.json())
			.then((data) => {
				const info = data.visualizers[visualizerKey];
				if (info) {
					yellowBrickInfoContent = {
						name: info.name,
						category: info.category,
						description: info.description,
						interpretation: info.interpretation,
						context: info.eta_context || '',
						whenToUse: info.when_to_use || '',
						docsUrl: info.docs_url || ''
					};
					yellowBrickInfoOpen = true;
				}
			});
	}

	async function onRunChange(runId: string) {
		updateProjectStore(selectedBatchRun, PROJECT, runId);
		await loadMetrics();
		const yellowBrickTabs = ['regression', 'features', 'target', 'diagnostics'];
		if (yellowBrickTabs.includes(activeMetricsTab)) {
			const currentViz = $selectedYellowBrickVisualizer[PROJECT];
			if (currentViz) {
				await loadVisualizer(currentYellowBrickCategory, currentViz);
			}
		}
	}

	// Handle outer tab changes (Prediction/Metrics)
	function onOuterTabChange(newTab: string) {
		activeTab = newTab;
	}

	// Reset visualization when switching to a YellowBrick tab
	function onMetricsTabChange(newTab: string) {
		activeMetricsTab = newTab;
		const yellowBrickTabs = ['regression', 'features', 'target', 'diagnostics'];
		if (yellowBrickTabs.includes(newTab)) {
			// Reset to "Select visualization..." when switching tabs
			updateProjectStore(selectedYellowBrickVisualizer, PROJECT, '');
			yellowBrickImages.update((imgs) => ({
				...imgs,
				[PROJECT]: {}
			}));
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

	function updateEstimates() {
		const form = $formData[PROJECT];
		if (form.origin_lat && form.origin_lon && form.destination_lat && form.destination_lon) {
			const dist = calcDistanceKm(
				Number(form.origin_lat),
				Number(form.origin_lon),
				Number(form.destination_lat),
				Number(form.destination_lon)
			);
			const distKm = Number(dist.toFixed(2));
			// Assume average speed of 40 km/h for initial estimate
			const travelTime = Math.round((dist / 40) * 3600);
			updateFormField(PROJECT, 'estimated_distance_km', distKm);
			updateFormField(PROJECT, 'initial_estimated_travel_time_seconds', travelTime);
		}
	}

	function updateDayOfWeek(dateStr: string) {
		if (dateStr) {
			const date = new Date(dateStr);
			const dayOfWeek = date.getDay(); // 0 = Sunday, 6 = Saturday
			updateFormField(PROJECT, 'day_of_week', dayOfWeek);
		}
	}

	function generateRandomCoordinates() {
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
		updateEstimates();
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
				// API returns estimated_travel_time_seconds for ETA predictions
				const eta = result.data.estimated_travel_time_seconds ||
				            result.data['Estimated Time of Arrival'] ||
				            result.data.eta_seconds || 0;
				if (eta > 0) {
					toast.success(`Predicted ETA: ${(eta / 60).toFixed(1)} minutes`);
				} else {
					toast.warning('Prediction returned zero - check input data');
				}
			}
		} finally {
			updateProjectStore(batchPredictionLoading, PROJECT, false);
		}
	}

	async function startTraining() {
		updateProjectStore(batchTrainingLoading, PROJECT, true);
		updateProjectStore(batchTrainingStatus, PROJECT, 'Starting training...');
		updateProjectStore(batchTrainingProgress, PROJECT, 0);
		updateProjectStore(batchTrainingStage, PROJECT, 'init');

		const mode = $batchTrainingMode[PROJECT];
		const maxRows = $batchTrainingMaxRows[PROJECT];
		const percentage = $batchTrainingDataPercentage[PROJECT];

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
		// Stop polling immediately to prevent race conditions
		stopStatusPolling();

		const result = await batchApi.stopTraining(PROJECT);

		// Reset all training state
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

			// Update CatBoost training log (iteration, total, remaining time)
			if (result.data.catboost_log) {
				updateProjectStore(batchTrainingCatboostLog, PROJECT, result.data.catboost_log);
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

	const currentMetrics = $derived($batchMlflowMetrics[PROJECT] || {});
	const runs = $derived($batchMlflowRuns[PROJECT] || []);
	const selectedRunId = $derived($selectedBatchRun[PROJECT] || '');
	const selectedRun = $derived(runs.find((r) => r.run_id === selectedRunId));
	const currentVisualizer = $derived($selectedYellowBrickVisualizer[PROJECT] || '');
	const currentImage = $derived($yellowBrickImages[PROJECT]?.[currentVisualizer] || '');
	const isImageLoading = $derived($yellowBrickLoading[PROJECT]?.[currentVisualizer] || false);
	const isTraining = $derived($batchTrainingLoading[PROJECT]);
	const modelAvailable = $derived($batchModelAvailable[PROJECT]);
	const experimentUrl = $derived($batchMlflowExperimentUrl[PROJECT] || '');
	const lastTrainedRunId = $derived($batchLastTrainedRunId[PROJECT] || '');
	const currentForm = $derived($formData[PROJECT] || {});
	const currentPrediction = $derived($batchPredictionResults[PROJECT]);
	const isLoading = $derived($batchPredictionLoading[PROJECT]);

	const mlflowUrl = $derived.by(() => {
		if (!experimentUrl) return '';
		if (!selectedRunId) return experimentUrl;
		if (experimentUrl.includes(`/runs/${selectedRunId}`)) return experimentUrl;
		const baseUrl = experimentUrl.replace(/\/runs\/[^/]+$/, '');
		return `${baseUrl}/runs/${selectedRunId}`;
	});

	function formatStartTime(startTime: string | null): string {
		if (!startTime) return '';
		const timestamp = typeof startTime === 'number' ? startTime : Date.parse(startTime);
		if (!Number.isNaN(timestamp)) {
			return new Date(timestamp).toISOString().replace('T', ' ').slice(0, 19);
		}
		return String(startTime);
	}

	function calcDistanceKm(oLat: number, oLon: number, dLat: number, dLon: number): number {
		const R = 6371;
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

	const estimatedDistanceKm = $derived.by(() => {
		const form = $formData[PROJECT];
		if (form?.estimated_distance_km) return Number(form.estimated_distance_km);
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
		if (form?.initial_estimated_travel_time_seconds)
			return Number(form.initial_estimated_travel_time_seconds);
		if (estimatedDistanceKm > 0) return Math.round((estimatedDistanceKm / 40) * 3600);
		return 0;
	});

	const etaSeconds = $derived(
		currentPrediction?.estimated_travel_time_seconds ||
		currentPrediction?.['Estimated Time of Arrival'] ||
		currentPrediction?.eta_seconds || 0
	);
	const etaMinutes = $derived(etaSeconds > 0 ? etaSeconds / 60 : 0);

	const etaPredictionData = $derived.by(() => [
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
	]);

	const etaPredictionLayout = {
		grid: { rows: 2, columns: 1, pattern: 'independent' },
		height: 250,
		margin: { l: 20, r: 20, t: 40, b: 20 },
		paper_bgcolor: 'transparent',
		plot_bgcolor: 'transparent'
	};

	const plotlyConfig = { displayModeBar: false, responsive: true };

	let mapContainer: HTMLDivElement | null = null;
	let leafletMap: any = null;
	let originMarker: any = null;
	let destMarker: any = null;
	let routeLine: any = null;
	let L: any = null;

	const originLat = $derived(Number(currentForm?.origin_lat) || 29.8);
	const originLon = $derived(Number(currentForm?.origin_lon) || -95.4);
	const destLat = $derived(Number(currentForm?.destination_lat) || 29.8);
	const destLon = $derived(Number(currentForm?.destination_lon) || -95.4);

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
		const centerLat = (originLat + destLat) / 2;
		const centerLon = (originLon + destLon) / 2;
		leafletMap = L.map(mapContainer).setView([centerLat, centerLon], 10);
		L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
			attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
		}).addTo(leafletMap);
		updateMapMarkers();
	}

	function updateMapMarkers() {
		if (!L || !leafletMap) return;
		if (originMarker) leafletMap.removeLayer(originMarker);
		if (destMarker) leafletMap.removeLayer(destMarker);
		if (routeLine) leafletMap.removeLayer(routeLine);
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
		originMarker = L.marker([originLat, originLon], { icon: blueIcon })
			.addTo(leafletMap)
			.bindPopup('Origin');
		destMarker = L.marker([destLat, destLon], { icon: redIcon })
			.addTo(leafletMap)
			.bindPopup('Destination');
		routeLine = L.polyline(
			[
				[originLat, originLon],
				[destLat, destLon]
			],
			{ color: '#333333', weight: 4, opacity: 0.8 }
		).addTo(leafletMap);
		const bounds = L.latLngBounds([
			[originLat, originLon],
			[destLat, destLon]
		]);
		leafletMap.fitBounds(bounds, { padding: [30, 30] });
	}

	$effect(() => {
		const _oLat = originLat;
		const _oLon = originLon;
		const _dLat = destLat;
		const _dLon = destLon;
		if (leafletMap && L) updateMapMarkers();
	});

	// Fix map vanishing - reinitialize when switching to prediction tab
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
</script>

<!-- 40%/60% Layout -->
<div class="flex gap-4">
	<!-- Left Column - Training Box + Form (40%) - Always Visible -->
	<div class="w-2/5 min-w-0 space-y-4">
		<!-- Batch ML Training Box -->
		<Card>
			<CardContent class="space-y-3 p-3">
				<!-- MLflow Run Header with Model Badge -->
				<div class="flex items-center gap-2">
					<GitBranch class="h-4 w-4 text-blue-600" />
					<span class="text-xs font-medium">MLflow Run</span>
					<span
						class="rounded bg-blue-100 px-1.5 py-0.5 text-[10px] font-medium text-blue-700 dark:bg-blue-900 dark:text-blue-300"
					>
						{MODEL_NAME}
					</span>
					{#if runsLoading}
						<Loader2 class="h-3 w-3 animate-spin text-muted-foreground" />
					{/if}
					<div class="flex-1"></div>
					{#if mlflowUrl}
						<a
							href={mlflowUrl}
							target="_blank"
							rel="noopener noreferrer"
							class="inline-flex items-center gap-1 rounded-md bg-cyan-100 px-2 py-1 text-xs font-medium text-cyan-700 hover:bg-cyan-200 dark:bg-cyan-900 dark:text-cyan-300 dark:hover:bg-cyan-800"
							title={selectedRunId
								? `Open run ${selectedRunId.slice(0, 8)} in MLflow`
								: 'Open experiment in MLflow'}
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
				<!-- Debug: runs count indicator -->
				<span class="text-[9px] text-muted-foreground/50">({runs.length} runs)</span>

				{#if runs.length > 0}
					<select
						class="w-full rounded-md border border-input bg-background px-2.5 py-1.5 text-xs shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/30"
						value={$selectedBatchRun[PROJECT] || ''}
						onchange={(e) => onRunChange(e.currentTarget.value)}
					>
						<option value="">Select MLflow run...</option>
						{#each runs as run}
							<option value={run.run_id}>{run.is_best ? '★ ' : ''}{run.run_id}</option>
						{/each}
					</select>
				{:else}
					<p class="text-xs text-muted-foreground">No runs available. Train a model first.</p>
				{/if}

				<hr class="border-border" />

				<!-- Training Section -->
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

						<!-- CatBoost Training Log (shown during training stage) -->
						{#if $batchTrainingStage[PROJECT] === 'training' && Object.keys($batchTrainingCatboostLog[PROJECT] || {}).length > 0}
							{@const log = $batchTrainingCatboostLog[PROJECT]}
							<div class="rounded bg-muted/50 p-2 space-y-1">
								<div class="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px]">
									<!-- Iteration -->
									<div class="flex items-center gap-1">
										<span class="text-muted-foreground w-[55px]">Iteration</span>
										<span class="font-bold text-blue-600">{log.iteration || '-'}</span>
									</div>
									<!-- Test (Loss) -->
									<div class="flex items-center gap-1">
										<span class="text-muted-foreground w-[55px]">RMSE</span>
										<span class="font-medium text-purple-600">{log.test ? Number(log.test).toFixed(4) : '-'}</span>
									</div>
									<!-- Best -->
									<div class="flex items-center gap-1">
										<span class="text-muted-foreground w-[55px]">Best</span>
										<span class="font-medium text-green-600">{log.best ? Number(log.best).toFixed(4) : '-'}</span>
									</div>
									<!-- Total Time -->
									<div class="flex items-center gap-1">
										<span class="text-muted-foreground w-[55px]">Total</span>
										<span class="font-medium">{log.total || '-'}</span>
									</div>
								</div>
								<!-- Remaining Time -->
								{#if log.remaining}
									<div class="flex items-center gap-1 text-[10px] pt-1 border-t border-border/50">
										<Clock class="h-3 w-3 text-orange-500" />
										<span class="text-muted-foreground">Remaining</span>
										<span class="font-medium text-orange-500">{log.remaining}</span>
									</div>
								{/if}
							</div>
						{/if}
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

		<!-- Form Card - Always Visible -->
		<Card>
			<CardContent class="space-y-3 pt-4">
				<div class="flex items-center gap-2">
					<Clock class="h-5 w-5 text-primary" />
					<h3 class="text-base font-bold">Trip Details</h3>
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
						<p class="text-xs text-muted-foreground">Driver ID ({FIELD_CONFIG.driver_id.min}-{FIELD_CONFIG.driver_id.max})</p>
						<Input
							type="number"
							value={(currentForm.driver_id as number) ?? FIELD_CONFIG.driver_id.min}
							min={FIELD_CONFIG.driver_id.min}
							max={FIELD_CONFIG.driver_id.max}
							step={FIELD_CONFIG.driver_id.step ?? 1}
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
							step={FIELD_CONFIG.vehicle_id.step ?? 1}
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
							oninput={(e) => {
								updateFormField(PROJECT, 'timestamp_date', e.currentTarget.value);
								updateDayOfWeek(e.currentTarget.value);
							}}
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
							step={FIELD_CONFIG.origin_lat.step ?? 0.000001}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.origin_lat.min;
								const clamped = clampFieldValue('origin_lat', val);
								updateFormField(PROJECT, 'origin_lat', clamped);
								e.currentTarget.value = String(clamped);
								updateEstimates();
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
							step={FIELD_CONFIG.origin_lon.step ?? 0.000001}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.origin_lon.min;
								const clamped = clampFieldValue('origin_lon', val);
								updateFormField(PROJECT, 'origin_lon', clamped);
								e.currentTarget.value = String(clamped);
								updateEstimates();
							}}
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
						<p class="text-xs text-muted-foreground">Dest Lat ({FIELD_CONFIG.destination_lat.min}-{FIELD_CONFIG.destination_lat.max})</p>
						<Input
							type="number"
							value={currentForm.destination_lat ?? ''}
							min={FIELD_CONFIG.destination_lat.min}
							max={FIELD_CONFIG.destination_lat.max}
							step={FIELD_CONFIG.destination_lat.step ?? 0.000001}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.destination_lat.min;
								const clamped = clampFieldValue('destination_lat', val);
								updateFormField(PROJECT, 'destination_lat', clamped);
								e.currentTarget.value = String(clamped);
								updateEstimates();
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
							step={FIELD_CONFIG.destination_lon.step ?? 0.000001}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.destination_lon.min;
								const clamped = clampFieldValue('destination_lon', val);
								updateFormField(PROJECT, 'destination_lon', clamped);
								e.currentTarget.value = String(clamped);
								updateEstimates();
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
							step={FIELD_CONFIG.hour_of_day.step ?? 1}
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
							step={FIELD_CONFIG.driver_rating.step ?? 0.1}
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
							step={FIELD_CONFIG.temperature_celsius.step ?? 0.1}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.temperature_celsius.min;
								const clamped = clampFieldValue('temperature_celsius', val);
								updateFormField(PROJECT, 'temperature_celsius', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>
				</div>

				<div class="space-y-1 text-xs text-muted-foreground">
					<p>Trip ID: {currentForm.trip_id || '-'}</p>
					<p>Estimated Distance: {estimatedDistanceKm.toFixed(2)} km</p>
					<p>Initial ETA: {initialEstimatedTravelTime} s ({(initialEstimatedTravelTime / 60).toFixed(1)} min)</p>
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
				<div class="flex gap-3">
					<Card class="h-[400px] w-1/2">
						<CardContent class="h-full p-4">
							<div class="flex h-full flex-col">
								<div class="mb-2 flex items-center gap-2">
									<MapPin class="h-4 w-4 text-primary" />
									<span class="text-sm font-bold">Origin and Destination</span>
								</div>
								<div
									bind:this={mapContainer}
									class="flex-1 overflow-hidden rounded-md"
									style="min-height: 280px;"
								></div>
								<div class="mt-2 space-y-1 text-xs text-muted-foreground">
									<p>Estimated Distance: {estimatedDistanceKm} km</p>
									<p>Initial Estimated Travel Time: {initialEstimatedTravelTime} s</p>
								</div>
							</div>
						</CardContent>
					</Card>

					<Card class="h-[400px] w-1/2">
						<CardContent class="h-full p-4">
							<div class="flex h-full flex-col">
								<div class="mb-2 flex items-center gap-2">
									<Clock class="h-4 w-4 text-primary" />
									<span class="text-sm font-bold">ETA - Prediction</span>
								</div>
								<div class="flex flex-1 items-center justify-center">
									{#if etaSeconds > 0}
										<Plotly
											data={etaPredictionData}
											layout={etaPredictionLayout}
											config={plotlyConfig}
										/>
									{:else if modelAvailable}
										<div
											class="flex items-center gap-2 rounded-lg border border-blue-200 bg-blue-50 p-4 text-sm text-blue-700 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-300"
										>
											<Info class="h-4 w-4" />
											<span
												>Click <strong>Predict</strong> to get the estimated time of arrival.</span
											>
										</div>
									{:else}
										<div
											class="flex items-center gap-2 rounded-lg border border-orange-200 bg-orange-50 p-4 text-sm text-orange-700 dark:border-orange-900 dark:bg-orange-950 dark:text-orange-300"
										>
											<AlertTriangle class="h-4 w-4" />
											<span>No trained model available. Click <strong>Train</strong> first.</span>
										</div>
									{/if}
								</div>
							</div>
						</CardContent>
					</Card>
				</div>
			</TabsContent>

			<!-- Metrics Tab -->
			<TabsContent value="metrics" class="mt-4">
				<div class="space-y-4">
					<div class="flex items-center justify-between">
						<h2 class="text-lg font-bold">Regression Metrics</h2>
						<Button variant="ghost" size="sm" onclick={loadMetrics} loading={metricsLoading}>
							<RefreshCw class="h-4 w-4" />
						</Button>
					</div>

					<Tabs value={activeMetricsTab} onValueChange={onMetricsTabChange} class="w-full">
						<TabsList class="grid w-full grid-cols-5">
							<TabsTrigger value="overview" class="px-1 text-[11px]">
								<LayoutDashboard class="mr-1 h-3 w-3" />
								Overview
							</TabsTrigger>
							<TabsTrigger value="regression" class="px-1 text-[11px]">
								<TrendingUp class="mr-1 h-3 w-3" />
								Regression
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
											<span class="text-xs text-blue-600 dark:text-blue-400"
												>(80% train / 20% test split)</span
											>
										</div>
									{/if}

									<!-- Primary Error Metrics (Lower is Better) -->
									<div class="flex items-center gap-2">
										<Target class="h-4 w-4 text-red-600" />
										<span class="text-sm font-bold">Primary Error Metrics</span>
										<span class="text-[10px] text-muted-foreground">(Lower is Better)</span>
									</div>
									<div class="grid grid-cols-4 gap-2">
										<MetricCard
											name="MAE"
											value={currentMetrics.mae as number}
											onInfoClick={() => openMetricInfo('mae')}
										/>
										<MetricCard
											name="RMSE"
											value={currentMetrics.rmse as number}
											onInfoClick={() => openMetricInfo('rmse')}
										/>
										<MetricCard
											name="MAPE"
											value={currentMetrics.mape as number}
											onInfoClick={() => openMetricInfo('mape')}
										/>
										<MetricCard
											name="SMAPE"
											value={currentMetrics.smape as number}
											onInfoClick={() => openMetricInfo('smape')}
										/>
									</div>

									<hr class="border-border" />

									<!-- Goodness of Fit Metrics (Higher is Better) -->
									<div class="flex items-center gap-2">
										<TrendingUp class="h-4 w-4 text-green-600" />
										<span class="text-sm font-bold">Goodness of Fit</span>
										<span class="text-[10px] text-muted-foreground">(Higher is Better)</span>
									</div>
									<div class="grid grid-cols-2 gap-2">
										<MetricCard
											name="R² Score"
											value={currentMetrics.r2_score as number}
											onInfoClick={() => openMetricInfo('r2_score')}
										/>
										<MetricCard
											name="Explained Var"
											value={currentMetrics.explained_variance as number}
											onInfoClick={() => openMetricInfo('explained_variance_score')}
										/>
									</div>

									<hr class="border-border" />

									<!-- Secondary Error Metrics -->
									<div class="flex items-center gap-2">
										<BarChart3 class="h-4 w-4 text-indigo-600" />
										<span class="text-sm font-bold">Secondary Error Metrics</span>
									</div>
									<div class="grid grid-cols-3 gap-2">
										<MetricCard
											name="MSE"
											value={currentMetrics.mse as number}
											onInfoClick={() => openMetricInfo('mean_squared_error')}
										/>
										<MetricCard
											name="Median AE"
											value={currentMetrics.median_ae as number}
											onInfoClick={() => openMetricInfo('median_absolute_error')}
										/>
										<MetricCard
											name="Max Error"
											value={currentMetrics.max_error as number}
											onInfoClick={() => openMetricInfo('max_error')}
										/>
									</div>

									<hr class="border-border" />

									<!-- D² Deviance Metrics (Higher is Better) -->
									<div class="flex items-center gap-2">
										<Lightbulb class="h-4 w-4 text-amber-600" />
										<span class="text-sm font-bold">D² Deviance Metrics</span>
										<span class="text-[10px] text-muted-foreground">(Higher is Better)</span>
									</div>
									<div class="grid grid-cols-3 gap-2">
										<MetricCard
											name="D² Absolute"
											value={currentMetrics.d2_absolute_error as number}
											onInfoClick={() => openMetricInfo('d2_absolute_error_score')}
										/>
										<MetricCard
											name="D² Pinball"
											value={currentMetrics.d2_pinball as number}
											onInfoClick={() => openMetricInfo('d2_pinball_score')}
										/>
										<MetricCard
											name="D² Tweedie"
											value={currentMetrics.d2_tweedie as number}
											onInfoClick={() => openMetricInfo('d2_tweedie_score')}
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

						<!-- Regression Tab -->
						<TabsContent value="regression" class="pt-4">
							<div class="space-y-4">
								{#if modelAvailable}
									<p class="text-xs text-muted-foreground">
										Select a regression analysis visualization.
									</p>
									<div class="flex items-center gap-2">
										<select
											class="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
											value={currentVisualizer}
											onchange={(e) => loadVisualizer('Regression', e.currentTarget.value)}
										>
											{#each YELLOWBRICK_CATEGORIES['Regression'] as viz}
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
					<div
						class="h-6 w-6 animate-spin rounded-full border-4 border-primary border-t-transparent"
					></div>
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
		<span
			>Train a model first to view visualizations. Use the Training box on the left and click
			Train.</span
		>
	</div>
{/snippet}

<!-- YellowBrick Info Dialog -->
{#if yellowBrickInfoOpen}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
		onclick={() => (yellowBrickInfoOpen = false)}
	>
		<div
			class="mx-4 max-h-[85vh] w-full max-w-lg overflow-y-auto rounded-lg bg-card p-6 shadow-xl"
			onclick={(e) => e.stopPropagation()}
		>
			<div class="mb-4 flex items-center justify-between">
				<div class="flex items-center gap-2">
					<BarChart3 class="h-5 w-5 text-primary" />
					<h2 class="text-lg font-bold">{yellowBrickInfoContent.name}</h2>
				</div>
				<button
					type="button"
					class="rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
					onclick={() => (yellowBrickInfoOpen = false)}
				>
					<span class="text-xl">&times;</span>
				</button>
			</div>

			<hr class="mb-4 border-border" />

			<div class="space-y-4">
				<span
					class="inline-block rounded-full bg-purple-100 px-2.5 py-0.5 text-xs font-medium text-purple-700 dark:bg-purple-900 dark:text-purple-300"
				>
					{yellowBrickInfoContent.category}
				</span>

				<div class="rounded-lg bg-muted/50 p-3">
					<div class="mb-1 flex items-center gap-1.5">
						<span class="text-sm">👁️</span>
						<span class="text-sm font-semibold">What it shows</span>
					</div>
					<p class="text-sm text-muted-foreground">{yellowBrickInfoContent.description}</p>
				</div>

				<div>
					<div class="mb-1 flex items-center gap-1.5">
						<span class="text-sm">🔍</span>
						<span class="text-sm font-semibold">How to read it</span>
					</div>
					<p class="text-sm text-muted-foreground">{@html yellowBrickInfoContent.interpretation}</p>
				</div>

				{#if yellowBrickInfoContent.context}
					<div class="rounded-lg bg-blue-50 p-3 dark:bg-blue-950">
						<div class="mb-1 flex items-center gap-1.5">
							<Clock class="h-4 w-4 text-blue-600" />
							<span class="text-sm font-semibold text-blue-700 dark:text-blue-300"
								>In ETA Prediction</span
							>
						</div>
						<p class="text-sm text-blue-600 dark:text-blue-400">
							{@html yellowBrickInfoContent.context}
						</p>
					</div>
				{/if}

				{#if yellowBrickInfoContent.whenToUse}
					<div>
						<div class="mb-1 flex items-center gap-1.5">
							<span class="text-sm">💡</span>
							<span class="text-sm font-semibold">When to use</span>
						</div>
						<p class="text-sm text-muted-foreground">{yellowBrickInfoContent.whenToUse}</p>
					</div>
				{/if}

				{#if yellowBrickInfoContent.docsUrl}
					<a
						href={yellowBrickInfoContent.docsUrl}
						target="_blank"
						rel="noopener noreferrer"
						class="inline-flex items-center gap-1.5 text-sm text-blue-600 hover:underline"
					>
						<ExternalLink class="h-3.5 w-3.5" />
						Official Documentation
					</a>
				{/if}
			</div>
		</div>
	</div>
{/if}
