<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { beforeNavigate } from '$app/navigation';
	import {
		Card,
		CardHeader,
		CardTitle,
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
	import { randomizeTFDForm } from '$lib/utils/randomize';
	import {
		RefreshCw,
		ExternalLink,
		LayoutDashboard,
		CheckCircle,
		Lightbulb,
		ScatterChart,
		Crosshair,
		Settings2,
		Info,
		CreditCard,
		Shuffle,
		Target,
		BarChart3,
		ShieldAlert,
		FlaskConical,
		GitBranch,
		Brain,
		Play,
		Square,
		Loader2,
		Settings,
		Database,
		Clock,
		Star
	} from 'lucide-svelte';
	import Plotly from 'svelte-plotly.js';
	import type { ProjectName } from '$types';
	import { cn } from '$lib/utils';

	const PROJECT: ProjectName = 'Transaction Fraud Detection';
	const MODEL_NAME = 'CatBoost Classifier';

	// Transform MLflow metrics from API format to simple format
	// API returns keys like: metrics.fbeta_score, metrics.roc_auc_score, etc. (matching Reflex)
	function transformMetrics(rawMetrics: Record<string, unknown>): Record<string, number> {
		// Helper to get metric value - matches Reflex's get_metric function
		const getMetric = (name: string): number => {
			const key = `metrics.${name}`;
			const val = rawMetrics[key];
			return val !== undefined && val !== null ? Number(val) : 0;
		};

		return {
			// Primary classification metrics (matching Reflex tfd_batch_dashboard_figures)
			fbeta_score: getMetric('fbeta_score'),
			roc_auc: getMetric('roc_auc_score'),
			precision: getMetric('precision_score'),
			recall: getMetric('recall_score'),
			accuracy: getMetric('accuracy_score'),
			f1_score: getMetric('f1_score'),

			// Probabilistic/loss metrics
			log_loss: getMetric('log_loss'),
			brier: getMetric('brier_score_loss'),

			// Additional classification metrics
			balanced_accuracy: getMetric('balanced_accuracy_score'),
			mcc: getMetric('matthews_corrcoef'),
			cohen_kappa: getMetric('cohen_kappa_score'),
			jaccard: getMetric('jaccard_score'),
			geometric_mean: getMetric('geometric_mean_score'),
			average_precision: getMetric('average_precision_score'),

			// D2 scores
			d2_log_loss: getMetric('d2_log_loss_score'),
			d2_brier: getMetric('d2_brier_score'),

			// Sample counts
			n_samples: getMetric('n_samples')
		};
	}

	// YellowBrick visualizer options organized by category (matching Reflex exactly)
	const YELLOWBRICK_CATEGORIES = {
		Classification: [
			{ value: '', label: 'Select visualization...' },
			{ value: 'ConfusionMatrix', label: 'Confusion Matrix' },
			{ value: 'ClassificationReport', label: 'Classification Report' },
			{ value: 'ROCAUC', label: 'ROC AUC Curve' },
			{ value: 'PrecisionRecallCurve', label: 'Precision-Recall Curve' },
			{ value: 'ClassPredictionError', label: 'Class Prediction Error' },
			{ value: 'DiscriminationThreshold', label: 'Discrimination Threshold' }
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
			{ value: 'ClassBalance', label: 'Class Balance' },
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
		]
	};

	let activeTab = $state('prediction');
	let activeMetricsTab = $state('overview');
	let metricsLoading = $state(false);
	let visualizerInfo = $state<Record<string, unknown>>({});
	let statusInterval: ReturnType<typeof setInterval> | null = null;
	let runsLoading = $state(false);

	// Form state
	let dropdownOptions = $state<Record<string, string[]>>({});
	let sampleLoading = $state(false);

	onMount(async () => {
		// Load dropdown options
		try {
			const response = await fetch('/data/dropdown_options_tfd.json');
			dropdownOptions = await response.json();
		} catch (e) {
			console.error('Failed to load dropdown options:', e);
		}

		// Initialize form if empty or missing required fields
		const existingForm = $formData[PROJECT] || {};
		if (!Object.keys(existingForm).length || !existingForm.merchant_id) {
			updateProjectStore(formData, PROJECT, randomizeTFDForm(dropdownOptions));
		}

		// Load YellowBrick info
		try {
			const response = await fetch('/data/yellowbrick_info_tfd.json');
			visualizerInfo = await response.json();
		} catch (e) {
			console.error('Failed to load visualizer info:', e);
		}

		// Initialize batch page with optimized single API call (matches Reflex init_batch_page)
		// This fetches: runs, model_available, experiment_url, metrics in one call
		await initBatchPage();

		// Get Delta Lake row count (for max_rows training option)
		const countResult = await batchApi.getDeltaLakeRowCount(PROJECT);
		if (countResult.data?.row_count) {
			updateProjectStore(batchDeltaLakeTotalRows, PROJECT, countResult.data.row_count);
		}

		// Don't load any visualizer on mount - let user select from dropdown
		// (placeholder option avoids automatic loading)
	});

	onDestroy(() => {
		stopStatusPolling();
		// Cancel any pending YellowBrick loading
		yellowBrickCancelRequested = true;
	});

	// Stop training when navigating away to prevent orphaned training processes
	beforeNavigate(async ({ to }) => {
		// Cancel YellowBrick loading
		yellowBrickCancelRequested = true;

		// Stop batch training if in progress
		if (isTraining) {
			await batchApi.stopTraining(PROJECT);
			updateProjectStore(batchTrainingLoading, PROJECT, false);
			updateProjectStore(batchTrainingStatus, PROJECT, '');
			updateProjectStore(batchTrainingProgress, PROJECT, 0);
			stopStatusPolling();
			toast.info('Training stopped - navigated away from page');
		}
	});

	// YellowBrick cancel flag (for stopping visualization loading)
	let yellowBrickCancelRequested = $state(false);

	/**
	 * Cancel the current YellowBrick visualization loading (matching Reflex cancel_yellowbrick_loading)
	 */
	function cancelYellowBrickLoading() {
		yellowBrickCancelRequested = true;
		updateProjectStore(yellowBrickLoading, PROJECT, {});
		updateProjectStore(selectedYellowBrickVisualizer, PROJECT, '');
		// Clear images for this project
		yellowBrickImages.update((imgs) => ({
			...imgs,
			[PROJECT]: {}
		}));
		toast.info('Visualization loading cancelled');
	}

	/**
	 * Initialize batch page with optimized single API call (matches Reflex init_batch_page).
	 * Fetches: runs, model_available, experiment_url, metrics in one call.
	 */
	async function initBatchPage() {
		runsLoading = true;
		metricsLoading = true;

		try {
			const runId = $selectedBatchRun[PROJECT] || undefined;
			const result = await batchApi.initBatchPage(PROJECT, runId);
			console.log('[TFD Batch] initBatchPage response:', result);

			if (result.error) {
				console.error('[TFD Batch] initBatchPage error:', result.error);
				toast.error(`Failed to initialize: ${result.error}`);
			} else if (result.data) {
				const data = result.data;
				console.log('[TFD Batch] Loaded data - runs:', data.runs?.length, 'model_available:', data.model_available);

				// Update runs
				const runs = data.runs || [];
				updateProjectStore(batchMlflowRuns, PROJECT, runs);

				// Model availability
				updateProjectStore(batchModelAvailable, PROJECT, data.model_available);

				// Experiment URL (for MLflow button)
				if (data.experiment_url) {
					updateProjectStore(batchMlflowExperimentUrl, PROJECT, data.experiment_url);
				}

				// Metrics
				if (data.metrics && !data.metrics._no_runs) {
					const transformed = transformMetrics(data.metrics);
					updateProjectStore(batchMlflowMetrics, PROJECT, transformed);
				}

				// Auto-select best run if none selected
				if (runs.length && !$selectedBatchRun[PROJECT]) {
					const bestRunId = data.best_run_id || runs[0].run_id;
					updateProjectStore(selectedBatchRun, PROJECT, bestRunId);
				}

				// Update run URL from metrics if available (for direct run link)
				if (data.metrics?.run_url) {
					updateProjectStore(batchMlflowExperimentUrl, PROJECT, data.metrics.run_url as string);
				}
			} else {
				console.warn('[TFD Batch] initBatchPage: No data in response');
			}
		} catch (err) {
			console.error('[TFD Batch] initBatchPage exception:', err);
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
			console.log('[TFD Batch] loadRuns response:', result);

			if (result.error) {
				console.error('[TFD Batch] loadRuns error:', result.error);
				toast.error(`Failed to load runs: ${result.error}`);
			} else if (result.data) {
				const runs = result.data.runs || [];
				console.log('[TFD Batch] Loaded runs:', runs.length, runs);
				updateProjectStore(batchMlflowRuns, PROJECT, runs);
				updateProjectStore(batchModelAvailable, PROJECT, runs.length > 0);

				if (runs.length === 0) {
					toast.info('No MLflow runs found. Train a model to create runs.');
				}
			} else {
				console.warn('[TFD Batch] loadRuns: No data in response');
				toast.warning('No data returned from server');
			}
		} catch (err) {
			console.error('[TFD Batch] loadRuns exception:', err);
			toast.error('Failed to load MLflow runs');
		} finally {
			runsLoading = false;
		}
	}

	/** Load runs after training completes - stores last trained run ID and auto-selects it */
	async function loadRunsAfterTraining() {
		runsLoading = true;
		try {
			const result = await batchApi.getMLflowRuns(PROJECT);
			console.log('[TFD Batch] loadRunsAfterTraining response:', result);

			if (result.error) {
				console.error('[TFD Batch] loadRunsAfterTraining error:', result.error);
				toast.error(`Failed to load runs after training: ${result.error}`);
			} else if (result.data) {
				const runs = result.data.runs || [];
				console.log('[TFD Batch] Runs after training:', runs.length, runs);
				updateProjectStore(batchMlflowRuns, PROJECT, runs);
				updateProjectStore(batchModelAvailable, PROJECT, runs.length > 0);

				if (runs.length > 0) {
					// The first run is the newest (just trained) - store and select it
					const newRunId = runs[0].run_id;
					console.log('[TFD Batch] Auto-selecting new run:', newRunId);
					updateProjectStore(batchLastTrainedRunId, PROJECT, newRunId);
					updateProjectStore(selectedBatchRun, PROJECT, newRunId);
					toast.success(`Loaded ${runs.length} MLflow runs`);
				}
			}
		} catch (err) {
			console.error('[TFD Batch] loadRunsAfterTraining exception:', err);
			toast.error('Failed to load MLflow runs after training');
		} finally {
			runsLoading = false;
		}
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

	async function loadMetrics() {
		metricsLoading = true;
		const runId = $selectedBatchRun[PROJECT] || undefined;
		const result = await batchApi.getMLflowMetrics(PROJECT, runId);
		if (result.data && !result.data._no_runs) {
			const transformed = transformMetrics(result.data);
			updateProjectStore(batchMlflowMetrics, PROJECT, transformed);

			// Update MLflow URL to link directly to the selected run (matches Reflex)
			if (result.data.run_url) {
				updateProjectStore(batchMlflowExperimentUrl, PROJECT, result.data.run_url);
			}
		}
		metricsLoading = false;
	}

	// Current YellowBrick category (metric_type) for API calls
	let currentYellowBrickCategory = $state('Classification');

	async function loadVisualizer(category: string, visualizerName: string) {
		// Skip loading if placeholder is selected (empty value)
		if (!visualizerName) {
			updateProjectStore(selectedYellowBrickVisualizer, PROJECT, '');
			return;
		}

		// Reset cancel flag before starting new load
		yellowBrickCancelRequested = false;

		// Update category and visualizer
		currentYellowBrickCategory = category;
		updateProjectStore(selectedYellowBrickVisualizer, PROJECT, visualizerName);
		updateProjectStore(yellowBrickLoading, PROJECT, { [visualizerName]: true });

		const runId = $selectedBatchRun[PROJECT] || undefined;
		// API expects: metric_type (category), metric_name (visualizerName)
		const result = await batchApi.getYellowBrickImage(PROJECT, category, visualizerName, runId);

		// Check if cancelled before updating UI (matching Reflex pattern)
		if (yellowBrickCancelRequested) {
			return;
		}

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

	function openMetricInfo(metricKey: string) {
		fetch('/data/metric_info_tfd.json')
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

	// YellowBrick info dialog state
	let yellowBrickInfoOpen = $state(false);
	let yellowBrickInfoContent = $state<{
		name: string;
		category: string;
		description: string;
		interpretation: string;
		context: string;
		whenToUse: string;
		docsUrl: string;
	}>({ name: '', category: '', description: '', interpretation: '', context: '', whenToUse: '', docsUrl: '' });

	function openYellowBrickInfo(visualizerKey: string) {
		fetch('/data/yellowbrick_info_tfd.json')
			.then((r) => r.json())
			.then((data) => {
				const info = data.visualizers[visualizerKey];
				if (info) {
					yellowBrickInfoContent = {
						name: info.name,
						category: info.category,
						description: info.description,
						interpretation: info.interpretation,
						context: info.fraud_context || '',
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
		// Only reload YellowBrick visualizer if on a YellowBrick tab (not overview)
		const yellowBrickTabs = ['classification', 'features', 'target', 'diagnostics'];
		if (yellowBrickTabs.includes(activeMetricsTab)) {
			const currentViz = $selectedYellowBrickVisualizer[PROJECT];
			if (currentViz) {
				await loadVisualizer(currentYellowBrickCategory, currentViz);
			}
		}
	}

	/**
	 * Handle metrics sub-tab change - reset visualization when switching tabs
	 */
	function onMetricsTabChange(newTab: string) {
		activeMetricsTab = newTab;
		// Reset visualization when switching to any YellowBrick tab
		// This ensures user sees "Select a visualizer" instead of stale visualization
		const yellowBrickTabs = ['classification', 'features', 'target', 'diagnostics'];
		if (yellowBrickTabs.includes(newTab)) {
			// Clear current visualizer selection and image
			updateProjectStore(selectedYellowBrickVisualizer, PROJECT, '');
			yellowBrickImages.update((imgs) => ({
				...imgs,
				[PROJECT]: {}
			}));
		}
	}

	// Training functions (matching Reflex's train_batch_model)
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
			// Update CatBoost training log (iteration, total, remaining time)
			if (result.data.catboost_log) {
				updateProjectStore(batchTrainingCatboostLog, PROJECT, result.data.catboost_log);
			}

			// Check for completion
			if (result.data.status === 'completed') {
				stopStatusPolling();
				updateProjectStore(batchTrainingLoading, PROJECT, false);
				updateProjectStore(batchTrainingStatus, PROJECT, 'Training complete!');
				updateProjectStore(batchTrainingProgress, PROJECT, 100);
				updateProjectStore(batchTrainingStage, PROJECT, 'complete');
				toast.success('Batch ML training complete');

				// Refresh runs (stores last trained run ID and auto-selects it)
				await loadRunsAfterTraining();
				// Re-initialize page to get updated experiment URL and metrics
				await initBatchPage();
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
				await loadMetrics();
			}
		}
	}

	// Form functions
	function loadRandomSample() {
		sampleLoading = true;
		try {
			const randomData = randomizeTFDForm(dropdownOptions);
			updateProjectStore(formData, PROJECT, randomData);
			updateProjectStore(batchPredictionResults, PROJECT, {});
			toast.success('Form randomized');
		} finally {
			sampleLoading = false;
		}
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
				if (result.data.prediction === 1 || (result.data.fraud_probability ?? 0) >= 0.5) {
					toast.warning('Transaction flagged as potentially fraudulent');
				} else {
					toast.success('Transaction appears legitimate');
				}
			}
		} finally {
			updateProjectStore(batchPredictionLoading, PROJECT, false);
		}
	}

	// Derived values
	const currentMetrics = $derived($batchMlflowMetrics[PROJECT] || {});
	const runs = $derived($batchMlflowRuns[PROJECT] || []);
	const selectedRunId = $derived($selectedBatchRun[PROJECT] || '');
	const selectedRun = $derived(runs.find((r) => r.run_id === selectedRunId));
	const currentVisualizer = $derived($selectedYellowBrickVisualizer[PROJECT] || '');
	const currentImage = $derived($yellowBrickImages[PROJECT]?.[currentVisualizer] || '');
	const isImageLoading = $derived($yellowBrickLoading[PROJECT]?.[currentVisualizer] || false);
	const isTraining = $derived($batchTrainingLoading[PROJECT]);
	const modelAvailable = $derived($batchModelAvailable[PROJECT]);
	const currentForm = $derived($formData[PROJECT] || {});
	const currentPrediction = $derived($batchPredictionResults[PROJECT]);
	const isLoading = $derived($batchPredictionLoading[PROJECT]);
	const experimentUrl = $derived($batchMlflowExperimentUrl[PROJECT] || '');
	const lastTrainedRunId = $derived($batchLastTrainedRunId[PROJECT] || '');

	// MLflow URL: experiment page when no run selected, run page when run is selected
	const mlflowUrl = $derived.by(() => {
		if (!experimentUrl) return '';
		if (!selectedRunId) return experimentUrl; // No run selected: open experiment page

		// If experimentUrl already contains the run ID, use it directly
		if (experimentUrl.includes(`/runs/${selectedRunId}`)) {
			return experimentUrl;
		}

		// Construct run URL by appending /runs/{run_id} to experiment URL
		// Remove any trailing /runs/... if present (to get base experiment URL)
		const baseUrl = experimentUrl.replace(/\/runs\/[^/]+$/, '');
		return `${baseUrl}/runs/${selectedRunId}`;
	});

	// Format start time for display
	function formatStartTime(startTime: string | null): string {
		if (!startTime) return '';
		const timestamp = typeof startTime === 'number' ? startTime : Date.parse(startTime);
		if (!Number.isNaN(timestamp)) {
			return new Date(timestamp).toISOString().replace('T', ' ').slice(0, 19);
		}
		return String(startTime);
	}

	// Fraud probability for gauge
	const fraudProbability = $derived(currentPrediction?.fraud_probability ?? 0);
	const notFraudProbability = $derived(1 - fraudProbability);
	const predictionText = $derived(fraudProbability >= 0.5 ? 'FRAUD' : 'NOT FRAUD');
	const predictionColor = $derived(fraudProbability >= 0.5 ? 'red' : 'green');

	// Risk level based on probability
	const riskLevel = $derived.by(() => {
		const prob = fraudProbability * 100;
		if (prob < 30) return { text: 'LOW RISK', color: '#22c55e' };
		if (prob < 70) return { text: 'MEDIUM RISK', color: '#eab308' };
		return { text: 'HIGH RISK', color: '#ef4444' };
	});

	const gaugeData = $derived.by(() => {
		return [
			{
				type: 'indicator',
				mode: 'gauge+number',
				value: Number((fraudProbability * 100).toFixed(2)),
				number: {
					suffix: '%',
					font: { size: 40, color: riskLevel.color }
				},
				title: {
					text: `<b>${riskLevel.text}</b><br><span style="font-size:12px;color:#888">Fraud Probability</span>`,
					font: { size: 18 }
				},
				gauge: {
					axis: {
						range: [0, 100],
						tickwidth: 1,
						tickcolor: 'hsl(var(--muted-foreground))',
						tickvals: [0, 25, 50, 75, 100],
						ticktext: ['0%', '25%', '50%', '75%', '100%']
					},
					bar: { color: riskLevel.color, thickness: 0.75 },
					bgcolor: 'rgba(0,0,0,0)',
					borderwidth: 2,
					bordercolor: '#333',
					steps: [
						{ range: [0, 30], color: 'rgba(34, 197, 94, 0.3)' },
						{ range: [30, 70], color: 'rgba(234, 179, 8, 0.3)' },
						{ range: [70, 100], color: 'rgba(239, 68, 68, 0.3)' }
					],
					threshold: { line: { color: '#333', width: 4 }, thickness: 0.8, value: fraudProbability * 100 }
				}
			}
		];
	});
	const gaugeLayout = {
		height: 280,
		margin: { t: 80, r: 30, b: 30, l: 30 },
		paper_bgcolor: 'transparent',
		plot_bgcolor: 'transparent'
	};
	const gaugeConfig = { displayModeBar: false, responsive: true };

</script>

<!-- 40%/60% Layout (matching Incremental ML) -->
<div class="flex gap-6">
	<!-- Left Column - Training Box + Form (40%) -->
	<div class="w-[40%] space-y-4">
		<!-- Batch ML Training Box (cloned from Reflex) -->
		<Card>
			<CardContent class="space-y-3 p-3">
				<!-- MLflow Run Section Header with Model Name Badge -->
				<div class="flex items-center gap-2">
					<GitBranch class="h-4 w-4 text-blue-600" />
					<span class="text-xs font-medium">MLflow Run</span>
					<!-- Model Name Badge -->
					<span class="rounded bg-blue-100 px-1.5 py-0.5 text-[10px] font-medium text-blue-700 dark:bg-blue-900 dark:text-blue-300">
						{MODEL_NAME}
					</span>
					{#if runsLoading}
						<Loader2 class="h-3 w-3 animate-spin text-muted-foreground" />
					{/if}
					<div class="flex-1"></div>
					<!-- MLflow Button: opens experiment page (no run) or run page (with run selected) -->
					{#if mlflowUrl}
						<a
							href={mlflowUrl}
							target="_blank"
							rel="noopener noreferrer"
							class="inline-flex items-center gap-1 rounded-md bg-cyan-100 px-2 py-1 text-xs font-medium text-cyan-700 hover:bg-cyan-200 dark:bg-cyan-900 dark:text-cyan-300 dark:hover:bg-cyan-800"
							title={selectedRunId ? `Open run ${selectedRunId.slice(0, 8)} in MLflow` : 'Open experiment in MLflow'}
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
				<!-- Debug: runs count indicator (can be removed after debugging) -->
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
									<!-- Test Score -->
									<div class="flex items-center gap-1">
										<span class="text-muted-foreground w-[55px]">Test</span>
										<span class="font-medium text-green-600">{log.test || '-'}</span>
									</div>
									<!-- Best Score -->
									<div class="flex items-center gap-1">
										<span class="text-muted-foreground w-[55px]">Best</span>
										<span class="font-medium text-purple-600">{log.best || '-'}</span>
									</div>
									<!-- Total Time -->
									<div class="flex items-center gap-1">
										<span class="text-muted-foreground w-[55px]">Total</span>
										<span class="font-medium text-cyan-600">{log.total || '-'}</span>
									</div>
								</div>
								<!-- Remaining Time (full width with icon) -->
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

		<!-- Form Card (cloned from Incremental ML) -->
		<Card>
			<CardContent class="space-y-3 pt-4">
				<!-- Form Legend -->
				<div class="flex items-center gap-2">
					<CreditCard class="h-5 w-5 text-primary" />
					<h3 class="text-base font-bold">Transaction Details</h3>
				</div>

				<!-- Predict + Randomize Buttons -->
				<div class="flex gap-2">
					<Button class="flex-1" onclick={predict} loading={isLoading} disabled={!modelAvailable || !selectedRunId}>
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
						<p class="text-xs text-muted-foreground">Amount</p>
						<Input
							type="number"
							value={currentForm.amount ?? ''}
							step="0.01"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'amount', parseFloat(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Account Age</p>
						<Input
							type="number"
							value={currentForm.account_age_days ?? ''}
							min="0"
							step="1"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'account_age_days', parseInt(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Currency</p>
						<Select
							value={(currentForm.currency as string) ?? ''}
							options={dropdownOptions.currency || ['USD']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'currency', e.currentTarget.value)}
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
						<p class="text-xs text-muted-foreground">Merchant ID</p>
						<Select
							value={(currentForm.merchant_id as string) || dropdownOptions.merchant_id?.[0] || 'merchant_1'}
							options={dropdownOptions.merchant_id?.slice(0, 50) || ['merchant_1']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'merchant_id', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Category</p>
						<Select
							value={(currentForm.product_category as string) ?? ''}
							options={dropdownOptions.product_category || ['electronics']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'product_category', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Type</p>
						<Select
							value={(currentForm.transaction_type as string) ?? ''}
							options={dropdownOptions.transaction_type || ['purchase']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'transaction_type', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Payment</p>
						<Select
							value={(currentForm.payment_method as string) ?? ''}
							options={dropdownOptions.payment_method || ['credit_card']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'payment_method', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Latitude</p>
						<Input
							type="number"
							value={currentForm.lat ?? ''}
							min="-90"
							max="90"
							step="0.0001"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'lat', parseFloat(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Longitude</p>
						<Input
							type="number"
							value={currentForm.lon ?? ''}
							min="-180"
							max="180"
							step="0.0001"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'lon', parseFloat(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Browser</p>
						<Select
							value={(currentForm.browser as string) ?? ''}
							options={dropdownOptions.browser || ['Chrome']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'browser', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">OS</p>
						<Select
							value={(currentForm.os as string) ?? ''}
							options={dropdownOptions.os || ['Windows']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'os', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">CVV</p>
						<label class="flex items-center gap-2 text-sm">
							<input
								type="checkbox"
								checked={currentForm.cvv_provided ?? false}
								onchange={(e) => updateFormField(PROJECT, 'cvv_provided', e.currentTarget.checked)}
								class="h-4 w-4 rounded border-input"
							/>
							Provided
						</label>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Billing</p>
						<label class="flex items-center gap-2 text-sm">
							<input
								type="checkbox"
								checked={currentForm.billing_address_match ?? false}
								onchange={(e) => updateFormField(PROJECT, 'billing_address_match', e.currentTarget.checked)}
								class="h-4 w-4 rounded border-input"
							/>
							Address Match
						</label>
					</div>
				</div>

				<!-- Display fields (read-only info) -->
				<div class="space-y-1 text-xs text-muted-foreground">
					<p>Transaction ID: {currentForm.transaction_id || '-'}</p>
					<p>User ID: {currentForm.user_id || '-'}</p>
					<p>IP Address: {currentForm.ip_address || '-'}</p>
				</div>
			</CardContent>
		</Card>
	</div>

	<!-- Right Column - Tabs for Prediction and Metrics (60%) -->
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
			</TabsList>

			<!-- MLflow Run Info Badge (matching Incremental ML style) -->
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
					{#if selectedRun?.is_best}
						<span
							class="inline-flex items-center gap-1 rounded-md bg-amber-500/10 px-2 py-1 text-xs font-medium text-amber-600"
						>
							Best Run
						</span>
					{/if}
					<span class="text-xs text-muted-foreground">
						Run:
						<code
							class="rounded bg-muted px-1"
							title={selectedRunId}
						>
							{selectedRunId.slice(0, 8)}
						</code>
					</span>
					{#if selectedRun?.start_time}
						<span class="text-xs text-muted-foreground">
							Started: {formatStartTime(selectedRun.start_time)}
						</span>
					{/if}
					{#if selectedRun?.total_rows}
						<span class="text-xs text-muted-foreground">
							Samples: {selectedRun.total_rows.toLocaleString()}
						</span>
					{/if}
				{:else}
					<span class="text-xs text-muted-foreground italic">
						No run selected - select a run to enable predictions
					</span>
				{/if}
			</div>

			<!-- Prediction Tab -->
			<TabsContent value="prediction" class="mt-4">
				<div class="space-y-4">
					<div class="flex items-center gap-2">
						<ShieldAlert class="h-5 w-5 text-primary" />
						<h2 class="text-xl font-bold">Prediction Result</h2>
					</div>

					{#if modelAvailable && currentPrediction}
						<Card>
							<CardContent class="space-y-4 pt-6">
								<div class="flex w-full items-center justify-center">
									<Plotly data={gaugeData} layout={{ ...gaugeLayout, width: 300 }} config={gaugeConfig} />
								</div>

								<div class="grid grid-cols-3 gap-3">
									<Card class="bg-muted/50">
										<CardContent class="p-3 text-center">
											<div class="mb-1 flex items-center justify-center gap-1">
												<ShieldAlert class="h-3.5 w-3.5" style="color: {predictionColor}" />
												<span class="text-xs text-muted-foreground">Classification</span>
											</div>
											<p class="text-lg font-bold" style="color: {predictionColor}">{predictionText}</p>
										</CardContent>
									</Card>

									<Card class="bg-muted/50">
										<CardContent class="p-3 text-center">
											<div class="mb-1 flex items-center justify-center gap-1">
												<span class="h-3.5 w-3.5 text-red-500">%</span>
												<span class="text-xs text-muted-foreground">Fraud</span>
											</div>
											<p class="text-lg font-bold text-red-500">{(fraudProbability * 100).toFixed(2)}%</p>
										</CardContent>
									</Card>

									<Card class="bg-muted/50">
										<CardContent class="p-3 text-center">
											<div class="mb-1 flex items-center justify-center gap-1">
												<CheckCircle class="h-3.5 w-3.5 text-green-500" />
												<span class="text-xs text-muted-foreground">Not Fraud</span>
											</div>
											<p class="text-lg font-bold text-green-500">{(notFraudProbability * 100).toFixed(2)}%</p>
										</CardContent>
									</Card>
								</div>
							</CardContent>
						</Card>
					{:else if modelAvailable}
						<div class="flex items-center gap-2 rounded-lg border border-blue-200 bg-blue-50 p-4 text-sm text-blue-700 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-300">
							<Info class="h-4 w-4" />
							<span>Fill in the transaction details and click <strong>Predict</strong> to get the fraud probability.</span>
						</div>
					{:else}
						<div class="flex items-center gap-2 rounded-lg border border-orange-200 bg-orange-50 p-4 text-sm text-orange-700 dark:border-orange-900 dark:bg-orange-950 dark:text-orange-300">
							<ShieldAlert class="h-4 w-4" />
							<span>No trained model available. Click <strong>Train</strong> to train the batch model first.</span>
						</div>
					{/if}
				</div>
			</TabsContent>

			<!-- Metrics Tab -->
			<TabsContent value="metrics" class="mt-4">
				<div class="space-y-4">
					<div class="flex items-center justify-between">
						<h2 class="text-lg font-bold">Classification Metrics</h2>
						<Button variant="ghost" size="sm" onclick={loadMetrics} loading={metricsLoading}>
							<RefreshCw class="h-4 w-4" />
						</Button>
					</div>

					<!-- Metrics Sub-tabs (with smaller font sizes) -->
					<Tabs value={activeMetricsTab} onValueChange={onMetricsTabChange} class="w-full">
						<TabsList class="grid w-full grid-cols-5">
							<TabsTrigger value="overview" class="text-[11px] px-1">
								<LayoutDashboard class="mr-1 h-3 w-3" />
								Overview
							</TabsTrigger>
							<TabsTrigger value="classification" class="text-[11px] px-1">
								<CheckCircle class="mr-1 h-3 w-3" />
								Classification
							</TabsTrigger>
							<TabsTrigger value="features" class="text-[11px] px-1">
								<ScatterChart class="mr-1 h-3 w-3" />
								Features
							</TabsTrigger>
							<TabsTrigger value="target" class="text-[11px] px-1">
								<Crosshair class="mr-1 h-3 w-3" />
								Target
							</TabsTrigger>
							<TabsTrigger value="diagnostics" class="text-[11px] px-1">
								<Settings2 class="mr-1 h-3 w-3" />
								Diagnostics
							</TabsTrigger>
						</TabsList>

						<!-- Overview Tab (cloned from Reflex transaction_fraud_detection_batch_metrics) -->
						<TabsContent value="overview" class="pt-4">
							<div class="space-y-4">
								{#if Object.keys(currentMetrics).length > 0}
									<!-- TRAINING DATA INFO -->
									{#if currentMetrics.n_samples > 0}
										<div class="flex items-center gap-2 rounded-md border border-blue-200 bg-blue-50 p-3 dark:border-blue-900 dark:bg-blue-950">
											<Database class="h-4 w-4 text-blue-600" />
											<span class="text-sm font-medium text-blue-700 dark:text-blue-300">Training Data:</span>
											<span class="rounded bg-blue-600 px-2 py-0.5 text-sm font-bold text-white">
												{(currentMetrics.n_samples as number)?.toLocaleString()} rows
											</span>
											<span class="text-xs text-blue-600 dark:text-blue-400">(80% train / 20% test split)</span>
										</div>
									{/if}

									<!-- PRIMARY METRICS (4) - Class-based, most important for fraud detection -->
									<div class="flex items-center gap-2">
										<Target class="h-4 w-4 text-blue-600" />
										<span class="text-sm font-bold">Primary Metrics</span>
										<span class="text-xs text-muted-foreground">(Class-based - most important for fraud detection)</span>
									</div>
									<div class="grid grid-cols-4 gap-2">
										<MetricCard name="Recall" value={currentMetrics.recall as number} onInfoClick={() => openMetricInfo('recall')} />
										<MetricCard name="Precision" value={currentMetrics.precision as number} onInfoClick={() => openMetricInfo('precision')} />
										<MetricCard name="F1 Score" value={currentMetrics.f1_score as number} onInfoClick={() => openMetricInfo('f1')} />
										<MetricCard name="F2 (β=2)" value={currentMetrics.fbeta_score as number} onInfoClick={() => openMetricInfo('fbeta')} />
									</div>

									<hr class="border-border" />

									<!-- RANKING METRICS (2) - Probability-based ranking ability -->
									<div class="flex items-center gap-2">
										<BarChart3 class="h-4 w-4 text-indigo-600" />
										<span class="text-sm font-bold">Ranking Metrics</span>
										<span class="text-xs text-muted-foreground">(Probability-based ranking ability)</span>
									</div>
									<div class="grid grid-cols-2 gap-2">
										<MetricCard name="ROC-AUC" value={currentMetrics.roc_auc as number} onInfoClick={() => openMetricInfo('rocauc')} />
										<MetricCard name="Avg Precision" value={currentMetrics.average_precision as number} onInfoClick={() => openMetricInfo('average_precision')} />
									</div>

									<hr class="border-border" />

									<!-- SECONDARY METRICS (6) - Gauges with additional monitoring insights -->
									<div class="flex items-center gap-2">
										<RefreshCw class="h-4 w-4 text-green-600" />
										<span class="text-sm font-bold">Secondary Metrics</span>
										<span class="text-xs text-muted-foreground">(Additional monitoring insights)</span>
									</div>
									<div class="grid grid-cols-6 gap-2">
										<MetricCard name="Accuracy" value={currentMetrics.accuracy as number} onInfoClick={() => openMetricInfo('accuracy')} />
										<MetricCard name="Balanced Acc" value={currentMetrics.balanced_accuracy as number} onInfoClick={() => openMetricInfo('balanced_accuracy')} />
										<MetricCard name="MCC" value={currentMetrics.mcc as number} onInfoClick={() => openMetricInfo('mcc')} />
										<MetricCard name="Cohen Kappa" value={currentMetrics.cohen_kappa as number} onInfoClick={() => openMetricInfo('cohen_kappa')} />
										<MetricCard name="Jaccard" value={currentMetrics.jaccard as number} onInfoClick={() => openMetricInfo('jaccard')} />
										<MetricCard name="G-Mean" value={currentMetrics.geometric_mean as number} onInfoClick={() => openMetricInfo('geometric_mean')} />
									</div>

									<hr class="border-border" />

									<!-- CALIBRATION METRICS (4) - Probability calibration quality -->
									<div class="flex items-center gap-2">
										<Settings class="h-4 w-4 text-purple-600" />
										<span class="text-sm font-bold">Calibration Metrics</span>
										<span class="text-xs text-muted-foreground">(Probability calibration quality)</span>
									</div>
									<div class="grid grid-cols-4 gap-2">
										<MetricCard name="Log Loss" value={currentMetrics.log_loss as number} onInfoClick={() => openMetricInfo('logloss')} />
										<MetricCard name="Brier Score" value={currentMetrics.brier as number} onInfoClick={() => openMetricInfo('brier')} />
										<MetricCard name="D² Log Loss" value={currentMetrics.d2_log_loss as number} onInfoClick={() => openMetricInfo('d2_logloss')} />
										<MetricCard name="D² Brier" value={currentMetrics.d2_brier as number} onInfoClick={() => openMetricInfo('d2_brier')} />
									</div>
								{:else}
									<div class="flex flex-col items-center justify-center py-12 text-center">
										<LayoutDashboard class="h-12 w-12 text-muted-foreground/50" />
										<p class="mt-4 text-sm text-muted-foreground">
											{runs.length === 0 ? 'Train a model first to see metrics' : 'Loading metrics...'}
										</p>
									</div>
								{/if}
							</div>
						</TabsContent>

						<!-- Classification Tab -->
						<TabsContent value="classification" class="pt-4">
							<div class="space-y-4">
								{#if modelAvailable}
									<p class="text-xs text-muted-foreground">Select a classification performance visualization.</p>
									<div class="flex items-center gap-2">
										<select
											class="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
											value={currentVisualizer}
											onchange={(e) => loadVisualizer('Classification', e.currentTarget.value)}
										>
											{#each YELLOWBRICK_CATEGORIES['Classification'] as viz}
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

						<!-- Feature Analysis Tab -->
						<TabsContent value="features" class="pt-4">
							<div class="space-y-4">
								{#if modelAvailable}
									<p class="text-xs text-muted-foreground">Select a feature analysis visualization.</p>
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

						<!-- Target Analysis Tab -->
						<TabsContent value="target" class="pt-4">
							<div class="space-y-4">
								{#if modelAvailable}
									<p class="text-xs text-muted-foreground">Select a target analysis visualization.</p>
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

						<!-- Model Diagnostics Tab -->
						<TabsContent value="diagnostics" class="pt-4">
							<div class="space-y-4">
								{#if modelAvailable}
									<p class="text-xs text-muted-foreground">Select a model diagnostics visualization.</p>
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
	<div class="flex min-h-[400px] items-center justify-center rounded-lg border border-border bg-muted/30">
		{#if isImageLoading}
			<!-- Loading state with Stop button (matching Reflex) -->
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
			<img src="data:image/png;base64,{currentImage}" alt={currentVisualizer} class="max-h-[500px] max-w-full" />
		{:else}
			<p class="text-sm text-muted-foreground">Select a visualizer</p>
		{/if}
	</div>
{/snippet}

{#snippet noModelWarning()}
	<div class="flex items-center gap-2 rounded-lg border border-blue-200 bg-blue-50 p-4 text-sm text-blue-700 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-300">
		<Info class="h-4 w-4" />
		<span>Train a model first to view visualizations. Use the Training box on the left and click Train.</span>
	</div>
{/snippet}

<!-- YellowBrick Info Dialog -->
{#if yellowBrickInfoOpen}
	<div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onclick={() => (yellowBrickInfoOpen = false)}>
		<div
			class="mx-4 max-h-[85vh] w-full max-w-lg overflow-y-auto rounded-lg bg-card p-6 shadow-xl"
			onclick={(e) => e.stopPropagation()}
		>
			<!-- Header -->
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
				<!-- Category Badge -->
				<span class="inline-block rounded-full bg-purple-100 px-2.5 py-0.5 text-xs font-medium text-purple-700 dark:bg-purple-900 dark:text-purple-300">
					{yellowBrickInfoContent.category}
				</span>

				<!-- What it shows -->
				<div class="rounded-lg bg-muted/50 p-3">
					<div class="mb-1 flex items-center gap-1.5">
						<span class="text-sm">👁️</span>
						<span class="text-sm font-semibold">What it shows</span>
					</div>
					<p class="text-sm text-muted-foreground">{yellowBrickInfoContent.description}</p>
				</div>

				<!-- How to read it -->
				<div>
					<div class="mb-1 flex items-center gap-1.5">
						<span class="text-sm">🔍</span>
						<span class="text-sm font-semibold">How to read it</span>
					</div>
					<p class="text-sm text-muted-foreground">{@html yellowBrickInfoContent.interpretation}</p>
				</div>

				<!-- In Fraud Detection -->
				{#if yellowBrickInfoContent.context}
					<div class="rounded-lg bg-blue-50 p-3 dark:bg-blue-950">
						<div class="mb-1 flex items-center gap-1.5">
							<ShieldAlert class="h-4 w-4 text-blue-600" />
							<span class="text-sm font-semibold text-blue-700 dark:text-blue-300">In Fraud Detection</span>
						</div>
						<p class="text-sm text-blue-600 dark:text-blue-400">{@html yellowBrickInfoContent.context}</p>
					</div>
				{/if}

				<!-- When to use -->
				{#if yellowBrickInfoContent.whenToUse}
					<div>
						<div class="mb-1 flex items-center gap-1.5">
							<span class="text-sm">💡</span>
							<span class="text-sm font-semibold">When to use</span>
						</div>
						<p class="text-sm text-muted-foreground">{yellowBrickInfoContent.whenToUse}</p>
					</div>
				{/if}

				<!-- Documentation link -->
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
