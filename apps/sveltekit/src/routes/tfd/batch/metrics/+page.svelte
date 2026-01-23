<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		Card,
		CardHeader,
		CardTitle,
		CardContent,
		Button,
		Select,
		FormField,
		MetricCard,
		BatchTrainingBox,
		BatchSubNav,
		Tabs,
		TabsList,
		TabsTrigger,
		TabsContent
	} from '$components/shared';
	import {
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
		updateProjectStore
	} from '$stores';
	import { toast } from '$stores/ui';
	import { metricInfoDialogOpen, metricInfoDialogContent } from '$stores';
	import * as batchApi from '$api/batch';
	import {
		RefreshCw,
		ExternalLink,
		LayoutDashboard,
		CheckCircle,
		Lightbulb,
		ScatterChart,
		Crosshair,
		Settings2,
		Info
	} from 'lucide-svelte';
	import type { ProjectName } from '$types';

	const PROJECT: ProjectName = 'Transaction Fraud Detection';
	const MODEL_NAME = 'CatBoost Classifier';

	// Transform MLflow metrics from API format (metrics.fbeta_score) to simple format (fbeta_score)
	function transformMetrics(rawMetrics: Record<string, unknown>): Record<string, number> {
		return {
			fbeta_score: Number(rawMetrics['metrics.fbeta_score'] ?? rawMetrics['metrics.FBeta'] ?? 0),
			roc_auc: Number(rawMetrics['metrics.roc_auc'] ?? rawMetrics['metrics.ROCAUC'] ?? 0),
			precision: Number(rawMetrics['metrics.precision'] ?? rawMetrics['metrics.Precision'] ?? 0),
			recall: Number(rawMetrics['metrics.recall'] ?? rawMetrics['metrics.Recall'] ?? 0),
			accuracy: Number(rawMetrics['metrics.accuracy'] ?? rawMetrics['metrics.Accuracy'] ?? 0),
			f1_score: Number(rawMetrics['metrics.f1_score'] ?? rawMetrics['metrics.F1'] ?? 0),
			log_loss: Number(rawMetrics['metrics.log_loss'] ?? rawMetrics['metrics.LogLoss'] ?? 0),
			n_samples: Number(rawMetrics['metrics.n_samples'] ?? rawMetrics['metrics.training_samples'] ?? 0)
		};
	}

	// YellowBrick visualizer options organized by category
	const YELLOWBRICK_CATEGORIES = {
		Classification: [
			{ value: 'ConfusionMatrix', label: 'Confusion Matrix' },
			{ value: 'ROCAUC', label: 'ROC AUC Curve' },
			{ value: 'PrecisionRecallCurve', label: 'Precision-Recall Curve' },
			{ value: 'ClassificationReport', label: 'Classification Report' },
			{ value: 'ClassPredictionError', label: 'Class Prediction Error' },
			{ value: 'DiscriminationThreshold', label: 'Discrimination Threshold' }
		],
		'Feature Analysis': [
			{ value: 'FeatureCorrelation', label: 'Feature Correlation' },
			{ value: 'Rank2D', label: 'Rank 2D' },
			{ value: 'PCA', label: 'PCA Decomposition' },
			{ value: 'RadViz', label: 'RadViz' }
		],
		Target: [
			{ value: 'ClassBalance', label: 'Class Balance' },
			{ value: 'FeatureTargetCorrelation', label: 'Feature-Target Correlation' }
		],
		'Model Selection': [
			{ value: 'LearningCurve', label: 'Learning Curve' },
			{ value: 'ValidationCurve', label: 'Validation Curve' },
			{ value: 'CVScores', label: 'Cross-Validation Scores' }
		]
	};

	let activeTab = $state('overview');
	let activeCategory = $state('Classification');
	let metricsLoading = $state(false);
	let visualizerInfo = $state<Record<string, unknown>>({});
	let statusInterval: ReturnType<typeof setInterval> | null = null;

	onMount(async () => {
		// Load YellowBrick info
		try {
			const response = await fetch('/data/yellowbrick_info_tfd.json');
			visualizerInfo = await response.json();
		} catch (e) {
			console.error('Failed to load visualizer info:', e);
		}

		// Get Delta Lake row count
		const countResult = await batchApi.getDeltaLakeRowCount(PROJECT);
		if (countResult.data?.row_count) {
			updateProjectStore(batchDeltaLakeTotalRows, PROJECT, countResult.data.row_count);
		}

		// Load MLflow runs if not already loaded
		if (!$batchMlflowRuns[PROJECT]?.length) {
			await loadRuns();
		}

		// Load initial metrics
		await loadMetrics();

		// Load initial visualizer
		if (!$selectedYellowBrickVisualizer[PROJECT]) {
			updateProjectStore(selectedYellowBrickVisualizer, PROJECT, 'ConfusionMatrix');
		}
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

	async function loadMetrics() {
		metricsLoading = true;
		const runId = $selectedBatchRun[PROJECT] || undefined;
		const result = await batchApi.getMLflowMetrics(PROJECT, runId);
		if (result.data && !result.data._no_runs) {
			// Transform metrics from API format (metrics.fbeta_score) to simple format (fbeta_score)
			const transformed = transformMetrics(result.data);
			updateProjectStore(batchMlflowMetrics, PROJECT, transformed);
		}
		metricsLoading = false;
	}

	async function loadVisualizer(visualizerName: string) {
		updateProjectStore(selectedYellowBrickVisualizer, PROJECT, visualizerName);
		updateProjectStore(yellowBrickLoading, PROJECT, { [visualizerName]: true });

		const runId = $selectedBatchRun[PROJECT] || undefined;
		const result = await batchApi.getYellowBrickImage(PROJECT, visualizerName, runId);

		if (result.data?.image_base64) {
			yellowBrickImages.update((imgs) => ({
				...imgs,
				[PROJECT]: {
					...imgs[PROJECT],
					[visualizerName]: result.data!.image_base64
				}
			}));
		} else if (result.error) {
			toast.error(`Failed to load ${visualizerName}`);
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

	async function onRunChange(runId: string) {
		updateProjectStore(selectedBatchRun, PROJECT, runId);
		await loadMetrics();
		const currentViz = $selectedYellowBrickVisualizer[PROJECT];
		if (currentViz) {
			await loadVisualizer(currentViz);
		}
	}

	// Training functions
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
				await loadMetrics();
			}
		}
	}

	const currentMetrics = $derived($batchMlflowMetrics[PROJECT] || {});
	const runs = $derived($batchMlflowRuns[PROJECT] || []);
	const currentVisualizer = $derived($selectedYellowBrickVisualizer[PROJECT] || 'ConfusionMatrix');
	const currentImage = $derived($yellowBrickImages[PROJECT]?.[currentVisualizer] || '');
	const isImageLoading = $derived($yellowBrickLoading[PROJECT]?.[currentVisualizer] || false);
	const isTraining = $derived($batchTrainingLoading[PROJECT]);
	const modelAvailable = $derived($batchModelAvailable[PROJECT]);
</script>

<!-- 40%/60% Layout -->
<div class="flex gap-6">
	<!-- Left Column - Training Box Only (40%) -->
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
	</div>

	<!-- Right Column - Batch Sub-Nav + Metrics Tabs (60%) -->
	<div class="w-[60%] space-y-4">
		<!-- Batch Sub-Nav -->
		<BatchSubNav projectKey="tfd" />

		<!-- Metrics Tabs -->
		<Tabs bind:value={activeTab} class="w-full">
			<TabsList class="grid w-full grid-cols-6">
				<TabsTrigger value="overview">Overview</TabsTrigger>
				<TabsTrigger value="classification">Classification</TabsTrigger>
				<TabsTrigger value="features">Feature Analysis</TabsTrigger>
				<TabsTrigger value="target">Target Analysis</TabsTrigger>
				<TabsTrigger value="diagnostics">Model Diagnostics</TabsTrigger>
				<TabsTrigger value="explainability">Explainability</TabsTrigger>
			</TabsList>

			<!-- Overview Tab -->
			<TabsContent value="overview" class="pt-4">
				<div class="space-y-4">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<LayoutDashboard class="h-5 w-5 text-primary" />
							<h2 class="text-xl font-bold">Classification Metrics Overview</h2>
						</div>
						<Button variant="ghost" size="sm" onclick={loadMetrics} loading={metricsLoading}>
							<RefreshCw class="h-4 w-4" />
						</Button>
					</div>

					{#if Object.keys(currentMetrics).length > 0}
						<div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
							<MetricCard
								name="F-Beta Score"
								value={currentMetrics.fbeta_score as number}
								onInfoClick={() => openMetricInfo('fbeta_score')}
							/>
							<MetricCard
								name="ROC AUC"
								value={currentMetrics.roc_auc as number}
								onInfoClick={() => openMetricInfo('roc_auc')}
							/>
							<MetricCard
								name="Precision"
								value={currentMetrics.precision as number}
								onInfoClick={() => openMetricInfo('precision')}
							/>
							<MetricCard
								name="Recall"
								value={currentMetrics.recall as number}
								onInfoClick={() => openMetricInfo('recall')}
							/>
							<MetricCard
								name="Accuracy"
								value={currentMetrics.accuracy as number}
								onInfoClick={() => openMetricInfo('accuracy')}
							/>
							<MetricCard
								name="F1 Score"
								value={currentMetrics.f1_score as number}
								onInfoClick={() => openMetricInfo('f1_score')}
							/>
							<MetricCard
								name="Log Loss"
								value={currentMetrics.log_loss as number}
								onInfoClick={() => openMetricInfo('log_loss')}
							/>
							<MetricCard name="Training Rows" value={currentMetrics.n_samples as number} decimals={0} />
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
					<div class="flex items-center gap-2">
						<CheckCircle class="h-5 w-5 text-primary" />
						<h2 class="text-xl font-bold">Classification</h2>
					</div>
					{#if modelAvailable}
						<p class="text-sm text-muted-foreground">
							Select a classification performance visualization.
						</p>
						<div class="flex items-center gap-2">
							<Select
								value={currentVisualizer}
								options={YELLOWBRICK_CATEGORIES['Classification'].map((v) => ({
									value: v.value,
									label: v.label
								}))}
								onchange={(e) => loadVisualizer(e.currentTarget.value)}
								class="flex-1"
							/>
							<Button variant="ghost" size="sm" title="Visualization info">
								<Info class="h-4 w-4" />
							</Button>
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
					<div class="flex items-center gap-2">
						<ScatterChart class="h-5 w-5 text-primary" />
						<h2 class="text-xl font-bold">Feature Analysis</h2>
					</div>
					{#if modelAvailable}
						<p class="text-sm text-muted-foreground">Select a feature analysis visualization.</p>
						<Select
							value={currentVisualizer}
							options={YELLOWBRICK_CATEGORIES['Feature Analysis'].map((v) => ({
								value: v.value,
								label: v.label
							}))}
							onchange={(e) => loadVisualizer(e.currentTarget.value)}
						/>
						{@render visualizationDisplay()}
					{:else}
						{@render noModelWarning()}
					{/if}
				</div>
			</TabsContent>

			<!-- Target Analysis Tab -->
			<TabsContent value="target" class="pt-4">
				<div class="space-y-4">
					<div class="flex items-center gap-2">
						<Crosshair class="h-5 w-5 text-primary" />
						<h2 class="text-xl font-bold">Target Analysis</h2>
					</div>
					{#if modelAvailable}
						<p class="text-sm text-muted-foreground">Select a target analysis visualization.</p>
						<Select
							value={currentVisualizer}
							options={YELLOWBRICK_CATEGORIES['Target'].map((v) => ({
								value: v.value,
								label: v.label
							}))}
							onchange={(e) => loadVisualizer(e.currentTarget.value)}
						/>
						{@render visualizationDisplay()}
					{:else}
						{@render noModelWarning()}
					{/if}
				</div>
			</TabsContent>

			<!-- Model Diagnostics Tab -->
			<TabsContent value="diagnostics" class="pt-4">
				<div class="space-y-4">
					<div class="flex items-center gap-2">
						<Settings2 class="h-5 w-5 text-primary" />
						<h2 class="text-xl font-bold">Model Diagnostics</h2>
					</div>
					{#if modelAvailable}
						<p class="text-sm text-muted-foreground">
							Select a model diagnostics visualization.
						</p>
						<Select
							value={currentVisualizer}
							options={YELLOWBRICK_CATEGORIES['Model Selection'].map((v) => ({
								value: v.value,
								label: v.label
							}))}
							onchange={(e) => loadVisualizer(e.currentTarget.value)}
						/>
						{@render visualizationDisplay()}
					{:else}
						{@render noModelWarning()}
					{/if}
				</div>
			</TabsContent>

			<!-- Explainability Tab -->
			<TabsContent value="explainability" class="pt-4">
				<div class="space-y-4">
					<div class="flex items-center gap-2">
						<Lightbulb class="h-5 w-5 text-primary" />
						<h2 class="text-xl font-bold">Explainability</h2>
					</div>
					<div
						class="flex items-center gap-2 rounded-lg border border-blue-200 bg-blue-50 p-4 text-sm text-blue-700 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-300"
					>
						<Info class="h-4 w-4" />
						<span>
							SHAP (SHapley Additive exPlanations) visualizations coming soon. This will include
							feature importance, summary plots, and individual prediction explanations.
						</span>
					</div>
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
			<div class="text-center">
				<div
					class="mx-auto h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"
				></div>
				<p class="mt-2 text-sm text-muted-foreground">Loading visualization...</p>
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
		<span>Train a model first to view visualizations. Go to the Prediction page and click Train.</span>
	</div>
{/snippet}
