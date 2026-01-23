<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		Card,
		CardContent,
		Button,
		Select,
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
		LayoutDashboard,
		TrendingUp,
		ScatterChart,
		Crosshair,
		Settings2,
		Lightbulb,
		Info
	} from 'lucide-svelte';
	import type { ProjectName } from '$types';

	const PROJECT: ProjectName = 'Estimated Time of Arrival';
	const MODEL_NAME = 'CatBoost Regressor';

	// Transform MLflow metrics from API format (metrics.mae) to simple format (mae)
	function transformMetrics(rawMetrics: Record<string, unknown>): Record<string, number> {
		return {
			mae: Number(rawMetrics['metrics.mae'] ?? rawMetrics['metrics.MAE'] ?? 0),
			rmse: Number(rawMetrics['metrics.rmse'] ?? rawMetrics['metrics.RMSE'] ?? 0),
			r2_score: Number(rawMetrics['metrics.r2_score'] ?? rawMetrics['metrics.R2'] ?? 0),
			mse: Number(rawMetrics['metrics.mse'] ?? rawMetrics['metrics.MSE'] ?? 0),
			mape: Number(rawMetrics['metrics.mape'] ?? rawMetrics['metrics.MAPE'] ?? 0),
			n_samples: Number(rawMetrics['metrics.n_samples'] ?? rawMetrics['metrics.training_samples'] ?? 0)
		};
	}

	// YellowBrick visualizer options organized by category (regression)
	const YELLOWBRICK_CATEGORIES = {
		Regression: [
			{ value: 'ResidualsPlot', label: 'Residuals Plot' },
			{ value: 'PredictionError', label: 'Prediction Error' },
			{ value: 'CooksDistance', label: 'Cooks Distance' }
		],
		'Feature Analysis': [
			{ value: 'FeatureCorrelation', label: 'Feature Correlation' },
			{ value: 'Rank2D', label: 'Rank 2D' },
			{ value: 'PCA', label: 'PCA Decomposition' },
			{ value: 'FeatureImportances', label: 'Feature Importances' }
		],
		Target: [
			{ value: 'TargetDistribution', label: 'Target Distribution' },
			{ value: 'FeatureTargetCorrelation', label: 'Feature-Target Correlation' }
		],
		'Model Selection': [
			{ value: 'LearningCurve', label: 'Learning Curve' },
			{ value: 'ValidationCurve', label: 'Validation Curve' },
			{ value: 'CVScores', label: 'Cross-Validation Scores' }
		]
	};

	let activeTab = $state('overview');
	let metricsLoading = $state(false);
	let statusInterval: ReturnType<typeof setInterval> | null = null;

	onMount(async () => {
		const countResult = await batchApi.getDeltaLakeRowCount(PROJECT);
		if (countResult.data?.row_count) {
			updateProjectStore(batchDeltaLakeTotalRows, PROJECT, countResult.data.row_count);
		}

		if (!$batchMlflowRuns[PROJECT]?.length) {
			await loadRuns();
		}

		await loadMetrics();

		if (!$selectedYellowBrickVisualizer[PROJECT]) {
			updateProjectStore(selectedYellowBrickVisualizer, PROJECT, 'ResidualsPlot');
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
			// Transform metrics from API format (metrics.mae) to simple format (mae)
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
	const currentVisualizer = $derived($selectedYellowBrickVisualizer[PROJECT] || 'ResidualsPlot');
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
		<BatchSubNav projectKey="eta" />

		<!-- Metrics Tabs -->
		<Tabs bind:value={activeTab} class="w-full">
			<TabsList class="grid w-full grid-cols-6">
				<TabsTrigger value="overview">Overview</TabsTrigger>
				<TabsTrigger value="regression">Regression</TabsTrigger>
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
							<h2 class="text-xl font-bold">Regression Metrics Overview</h2>
						</div>
						<Button variant="ghost" size="sm" onclick={loadMetrics} loading={metricsLoading}>
							<RefreshCw class="h-4 w-4" />
						</Button>
					</div>

					{#if Object.keys(currentMetrics).length > 0}
						<div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
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
								name="R² Score"
								value={currentMetrics.r2_score as number}
								onInfoClick={() => openMetricInfo('r2_score')}
							/>
							<MetricCard
								name="MSE"
								value={currentMetrics.mse as number}
								onInfoClick={() => openMetricInfo('mse')}
							/>
							<MetricCard
								name="MAPE"
								value={currentMetrics.mape as number}
								onInfoClick={() => openMetricInfo('mape')}
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

			<!-- Regression Tab -->
			<TabsContent value="regression" class="pt-4">
				<div class="space-y-4">
					<div class="flex items-center gap-2">
						<TrendingUp class="h-5 w-5 text-primary" />
						<h2 class="text-xl font-bold">Regression</h2>
					</div>
					{#if modelAvailable}
						<p class="text-sm text-muted-foreground">
							Select a regression performance visualization.
						</p>
						<Select
							value={currentVisualizer}
							options={YELLOWBRICK_CATEGORIES['Regression'].map((v) => ({
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
