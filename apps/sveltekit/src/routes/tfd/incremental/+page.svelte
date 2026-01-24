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
	import { randomizeTFDForm, FIELD_CONFIG, clampFieldValue } from '$lib/utils/randomize';
	import {
		Shuffle,
		ShieldAlert,
		CheckCircle,
		CreditCard,
		Target,
		BarChart3,
		FlaskConical,
		RefreshCw
	} from 'lucide-svelte';
	import Plotly from 'svelte-plotly.js';
	import type { DropdownOptions, ProjectName } from '$types';

	const PROJECT: ProjectName = 'Transaction Fraud Detection';
	const MODEL_NAME = 'Adaptive Random Forest Classifier (River)';

	// Transform MLflow metrics from API format (metrics.FBeta) to simple format (fbeta)
	function transformMetrics(rawMetrics: Record<string, unknown>): Record<string, number> {
		return {
			fbeta: Number(rawMetrics['metrics.FBeta'] ?? rawMetrics['metrics.fbeta'] ?? 0),
			rocauc: Number(rawMetrics['metrics.ROCAUC'] ?? rawMetrics['metrics.rocauc'] ?? 0),
			precision: Number(rawMetrics['metrics.Precision'] ?? rawMetrics['metrics.precision'] ?? 0),
			recall: Number(rawMetrics['metrics.Recall'] ?? rawMetrics['metrics.recall'] ?? 0),
			rolling_rocauc: Number(rawMetrics['metrics.RollingROCAUC'] ?? rawMetrics['metrics.rolling_rocauc'] ?? 0),
			f1: Number(rawMetrics['metrics.F1'] ?? rawMetrics['metrics.f1'] ?? 0),
			accuracy: Number(rawMetrics['metrics.Accuracy'] ?? rawMetrics['metrics.accuracy'] ?? 0),
			geometric_mean: Number(rawMetrics['metrics.GeometricMean'] ?? rawMetrics['metrics.geometric_mean'] ?? 0),
			cohen_kappa: Number(rawMetrics['metrics.CohenKappa'] ?? rawMetrics['metrics.cohen_kappa'] ?? 0),
			jaccard: Number(rawMetrics['metrics.Jaccard'] ?? rawMetrics['metrics.jaccard'] ?? 0),
			logloss: Number(rawMetrics['metrics.LogLoss'] ?? rawMetrics['metrics.logloss'] ?? 0),
			mcc: Number(rawMetrics['metrics.MCC'] ?? rawMetrics['metrics.mcc'] ?? 0),
			balanced_accuracy: Number(rawMetrics['metrics.BalancedAccuracy'] ?? rawMetrics['metrics.balanced_accuracy'] ?? 0)
		};
	}

	// Local state
	let trainingLoading = $state(false);
	let sampleLoading = $state(false);
	let metricsInterval: ReturnType<typeof setInterval> | null = null;
	let activeTab = $state('prediction');
	let reportMetrics = $state<{
		confusion_matrix?: {
			available: boolean;
			tn?: number;
			fp?: number;
			fn?: number;
			tp?: number;
			total?: number;
			error?: string;
		};
		classification_report?: {
			available: boolean;
			report?: string;
			error?: string;
		};
	}>({});

	// Load dropdown options from static JSON
	let dropdownOptions = $state<DropdownOptions>({});

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

		// Check training status
		await checkTrainingStatus();
		await fetchMetrics();

		// Start metrics polling if training is active
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
				mlflowRunInfo = {
					...mlflowRunInfo,
					is_live: true,
					status: 'RUNNING'
				};
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
			// Transform metrics from API format (metrics.FBeta) to simple format (fbeta)
			const transformed = transformMetrics(result.data);
			updateProjectStore(mlflowMetrics, PROJECT, transformed);
		}
		const reportResult = await incrementalApi.getReportMetrics(PROJECT);
		if (reportResult.data) {
			reportMetrics = reportResult.data;
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
				// Check if any training is already active (global training lock)
				const statusResult = await incrementalApi.getTrainingStatus();
				if (statusResult.data?.is_active && statusResult.data?.project_name !== PROJECT) {
					toast.error(
						`Cannot start training: ${statusResult.data.project_name} training is already running. Stop it first.`
					);
					return;
				}

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

	function loadRandomSample() {
		sampleLoading = true;
		try {
			const randomData = randomizeTFDForm(dropdownOptions);
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
				const sourceLabel = result.data.model_source
					? result.data.model_source.toUpperCase()
					: 'MLFLOW';
				if (result.data.prediction === 1 || (result.data.fraud_probability ?? 0) >= 0.5) {
					toast.warning(`Transaction flagged as potentially fraudulent (${sourceLabel})`);
				} else {
					toast.success(`Transaction appears legitimate (${sourceLabel})`);
				}
				await fetchMetrics();
			}
		} finally {
			updateProjectStore(predictionLoading, PROJECT, false);
		}
	}

	function openMetricInfo(metricKey: string) {
		fetch('/data/incremental_metric_info_tfd.json')
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
	const hasPrediction = $derived(
		currentPrediction && Object.keys(currentPrediction).length > 0
	);

	// Derived values for prediction display
	const fraudProbability = $derived(currentPrediction?.fraud_probability ?? 0);
	const isFraud = $derived(
		currentPrediction?.prediction === 1 || (currentPrediction?.fraud_probability ?? 0) >= 0.5
	);
	const predictionText = $derived(isFraud ? 'FRAUD' : 'NOT FRAUD');
	const predictionColor = $derived(isFraud ? 'red' : 'green');

	// Risk level based on probability (matching Reflex implementation)
	const riskLevel = $derived.by(() => {
		const prob = fraudProbability * 100;
		if (prob < 30) return { text: 'LOW RISK', color: '#22c55e' }; // green
		if (prob < 70) return { text: 'MEDIUM RISK', color: '#eab308' }; // yellow
		return { text: 'HIGH RISK', color: '#ef4444' }; // red
	});

	const gaugeData = $derived.by(() => {
		return [
			{
				type: 'indicator',
				mode: 'gauge+number',
				value: Number((fraudProbability * 100).toFixed(2)),
				number: {
					suffix: '%',
					font: {
						size: 40,
						color: riskLevel.color
					}
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
					bar: {
						color: riskLevel.color,
						thickness: 0.75
					},
					bgcolor: 'rgba(0,0,0,0)',
					borderwidth: 2,
					bordercolor: '#333',
					steps: [
						{ range: [0, 30], color: 'rgba(34, 197, 94, 0.3)' }, // green zone
						{ range: [30, 70], color: 'rgba(234, 179, 8, 0.3)' }, // yellow zone
						{ range: [70, 100], color: 'rgba(239, 68, 68, 0.3)' } // red zone
					],
					threshold: {
						line: { color: '#333', width: 4 },
						thickness: 0.8,
						value: fraudProbability * 100
					}
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

	// MCC Gauge (range -1 to 1)
	const mccGaugeData = $derived.by(() => {
		const value = currentMetrics?.mcc ?? 0;
		return [
			{
				type: 'indicator',
				mode: 'gauge+number',
				value: value,
				number: { valueformat: '.3f', font: { size: 14 } },
				domain: { x: [0.1, 0.9], y: [0.15, 0.85] },
				gauge: {
					axis: { range: [-1, 1], tickwidth: 1, tickfont: { size: 8 }, nticks: 5 },
					bar: { color: '#1e40af', thickness: 0.5 },
					borderwidth: 1,
					steps: [
						{ range: [-1, 0], color: 'rgba(239, 68, 68, 0.3)' },
						{ range: [0, 0.4], color: 'rgba(234, 179, 8, 0.3)' },
						{ range: [0.4, 0.6], color: 'rgba(34, 197, 94, 0.3)' },
						{ range: [0.6, 1], color: 'rgba(59, 130, 246, 0.3)' }
					],
					threshold: { value: 0.5, line: { color: 'black', width: 1 }, thickness: 0.6 }
				}
			}
		];
	});

	// Balanced Accuracy Gauge (range 0 to 1)
	const balancedAccGaugeData = $derived.by(() => {
		const value = currentMetrics?.balanced_accuracy ?? 0;
		return [
			{
				type: 'indicator',
				mode: 'gauge+number',
				value: value,
				number: { valueformat: '.3f', font: { size: 14 } },
				domain: { x: [0.1, 0.9], y: [0.15, 0.85] },
				gauge: {
					axis: { range: [0, 1], tickwidth: 1, tickfont: { size: 8 }, nticks: 5 },
					bar: { color: '#1e40af', thickness: 0.5 },
					borderwidth: 1,
					steps: [
						{ range: [0, 0.5], color: 'rgba(239, 68, 68, 0.3)' },
						{ range: [0.5, 0.7], color: 'rgba(234, 179, 8, 0.3)' },
						{ range: [0.7, 0.85], color: 'rgba(34, 197, 94, 0.3)' },
						{ range: [0.85, 1], color: 'rgba(59, 130, 246, 0.3)' }
					],
					threshold: { value: 0.8, line: { color: 'black', width: 1 }, thickness: 0.6 }
				}
			}
		];
	});

	const metricGaugeLayout = {
		width: 120,
		height: 70,
		margin: { l: 5, r: 5, t: 5, b: 5 },
		paper_bgcolor: 'transparent',
		plot_bgcolor: 'transparent'
	};

	// Confusion Matrix Heatmap
	const confusionMatrixData = $derived.by(() => {
		const cm = reportMetrics.confusion_matrix;
		if (!cm?.available) {
			return [];
		}
		const tn = cm.tn ?? 0;
		const fp = cm.fp ?? 0;
		const fn = cm.fn ?? 0;
		const tp = cm.tp ?? 0;
		return [
			{
				type: 'heatmap',
				z: [[tn, fp], [fn, tp]],
				x: ['Pred: 0', 'Pred: 1'],
				y: ['Actual: 0', 'Actual: 1'],
				colorscale: 'Blues',
				text: [[`TN<br>${tn.toLocaleString()}`, `FP<br>${fp.toLocaleString()}`], [`FN<br>${fn.toLocaleString()}`, `TP<br>${tp.toLocaleString()}`]],
				texttemplate: '%{text}',
				textfont: { size: 12 },
				showscale: false
			}
		];
	});

	const confusionMatrixLayout = {
		width: 140,
		height: 100,
		margin: { l: 35, r: 5, t: 5, b: 20 },
		paper_bgcolor: 'transparent',
		plot_bgcolor: 'transparent',
		yaxis: { autorange: 'reversed', tickfont: { size: 8 } },
		xaxis: { tickfont: { size: 8 } }
	};

	// Classification Report Heatmap
	const classificationReportData = $derived.by(() => {
		const cm = reportMetrics.confusion_matrix;
		if (!cm?.available) {
			return [];
		}
		const tn = cm.tn ?? 0;
		const fp = cm.fp ?? 0;
		const fn = cm.fn ?? 0;
		const tp = cm.tp ?? 0;

		// Calculate per-class metrics
		const prec_0 = (tn + fn) > 0 ? tn / (tn + fn) : 0;
		const rec_0 = (tn + fp) > 0 ? tn / (tn + fp) : 0;
		const f1_0 = (prec_0 + rec_0) > 0 ? 2 * prec_0 * rec_0 / (prec_0 + rec_0) : 0;
		const support_0 = tn + fp;

		const prec_1 = (tp + fp) > 0 ? tp / (tp + fp) : 0;
		const rec_1 = (tp + fn) > 0 ? tp / (tp + fn) : 0;
		const f1_1 = (prec_1 + rec_1) > 0 ? 2 * prec_1 * rec_1 / (prec_1 + rec_1) : 0;
		const support_1 = tp + fn;

		return [
			{
				type: 'heatmap',
				z: [[prec_0, rec_0, f1_0], [prec_1, rec_1, f1_1]],
				x: ['Precision', 'Recall', 'F1'],
				y: [`0 (n=${support_0.toLocaleString()})`, `1 (n=${support_1.toLocaleString()})`],
				colorscale: 'YlOrRd',
				text: [[prec_0.toFixed(2), rec_0.toFixed(2), f1_0.toFixed(2)], [prec_1.toFixed(2), rec_1.toFixed(2), f1_1.toFixed(2)]],
				texttemplate: '%{text}',
				textfont: { size: 14, color: 'black' },
				showscale: true,
				zmin: 0,
				zmax: 1,
				colorbar: { len: 0.8, thickness: 10 }
			}
		];
	});

	const classificationReportLayout = {
		width: 280,
		height: 120,
		margin: { l: 55, r: 40, t: 10, b: 25 },
		paper_bgcolor: 'transparent',
		plot_bgcolor: 'transparent',
		yaxis: { autorange: 'reversed', tickfont: { size: 10 } },
		xaxis: { tickfont: { size: 10 } }
	};
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
					<CreditCard class="h-5 w-5 text-primary" />
					<h3 class="text-base font-bold">Transaction Details</h3>
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
						<p class="text-xs text-muted-foreground">Amount ({FIELD_CONFIG.amount.min}-{FIELD_CONFIG.amount.max})</p>
						<Input
							type="number"
							value={currentForm.amount ?? FIELD_CONFIG.amount.min}
							min={FIELD_CONFIG.amount.min}
							max={FIELD_CONFIG.amount.max}
							step={FIELD_CONFIG.amount.step}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.amount.min;
								const clamped = clampFieldValue('amount', val);
								updateFormField(PROJECT, 'amount', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Account Age ({FIELD_CONFIG.account_age_days.min}-{FIELD_CONFIG.account_age_days.max})</p>
						<Input
							type="number"
							value={currentForm.account_age_days ?? FIELD_CONFIG.account_age_days.min}
							min={FIELD_CONFIG.account_age_days.min}
							max={FIELD_CONFIG.account_age_days.max}
							step="1"
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseInt(e.currentTarget.value) || FIELD_CONFIG.account_age_days.min;
								const clamped = clampFieldValue('account_age_days', val);
								updateFormField(PROJECT, 'account_age_days', clamped);
								e.currentTarget.value = String(clamped);
							}}
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
						<p class="text-xs text-muted-foreground">Merchant ID ({FIELD_CONFIG.merchant_id.min}-{FIELD_CONFIG.merchant_id.max})</p>
						<Input
							type="number"
							value={(currentForm.merchant_id as number) ?? FIELD_CONFIG.merchant_id.min}
							min={FIELD_CONFIG.merchant_id.min}
							max={FIELD_CONFIG.merchant_id.max}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseInt(e.currentTarget.value) || FIELD_CONFIG.merchant_id.min;
								const clamped = clampFieldValue('merchant_id', val);
								updateFormField(PROJECT, 'merchant_id', clamped);
								e.currentTarget.value = String(clamped);
							}}
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
						<p class="text-xs text-muted-foreground">Latitude ({FIELD_CONFIG.lat.min}-{FIELD_CONFIG.lat.max})</p>
						<Input
							type="number"
							value={currentForm.lat ?? FIELD_CONFIG.lat.min}
							min={FIELD_CONFIG.lat.min}
							max={FIELD_CONFIG.lat.max}
							step={FIELD_CONFIG.lat.step}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.lat.min;
								const clamped = clampFieldValue('lat', val);
								updateFormField(PROJECT, 'lat', clamped);
								e.currentTarget.value = String(clamped);
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Longitude ({FIELD_CONFIG.lon.min}-{FIELD_CONFIG.lon.max})</p>
						<Input
							type="number"
							value={currentForm.lon ?? FIELD_CONFIG.lon.min}
							min={FIELD_CONFIG.lon.min}
							max={FIELD_CONFIG.lon.max}
							step={FIELD_CONFIG.lon.step}
							class="h-8 text-sm"
							oninput={(e) => {
								const val = parseFloat(e.currentTarget.value) || FIELD_CONFIG.lon.min;
								const clamped = clampFieldValue('lon', val);
								updateFormField(PROJECT, 'lon', clamped);
								e.currentTarget.value = String(clamped);
							}}
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
								onchange={(e) =>
									updateFormField(PROJECT, 'billing_address_match', e.currentTarget.checked)}
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

	<!-- RIGHT COLUMN (60%) - Tabs for Prediction and Metrics -->
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

			<!-- Prediction Tab -->
			<TabsContent value="prediction" class="mt-4">
				<div class="space-y-4 pt-4">
					<div class="flex items-center gap-2">
						<ShieldAlert class="h-5 w-5 text-primary" />
						<h2 class="text-xl font-bold">Prediction Result</h2>
					</div>

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
								<code
									class="rounded bg-muted px-1"
									title={mlflowRunInfo.run_id_full}
								>
									{mlflowRunInfo.run_id}
								</code>
							</span>
						{/if}
						{#if mlflowRunInfo.start_time}
							<span class="text-xs text-muted-foreground">Started: {mlflowRunInfo.start_time}</span>
						{/if}
					</div>

					<Card>
						<CardContent class="p-6">
							<div class="space-y-4">
						<div class="flex w-full items-center justify-center">
							<Plotly
								data={gaugeData}
								layout={{ ...gaugeLayout, width: 300 }}
								config={gaugeConfig}
							/>
						</div>

								<div class="grid grid-cols-3 gap-2">
									<Card>
										<CardContent class="p-3 text-center">
											<div class="flex items-center justify-center gap-1 text-xs text-muted-foreground">
												<ShieldAlert class="h-3 w-3" style="color: {predictionColor}" />
												Classification
											</div>
											<p class="mt-1 text-lg font-bold" style="color: {predictionColor}">
												{hasPrediction ? predictionText : '--'}
											</p>
										</CardContent>
									</Card>
									<Card>
										<CardContent class="p-3 text-center">
											<div class="flex items-center justify-center gap-1 text-xs text-muted-foreground">
												<span class="h-3 w-3 text-red-500">%</span>
												Fraud
											</div>
											<p class="mt-1 text-lg font-bold text-red-500">
												{hasPrediction ? `${(fraudProbability * 100).toFixed(2)}%` : '--'}
											</p>
										</CardContent>
									</Card>
									<Card>
										<CardContent class="p-3 text-center">
											<div class="flex items-center justify-center gap-1 text-xs text-muted-foreground">
												<CheckCircle class="h-3 w-3 text-green-500" />
												Not Fraud
											</div>
											<p class="mt-1 text-lg font-bold text-green-500">
												{hasPrediction ? `${((1 - fraudProbability) * 100).toFixed(2)}%` : '--'}
											</p>
										</CardContent>
									</Card>
								</div>
							</div>
						</CardContent>
					</Card>
				</div>
			</TabsContent>

			<!-- Metrics Tab -->
			<TabsContent value="metrics" class="mt-4 space-y-4">
				<div class="flex items-center justify-between">
					<h2 class="text-lg font-bold">Classification Metrics</h2>
					<Button variant="ghost" size="sm" onclick={() => fetchMetrics(true)}>
						<RefreshCw class="h-4 w-4" />
					</Button>
				</div>

				<!-- MLflow Run Info Badge (same as Prediction tab) -->
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
							<code
								class="rounded bg-muted px-1"
								title={mlflowRunInfo.run_id_full}
							>
								{mlflowRunInfo.run_id}
							</code>
						</span>
					{/if}
					{#if mlflowRunInfo.start_time}
						<span class="text-xs text-muted-foreground">Started: {mlflowRunInfo.start_time}</span>
					{/if}
				</div>

				{#if Object.keys(currentMetrics || {}).length > 0}
					<!-- ROW 1: Primary KPIs -->
					<div class="grid grid-cols-5 gap-2">
						<MetricCard
							name="F-Beta"
							value={currentMetrics.fbeta ?? currentMetrics.fbeta_score}
							onInfoClick={() => openMetricInfo('fbeta')}
						/>
						<MetricCard
							name="ROC AUC"
							value={currentMetrics.rocauc ?? currentMetrics.roc_auc}
							onInfoClick={() => openMetricInfo('rocauc')}
						/>
						<MetricCard
							name="Precision"
							value={currentMetrics.precision}
							onInfoClick={() => openMetricInfo('precision')}
						/>
						<MetricCard
							name="Recall"
							value={currentMetrics.recall}
							onInfoClick={() => openMetricInfo('recall')}
						/>
						<MetricCard
							name="Rolling ROC"
							value={currentMetrics.rolling_rocauc ?? currentMetrics.rolling_roc_auc}
							onInfoClick={() => openMetricInfo('rolling_rocauc')}
						/>
					</div>

					<!-- ROW 2: Secondary metrics -->
					<div class="grid grid-cols-6 gap-2">
						<MetricCard
							name="F1"
							value={currentMetrics.f1}
							onInfoClick={() => openMetricInfo('f1')}
						/>
						<MetricCard
							name="Accuracy"
							value={currentMetrics.accuracy}
							onInfoClick={() => openMetricInfo('accuracy')}
						/>
						<MetricCard
							name="Geo Mean"
							value={currentMetrics.geometric_mean}
							onInfoClick={() => openMetricInfo('geometric_mean')}
						/>
						<MetricCard
							name="Cohen k"
							value={currentMetrics.cohen_kappa}
							onInfoClick={() => openMetricInfo('cohen_kappa')}
						/>
						<MetricCard
							name="Jaccard"
							value={currentMetrics.jaccard}
							onInfoClick={() => openMetricInfo('jaccard')}
						/>
						<MetricCard
							name="LogLoss"
							value={currentMetrics.logloss}
							onInfoClick={() => openMetricInfo('logloss')}
						/>
					</div>

					<!-- ROW 3: Gauges (Plotly) -->
					<div class="grid grid-cols-2 gap-2">
						<Card class="overflow-hidden">
							<CardContent class="p-2 overflow-hidden">
								<div class="flex items-center justify-between mb-1">
									<span class="text-xs font-medium text-muted-foreground">MCC</span>
									<button
										class="rounded-full p-0.5 text-muted-foreground hover:bg-blue-500/10 hover:text-blue-600"
										onclick={() => openMetricInfo('mcc')}
										title="More info"
									>
										<span class="text-xs">ⓘ</span>
									</button>
								</div>
								<div class="flex w-full items-center justify-center">
									<Plotly
										data={mccGaugeData}
										layout={metricGaugeLayout}
										config={gaugeConfig}
									/>
								</div>
							</CardContent>
						</Card>
						<Card class="overflow-hidden">
							<CardContent class="p-2 overflow-hidden">
								<div class="flex items-center justify-between mb-1">
									<span class="text-xs font-medium text-muted-foreground">Balanced Accuracy</span>
									<button
										class="rounded-full p-0.5 text-muted-foreground hover:bg-blue-500/10 hover:text-blue-600"
										onclick={() => openMetricInfo('balanced_accuracy')}
										title="More info"
									>
										<span class="text-xs">ⓘ</span>
									</button>
								</div>
								<div class="flex w-full items-center justify-center">
									<Plotly
										data={balancedAccGaugeData}
										layout={metricGaugeLayout}
										config={gaugeConfig}
									/>
								</div>
							</CardContent>
						</Card>
					</div>

					<!-- ROW 4: Confusion Matrix + Classification Report (Plotly Heatmaps) -->
					<div class="grid grid-cols-2 gap-2">
						<Card class="overflow-hidden">
							<CardContent class="p-2 overflow-hidden">
								<div class="flex items-center justify-between mb-1">
									<span class="text-xs font-medium text-muted-foreground">Confusion Matrix</span>
									<button
										class="rounded-full p-0.5 text-muted-foreground hover:bg-blue-500/10 hover:text-blue-600"
										onclick={() => openMetricInfo('confusion_matrix')}
										title="More info"
									>
										<span class="text-xs">ⓘ</span>
									</button>
								</div>
								{#if confusionMatrixData.length > 0}
									<div class="flex w-full items-center justify-center">
										<Plotly
											data={confusionMatrixData}
											layout={confusionMatrixLayout}
											config={gaugeConfig}
										/>
									</div>
								{:else}
									<div class="flex h-24 items-center justify-center">
										<p class="text-xs text-muted-foreground">
											{reportMetrics.confusion_matrix?.error || 'Not available yet'}
										</p>
									</div>
								{/if}
							</CardContent>
						</Card>
						<Card class="overflow-hidden">
							<CardContent class="p-2 overflow-hidden">
								<div class="flex items-center justify-between mb-1">
									<span class="text-xs font-medium text-muted-foreground">Classification Report</span>
									<button
										class="rounded-full p-0.5 text-muted-foreground hover:bg-blue-500/10 hover:text-blue-600"
										onclick={() => openMetricInfo('classification_report')}
										title="More info"
									>
										<span class="text-xs">ⓘ</span>
									</button>
								</div>
								{#if classificationReportData.length > 0}
									<div class="flex w-full items-center justify-center">
										<Plotly
											data={classificationReportData}
											layout={classificationReportLayout}
											config={gaugeConfig}
										/>
									</div>
								{:else}
									<div class="flex h-24 items-center justify-center">
										<p class="text-xs text-muted-foreground">
											{reportMetrics.classification_report?.error || 'Not available yet'}
										</p>
									</div>
								{/if}
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
		</Tabs>
	</div>
</div>
