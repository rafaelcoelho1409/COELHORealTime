<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		Card,
		CardHeader,
		CardTitle,
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
	import { randomizeTFDForm } from '$lib/utils/randomize';
	import { CreditCard, Shuffle, AlertTriangle, CheckCircle, Percent } from 'lucide-svelte';
	import Plotly from 'svelte-plotly.js';
	import type { ProjectName } from '$types';

	const PROJECT: ProjectName = 'Transaction Fraud Detection';
	const MODEL_NAME = 'CatBoost Classifier';

	let dropdownOptions = $state<Record<string, string[]>>({});
	let sampleLoading = $state(false);
	let statusInterval: ReturnType<typeof setInterval> | null = null;

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

		// Get Delta Lake row count
		const countResult = await batchApi.getDeltaLakeRowCount(PROJECT);
		if (countResult.data?.row_count) {
			updateProjectStore(batchDeltaLakeTotalRows, PROJECT, countResult.data.row_count);
		}

		// Load MLflow runs
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

			// Check if training completed
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
				toast.error(result.error);
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

	const currentForm = $derived($formData[PROJECT] || {});
	const currentPrediction = $derived($batchPredictionResults[PROJECT]);
	const isLoading = $derived($batchPredictionLoading[PROJECT]);
	const isTraining = $derived($batchTrainingLoading[PROJECT]);
	const runs = $derived($batchMlflowRuns[PROJECT] || []);
	const modelAvailable = $derived($batchModelAvailable[PROJECT]);

	// Fraud probability for gauge
	const fraudProbability = $derived(currentPrediction?.fraud_probability ?? 0);
	const notFraudProbability = $derived(1 - fraudProbability);
	const predictionText = $derived(fraudProbability >= 0.5 ? 'FRAUD' : 'NOT FRAUD');
	const predictionColor = $derived(fraudProbability >= 0.5 ? 'red' : 'green');

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
</script>

<!-- 40%/60% Layout -->
<div class="flex gap-6">
	<!-- Left Column - Training Box + Form (40%) -->
	<div class="w-[40%] space-y-4">
		<!-- Training Box -->
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

		<!-- Form Card -->
		<Card>
			<CardContent class="space-y-3 pt-4">
				<!-- Form Legend -->
				<div class="flex items-center gap-2">
					<CreditCard class="h-5 w-5 text-primary" />
					<h3 class="text-base font-bold">Transaction Details</h3>
				</div>

				<!-- Predict + Randomize Buttons -->
				<div class="flex gap-2">
					<Button class="flex-1" onclick={predict} loading={isLoading} disabled={!modelAvailable}>
						Predict
					</Button>
					<Button
						variant="outline"
						class="flex-1"
						onclick={loadRandomSample}
						loading={sampleLoading}
					>
						<Shuffle class="mr-2 h-3.5 w-3.5" />
						Randomize
					</Button>
				</div>

				<hr class="border-border" />

				<!-- 3-column grid for form fields -->
				<div class="grid grid-cols-3 gap-2">
					<!-- Amount -->
					<FormField label="Amount" id="amount" class="space-y-1">
						<Input
							id="amount"
							type="number"
							value={currentForm.amount ?? ''}
							step="0.01"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'amount', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<!-- Account Age -->
					<FormField label="Account Age" id="account_age_days" class="space-y-1">
						<Input
							id="account_age_days"
							type="number"
							value={currentForm.account_age_days ?? ''}
							min="0"
							class="h-8 text-sm"
							oninput={(e) =>
								updateFormField(PROJECT, 'account_age_days', parseInt(e.currentTarget.value))}
						/>
					</FormField>

					<!-- Currency -->
					<FormField label="Currency" id="currency" class="space-y-1">
						<Select
							id="currency"
							value={(currentForm.currency as string) ?? 'USD'}
							options={dropdownOptions.currency || ['USD']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'currency', e.currentTarget.value)}
						/>
					</FormField>

					<!-- Date -->
					<FormField label="Date" id="timestamp_date" class="space-y-1">
						<Input
							id="timestamp_date"
							type="date"
							value={(currentForm.timestamp_date as string) ?? ''}
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'timestamp_date', e.currentTarget.value)}
						/>
					</FormField>

					<!-- Time -->
					<FormField label="Time" id="timestamp_time" class="space-y-1">
						<Input
							id="timestamp_time"
							type="time"
							value={(currentForm.timestamp_time as string) ?? ''}
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'timestamp_time', e.currentTarget.value)}
						/>
					</FormField>

					<!-- Merchant ID -->
					<FormField label="Merchant ID" id="merchant_id" class="space-y-1">
						<Select
							id="merchant_id"
							value={
								(currentForm.merchant_id as string) ||
								dropdownOptions.merchant_id?.[0] ||
								'merchant_1'
							}
							options={dropdownOptions.merchant_id?.slice(0, 50) || ['merchant_1']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'merchant_id', e.currentTarget.value)}
						/>
					</FormField>

					<!-- Category -->
					<FormField label="Category" id="product_category" class="space-y-1">
						<Select
							id="product_category"
							value={(currentForm.product_category as string) ?? ''}
							options={dropdownOptions.product_category || ['electronics']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'product_category', e.currentTarget.value)}
						/>
					</FormField>

					<!-- Type -->
					<FormField label="Type" id="transaction_type" class="space-y-1">
						<Select
							id="transaction_type"
							value={(currentForm.transaction_type as string) ?? ''}
							options={dropdownOptions.transaction_type || ['purchase']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'transaction_type', e.currentTarget.value)}
						/>
					</FormField>

					<!-- Payment -->
					<FormField label="Payment" id="payment_method" class="space-y-1">
						<Select
							id="payment_method"
							value={(currentForm.payment_method as string) ?? ''}
							options={dropdownOptions.payment_method || ['credit_card']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'payment_method', e.currentTarget.value)}
						/>
					</FormField>

					<!-- Latitude -->
					<FormField label="Latitude" id="lat" class="space-y-1">
						<Input
							id="lat"
							type="number"
							value={currentForm.lat ?? ''}
							step="0.0001"
							min="-90"
							max="90"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'lat', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<!-- Longitude -->
					<FormField label="Longitude" id="lon" class="space-y-1">
						<Input
							id="lon"
							type="number"
							value={currentForm.lon ?? ''}
							step="0.0001"
							min="-180"
							max="180"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'lon', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<!-- Browser -->
					<FormField label="Browser" id="browser" class="space-y-1">
						<Select
							id="browser"
							value={(currentForm.browser as string) ?? ''}
							options={dropdownOptions.browser || ['Chrome']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'browser', e.currentTarget.value)}
						/>
					</FormField>

					<!-- OS -->
					<FormField label="OS" id="os" class="space-y-1">
						<Select
							id="os"
							value={(currentForm.os as string) ?? ''}
							options={dropdownOptions.os || ['Windows']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'os', e.currentTarget.value)}
						/>
					</FormField>

					<!-- CVV Provided -->
					<FormField label="CVV" id="cvv_provided" class="space-y-1">
						<label class="flex items-center gap-2 text-sm">
							<input
								type="checkbox"
								checked={currentForm.cvv_provided ?? false}
								onchange={(e) => updateFormField(PROJECT, 'cvv_provided', e.currentTarget.checked)}
								class="h-4 w-4 rounded border-input"
							/>
							Provided
						</label>
					</FormField>

					<!-- Billing Match -->
					<FormField label="Billing" id="billing_address_match" class="space-y-1">
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
					</FormField>
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

	<!-- Right Column - Batch Sub-Nav + Prediction Result (60%) -->
	<div class="w-[60%] space-y-4">
		<!-- Batch Sub-Nav -->
		<BatchSubNav projectKey="tfd" />

		<!-- Prediction Result Section -->
		<div class="space-y-4">
			<div class="flex items-center gap-2">
				<AlertTriangle class="h-5 w-5 text-primary" />
				<h2 class="text-xl font-bold">Prediction Result</h2>
			</div>

			<Card>
				<CardContent class="space-y-4 pt-6">
					<!-- Gauge Chart (Plotly) -->
					<div class="flex w-full items-center justify-center">
						<Plotly
							data={gaugeData}
							layout={{ ...gaugeLayout, width: 300 }}
							config={gaugeConfig}
						/>
					</div>

					<!-- Prediction Summary Cards -->
					<div class="grid grid-cols-3 gap-3">
						<!-- Classification -->
						<Card class="bg-muted/50">
							<CardContent class="p-3 text-center">
								<div class="mb-1 flex items-center justify-center gap-1">
									<AlertTriangle class="h-3.5 w-3.5" style="color: {predictionColor}" />
									<span class="text-xs text-muted-foreground">Classification</span>
								</div>
								<p class="text-lg font-bold" style="color: {predictionColor}">
									{predictionText}
								</p>
							</CardContent>
						</Card>

						<!-- Fraud Probability -->
						<Card class="bg-muted/50">
							<CardContent class="p-3 text-center">
								<div class="mb-1 flex items-center justify-center gap-1">
									<Percent class="h-3.5 w-3.5 text-red-500" />
									<span class="text-xs text-muted-foreground">Fraud</span>
								</div>
								<p class="text-lg font-bold text-red-500">
									{(fraudProbability * 100).toFixed(2)}%
								</p>
							</CardContent>
						</Card>

						<!-- Not Fraud Probability -->
						<Card class="bg-muted/50">
							<CardContent class="p-3 text-center">
								<div class="mb-1 flex items-center justify-center gap-1">
									<CheckCircle class="h-3.5 w-3.5 text-green-500" />
									<span class="text-xs text-muted-foreground">Not Fraud</span>
								</div>
								<p class="text-lg font-bold text-green-500">
									{(notFraudProbability * 100).toFixed(2)}%
								</p>
							</CardContent>
						</Card>
					</div>

					<!-- Warning if no model available -->
					{#if !modelAvailable}
						<div
							class="flex items-center gap-2 rounded-lg border border-orange-200 bg-orange-50 p-3 text-sm text-orange-700 dark:border-orange-900 dark:bg-orange-950 dark:text-orange-300"
						>
							<AlertTriangle class="h-4 w-4" />
							<span>No trained model available. Click <strong>Train</strong> to train the batch model first.</span>
						</div>
					{/if}
				</CardContent>
			</Card>
		</div>
	</div>
</div>
