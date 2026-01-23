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
	import { randomizeETAForm } from '$lib/utils/randomize';
	import {
		Shuffle,
		Clock,
		MapPin,
		Target,
		BarChart3,
		FlaskConical,
		RefreshCw
	} from 'lucide-svelte';
	import type { DropdownOptions, ProjectName } from '$types';

	const PROJECT: ProjectName = 'Estimated Time of Arrival';
	const MODEL_NAME = 'Adaptive Random Forest Regressor (River)';

	// Transform MLflow metrics from API format (metrics.MAE) to simple format (mae)
	function transformMetrics(rawMetrics: Record<string, unknown>): Record<string, number> {
		return {
			mae: Number(rawMetrics['metrics.MAE'] ?? rawMetrics['metrics.mae'] ?? 0),
			rmse: Number(rawMetrics['metrics.RMSE'] ?? rawMetrics['metrics.rmse'] ?? 0),
			r2: Number(rawMetrics['metrics.R2'] ?? rawMetrics['metrics.r2'] ?? 0),
			mse: Number(rawMetrics['metrics.MSE'] ?? rawMetrics['metrics.mse'] ?? 0),
			rmsle: Number(rawMetrics['metrics.RMSLE'] ?? rawMetrics['metrics.rmsle'] ?? 0),
			smape: Number(rawMetrics['metrics.SMAPE'] ?? rawMetrics['metrics.smape'] ?? 0),
			mape: Number(rawMetrics['metrics.MAPE'] ?? rawMetrics['metrics.mape'] ?? 0),
			rolling_mae: Number(rawMetrics['metrics.RollingMAE'] ?? rawMetrics['metrics.rolling_mae'] ?? 0),
			time_rolling_mae: Number(rawMetrics['metrics.TimeRollingMAE'] ?? rawMetrics['metrics.time_rolling_mae'] ?? 0)
		};
	}

	let trainingLoading = $state(false);
	let sampleLoading = $state(false);
	let metricsInterval: ReturnType<typeof setInterval> | null = null;
	let activeTab = $state('prediction');
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

	// Derived values for display
	let estimatedDistanceKm = $state<number>(0);
	let initialEstimatedTravelTime = $state<number>(0);

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
			// Transform metrics from API format (metrics.MAE) to simple format (mae)
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

	function calculateDistanceKm(
		originLat: number,
		originLon: number,
		destLat: number,
		destLon: number
	): number {
		const R = 6371; // Earth's radius in km
		const dLat = ((destLat - originLat) * Math.PI) / 180;
		const dLon = ((destLon - originLon) * Math.PI) / 180;
		const a =
			Math.sin(dLat / 2) * Math.sin(dLat / 2) +
			Math.cos((originLat * Math.PI) / 180) *
				Math.cos((destLat * Math.PI) / 180) *
				Math.sin(dLon / 2) *
				Math.sin(dLon / 2);
		const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
		return R * c;
	}

	function updateEstimates() {
		const form = $formData[PROJECT];
		if (form.origin_lat && form.origin_lon && form.destination_lat && form.destination_lon) {
			const dist = calculateDistanceKm(
				Number(form.origin_lat),
				Number(form.origin_lon),
				Number(form.destination_lat),
				Number(form.destination_lon)
			);
			estimatedDistanceKm = Number(dist.toFixed(2));
			// Assume average speed of 40 km/h for initial estimate
			initialEstimatedTravelTime = Math.round((dist / 40) * 3600);
			updateFormField(PROJECT, 'estimated_distance_km', estimatedDistanceKm);
			updateFormField(
				PROJECT,
				'initial_estimated_travel_time_seconds',
				initialEstimatedTravelTime
			);
		}
	}

	function loadRandomSample() {
		sampleLoading = true;
		try {
			const randomData = randomizeETAForm(dropdownOptions);
			updateProjectStore(formData, PROJECT, randomData);
			updateProjectStore(predictionResults, PROJECT, {});
			estimatedDistanceKm = Number(randomData.estimated_distance_km ?? 0);
			initialEstimatedTravelTime = Number(
				randomData.initial_estimated_travel_time_seconds ?? 0
			);
			updateEstimates();
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

		updateFormField(PROJECT, 'origin_lat', Number(originLat.toFixed(4)));
		updateFormField(PROJECT, 'origin_lon', Number(originLon.toFixed(4)));
		updateFormField(PROJECT, 'destination_lat', Number(destLat.toFixed(4)));
		updateFormField(PROJECT, 'destination_lon', Number(destLon.toFixed(4)));
		updateEstimates();
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
				const eta = result.data['Estimated Time of Arrival'] || result.data.eta_minutes;
				const sourceLabel = result.data.model_source
					? result.data.model_source.toUpperCase()
					: 'MLFLOW';
				toast.success(`Predicted ETA: ${eta?.toFixed(1)} minutes (${sourceLabel})`);
				await fetchMetrics();
			}
		} finally {
			updateProjectStore(predictionLoading, PROJECT, false);
		}
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
	const etaMinutes = $derived(
		currentPrediction?.['Estimated Time of Arrival'] || currentPrediction?.eta_minutes || 0
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
					<Clock class="h-5 w-5 text-primary" />
					<h3 class="text-base font-bold">Trip Details</h3>
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
						<p class="text-xs text-muted-foreground">Driver ID</p>
						<Select
							value={(currentForm.driver_id as string) ?? ''}
							options={dropdownOptions.driver_id?.slice(0, 50) || ['driver_1000']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'driver_id', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Vehicle ID</p>
						<Select
							value={(currentForm.vehicle_id as string) ?? ''}
							options={dropdownOptions.vehicle_id?.slice(0, 50) || ['vehicle_1000']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'vehicle_id', e.currentTarget.value)}
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
						<p class="text-xs text-muted-foreground">Origin Lat</p>
						<Input
							type="number"
							value={currentForm.origin_lat ?? ''}
							min="29.5"
							max="30.1"
							step="0.0001"
							class="h-8 text-sm"
							oninput={(e) => {
								updateFormField(PROJECT, 'origin_lat', parseFloat(e.currentTarget.value));
								updateEstimates();
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Origin Lon</p>
						<Input
							type="number"
							value={currentForm.origin_lon ?? ''}
							min="-95.8"
							max="-95.0"
							step="0.0001"
							class="h-8 text-sm"
							oninput={(e) => {
								updateFormField(PROJECT, 'origin_lon', parseFloat(e.currentTarget.value));
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
						<p class="text-xs text-muted-foreground">Dest Lat</p>
						<Input
							type="number"
							value={currentForm.destination_lat ?? ''}
							min="29.5"
							max="30.1"
							step="0.0001"
							class="h-8 text-sm"
							oninput={(e) => {
								updateFormField(PROJECT, 'destination_lat', parseFloat(e.currentTarget.value));
								updateEstimates();
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Dest Lon</p>
						<Input
							type="number"
							value={currentForm.destination_lon ?? ''}
							min="-95.8"
							max="-95.0"
							step="0.0001"
							class="h-8 text-sm"
							oninput={(e) => {
								updateFormField(PROJECT, 'destination_lon', parseFloat(e.currentTarget.value));
								updateEstimates();
							}}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Hour</p>
						<Input
							type="number"
							value={currentForm.hour_of_day ?? ''}
							min="0"
							max="23"
							step="1"
							class="h-8 text-sm"
							oninput={(e) =>
								updateFormField(PROJECT, 'hour_of_day', parseInt(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Rating</p>
						<Input
							type="number"
							value={currentForm.driver_rating ?? ''}
							min="3.5"
							max="5.0"
							step="0.1"
							class="h-8 text-sm"
							oninput={(e) =>
								updateFormField(PROJECT, 'driver_rating', parseFloat(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Temp C</p>
						<Input
							type="number"
							value={currentForm.temperature_celsius ?? ''}
							min="-50"
							max="50"
							step="0.1"
							class="h-8 text-sm"
							oninput={(e) =>
								updateFormField(PROJECT, 'temperature_celsius', parseFloat(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Traffic Factor</p>
						<Input
							type="number"
							value={currentForm.debug_traffic_factor ?? ''}
							min="0.3"
							max="1.9"
							step="0.1"
							class="h-8 text-sm"
							oninput={(e) =>
								updateFormField(PROJECT, 'debug_traffic_factor', parseFloat(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Weather Factor</p>
						<Input
							type="number"
							value={currentForm.debug_weather_factor ?? ''}
							min="1.0"
							max="2.0"
							step="0.1"
							class="h-8 text-sm"
							oninput={(e) =>
								updateFormField(PROJECT, 'debug_weather_factor', parseFloat(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Driver Factor</p>
						<Input
							type="number"
							value={currentForm.debug_driver_factor ?? ''}
							min="0.85"
							max="1.15"
							step="0.01"
							class="h-8 text-sm"
							oninput={(e) =>
								updateFormField(PROJECT, 'debug_driver_factor', parseFloat(e.currentTarget.value))}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Incident (s)</p>
						<Input
							type="number"
							value={currentForm.debug_incident_delay_seconds ?? ''}
							min="0"
							max="1800"
							step="1"
							class="h-8 text-sm"
							oninput={(e) =>
								updateFormField(
									PROJECT,
									'debug_incident_delay_seconds',
									parseInt(e.currentTarget.value)
								)}
						/>
					</div>
				</div>

				<!-- Display fields (read-only info) -->
				<div class="space-y-1 text-xs text-muted-foreground">
					<p>Trip ID: {currentForm.trip_id || '-'}</p>
					<p>Estimated Distance: {estimatedDistanceKm} km</p>
					<p>Initial Estimated Travel Time: {initialEstimatedTravelTime} s</p>
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
					<div class="flex gap-3">
						<Card class="h-[400px] w-1/2">
							<CardContent class="h-full p-4">
								<div class="flex h-full flex-col">
									<div class="mb-2 flex items-center gap-2">
										<MapPin class="h-4 w-4 text-primary" />
										<span class="text-sm font-bold">Origin and Destination</span>
									</div>
									<div
										class="flex-1 rounded-md bg-gradient-to-br from-blue-100 to-blue-50 p-4 dark:from-blue-900/20 dark:to-blue-800/10"
									>
										<div class="flex h-full items-center justify-center">
											<div class="space-y-2 text-center text-sm text-muted-foreground">
												<MapPin class="mx-auto h-8 w-8" />
												<p>Map visualization</p>
												<p class="text-xs">
													Origin: ({currentForm.origin_lat || '-'}, {currentForm.origin_lon || '-'})
												</p>
												<p class="text-xs">
													Dest: ({currentForm.destination_lat || '-'}, {currentForm.destination_lon ||
														'-'})
												</p>
											</div>
										</div>
									</div>
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

								<div class="mb-4 flex flex-wrap items-center gap-2 rounded-md bg-muted/50 p-2">
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

									<div class="flex flex-1 items-center justify-center">
										<div class="text-center">
											<div
												class="mx-auto flex h-32 w-32 items-center justify-center rounded-full border-8 border-blue-500 bg-blue-50 dark:bg-blue-900/20"
											>
												<div>
													<span class="text-3xl font-bold text-blue-600">
														{etaMinutes.toFixed(1)}
													</span>
													<p class="text-xs text-muted-foreground">minutes</p>
												</div>
											</div>
											<p class="mt-4 text-sm text-muted-foreground">Estimated Time of Arrival</p>
										</div>
									</div>
								</div>
							</CardContent>
						</Card>
					</div>
				</div>
			</TabsContent>

			<!-- Metrics Tab -->
			<TabsContent value="metrics" class="mt-4 space-y-4">
				<div class="flex items-center justify-between">
					<h2 class="text-lg font-bold">Regression Metrics</h2>
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
							name="MAE"
							value={currentMetrics.mae}
							onInfoClick={() => openMetricInfo('mae')}
						/>
						<MetricCard
							name="RMSE"
							value={currentMetrics.rmse}
							onInfoClick={() => openMetricInfo('rmse')}
						/>
						<MetricCard
							name="R² Score"
							value={currentMetrics.r2}
							onInfoClick={() => openMetricInfo('r2')}
						/>
						<MetricCard
							name="Rolling MAE"
							value={currentMetrics.rolling_mae}
							onInfoClick={() => openMetricInfo('rolling_mae')}
						/>
					</div>

					<!-- ROW 2: Additional metrics -->
					<div class="grid grid-cols-4 gap-2">
						<MetricCard
							name="MSE"
							value={currentMetrics.mse}
							onInfoClick={() => openMetricInfo('mse')}
						/>
						<MetricCard
							name="RMSLE"
							value={currentMetrics.rmsle}
							onInfoClick={() => openMetricInfo('rmsle')}
						/>
						<MetricCard
							name="SMAPE"
							value={currentMetrics.smape}
							onInfoClick={() => openMetricInfo('smape')}
						/>
						<MetricCard
							name="Time Rolling MAE"
							value={currentMetrics.time_rolling_mae}
							onInfoClick={() => openMetricInfo('time_rolling_mae')}
						/>
					</div>

					<!-- ROW 3: Gauges -->
					<div class="grid grid-cols-2 gap-4">
						<Card>
							<CardContent class="p-4">
								<div class="flex items-center justify-between">
									<span class="text-sm font-medium">R² Score</span>
									<button
										class="text-muted-foreground hover:text-foreground"
										onclick={() => openMetricInfo('r2')}
									>
										<span class="text-xs">ⓘ</span>
									</button>
								</div>
								<div class="mt-2 flex items-center gap-2">
									<div class="h-2 flex-1 rounded-full bg-muted">
										<div
											class="h-2 rounded-full bg-primary transition-all"
											style="width: {Math.max(0, (currentMetrics.r2 || 0)) * 100}%"
										></div>
									</div>
									<span class="text-sm font-bold">{(currentMetrics.r2 || 0).toFixed(4)}</span>
								</div>
							</CardContent>
						</Card>
						<Card>
							<CardContent class="p-4">
								<div class="flex items-center justify-between">
									<span class="text-sm font-medium">MAPE</span>
									<button
										class="text-muted-foreground hover:text-foreground"
										onclick={() => openMetricInfo('mape')}
									>
										<span class="text-xs">ⓘ</span>
									</button>
								</div>
								<div class="mt-2 flex items-center gap-2">
									<div class="h-2 flex-1 rounded-full bg-muted">
										<div
											class="h-2 rounded-full bg-orange-500 transition-all"
											style="width: {Math.min(100, (currentMetrics.mape || 0))}%"
										></div>
									</div>
									<span class="text-sm font-bold">{(currentMetrics.mape || 0).toFixed(2)}%</span>
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
		</Tabs>
	</div>
</div>
