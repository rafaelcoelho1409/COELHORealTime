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
	import { randomizeETAForm } from '$lib/utils/randomize';
	import { Clock, Shuffle, MapPin, AlertTriangle, Info } from 'lucide-svelte';
	import type { ProjectName } from '$types';

	const PROJECT: ProjectName = 'Estimated Time of Arrival';
	const MODEL_NAME = 'CatBoost Regressor';

	let dropdownOptions = $state<Record<string, string[]>>({});
	let sampleLoading = $state(false);
	let statusInterval: ReturnType<typeof setInterval> | null = null;

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

	// Calculate estimated distance using Haversine formula
	const estimatedDistanceKm = $derived(() => {
		const lat1 = currentForm.origin_lat as number;
		const lon1 = currentForm.origin_lon as number;
		const lat2 = currentForm.destination_lat as number;
		const lon2 = currentForm.destination_lon as number;
		if (!lat1 || !lon1 || !lat2 || !lon2) return 0;

		const R = 6371;
		const dLat = ((lat2 - lat1) * Math.PI) / 180;
		const dLon = ((lon2 - lon1) * Math.PI) / 180;
		const a =
			Math.sin(dLat / 2) * Math.sin(dLat / 2) +
			Math.cos((lat1 * Math.PI) / 180) *
				Math.cos((lat2 * Math.PI) / 180) *
				Math.sin(dLon / 2) *
				Math.sin(dLon / 2);
		const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
		return R * c;
	});

	const etaMinutes = $derived(
		currentPrediction?.['Estimated Time of Arrival'] || currentPrediction?.eta_minutes || 0
	);
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
					<Clock class="h-5 w-5 text-primary" />
					<h3 class="text-base font-bold">Trip Details</h3>
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
					<FormField label="Driver ID" id="driver_id" class="space-y-1">
						<Select
							id="driver_id"
							value={(currentForm.driver_id as string) ?? ''}
							options={dropdownOptions.driver_id?.slice(0, 50) || ['driver_1000']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'driver_id', e.currentTarget.value)}
						/>
					</FormField>

					<FormField label="Vehicle ID" id="vehicle_id" class="space-y-1">
						<Select
							id="vehicle_id"
							value={(currentForm.vehicle_id as string) ?? ''}
							options={dropdownOptions.vehicle_id?.slice(0, 50) || ['vehicle_100']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'vehicle_id', e.currentTarget.value)}
						/>
					</FormField>

					<FormField label="Weather" id="weather" class="space-y-1">
						<Select
							id="weather"
							value={(currentForm.weather as string) ?? 'Clear'}
							options={dropdownOptions.weather || ['Clear']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'weather', e.currentTarget.value)}
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

					<FormField label="Vehicle Type" id="vehicle_type" class="space-y-1">
						<Select
							id="vehicle_type"
							value={(currentForm.vehicle_type as string) ?? 'Sedan'}
							options={dropdownOptions.vehicle_type || ['Sedan']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'vehicle_type', e.currentTarget.value)}
						/>
					</FormField>

					<FormField label="Origin Lat" id="origin_lat" class="space-y-1">
						<Input
							id="origin_lat"
							type="number"
							value={currentForm.origin_lat ?? ''}
							step="0.0001"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'origin_lat', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Origin Lon" id="origin_lon" class="space-y-1">
						<Input
							id="origin_lon"
							type="number"
							value={currentForm.origin_lon ?? ''}
							step="0.0001"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'origin_lon', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Coords" id="random_coords" class="space-y-1">
						<Button variant="outline" size="sm" class="h-8 w-full text-xs" onclick={generateRandomCoordinates}>
							<Shuffle class="mr-1 h-3 w-3" />
							Random
						</Button>
					</FormField>

					<FormField label="Dest Lat" id="destination_lat" class="space-y-1">
						<Input
							id="destination_lat"
							type="number"
							value={currentForm.destination_lat ?? ''}
							step="0.0001"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'destination_lat', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Dest Lon" id="destination_lon" class="space-y-1">
						<Input
							id="destination_lon"
							type="number"
							value={currentForm.destination_lon ?? ''}
							step="0.0001"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'destination_lon', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Hour" id="hour_of_day" class="space-y-1">
						<Input
							id="hour_of_day"
							type="number"
							value={currentForm.hour_of_day ?? ''}
							min="0"
							max="23"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'hour_of_day', parseInt(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Rating" id="driver_rating" class="space-y-1">
						<Input
							id="driver_rating"
							type="number"
							value={currentForm.driver_rating ?? ''}
							min="3.5"
							max="5"
							step="0.1"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'driver_rating', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Temp C" id="temperature_celsius" class="space-y-1">
						<Input
							id="temperature_celsius"
							type="number"
							value={currentForm.temperature_celsius ?? ''}
							step="0.1"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'temperature_celsius', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Traffic Factor" id="debug_traffic_factor" class="space-y-1">
						<Input
							id="debug_traffic_factor"
							type="number"
							value={currentForm.debug_traffic_factor ?? ''}
							min="0.3"
							max="1.9"
							step="0.1"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'debug_traffic_factor', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Weather Factor" id="debug_weather_factor" class="space-y-1">
						<Input
							id="debug_weather_factor"
							type="number"
							value={currentForm.debug_weather_factor ?? ''}
							min="1.0"
							max="2.0"
							step="0.1"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'debug_weather_factor', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Driver Factor" id="debug_driver_factor" class="space-y-1">
						<Input
							id="debug_driver_factor"
							type="number"
							value={currentForm.debug_driver_factor ?? ''}
							min="0.85"
							max="1.15"
							step="0.01"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'debug_driver_factor', parseFloat(e.currentTarget.value))}
						/>
					</FormField>

					<FormField label="Incident (s)" id="debug_incident_delay_seconds" class="space-y-1">
						<Input
							id="debug_incident_delay_seconds"
							type="number"
							value={currentForm.debug_incident_delay_seconds ?? ''}
							min="0"
							max="1800"
							class="h-8 text-sm"
							oninput={(e) => updateFormField(PROJECT, 'debug_incident_delay_seconds', parseInt(e.currentTarget.value))}
						/>
					</FormField>
				</div>

				<div class="space-y-1 text-xs text-muted-foreground">
					<p>Trip ID: {currentForm.trip_id || '-'}</p>
					<p>Estimated Distance: {estimatedDistanceKm().toFixed(2)} km</p>
					<p>Initial Est. Travel Time: {Math.round(estimatedDistanceKm() * 2)} s</p>
				</div>
			</CardContent>
		</Card>
	</div>

	<!-- Right Column - Batch Sub-Nav + Prediction Result (60%) -->
	<div class="w-[60%] space-y-4">
		<BatchSubNav projectKey="eta" />

		<!-- Prediction Result - Side by Side -->
		<div class="grid grid-cols-2 gap-3">
			<!-- Origin and Destination Card -->
			<Card class="h-[400px]">
				<CardContent class="flex h-full flex-col p-4">
					<div class="mb-2 flex items-center gap-2">
						<MapPin class="h-4 w-4 text-primary" />
						<span class="text-sm font-bold">Origin and Destination</span>
					</div>
					<!-- Map placeholder -->
					<div class="flex flex-1 items-center justify-center rounded-lg bg-muted/30">
						<div class="text-center text-muted-foreground">
							<MapPin class="mx-auto h-12 w-12 opacity-50" />
							<p class="mt-2 text-sm">Map visualization</p>
							<p class="text-xs">
								{currentForm.origin_lat ?? 0}, {currentForm.origin_lon ?? 0}
							</p>
							<p class="text-xs">→</p>
							<p class="text-xs">
								{currentForm.destination_lat ?? 0}, {currentForm.destination_lon ?? 0}
							</p>
						</div>
					</div>
					<div class="mt-2 space-y-1 text-xs text-muted-foreground">
						<p>Estimated Distance: {estimatedDistanceKm().toFixed(2)} km</p>
						<p>Initial Est. Travel Time: {Math.round(estimatedDistanceKm() * 2)} s</p>
					</div>
				</CardContent>
			</Card>

			<!-- ETA Prediction Card -->
			<Card class="h-[400px]">
				<CardContent class="flex h-full flex-col p-4">
					<div class="mb-2 flex items-center gap-2">
						<Clock class="h-4 w-4 text-primary" />
						<span class="text-sm font-bold">ETA - Prediction</span>
					</div>
					<div class="flex flex-1 items-center justify-center">
						{#if etaMinutes > 0}
							<!-- ETA Display -->
							<div class="text-center">
								<svg viewBox="0 0 200 120" class="mx-auto h-40 w-64">
									<!-- Background arc -->
									<path
										d="M 20 100 A 80 80 0 0 1 180 100"
										fill="none"
										stroke="currentColor"
										stroke-width="16"
										class="text-muted"
									/>
									<!-- Progress arc -->
									<path
										d="M 20 100 A 80 80 0 0 1 180 100"
										fill="none"
										stroke="#3b82f6"
										stroke-width="16"
										stroke-dasharray="{Math.min(etaMinutes / 60, 1) * 251.2} 251.2"
										stroke-linecap="round"
									/>
									<circle cx="100" cy="100" r="6" fill="currentColor" class="text-foreground" />
									<text x="100" y="70" text-anchor="middle" class="fill-current text-3xl font-bold">
										{etaMinutes.toFixed(1)}
									</text>
									<text x="100" y="90" text-anchor="middle" class="fill-muted-foreground text-sm">
										minutes
									</text>
								</svg>
							</div>
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
				</CardContent>
			</Card>
		</div>
	</div>
</div>
