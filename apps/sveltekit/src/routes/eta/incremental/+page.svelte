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
		MLTrainingSwitch,
		MetricCard,
		Tabs,
		TabsList,
		TabsTrigger,
		TabsContent
	} from '$components/shared';
	import Plotly from 'svelte-plotly.js';
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

	// Derived values for display - auto-calculate from form data
	const estimatedDistanceKm = $derived.by(() => {
		const form = $formData[PROJECT];
		// First try to get from form data
		if (form?.estimated_distance_km) {
			return Number(form.estimated_distance_km);
		}
		// Otherwise calculate from coordinates
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
		// First try to get from form data
		if (form?.initial_estimated_travel_time_seconds) {
			return Number(form.initial_estimated_travel_time_seconds);
		}
		// Otherwise calculate from distance (assume 40 km/h average speed)
		if (estimatedDistanceKm > 0) {
			return Math.round((estimatedDistanceKm / 40) * 3600);
		}
		return 0;
	});

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

	function loadRandomSample() {
		sampleLoading = true;
		try {
			const randomData = randomizeETAForm(dropdownOptions);
			updateProjectStore(formData, PROJECT, randomData);
			updateProjectStore(predictionResults, PROJECT, {});
			// Distance and travel time are now derived automatically from form data
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

	// Watch for coordinate changes - explicitly reference coordinates to create reactive dependency
	$effect(() => {
		// Reference all coordinates to track changes
		const _oLat = originLat;
		const _oLon = originLon;
		const _dLat = destLat;
		const _dLon = destLon;

		if (leafletMap && L) {
			updateMapMarkers();
		}
	});

	// Initialize or reinitialize map when tab switches to prediction
	$effect(() => {
		// Track activeTab to trigger when switching tabs
		const currentTab = activeTab;

		if (browser && currentTab === 'prediction') {
			// Small delay to ensure DOM is ready after tab switch
			setTimeout(() => {
				if (mapContainer) {
					// Clean up old map instance if it exists
					if (leafletMap) {
						try {
							leafletMap.remove();
						} catch (e) {
							// Ignore errors when removing invalid map
						}
						leafletMap = null;
					}
					initMap();
				}
			}, 50);
		}
	});

	// Cleanup map on component destroy
	onDestroy(() => {
		if (leafletMap) {
			try {
				leafletMap.remove();
			} catch (e) {
				// Ignore errors
			}
			leafletMap = null;
		}
	});
</script>

<div class="flex gap-4">
	<!-- LEFT COLUMN (40%) - Training Switch + Form -->
	<div class="w-2/5 min-w-0 space-y-4">
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
							value={(currentForm.driver_id as string) || ''}
							options={dropdownOptions.driver_id?.slice(0, 50) || ['driver_1000']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'driver_id', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Vehicle ID</p>
						<Select
							value={(currentForm.vehicle_id as string) || ''}
							options={dropdownOptions.vehicle_id?.slice(0, 50) || ['vehicle_1000']}
							class="h-8 text-sm"
							onchange={(e) => updateFormField(PROJECT, 'vehicle_id', e.currentTarget.value)}
						/>
					</div>

					<div class="space-y-1">
						<p class="text-xs text-muted-foreground">Weather</p>
						<Select
							value={(currentForm.weather as string) || 'Clear'}
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
							value={(currentForm.vehicle_type as string) || 'Sedan'}
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
	<div class="w-3/5 min-w-0">
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
										<Plotly
											data={etaPredictionData}
											layout={etaPredictionLayout}
											config={plotlyConfig}
										/>
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
					<Button variant="ghost" size="sm" onclick={() => fetchMetrics(true)}>
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
