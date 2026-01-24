import { writable, derived, type Writable } from 'svelte/store';
import type {
	ProjectName,
	ProjectKey,
	FormData,
	PredictionResult,
	MLflowMetrics,
	MLflowRunInfo,
	IncrementalSample,
	DropdownOptions
} from '$types';

// =============================================================================
// Project State
// =============================================================================
export const currentProject = writable<ProjectName>('Transaction Fraud Detection');
export const currentProjectKey = writable<ProjectKey>('tfd');

// =============================================================================
// Form Data (per project)
// =============================================================================
function createProjectStore<T>(defaultValue: T): Writable<Record<ProjectName, T>> {
	return writable({
		'Transaction Fraud Detection': defaultValue,
		'Estimated Time of Arrival': defaultValue,
		'E-Commerce Customer Interactions': defaultValue
	} as Record<ProjectName, T>);
}

export const formData = createProjectStore<FormData>({});

// =============================================================================
// Dropdown Options (loaded from JSON)
// =============================================================================
export const dropdownOptions = createProjectStore<DropdownOptions>({});

// =============================================================================
// Prediction Results
// =============================================================================
export const predictionResults = createProjectStore<PredictionResult>({});
export const predictionLoading = createProjectStore<boolean>(false);

// =============================================================================
// Incremental ML State
// =============================================================================
export const incrementalMlEnabled = createProjectStore<boolean>(false);
export const incrementalMlSample = createProjectStore<IncrementalSample>({});
export const incrementalModelAvailable = createProjectStore<boolean>(false);
export const incrementalModelLastTrained = createProjectStore<string>('');

// Global training state (only one model trains at a time)
export const activatedModel = writable<string>('');
export const mlTrainingEnabled = writable<boolean>(false);

// =============================================================================
// MLflow Metrics (Incremental)
// =============================================================================
export const mlflowMetrics = createProjectStore<MLflowMetrics>({});
export const mlflowExperimentUrl = createProjectStore<string>('');

// =============================================================================
// Batch ML State
// =============================================================================
export const batchTrainingLoading = createProjectStore<boolean>(false);
export const batchModelAvailable = createProjectStore<boolean>(false);
export const batchMlflowMetrics = createProjectStore<MLflowMetrics>({});
export const batchMlflowExperimentUrl = createProjectStore<string>('');
export const batchPredictionResults = createProjectStore<PredictionResult>({});
export const batchPredictionLoading = createProjectStore<boolean>(false);

// Batch training configuration
export const batchTrainingMode = createProjectStore<'percentage' | 'max_rows'>('max_rows');
export const batchTrainingDataPercentage = createProjectStore<number>(100);
export const batchTrainingMaxRows = createProjectStore<number>(10000);
export const batchDeltaLakeTotalRows = createProjectStore<number>(0);

// Batch training live status
export const batchTrainingStatus = createProjectStore<string>('');
export const batchTrainingProgress = createProjectStore<number>(0);
export const batchTrainingStage = createProjectStore<string>('');
export const batchTrainingMetricsPreview = createProjectStore<Record<string, number>>({});
export const batchTrainingCatboostLog = createProjectStore<Record<string, number | string>>({});
export const batchTrainingKMeansLog = createProjectStore<Record<string, number | string>>({});
export const batchTrainingTotalRows = createProjectStore<number>(0);

// MLflow run selection
export const batchMlflowRuns = createProjectStore<MLflowRunInfo[]>([]);
export const selectedBatchRun = createProjectStore<string>('');
export const batchRunsLoading = createProjectStore<boolean>(false);
export const batchLastTrainedRunId = createProjectStore<string>('');

// =============================================================================
// YellowBrick Visualizations
// =============================================================================
export const yellowBrickImages = createProjectStore<Record<string, string>>({});
export const yellowBrickLoading = createProjectStore<Record<string, boolean>>({});
export const selectedYellowBrickVisualizer = createProjectStore<string>('');

// =============================================================================
// Derived Stores for Current Project
// =============================================================================
export const currentFormData = derived(
	[formData, currentProject],
	([$formData, $currentProject]) => $formData[$currentProject]
);

export const currentDropdownOptions = derived(
	[dropdownOptions, currentProject],
	([$dropdownOptions, $currentProject]) => $dropdownOptions[$currentProject]
);

export const currentPredictionResults = derived(
	[predictionResults, currentProject],
	([$predictionResults, $currentProject]) => $predictionResults[$currentProject]
);

export const currentMlflowMetrics = derived(
	[mlflowMetrics, currentProject],
	([$mlflowMetrics, $currentProject]) => $mlflowMetrics[$currentProject]
);

export const currentBatchMlflowMetrics = derived(
	[batchMlflowMetrics, currentProject],
	([$batchMlflowMetrics, $currentProject]) => $batchMlflowMetrics[$currentProject]
);

export const currentBatchPredictionResults = derived(
	[batchPredictionResults, currentProject],
	([$batchPredictionResults, $currentProject]) => $batchPredictionResults[$currentProject]
);

export const currentIncrementalMlEnabled = derived(
	[incrementalMlEnabled, currentProject],
	([$incrementalMlEnabled, $currentProject]) => $incrementalMlEnabled[$currentProject]
);

export const currentBatchTrainingLoading = derived(
	[batchTrainingLoading, currentProject],
	([$batchTrainingLoading, $currentProject]) => $batchTrainingLoading[$currentProject]
);

export const currentBatchTrainingStatus = derived(
	[batchTrainingStatus, currentProject],
	([$batchTrainingStatus, $currentProject]) => $batchTrainingStatus[$currentProject]
);

export const currentBatchTrainingProgress = derived(
	[batchTrainingProgress, currentProject],
	([$batchTrainingProgress, $currentProject]) => $batchTrainingProgress[$currentProject]
);

export const currentBatchMlflowRuns = derived(
	[batchMlflowRuns, currentProject],
	([$batchMlflowRuns, $currentProject]) => $batchMlflowRuns[$currentProject]
);

// =============================================================================
// Helper Functions
// =============================================================================
export function updateProjectStore<T>(
	store: Writable<Record<ProjectName, T>>,
	project: ProjectName,
	value: T
) {
	store.update((s) => ({ ...s, [project]: value }));
}

export function updateFormField(project: ProjectName, field: string, value: unknown) {
	formData.update((s) => ({
		...s,
		[project]: { ...s[project], [field]: value }
	}));
}

export function resetFormData(project: ProjectName) {
	formData.update((s) => ({ ...s, [project]: {} }));
}

export function setCurrentProject(project: ProjectName, key: ProjectKey) {
	currentProject.set(project);
	currentProjectKey.set(key);
}
