import { API, apiGet, apiPost } from './client';
import type {
	ProjectName,
	MLflowMetrics,
	MLflowRunInfo,
	PredictResponse,
	ModelStatusResponse,
	TrainingResponse,
	BatchTrainingStatus,
	FormData
} from '$types';
import { MLFLOW_MODEL_NAMES } from '$types';
import {
	buildTFDPredictPayload,
	buildETAPredictPayload,
	buildECCIPredictPayload
} from '../utils/randomize';

// =============================================================================
// Training Script Mapping (matches Reflex _batch_training_scripts)
// =============================================================================
const BATCH_TRAINING_SCRIPTS: Record<ProjectName, string> = {
	'Transaction Fraud Detection': 'ml_training/sklearn/tfd.py',
	'Estimated Time of Arrival': 'ml_training/sklearn/eta.py',
	'E-Commerce Customer Interactions': 'ml_training/sklearn/ecci.py'
};

// =============================================================================
// Training Control
// =============================================================================

/**
 * Start batch model training using /switch-model endpoint (matches Reflex)
 */
export async function startTraining(
	projectName: ProjectName,
	options: {
		mode: 'percentage' | 'max_rows';
		percentage?: number;
		maxRows?: number;
	}
) {
	const trainingScript = BATCH_TRAINING_SCRIPTS[projectName];

	// Build payload matching Reflex's train_batch_model
	const payload: Record<string, unknown> = {
		model_key: trainingScript
	};

	if (options.mode === 'percentage' && options.percentage && options.percentage < 100) {
		payload.sample_frac = options.percentage / 100.0; // Convert to 0.0-1.0
	} else if (options.mode === 'max_rows' && options.maxRows) {
		payload.max_rows = options.maxRows;
	}

	return apiPost<TrainingResponse>(`${API.BATCH}/switch-model`, payload);
}

/**
 * Stop batch training using /stop-training endpoint (matches Reflex)
 */
export async function stopTraining(_projectName: ProjectName) {
	return apiPost<{ status: string; message?: string }>(`${API.BATCH}/stop-training`, {});
}

/**
 * Get training status using /status endpoint (matches Reflex)
 */
export async function getTrainingStatus(_projectName: ProjectName) {
	return apiGet<BatchTrainingStatus>(`${API.BATCH}/status`);
}

// =============================================================================
// Page Initialization (optimized single call - matches Reflex init_batch_page)
// =============================================================================

export interface BatchInitResponse {
	runs: MLflowRunInfo[];
	model_available: boolean;
	experiment_url?: string;
	metrics: Record<string, unknown>;
	total_rows: number;
	best_run_id?: string;
}

/**
 * Initialize batch ML page with optimized single API call.
 * Fetches all data in parallel: runs, model availability, metrics, experiment URL.
 * This matches Reflex's init_batch_page function.
 */
export async function initBatchPage(projectName: ProjectName, runId?: string) {
	const endpoint = `${API.BATCH}/init`;
	const payload = {
		project_name: projectName,
		run_id: runId
	};
	console.log('[BatchAPI] initBatchPage - endpoint:', endpoint, 'payload:', payload);
	const result = await apiPost<BatchInitResponse>(endpoint, payload);
	console.log('[BatchAPI] initBatchPage - result:', result);
	return result;
}

// =============================================================================
// Model Availability & Metrics
// =============================================================================

/**
 * Check if a trained batch model is available in MLflow
 */
export async function checkModelAvailable(projectName: ProjectName) {
	const modelName = MLFLOW_MODEL_NAMES.batch[projectName];
	return apiPost<ModelStatusResponse>(`${API.BATCH}/model-available`, {
		project_name: projectName,
		model_name: modelName
	});
}

/**
 * Get MLflow metrics for a batch model.
 * Response includes run_url which should be used to update the experiment URL.
 */
export async function getMLflowMetrics(projectName: ProjectName, runId?: string) {
	const modelName = MLFLOW_MODEL_NAMES.batch[projectName];
	return apiPost<
		Record<string, unknown> & {
			_no_runs?: boolean;
			run_url?: string;
			run_id?: string;
			params?: Record<string, string>;
		}
	>(`${API.BATCH}/mlflow-metrics`, {
		project_name: projectName,
		model_name: modelName,
		run_id: runId
	});
}

/**
 * Get all MLflow runs for a project (for run selector dropdown)
 */
export async function getMLflowRuns(projectName: ProjectName) {
	const modelName = MLFLOW_MODEL_NAMES.batch[projectName];
	const endpoint = `${API.BATCH}/mlflow-runs`;
	const payload = {
		project_name: projectName,
		model_name: modelName
	};
	console.log('[BatchAPI] getMLflowRuns - endpoint:', endpoint, 'payload:', payload);
	const result = await apiPost<{ runs: MLflowRunInfo[] }>(endpoint, payload);
	console.log('[BatchAPI] getMLflowRuns - result:', result);
	return result;
}

// =============================================================================
// Predictions
// =============================================================================

/**
 * Make a prediction using the batch model
 */
export async function predict(projectName: ProjectName, formData: FormData, runId?: string) {
	const modelName = MLFLOW_MODEL_NAMES.batch[projectName];

	// Build proper payload based on project (matching incremental API)
	let payload: Record<string, unknown>;
	switch (projectName) {
		case 'Transaction Fraud Detection':
			payload = buildTFDPredictPayload(formData);
			break;
		case 'Estimated Time of Arrival':
			payload = buildETAPredictPayload(formData);
			break;
		case 'E-Commerce Customer Interactions':
			payload = buildECCIPredictPayload(formData);
			break;
		default:
			payload = formData as Record<string, unknown>;
	}

	return apiPost<PredictResponse>(`${API.BATCH}/predict`, {
		project_name: projectName,
		model_name: modelName,
		run_id: runId,
		...payload
	});
}

// =============================================================================
// YellowBrick Visualizations
// =============================================================================

/**
 * Get YellowBrick visualization image (matching Reflex's fetch_yellowbrick_metric)
 * Note: Some visualizations like Manifold can take several minutes
 */
export async function getYellowBrickImage(
	projectName: ProjectName,
	metricType: string,
	metricName: string,
	runId?: string
) {
	return apiPost<{ image_base64: string; error?: string }>(
		`${API.BATCH}/yellowbrick-metric`,
		{
			project_name: projectName,
			metric_type: metricType,
			metric_name: metricName,
			run_id: runId
		}
	);
}

/**
 * Get sklearn visualization image
 */
export async function getSklearnImage(
	projectName: ProjectName,
	metricType: string,
	metricName: string,
	runId?: string
) {
	return apiPost<{ image_base64: string; error?: string }>(`${API.BATCH}/sklearn-metric`, {
		project_name: projectName,
		metric_type: metricType,
		metric_name: metricName,
		run_id: runId
	});
}

// =============================================================================
// Delta Lake Info
// =============================================================================

/**
 * Get Delta Lake table row count
 */
export async function getDeltaLakeRowCount(projectName: ProjectName) {
	return apiGet<{ row_count: number }>(
		`${API.BATCH}/delta-lake-count/${encodeURIComponent(projectName)}`
	);
}

// =============================================================================
// Sample Data
// =============================================================================

/**
 * Get a random sample from Delta Lake for form population
 */
export async function getSample(projectName: ProjectName) {
	return apiPost<{ sample: Record<string, unknown> }>(`${API.BATCH}/sample`, {
		project_name: projectName
	});
}

// =============================================================================
// Cluster Analytics (ECCI Batch ML)
// =============================================================================

/**
 * Get cluster sample counts from batch ML training data
 */
export async function getClusterCounts(projectName: ProjectName, runId?: string) {
	return apiPost<{
		cluster_counts: Record<string, number>;
		run_id?: string;
		total_samples?: number;
		error?: string;
	}>(`${API.BATCH}/cluster-counts`, {
		project_name: projectName,
		run_id: runId
	});
}

/**
 * Get cluster feature counts from batch ML training data
 */
export async function getClusterFeatureCounts(
	projectName: ProjectName,
	featureName: string,
	runId?: string
) {
	return apiPost<{
		feature_counts: Record<string, Record<string, number>>;
		run_id?: string;
		total_samples?: number;
		error?: string;
	}>(`${API.BATCH}/cluster-feature-counts`, {
		project_name: projectName,
		feature_name: featureName,
		run_id: runId
	});
}
