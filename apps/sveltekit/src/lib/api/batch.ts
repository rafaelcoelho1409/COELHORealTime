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

// =============================================================================
// Training Control
// =============================================================================

/**
 * Start batch model training
 */
export async function startTraining(
	projectName: ProjectName,
	options: {
		mode: 'percentage' | 'max_rows';
		percentage?: number;
		maxRows?: number;
	}
) {
	const modelName = MLFLOW_MODEL_NAMES.batch[projectName];
	return apiPost<TrainingResponse>(`${API.BATCH}/train`, {
		project_name: projectName,
		model_name: modelName,
		training_mode: options.mode,
		data_percentage: options.percentage,
		max_rows: options.maxRows
	});
}

/**
 * Stop batch training
 */
export async function stopTraining(projectName: ProjectName) {
	return apiPost<{ message: string }>(`${API.BATCH}/stop`, {
		project_name: projectName
	});
}

/**
 * Get training status
 */
export async function getTrainingStatus(projectName: ProjectName) {
	return apiGet<BatchTrainingStatus>(
		`${API.BATCH}/training-status/${encodeURIComponent(projectName)}`
	);
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
 * Get MLflow metrics for a batch model
 */
export async function getMLflowMetrics(projectName: ProjectName, runId?: string) {
	const modelName = MLFLOW_MODEL_NAMES.batch[projectName];
	return apiPost<{
		metrics: MLflowMetrics;
		run_id?: string;
		params?: Record<string, string>;
	}>(`${API.BATCH}/mlflow-metrics`, {
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
	return apiPost<{ runs: MLflowRunInfo[] }>(`${API.BATCH}/mlflow-runs`, {
		project_name: projectName,
		model_name: modelName
	});
}

// =============================================================================
// Predictions
// =============================================================================

/**
 * Make a prediction using the batch model
 */
export async function predict(projectName: ProjectName, formData: FormData, runId?: string) {
	const modelName = MLFLOW_MODEL_NAMES.batch[projectName];
	return apiPost<PredictResponse>(`${API.BATCH}/predict`, {
		project_name: projectName,
		model_name: modelName,
		run_id: runId,
		...formData
	});
}

// =============================================================================
// YellowBrick Visualizations
// =============================================================================

/**
 * Get YellowBrick visualization image
 */
export async function getYellowBrickImage(
	projectName: ProjectName,
	visualizerName: string,
	runId?: string
) {
	const modelName = MLFLOW_MODEL_NAMES.batch[projectName];
	return apiPost<{ image_base64: string; visualizer_name: string }>(
		`${API.BATCH}/yellowbrick`,
		{
			project_name: projectName,
			model_name: modelName,
			visualizer_name: visualizerName,
			run_id: runId
		}
	);
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
