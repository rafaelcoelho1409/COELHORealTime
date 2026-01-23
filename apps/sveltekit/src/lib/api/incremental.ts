import { API, apiGet, apiPost } from './client';
import type {
	ProjectName,
	MLflowMetrics,
	PredictResponse,
	ModelStatusResponse,
	TrainingStatusResponse,
	FormData
} from '$types';
import { MLFLOW_MODEL_NAMES } from '$types';
import {
	buildTFDPredictPayload,
	buildETAPredictPayload,
	buildECCIPredictPayload
} from '../utils/randomize.ts';

// =============================================================================
// Training Control
// =============================================================================

/**
 * Switch incremental model training on/off
 */
export async function switchModel(modelKey: string) {
	return apiPost<{ message: string; pid?: number }>(`${API.INCREMENTAL}/switch-model`, {
		model_key: modelKey
	});
}

/**
 * Stop all incremental training
 */
export async function stopTraining() {
	return switchModel('none');
}

/**
 * Get current training status
 */
export async function getTrainingStatus() {
	return apiGet<TrainingStatusResponse>(`${API.INCREMENTAL}/status`);
}

/**
 * Check if training is active for a specific project
 */
export async function getProjectTrainingStatus(projectName: ProjectName) {
	return apiGet<{
		project_name: string;
		model_name: string;
		is_active: boolean;
		model_source: 'live' | 'mlflow';
	}>(`${API.INCREMENTAL}/training-status/${encodeURIComponent(projectName)}`);
}

// =============================================================================
// Model Availability & Metrics
// =============================================================================

/**
 * Check if a trained model is available in MLflow
 */
export async function checkModelAvailable(projectName: ProjectName) {
	const modelName = MLFLOW_MODEL_NAMES.incremental[projectName];
	return apiPost<ModelStatusResponse>(`${API.INCREMENTAL}/model-available`, {
		project_name: projectName,
		model_name: modelName
	});
}

/**
 * Get MLflow metrics for a model
 */
export async function getMLflowMetrics(projectName: ProjectName, forceRefresh = false) {
	const modelName = MLFLOW_MODEL_NAMES.incremental[projectName];
	return apiPost<{
		metrics: MLflowMetrics;
		run_id?: string;
		is_live?: boolean;
	}>(`${API.INCREMENTAL}/mlflow-metrics`, {
		project_name: projectName,
		model_name: modelName,
		force_refresh: forceRefresh
	});
}

/**
 * Get report metrics (confusion matrix, classification report)
 */
export async function getReportMetrics(projectName: ProjectName) {
	const modelName = MLFLOW_MODEL_NAMES.incremental[projectName];
	return apiPost<{
		available: boolean;
		run_id?: string;
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
	}>(`${API.INCREMENTAL}/report-metrics`, {
		project_name: projectName,
		model_name: modelName
	});
}

// =============================================================================
// Predictions
// =============================================================================

/**
 * Make a prediction using the incremental model
 */
export async function predict(projectName: ProjectName, formData: FormData) {
	const modelName = MLFLOW_MODEL_NAMES.incremental[projectName];

	// Build proper payload based on project
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
			payload = formData;
	}

	return apiPost<PredictResponse>(`${API.INCREMENTAL}/predict`, {
		project_name: projectName,
		model_name: modelName,
		...payload
	});
}

// =============================================================================
// Sample Data
// =============================================================================

/**
 * Get a random sample from Delta Lake for form population
 */
export async function getSample(projectName: ProjectName) {
	return apiPost<{ sample: Record<string, unknown> }>(`${API.INCREMENTAL}/sample`, {
		project_name: projectName
	});
}

// =============================================================================
// ECCI Cluster Analytics
// =============================================================================

export async function getClusterCounts() {
	return apiGet<Record<string, number>>(`${API.INCREMENTAL}/cluster-counts`);
}

export async function getClusterFeatureCounts(columnName: string) {
	return apiPost<Record<string, Record<string, number>>>(`${API.INCREMENTAL}/cluster-feature-counts`, {
		column_name: columnName
	});
}

// =============================================================================
// Training Script Mapping
// =============================================================================
export const TRAINING_SCRIPTS: Record<ProjectName, string> = {
	'Transaction Fraud Detection': 'ml_training/river/tfd.py',
	'Estimated Time of Arrival': 'ml_training/river/eta.py',
	'E-Commerce Customer Interactions': 'ml_training/river/ecci.py'
};

/**
 * Start incremental training for a project
 */
export async function startTraining(projectName: ProjectName) {
	const script = TRAINING_SCRIPTS[projectName];
	return switchModel(script);
}
