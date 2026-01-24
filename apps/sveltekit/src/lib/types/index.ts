// Project names
export type ProjectName = 'Transaction Fraud Detection' | 'Estimated Time of Arrival' | 'E-Commerce Customer Interactions';
export type ProjectKey = 'tfd' | 'eta' | 'ecci';

// Project name to key mapping
export const PROJECT_NAME_TO_KEY: Record<ProjectName, ProjectKey> = {
	'Transaction Fraud Detection': 'tfd',
	'Estimated Time of Arrival': 'eta',
	'E-Commerce Customer Interactions': 'ecci'
};

export const PROJECT_KEY_TO_NAME: Record<ProjectKey, ProjectName> = {
	tfd: 'Transaction Fraud Detection',
	eta: 'Estimated Time of Arrival',
	ecci: 'E-Commerce Customer Interactions'
};

// MLflow Run Info
export interface MLflowRunInfo {
	run_id: string;
	run_name: string;
	start_time: string | null;
	end_time: string | null;
	metrics: Record<string, number>;
	params: Record<string, string>;
	total_rows: number;
	is_best: boolean;
}

// Metric Info from JSON data files
export interface MetricInfo {
	name: string;
	formula: string;
	explanation: string;
	context: string;
	range: string;
	optimal: string;
	docs_url: {
		incremental: string;
		batch: string;
	};
}

export interface MetricInfoData {
	metrics: Record<string, MetricInfo>;
}

// YellowBrick Visualizer Info
export interface YellowBrickInfo {
	name: string;
	category: string;
	description: string;
	interpretation: string;
	guidance: Record<string, string>;
	docs_url: string;
}

export interface YellowBrickInfoData {
	visualizers: Record<string, YellowBrickInfo>;
}

// Dropdown options structure
export type DropdownOptions = Record<string, string[]>;

// Form data structure
export type FormData = Record<string, string | number | boolean>;

// Prediction results
export interface PredictionResult {
	prediction?: number | string;
	probability?: number;
	fraud_probability?: number;
	legitimate_probability?: number;
	eta_minutes?: number;
	cluster_id?: number;
	cluster_center?: number[];
	model_source?: string;
	error?: string;
}

// MLflow metrics
export interface MLflowMetrics {
	[key: string]: number | string;
}

// SQL query results
export interface SQLQueryResult {
	columns: string[];
	data: Record<string, unknown>[];
	row_count: number;
	execution_time_ms?: number;
}

// Batch training status (matching Reflex /status endpoint)
export interface BatchTrainingStatus {
	status: string; // 'running', 'completed', 'failed', 'idle'
	status_message?: string;
	progress_percent?: number;
	current_stage?: string;
	metrics_preview?: Record<string, number>;
	catboost_log?: Record<string, number | string>;
	kmeans_log?: Record<string, number | string>;
	total_rows?: number;
	error?: string;
}

// Training mode
export type TrainingMode = 'percentage' | 'max_rows';

// YellowBrick image response
export interface YellowBrickImage {
	image_base64: string;
	visualizer_name: string;
}

// API response types
export interface ApiResponse<T> {
	data?: T;
	error?: string;
	status?: number;
}

export interface TrainingResponse {
	success?: boolean;
	status?: string; // 'started', 'error', etc.
	message?: string;
	run_id?: string;
	error?: string;
}

export interface PredictResponse {
	prediction: number | string;
	probability?: number;
	fraud_probability?: number;
	legitimate_probability?: number;
	eta_minutes?: number;
	cluster_id?: number;
	cluster_center?: number[];
	model_source?: string;
}

export interface ModelStatusResponse {
	available: boolean;
	last_trained?: string;
	trained_at?: string;
	run_id?: string;
	model_name?: string;
	experiment_id?: string | number;
	experiment_url?: string;
}

export interface TrainingStatusResponse {
	training: boolean;
	model_key?: string;
	project_name?: string;
}

// Incremental ML sample data
export interface IncrementalSample {
	[key: string]: unknown;
}

// Tab names
export type TabName = 'incremental_ml' | 'batch_ml' | 'delta_lake_sql';

// Batch sub-tab names
export type BatchSubTab = 'prediction' | 'metrics';

// Toast notification types
export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
	id: string;
	type: ToastType;
	message: string;
	duration?: number;
}

// Page route mapping
export const PAGE_ROUTES = {
	home: '/',
	tfd: {
		incremental: '/tfd/incremental',
		batch: {
			prediction: '/tfd/batch/prediction',
			metrics: '/tfd/batch/metrics'
		},
		sql: '/tfd/sql'
	},
	eta: {
		incremental: '/eta/incremental',
		batch: {
			prediction: '/eta/batch/prediction',
			metrics: '/eta/batch/metrics'
		},
		sql: '/eta/sql'
	},
	ecci: {
		incremental: '/ecci/incremental',
		batch: {
			prediction: '/ecci/batch/prediction',
			metrics: '/ecci/batch/metrics'
		},
		sql: '/ecci/sql'
	}
} as const;

// Model names for display
export const MODEL_NAMES = {
	incremental: {
		'Transaction Fraud Detection': 'Adaptive Random Forest Classifier (River)',
		'Estimated Time of Arrival': 'Adaptive Random Forest Regressor (River)',
		'E-Commerce Customer Interactions': 'DBSTREAM Clustering (River)'
	},
	batch: {
		'Transaction Fraud Detection': 'CatBoost Classifier',
		'Estimated Time of Arrival': 'CatBoost Regressor',
		'E-Commerce Customer Interactions': 'KMeans (Scikit-Learn)'
	}
} as const;

// MLflow model keys
export const MLFLOW_MODEL_NAMES = {
	incremental: {
		'Transaction Fraud Detection': 'ARFClassifier',
		'Estimated Time of Arrival': 'ARFRegressor',
		'E-Commerce Customer Interactions': 'DBSTREAM'
	},
	batch: {
		'Transaction Fraud Detection': 'CatBoostClassifier',
		'Estimated Time of Arrival': 'CatBoostRegressor',
		'E-Commerce Customer Interactions': 'KMeans'
	}
} as const;
