import { writable, derived } from 'svelte/store';
import type { ProjectName, SQLQueryResult } from '$types';
import { currentProject } from './shared';

// =============================================================================
// SQL Query State (per project)
// =============================================================================
export const sqlQueryInput = writable<Record<ProjectName, string>>({
	'Transaction Fraud Detection': 'SELECT * FROM data LIMIT 100',
	'Estimated Time of Arrival': 'SELECT * FROM data LIMIT 100',
	'E-Commerce Customer Interactions': 'SELECT * FROM data LIMIT 100'
});

export const sqlQueryResults = writable<Record<ProjectName, SQLQueryResult>>({
	'Transaction Fraud Detection': { columns: [], data: [], row_count: 0 },
	'Estimated Time of Arrival': { columns: [], data: [], row_count: 0 },
	'E-Commerce Customer Interactions': { columns: [], data: [], row_count: 0 }
});

export const sqlLoading = writable<Record<ProjectName, boolean>>({
	'Transaction Fraud Detection': false,
	'Estimated Time of Arrival': false,
	'E-Commerce Customer Interactions': false
});

export const sqlError = writable<Record<ProjectName, string>>({
	'Transaction Fraud Detection': '',
	'Estimated Time of Arrival': '',
	'E-Commerce Customer Interactions': ''
});

export const sqlExecutionTime = writable<Record<ProjectName, number>>({
	'Transaction Fraud Detection': 0,
	'Estimated Time of Arrival': 0,
	'E-Commerce Customer Interactions': 0
});

// =============================================================================
// SQL Query Templates
// =============================================================================
export const SQL_QUERY_TEMPLATES = {
	'Transaction Fraud Detection': [
		{ name: 'Sample Data', query: 'SELECT * FROM data LIMIT 100' },
		{ name: 'Fraud Cases', query: 'SELECT * FROM data WHERE is_fraud = 1 LIMIT 100' },
		{ name: 'High Value (>$1000)', query: 'SELECT * FROM data WHERE amount > 1000 LIMIT 100' },
		{
			name: 'Fraud by Merchant',
			query:
				'SELECT merchant_id, COUNT(*) as fraud_count FROM data WHERE is_fraud = 1 GROUP BY merchant_id ORDER BY fraud_count DESC LIMIT 20'
		},
		{
			name: 'Transaction Types',
			query:
				'SELECT transaction_type, COUNT(*) as count, AVG(amount) as avg_amount FROM data GROUP BY transaction_type ORDER BY count DESC'
		},
		{ name: 'Row Count', query: 'SELECT COUNT(*) as total_rows FROM data' }
	],
	'Estimated Time of Arrival': [
		{ name: 'Sample Data', query: 'SELECT * FROM data LIMIT 100' },
		{ name: 'Long Trips (>50km)', query: 'SELECT * FROM data WHERE estimated_distance_km > 50 LIMIT 100' },
		{
			name: 'Weather Impact',
			query:
				'SELECT weather, COUNT(*) as trips, AVG(simulated_actual_travel_time_seconds) as avg_time FROM data GROUP BY weather ORDER BY trips DESC'
		},
		{
			name: 'Driver Performance',
			query:
				'SELECT driver_id, COUNT(*) as trips, AVG(driver_rating) as avg_rating FROM data GROUP BY driver_id ORDER BY trips DESC LIMIT 20'
		},
		{
			name: 'Vehicle Types',
			query:
				'SELECT vehicle_type, COUNT(*) as count, AVG(estimated_distance_km) as avg_distance FROM data GROUP BY vehicle_type ORDER BY count DESC'
		},
		{ name: 'Row Count', query: 'SELECT COUNT(*) as total_rows FROM data' }
	],
	'E-Commerce Customer Interactions': [
		{ name: 'Sample Data', query: 'SELECT * FROM data LIMIT 100' },
		{ name: 'Purchases', query: "SELECT * FROM data WHERE event_type = 'purchase' LIMIT 100" },
		{
			name: 'By Category',
			query:
				'SELECT product_category, COUNT(*) as count FROM data GROUP BY product_category ORDER BY count DESC'
		},
		{
			name: 'Event Types',
			query: 'SELECT event_type, COUNT(*) as count FROM data GROUP BY event_type ORDER BY count DESC'
		},
		{
			name: 'Top Products',
			query:
				"SELECT product_id, COUNT(*) as views, SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchases FROM data GROUP BY product_id ORDER BY views DESC LIMIT 20"
		},
		{ name: 'Row Count', query: 'SELECT COUNT(*) as total_rows FROM data' }
	]
} as const;

// =============================================================================
// Derived Stores for Current Project
// =============================================================================
export const currentSqlQuery = derived(
	[sqlQueryInput, currentProject],
	([$sqlQueryInput, $currentProject]) => $sqlQueryInput[$currentProject]
);

export const currentSqlResults = derived(
	[sqlQueryResults, currentProject],
	([$sqlQueryResults, $currentProject]) => $sqlQueryResults[$currentProject]
);

export const currentSqlLoading = derived(
	[sqlLoading, currentProject],
	([$sqlLoading, $currentProject]) => $sqlLoading[$currentProject]
);

export const currentSqlError = derived(
	[sqlError, currentProject],
	([$sqlError, $currentProject]) => $sqlError[$currentProject]
);

export const currentSqlTemplates = derived(
	currentProject,
	($currentProject) => SQL_QUERY_TEMPLATES[$currentProject]
);

// =============================================================================
// Helper Functions
// =============================================================================
export function updateSqlQuery(project: ProjectName, query: string) {
	sqlQueryInput.update((s) => ({ ...s, [project]: query }));
}

export function updateSqlResults(project: ProjectName, results: SQLQueryResult) {
	sqlQueryResults.update((s) => ({ ...s, [project]: results }));
}

export function setSqlLoading(project: ProjectName, loading: boolean) {
	sqlLoading.update((s) => ({ ...s, [project]: loading }));
}

export function setSqlError(project: ProjectName, error: string) {
	sqlError.update((s) => ({ ...s, [project]: error }));
}

export function setSqlExecutionTime(project: ProjectName, time: number) {
	sqlExecutionTime.update((s) => ({ ...s, [project]: time }));
}
