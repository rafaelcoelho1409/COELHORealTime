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
		{ name: 'All Data (100 rows)', query: 'SELECT * FROM data LIMIT 100' },
		{ name: 'Fraudulent Transactions', query: "SELECT * FROM data WHERE is_fraud = true LIMIT 100" },
		{ name: 'Legitimate Transactions', query: "SELECT * FROM data WHERE is_fraud = false LIMIT 100" },
		{ name: 'High Value Transactions', query: 'SELECT * FROM data WHERE amount > 1000 ORDER BY amount DESC LIMIT 100' },
		{ name: 'Count by Fraud Status', query: 'SELECT is_fraud, COUNT(*) as count FROM data GROUP BY is_fraud' },
		{ name: 'Transaction Stats', query: 'SELECT COUNT(*) as total, AVG(amount) as avg_amount, MAX(amount) as max_amount FROM data' }
	],
	'Estimated Time of Arrival': [
		{ name: 'All Data (100 rows)', query: 'SELECT * FROM data LIMIT 100' },
		{ name: 'Long Trips (>30min)', query: 'SELECT * FROM data WHERE actual_eta_minutes > 30 ORDER BY actual_eta_minutes DESC LIMIT 100' },
		{ name: 'Short Trips (<10min)', query: 'SELECT * FROM data WHERE actual_eta_minutes < 10 LIMIT 100' },
		{ name: 'Trip Stats', query: 'SELECT COUNT(*) as total, AVG(actual_eta_minutes) as avg_eta, MAX(actual_eta_minutes) as max_eta FROM data' },
		{ name: 'By Hour of Day', query: 'SELECT HOUR(timestamp) as hour, COUNT(*) as count, AVG(actual_eta_minutes) as avg_eta FROM data GROUP BY HOUR(timestamp) ORDER BY hour' }
	],
	'E-Commerce Customer Interactions': [
		{ name: 'All Data (100 rows)', query: 'SELECT * FROM data LIMIT 100' },
		{ name: 'High Spenders', query: 'SELECT * FROM data ORDER BY total_spend DESC LIMIT 100' },
		{ name: 'Frequent Customers', query: 'SELECT * FROM data ORDER BY purchase_frequency DESC LIMIT 100' },
		{ name: 'Customer Stats', query: 'SELECT COUNT(*) as total, AVG(total_spend) as avg_spend, AVG(purchase_frequency) as avg_frequency FROM data' },
		{ name: 'By Region', query: 'SELECT region, COUNT(*) as count, AVG(total_spend) as avg_spend FROM data GROUP BY region ORDER BY count DESC' }
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
