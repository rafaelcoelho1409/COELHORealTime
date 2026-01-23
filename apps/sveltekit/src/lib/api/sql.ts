import { API, apiPost } from './client';
import type { ProjectName, SQLQueryResult, ApiResponse } from '$types';

/**
 * Execute a SQL query against Delta Lake with optional abort signal
 */
export async function executeQuery(
	projectName: ProjectName,
	query: string,
	limit = 100,
	signal?: AbortSignal
): Promise<ApiResponse<SQLQueryResult>> {
	const url = `${API.SQL}/query`;
	const body = {
		project_name: projectName,
		query,
		limit
	};

	try {
		const response = await fetch(url, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body),
			signal
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => ({}));
			return {
				error: errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
				status: response.status
			};
		}

		const data = await response.json();
		return { data, status: response.status };
	} catch (error) {
		if (error instanceof Error && error.name === 'AbortError') {
			return { error: 'Query cancelled by user' };
		}
		const message = error instanceof Error ? error.message : 'Unknown error occurred';
		return { error: message };
	}
}

/**
 * Get table schema for a project's Delta Lake table
 */
export async function getTableSchema(projectName: ProjectName) {
	return apiPost<{
		columns: Array<{
			name: string;
			type: string;
			nullable: boolean;
		}>;
		approximate_row_count?: number;
	}>(`${API.SQL}/schema`, {
		project_name: projectName
	});
}

/**
 * Get available tables/views
 */
export async function getTables(projectName: ProjectName) {
	return apiPost<{ tables: string[] }>(`${API.SQL}/tables`, {
		project_name: projectName
	});
}
