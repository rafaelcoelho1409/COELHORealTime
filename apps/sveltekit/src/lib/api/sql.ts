import { API, apiPost } from './client';
import type { ProjectName, SQLQueryResult } from '$types';

/**
 * Execute a SQL query against Delta Lake
 */
export async function executeQuery(projectName: ProjectName, query: string) {
	return apiPost<SQLQueryResult>(`${API.SQL}/query`, {
		project_name: projectName,
		query
	});
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
