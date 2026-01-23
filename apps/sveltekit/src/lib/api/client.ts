import type { ApiResponse } from '$types';

// Base URL for FastAPI backend
const BASE_URL = typeof window !== 'undefined'
	? (import.meta.env.PUBLIC_FASTAPI_URL || 'http://localhost:8001')
	: 'http://localhost:8001';

// API Router prefixes
export const API = {
	INCREMENTAL: `${BASE_URL}/api/v1/incremental`,
	BATCH: `${BASE_URL}/api/v1/batch`,
	SQL: `${BASE_URL}/api/v1/sql`
} as const;

// Default fetch options
const defaultOptions: RequestInit = {
	headers: {
		'Content-Type': 'application/json'
	}
};

// Generic fetch wrapper with error handling
export async function apiFetch<T>(
	url: string,
	options: RequestInit = {}
): Promise<ApiResponse<T>> {
	try {
		const response = await fetch(url, {
			...defaultOptions,
			...options,
			headers: {
				...defaultOptions.headers,
				...options.headers
			}
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
		const message = error instanceof Error ? error.message : 'Unknown error occurred';
		return { error: message };
	}
}

// GET request
export async function apiGet<T>(endpoint: string): Promise<ApiResponse<T>> {
	return apiFetch<T>(endpoint, { method: 'GET' });
}

// POST request
export async function apiPost<T>(endpoint: string, body?: unknown): Promise<ApiResponse<T>> {
	return apiFetch<T>(endpoint, {
		method: 'POST',
		body: body ? JSON.stringify(body) : undefined
	});
}

// PUT request
export async function apiPut<T>(endpoint: string, body?: unknown): Promise<ApiResponse<T>> {
	return apiFetch<T>(endpoint, {
		method: 'PUT',
		body: body ? JSON.stringify(body) : undefined
	});
}

// DELETE request
export async function apiDelete<T>(endpoint: string): Promise<ApiResponse<T>> {
	return apiFetch<T>(endpoint, { method: 'DELETE' });
}
