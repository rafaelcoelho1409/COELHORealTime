export { cn } from './cn';

/**
 * Format a number to a fixed number of decimal places
 */
export function formatNumber(value: number | undefined, decimals = 4): string {
	if (value === undefined || value === null || isNaN(value)) {
		return '-';
	}
	return value.toFixed(decimals);
}

/**
 * Format a percentage value
 */
export function formatPercent(value: number | undefined, decimals = 2): string {
	if (value === undefined || value === null || isNaN(value)) {
		return '-';
	}
	return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format a date/time string
 */
export function formatDateTime(dateStr: string | null | undefined): string {
	if (!dateStr) return '-';
	try {
		const date = new Date(dateStr);
		return date.toLocaleString();
	} catch {
		return dateStr;
	}
}

/**
 * Debounce function
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
	fn: T,
	delay: number
): (...args: Parameters<T>) => void {
	let timeoutId: ReturnType<typeof setTimeout>;
	return (...args: Parameters<T>) => {
		clearTimeout(timeoutId);
		timeoutId = setTimeout(() => fn(...args), delay);
	};
}

/**
 * Sleep for a given number of milliseconds
 */
export function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Generate a random ID
 */
export function generateId(): string {
	return Math.random().toString(36).substring(2, 11);
}
