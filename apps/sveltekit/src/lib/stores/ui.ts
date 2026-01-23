import { writable, derived } from 'svelte/store';
import type { Toast, ToastType, TabName, BatchSubTab } from '$types';

// =============================================================================
// Toast Notifications
// =============================================================================
export const toasts = writable<Toast[]>([]);

let toastId = 0;

export function addToast(type: ToastType, message: string, duration = 5000): string {
	const id = `toast-${++toastId}`;
	const toast: Toast = { id, type, message, duration };

	toasts.update((t) => [...t, toast]);

	if (duration > 0) {
		setTimeout(() => removeToast(id), duration);
	}

	return id;
}

export function removeToast(id: string) {
	toasts.update((t) => t.filter((toast) => toast.id !== id));
}

export function clearToasts() {
	toasts.set([]);
}

// Convenience functions
export const toast = {
	success: (message: string, duration?: number) => addToast('success', message, duration),
	error: (message: string, duration?: number) => addToast('error', message, duration ?? 8000),
	warning: (message: string, duration?: number) => addToast('warning', message, duration),
	info: (message: string, duration?: number) => addToast('info', message, duration)
};

// =============================================================================
// Tab Navigation
// =============================================================================
export const currentTab = writable<TabName>('incremental_ml');
export const currentBatchSubTab = writable<BatchSubTab>('prediction');

// =============================================================================
// Dialog State
// =============================================================================
export const metricInfoDialogOpen = writable<boolean>(false);
export const metricInfoDialogContent = writable<{
	name: string;
	formula: string;
	explanation: string;
	context: string;
	range: string;
	optimal: string;
	docsUrl: string;
} | null>(null);

export const yellowBrickInfoDialogOpen = writable<boolean>(false);
export const yellowBrickInfoDialogContent = writable<{
	name: string;
	category: string;
	description: string;
	interpretation: string;
	guidance: Record<string, string>;
	docsUrl: string;
} | null>(null);

// =============================================================================
// Loading States
// =============================================================================
export const globalLoading = writable<boolean>(false);
export const pageLoading = writable<boolean>(false);

// =============================================================================
// Sidebar State
// =============================================================================
export const sidebarOpen = writable<boolean>(false);

// =============================================================================
// Theme
// =============================================================================
export const theme = writable<'light' | 'dark'>('light');

export function toggleTheme() {
	theme.update((t) => (t === 'light' ? 'dark' : 'light'));
}

// =============================================================================
// Derived Helper Stores
// =============================================================================
export const hasToasts = derived(toasts, ($toasts) => $toasts.length > 0);
