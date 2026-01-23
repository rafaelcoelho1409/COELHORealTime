<script lang="ts">
	import { toasts, removeToast } from '$stores';
	import { cn } from '$lib/utils';
	import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-svelte';
	import { fly } from 'svelte/transition';

	const icons = {
		success: CheckCircle,
		error: AlertCircle,
		warning: AlertTriangle,
		info: Info
	};

	const styles = {
		success:
			'bg-green-50 text-green-900 border-green-200 dark:bg-green-950 dark:text-green-100 dark:border-green-800',
		error:
			'bg-red-50 text-red-900 border-red-200 dark:bg-red-950 dark:text-red-100 dark:border-red-800',
		warning:
			'bg-amber-50 text-amber-900 border-amber-200 dark:bg-amber-950 dark:text-amber-100 dark:border-amber-800',
		info: 'bg-blue-50 text-blue-900 border-blue-200 dark:bg-blue-950 dark:text-blue-100 dark:border-blue-800'
	};

	const iconStyles = {
		success: 'text-green-600 dark:text-green-400',
		error: 'text-red-600 dark:text-red-400',
		warning: 'text-amber-600 dark:text-amber-400',
		info: 'text-blue-600 dark:text-blue-400'
	};

	const progressStyles = {
		success: 'bg-green-500',
		error: 'bg-red-500',
		warning: 'bg-amber-500',
		info: 'bg-blue-500'
	};
</script>

<div
	class="pointer-events-none fixed bottom-0 right-0 z-[100] flex flex-col items-end gap-3 p-6"
	aria-live="polite"
>
	{#each $toasts as toast (toast.id)}
		{@const Icon = icons[toast.type]}
		<div
			class={cn(
			'pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-lg border p-4 shadow-lg backdrop-blur-sm',
				styles[toast.type]
			)}
			transition:fly={{ x: 100, duration: 300 }}
		>
			<div
				class={cn(
					'flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-current/10',
					iconStyles[toast.type]
				)}
			>
				<Icon class={cn('h-4 w-4', iconStyles[toast.type])} />
			</div>
			<div class="flex-1 pt-0.5">
				<p class="text-sm font-medium">{toast.message}</p>
			</div>
			<button
				type="button"
			class="flex-shrink-0 rounded-md p-1.5 opacity-70 transition-all hover:bg-black/5 hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring dark:hover:bg-white/10"
				onclick={() => removeToast(toast.id)}
			>
				<span class="sr-only">Dismiss</span>
				<X class="h-4 w-4" />
			</button>
		</div>
	{/each}
</div>
