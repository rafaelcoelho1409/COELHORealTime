<script lang="ts">
	import { metricInfoDialogOpen, metricInfoDialogContent } from '$stores';
	import { X, ExternalLink, Calculator, HelpCircle, Target, TrendingUp } from 'lucide-svelte';
	import { scale, fade } from 'svelte/transition';
	import { onMount } from 'svelte';

	let katexLoaded = $state(false);

	onMount(() => {
		// Load KaTeX CSS and JS for LaTeX rendering
		if (!document.getElementById('katex-css')) {
			const link = document.createElement('link');
			link.id = 'katex-css';
			link.rel = 'stylesheet';
			link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
			document.head.appendChild(link);
		}

		if (!(window as any).katex) {
			const script = document.createElement('script');
			script.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js';
			script.onload = () => {
				katexLoaded = true;
			};
			document.head.appendChild(script);
		} else {
			katexLoaded = true;
		}
	});

	function close() {
		metricInfoDialogOpen.set(false);
		metricInfoDialogContent.set(null);
	}

	function handleBackdropClick(e: MouseEvent) {
		if (e.target === e.currentTarget) {
			close();
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') {
			close();
		}
	}

	// Simple markdown parser for bold, italic, code, and links
	function parseMarkdown(text: string): string {
		if (!text) return '';
		return text
			// Bold: **text** or __text__
			.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
			.replace(/__(.+?)__/g, '<strong>$1</strong>')
			// Italic: *text* or _text_ (but not inside words)
			.replace(/(?<!\w)\*([^*]+)\*(?!\w)/g, '<em>$1</em>')
			// Inline code: `code`
			.replace(/`(.+?)`/g, '<code class="rounded bg-muted px-1 py-0.5 font-mono text-xs">$1</code>')
			// Links: [text](url)
			.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-primary underline hover:text-primary/80">$1</a>')
			// Line breaks
			.replace(/\n/g, '<br>');
	}

	// Strip LaTeX delimiters ($$...$$ or $...$)
	function stripLatexDelimiters(formula: string): string {
		if (!formula) return '';
		// Remove $$ delimiters (block math)
		let cleaned = formula.trim();
		if (cleaned.startsWith('$$') && cleaned.endsWith('$$')) {
			cleaned = cleaned.slice(2, -2).trim();
		} else if (cleaned.startsWith('$') && cleaned.endsWith('$')) {
			cleaned = cleaned.slice(1, -1).trim();
		}
		return cleaned;
	}

	// Render LaTeX formula using KaTeX
	function renderLatex(formula: string): string {
		if (!formula) return '';

		const cleaned = stripLatexDelimiters(formula);

		if (!(window as any).katex) {
			return `<code class="font-mono text-sm">${cleaned}</code>`;
		}
		try {
			return (window as any).katex.renderToString(cleaned, {
				throwOnError: false,
				displayMode: true,
				output: 'html',
				strict: false
			});
		} catch (e) {
			console.error('KaTeX error:', e);
			return `<code class="font-mono text-sm">${cleaned}</code>`;
		}
	}

	// Check if the formula contains LaTeX
	function hasLatex(formula: string): boolean {
		if (!formula) return false;
		return formula.includes('$$') || formula.includes('\\frac') || formula.includes('\\text') ||
		       formula.includes('\\cdot') || formula.includes('\\sqrt') || formula.includes('^{') ||
		       formula.includes('_{');
	}
</script>

<svelte:window onkeydown={handleKeydown} />

{#if $metricInfoDialogOpen && $metricInfoDialogContent}
	<!-- Backdrop -->
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
		role="button"
		tabindex="-1"
		onclick={handleBackdropClick}
		onkeydown={(e) => e.key === 'Enter' && close()}
		transition:fade={{ duration: 200 }}
	>
		<!-- Dialog -->
		<div
			class="relative w-full max-w-lg rounded-lg border border-border bg-card shadow-2xl"
			transition:scale={{ duration: 200, start: 0.95 }}
		>
			<!-- Header -->
			<div
				class="flex items-center justify-between border-b border-border bg-muted/30 px-6 py-4 rounded-t-lg"
			>
				<div class="flex items-center gap-3">
					<div class="flex h-10 w-10 items-center justify-center rounded-md bg-primary/10">
						<Calculator class="h-5 w-5 text-primary" />
					</div>
					<div>
						<h2 class="text-lg font-semibold text-foreground">
							{$metricInfoDialogContent.name}
						</h2>
						<p class="text-xs text-muted-foreground">Metric Details</p>
					</div>
				</div>
				<button
					type="button"
					class="rounded-md p-2 text-muted-foreground transition-all hover:bg-muted hover:text-foreground"
					onclick={close}
				>
					<X class="h-5 w-5" />
				</button>
			</div>

			<!-- Content -->
			<div class="p-6 space-y-5 max-h-[70vh] overflow-y-auto">
				<!-- Formula -->
				{#if $metricInfoDialogContent.formula}
					<div>
						<div class="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
							<Calculator class="h-3.5 w-3.5" />
							Formula
						</div>
						<div
							class="mt-2 overflow-x-auto rounded-md border border-border bg-muted/50 p-4 text-center"
						>
							{#if hasLatex($metricInfoDialogContent.formula) && katexLoaded}
								<div class="katex-formula">
									{@html renderLatex($metricInfoDialogContent.formula)}
								</div>
							{:else}
								<code class="font-mono text-sm text-foreground">{stripLatexDelimiters($metricInfoDialogContent.formula)}</code>
							{/if}
						</div>
					</div>
				{/if}

				<!-- Explanation -->
				<div>
					<div class="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
						<HelpCircle class="h-3.5 w-3.5" />
						Explanation
					</div>
					<div class="mt-2 text-sm leading-relaxed text-foreground">
						{@html parseMarkdown($metricInfoDialogContent.explanation || '')}
					</div>
				</div>

				<!-- Context -->
				{#if $metricInfoDialogContent.context}
					<div>
						<div class="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
							<TrendingUp class="h-3.5 w-3.5" />
							Context
						</div>
						<div class="mt-2 text-sm leading-relaxed text-foreground">
							{@html parseMarkdown($metricInfoDialogContent.context)}
						</div>
					</div>
				{/if}

				<!-- Range & Optimal -->
				{#if $metricInfoDialogContent.range || $metricInfoDialogContent.optimal}
					<div class="grid grid-cols-2 gap-4">
						{#if $metricInfoDialogContent.range}
							<div class="rounded-md border border-border bg-muted/30 p-4">
								<div class="text-xs font-medium uppercase tracking-wider text-muted-foreground">
									Range
								</div>
								<p class="mt-1 text-sm font-semibold text-foreground">
									{$metricInfoDialogContent.range}
								</p>
							</div>
						{/if}

						{#if $metricInfoDialogContent.optimal}
							<div class="rounded-md border border-green-500/30 bg-green-500/10 p-4">
								<div class="flex items-center gap-1 text-xs font-medium uppercase tracking-wider text-green-600 dark:text-green-400">
									<Target class="h-3 w-3" />
									Optimal
								</div>
								<p class="mt-1 text-sm font-semibold text-green-700 dark:text-green-300">
									{$metricInfoDialogContent.optimal}
								</p>
							</div>
						{/if}
					</div>
				{/if}

				<!-- Documentation Link -->
				{#if $metricInfoDialogContent.docsUrl}
					<div class="pt-2 border-t border-border">
						<a
							href={$metricInfoDialogContent.docsUrl}
							target="_blank"
							rel="noopener noreferrer"
							class="flex w-full items-center justify-center gap-2 rounded-md bg-primary/10 px-4 py-3 text-sm font-medium text-primary transition-all hover:bg-primary/20"
						>
							<ExternalLink class="h-4 w-4" />
							View Documentation
						</a>
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	:global(.katex-formula .katex) {
		font-size: 1.1em;
	}
	:global(.katex-formula .katex-display) {
		margin: 0;
	}
</style>
