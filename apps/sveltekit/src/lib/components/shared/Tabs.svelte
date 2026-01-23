<script lang="ts" module>
	import { setContext, getContext } from 'svelte';

	export interface TabsContext {
		value: string;
		setValue: (value: string) => void;
	}

	export const TABS_CONTEXT_KEY = Symbol('tabs');
</script>

<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		value: string;
		onValueChange?: (value: string) => void;
		children: Snippet;
		class?: string;
	}

	let { value = $bindable(), onValueChange, children, class: className = '' }: Props = $props();

	function setValue(newValue: string) {
		value = newValue;
		onValueChange?.(newValue);
	}

	setContext<TabsContext>(TABS_CONTEXT_KEY, {
		get value() {
			return value;
		},
		setValue
	});
</script>

<div class={className}>
	{@render children()}
</div>
