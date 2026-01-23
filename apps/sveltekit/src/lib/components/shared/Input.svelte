<script lang="ts">
	import { cn } from '$lib/utils';

	interface Props {
		type?: 'text' | 'number' | 'email' | 'password' | 'search' | 'date' | 'time' | 'datetime-local' | 'tel' | 'url';
		value?: string | number;
		placeholder?: string;
		disabled?: boolean;
		readonly?: boolean;
		required?: boolean;
		id?: string;
		name?: string;
		min?: number | string;
		max?: number | string;
		step?: number | string;
		pattern?: string;
		autocomplete?: string;
		class?: string;
		oninput?: (e: Event & { currentTarget: HTMLInputElement }) => void;
		onchange?: (e: Event & { currentTarget: HTMLInputElement }) => void;
		onblur?: (e: Event & { currentTarget: HTMLInputElement }) => void;
		onfocus?: (e: Event & { currentTarget: HTMLInputElement }) => void;
	}

	let {
		type = 'text',
		value = '',
		placeholder,
		disabled = false,
		readonly = false,
		required = false,
		id,
		name,
		min,
		max,
		step,
		pattern,
		autocomplete,
		class: className,
		oninput,
		onchange,
		onblur,
		onfocus
	}: Props = $props();
</script>

<input
	{type}
	{value}
	{placeholder}
	{disabled}
	{readonly}
	{required}
	{id}
	{name}
	{min}
	{max}
	{step}
	{pattern}
	{autocomplete}
	class={cn(
		'flex h-10 w-full rounded-md border border-input bg-background px-4 py-2 text-sm transition-all duration-200',
		'shadow-sm hover:shadow',
		'ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium',
		'placeholder:text-muted-foreground',
		'hover:border-muted-foreground/50 hover:bg-background/80',
		'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:border-primary focus-visible:shadow-md',
		'disabled:cursor-not-allowed disabled:opacity-50 disabled:bg-muted disabled:hover:shadow-none',
		'read-only:bg-muted/50 read-only:hover:border-input',
		className
	)}
	{oninput}
	{onchange}
	{onblur}
	{onfocus}
/>
