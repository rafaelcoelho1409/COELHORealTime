<script lang="ts">
	import {
		Home,
		ChevronDown,
		ChevronRight,
		CreditCard,
		Clock,
		ShoppingCart,
		Activity,
		Layers,
		Database,
		ExternalLink
	} from 'lucide-svelte';

	let applicationsOpen = $state(false);
	let servicesOpen = $state(false);
	let tfdSubOpen = $state(false);
	let etaSubOpen = $state(false);
	let ecciSubOpen = $state(false);

	function closeAllMenus() {
		applicationsOpen = false;
		servicesOpen = false;
		tfdSubOpen = false;
		etaSubOpen = false;
		ecciSubOpen = false;
	}

	// Application menu items
	const applications = [
		{
			name: 'Transaction Fraud Detection',
			icon: CreditCard,
			color: 'text-red-500',
			bgColor: 'bg-red-500/10',
			description: 'Real-time fraud detection',
			links: [
				{ name: 'Incremental ML', href: '/tfd/incremental', icon: Activity },
				{ name: 'Batch ML', href: '/tfd/batch/prediction', icon: Layers },
				{ name: 'Delta Lake SQL', href: '/tfd/sql', icon: Database }
			]
		},
		{
			name: 'Estimated Time of Arrival',
			icon: Clock,
			color: 'text-blue-500',
			bgColor: 'bg-blue-500/10',
			description: 'Delivery time prediction',
			links: [
				{ name: 'Incremental ML', href: '/eta/incremental', icon: Activity },
				{ name: 'Batch ML', href: '/eta/batch/prediction', icon: Layers },
				{ name: 'Delta Lake SQL', href: '/eta/sql', icon: Database }
			]
		},
		{
			name: 'E-Commerce Customer Interactions',
			icon: ShoppingCart,
			color: 'text-green-500',
			bgColor: 'bg-green-500/10',
			description: 'Customer clustering & analysis',
			links: [
				{ name: 'Incremental ML', href: '/ecci/incremental', icon: Activity },
				{ name: 'Batch ML', href: '/ecci/batch/prediction', icon: Layers },
				{ name: 'Delta Lake SQL', href: '/ecci/sql', icon: Database }
			]
		}
	];

	// Service links
	const services = [
		{
			name: 'FastAPI',
			href: 'http://localhost:8001/docs',
			icon: 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg',
			description: 'API Documentation'
		},
		{
			name: 'Spark',
			href: 'http://localhost:4040',
			icon: 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/apachespark/apachespark-original.svg',
			description: 'Spark UI'
		},
		{
			name: 'MLflow',
			href: 'http://localhost:5001',
			icon: 'https://cdn.simpleicons.org/mlflow/0194E2',
			description: 'Experiment Tracking'
		},
		{
			name: 'MinIO',
			href: 'http://localhost:9001',
			icon: 'https://cdn.simpleicons.org/minio/C72E49',
			description: 'Object Storage'
		},
		{
			name: 'Prometheus',
			href: 'http://localhost:9090',
			icon: 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/prometheus/prometheus-original.svg',
			description: 'Metrics'
		},
		{
			name: 'Grafana',
			href: 'http://localhost:3001',
			icon: 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/grafana/grafana-original.svg',
			description: 'Dashboards'
		},
		{
			name: 'Alertmanager',
			href: 'http://localhost:9094',
			icon: 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/prometheus/prometheus-original.svg',
			description: 'Alert Management'
		}
	];

	let activeAppIndex = $state<number | null>(null);
</script>

<svelte:window
	onclick={(e) => {
		const target = e.target as HTMLElement;
		if (!target.closest('.dropdown-container')) {
			closeAllMenus();
		}
	}}
/>

<nav class="sticky top-0 z-[1000] w-full border-b border-white/10 bg-slate-950/95 shadow-xl">
	<div class="mx-auto flex max-w-[1400px] items-center justify-between px-6 py-2">
		<!-- Left section - Logo and Title -->
		<a href="/" class="group flex items-center gap-6">
			<div class="relative">
				<div
					class="absolute -inset-1 rounded-md bg-gradient-to-r from-amber-200/30 via-amber-400/30 to-amber-200/30 opacity-50 blur-sm transition-opacity group-hover:opacity-75"
				></div>
				<img
					src="/coelho_realtime_logo.png"
					alt="COELHO RealTime"
					class="relative h-auto w-20 rounded-md object-contain"
				/>
			</div>
			<span class="font-brand text-3xl font-semibold uppercase tracking-[0.14em] text-slate-100">
				COELHO RealTime
			</span>
		</a>

		<!-- Right section - Navigation -->
		<div class="flex items-center gap-2">
			<!-- Home Link -->
			<a
				href="/"
				class="flex items-center gap-2 rounded-md px-3 py-2 text-slate-300 transition-all hover:bg-white/10 hover:text-white"
			>
				<Home class="h-4 w-4" />
				<span class="text-sm font-medium">Home</span>
			</a>

			<!-- Applications Dropdown -->
			<div class="dropdown-container relative">
				<button
					type="button"
					class="flex items-center gap-2 rounded-md bg-white/5 px-4 py-2 text-sm font-medium text-slate-200 ring-1 ring-white/10 transition-all hover:bg-white/10 hover:text-white hover:ring-white/20"
					onclick={() => {
						applicationsOpen = !applicationsOpen;
						servicesOpen = false;
						activeAppIndex = null;
					}}
				>
					<Activity class="h-4 w-4 text-purple-400" />
					<span>Applications</span>
					<ChevronDown
						class="h-4 w-4 text-slate-400 transition-transform {applicationsOpen
							? 'rotate-180'
							: ''}"
					/>
				</button>

				{#if applicationsOpen}
					<div
						class="absolute right-0 top-full mt-2 flex overflow-hidden rounded-lg border border-white/10 bg-slate-900/95 shadow-2xl backdrop-blur-xl"
					>
						<!-- App List -->
						<div class="w-72 border-r border-white/10 p-2">
							{#each applications as app, index}
								{@const AppIcon = app.icon}
								<button
									type="button"
								class="flex w-full items-center gap-3 rounded-md px-3 py-3 text-left transition-all {activeAppIndex ===
									index
										? 'bg-white/10'
										: 'hover:bg-white/5'}"
									onmouseenter={() => (activeAppIndex = index)}
									onclick={() => (activeAppIndex = index)}
								>
								<div class="rounded-md p-2 {app.bgColor}">
										<AppIcon class="h-5 w-5 {app.color}" />
									</div>
									<div class="flex-1">
										<div class="text-sm font-medium text-white">{app.name}</div>
										<div class="text-xs text-slate-400">{app.description}</div>
									</div>
									<ChevronRight class="h-4 w-4 text-slate-500" />
								</button>
							{/each}
						</div>

						<!-- Sub-links Panel -->
						<div class="w-56 bg-slate-800/50 p-2">
							{#if activeAppIndex !== null}
								{@const activeApp = applications[activeAppIndex]}
								<div class="mb-2 px-3 py-2">
									<div class="text-xs font-semibold uppercase tracking-wider text-slate-500">
										Navigate to
									</div>
								</div>
								{#each activeApp.links as link}
									{@const LinkIcon = link.icon}
									<a
										href={link.href}
								class="flex items-center gap-3 rounded-md px-3 py-2.5 text-slate-300 transition-all hover:bg-white/10 hover:text-white"
										onclick={closeAllMenus}
									>
										<LinkIcon class="h-4 w-4 {activeApp.color}" />
										<span class="text-sm font-medium">{link.name}</span>
									</a>
								{/each}
							{:else}
								<div class="flex h-full items-center justify-center p-6 text-center">
									<div class="text-sm text-slate-500">
										Hover over an application to see options
									</div>
								</div>
							{/if}
						</div>
					</div>
				{/if}
			</div>

			<!-- Services Dropdown -->
			<div class="dropdown-container relative">
				<button
					type="button"
					class="flex items-center gap-2 rounded-md bg-white/5 px-4 py-2 text-sm font-medium text-slate-200 ring-1 ring-white/10 transition-all hover:bg-white/10 hover:text-white hover:ring-white/20"
					onclick={() => {
						servicesOpen = !servicesOpen;
						applicationsOpen = false;
					}}
				>
					<Database class="h-4 w-4 text-emerald-400" />
					<span>Services</span>
					<ChevronDown
						class="h-4 w-4 text-slate-400 transition-transform {servicesOpen ? 'rotate-180' : ''}"
					/>
				</button>

				{#if servicesOpen}
					<div
						class="absolute right-0 top-full mt-2 w-64 overflow-hidden rounded-lg border border-white/10 bg-slate-900/95 p-2 shadow-2xl backdrop-blur-xl"
					>
						<div class="mb-2 px-3 py-2">
							<div class="text-xs font-semibold uppercase tracking-wider text-slate-500">
								External Services
							</div>
						</div>
						<div class="grid gap-1">
							{#each services as service}
							<a
								href={service.href}
								target="_blank"
								rel="noopener noreferrer"
								class="group flex items-center gap-3 rounded-md px-3 py-2.5 transition-all hover:bg-white/10"
								onclick={closeAllMenus}
							>
								<div class="flex h-8 w-8 items-center justify-center rounded-md bg-white/5">
										<img src={service.icon} alt={service.name} class="h-5 w-5" />
									</div>
									<div class="flex-1">
										<div class="text-sm font-medium text-white">{service.name}</div>
										<div class="text-xs text-slate-500">{service.description}</div>
									</div>
									<ExternalLink
										class="h-3.5 w-3.5 text-slate-600 transition-colors group-hover:text-slate-400"
									/>
								</a>
							{/each}
						</div>
					</div>
				{/if}
			</div>
		</div>
	</div>

	<!-- Gradient line at bottom -->
	<div class="h-[1px] w-full bg-gradient-to-r from-transparent via-amber-400/40 to-transparent"></div>
</nav>
