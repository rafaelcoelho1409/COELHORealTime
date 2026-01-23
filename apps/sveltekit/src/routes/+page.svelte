<script lang="ts">
	import {
		ArrowRight,
		Activity,
		Database,
		LineChart,
		Zap,
		Layers,
		GitBranch,
		BarChart3,
		Shield,
		Clock,
		ShoppingCart,
		Cpu,
		HardDrive,
		Gauge,
		PanelTop,
		Bell,
		Server,
		Workflow,
		Eye,
		FileCode,
		ExternalLink
	} from 'lucide-svelte';

	const projects = [
		{
			name: 'Transaction Fraud Detection',
			shortName: 'TFD',
			href: '/tfd/incremental',
			description: 'Real-time fraud detection using streaming ML. Identify fraudulent transactions as they happen with incremental learning models that adapt to new patterns.',
			color: 'from-amber-500 to-yellow-600',
			bgColor: 'bg-amber-500/10',
			borderColor: 'hover:border-amber-500/40',
			icon: Shield
		},
		{
			name: 'Estimated Time of Arrival',
			shortName: 'ETA',
			href: '/eta/incremental',
			description: 'Predict delivery times with high accuracy. Leverage streaming data and incremental regression models to provide real-time ETA predictions.',
			color: 'from-slate-500 to-slate-700',
			bgColor: 'bg-slate-500/10',
			borderColor: 'hover:border-slate-500/40',
			icon: Clock
		},
		{
			name: 'E-Commerce Customer Interactions',
			shortName: 'ECCI',
			href: '/ecci/incremental',
			description: 'Customer segmentation and behavior analysis. Cluster customers in real-time based on interactions and purchasing patterns.',
			color: 'from-stone-500 to-amber-600',
			bgColor: 'bg-stone-500/10',
			borderColor: 'hover:border-stone-500/40',
			icon: ShoppingCart
		}
	];

	// ML Capabilities
	const mlFeatures = [
		{
			name: 'Incremental ML',
			description: 'Stream-based learning with River. Models update continuously as new data arrives, perfect for real-time predictions without retraining.',
			icon: Activity,
			color: 'text-amber-400',
			bgColor: 'bg-amber-400/10'
		},
		{
			name: 'Batch Training',
			description: 'Traditional ML with Scikit-Learn and CatBoost. Train powerful models on historical data with full control over hyperparameters.',
			icon: Layers,
			color: 'text-slate-300',
			bgColor: 'bg-slate-500/10'
		},
		{
			name: 'YellowBrick Visualizations',
			description: 'Interactive model diagnostics with K-Elbow, Silhouette, ROC curves, feature importance, and more for deep model insights.',
			icon: Eye,
			color: 'text-amber-500',
			bgColor: 'bg-amber-500/10'
		},
		{
			name: 'MLflow Experiment Tracking',
			description: 'Complete experiment lifecycle management. Track parameters, metrics, artifacts, and compare model versions effortlessly.',
			icon: GitBranch,
			color: 'text-stone-400',
			bgColor: 'bg-stone-500/10'
		}
	];

	// Data Infrastructure
	const dataFeatures = [
		{
			name: 'Apache Kafka',
			description: 'High-throughput event streaming platform. Ingest millions of events per second with guaranteed delivery and exactly-once semantics.',
			icon: Workflow,
			color: 'text-slate-300',
			bgColor: 'bg-slate-500/10',
			href: 'http://localhost:9092'
		},
		{
			name: 'Apache Spark',
			description: 'Distributed data processing at scale. Handle massive datasets with in-memory computing and unified analytics engine.',
			icon: Zap,
			color: 'text-amber-400',
			bgColor: 'bg-amber-400/10',
			href: 'http://localhost:4040'
		},
		{
			name: 'Delta Lake',
			description: 'ACID transactions on data lakes. Time travel, schema enforcement, and audit history for reliable data pipelines.',
			icon: Database,
			color: 'text-stone-400',
			bgColor: 'bg-stone-500/10'
		},
		{
			name: 'DuckDB SQL',
			description: 'Lightning-fast analytical queries directly on Delta Lake. Full SQL support with columnar storage optimization.',
			icon: FileCode,
			color: 'text-amber-500',
			bgColor: 'bg-amber-500/10'
		}
	];

	// Infrastructure Services
	const infraFeatures = [
		{
			name: 'FastAPI Backend',
			description: 'High-performance async API layer. Auto-generated OpenAPI docs, type validation, and blazing fast response times.',
			icon: Server,
			color: 'text-slate-300',
			bgColor: 'bg-slate-500/10',
			href: 'http://localhost:8001/docs'
		},
		{
			name: 'MinIO Object Storage',
			description: 'S3-compatible storage for ML artifacts. Store models, datasets, and visualizations with enterprise-grade durability.',
			icon: HardDrive,
			color: 'text-stone-400',
			bgColor: 'bg-stone-500/10',
			href: 'http://localhost:9001'
		},
		{
			name: 'Prometheus Metrics',
			description: 'Time-series monitoring and alerting. Collect, query, and visualize metrics from all platform components.',
			icon: Gauge,
			color: 'text-amber-400',
			bgColor: 'bg-amber-400/10',
			href: 'http://localhost:9090'
		},
		{
			name: 'Grafana Dashboards',
			description: 'Beautiful real-time visualizations. Monitor system health, model performance, and business KPIs in one place.',
			icon: PanelTop,
			color: 'text-amber-300',
			bgColor: 'bg-amber-300/10',
			href: 'http://localhost:3001'
		},
		{
			name: 'Alertmanager',
			description: 'Intelligent alert routing and grouping. Get notified on critical events via email, Slack, or PagerDuty.',
			icon: Bell,
			color: 'text-stone-300',
			bgColor: 'bg-stone-500/10',
			href: 'http://localhost:9094'
		},
		{
			name: 'Kubernetes (K3d)',
			description: 'Container orchestration for scalability. Auto-scaling, self-healing, and declarative deployments with Helm charts.',
			icon: Cpu,
			color: 'text-slate-300',
			bgColor: 'bg-slate-500/10'
		}
	];
</script>

<svelte:head>
	<title>COELHO RealTime - Real-Time ML Platform</title>
</svelte:head>

<div class="min-h-screen">
	<!-- Hero Section -->
	<div class="relative overflow-hidden bg-gradient-to-b from-slate-950 via-slate-900 to-background py-20">
		<!-- Background Pattern -->
		<div class="absolute inset-0 opacity-10">
			<div class="absolute inset-0" style="background-image: radial-gradient(circle at 1px 1px, white 1px, transparent 0); background-size: 40px 40px;"></div>
		</div>

		<div class="relative mx-auto max-w-6xl px-3 sm:px-5 lg:px-6">
			<div class="text-center">
				<!-- Logo with glow -->
				<div class="relative mx-auto w-fit">
					<div class="absolute -inset-4 rounded-2xl bg-gradient-to-r from-amber-200/30 via-amber-400/30 to-amber-200/30 opacity-30 blur-xl"></div>
					<img
						src="/coelho_realtime_logo.png"
						alt="COELHO RealTime"
						class="relative mx-auto h-auto w-36 rounded-2xl object-contain"
					/>
				</div>

				<!-- Title with gradient -->
				<h1 class="mt-8 text-5xl font-bold tracking-tight sm:text-6xl">
					<span
						class="font-brand bg-gradient-to-r from-amber-200 via-amber-300 to-amber-500 bg-clip-text text-transparent"
					>
						COELHO RealTime
					</span>
				</h1>

				<p class="mx-auto mt-6 max-w-3xl text-xl text-slate-300">
					Enterprise-grade Real-Time Machine Learning Platform combining streaming data processing,
					incremental learning, and batch model training in a unified Kubernetes-native architecture.
				</p>

				<!-- Stats -->
				<div class="mt-10 flex flex-wrap justify-center gap-8">
					<div class="text-center">
						<div class="text-3xl font-bold text-white">3</div>
						<div class="text-sm text-slate-400">ML Projects</div>
					</div>
					<div class="text-center">
						<div class="text-3xl font-bold text-white">2</div>
						<div class="text-sm text-slate-400">ML Paradigms</div>
					</div>
					<div class="text-center">
						<div class="text-3xl font-bold text-white">10+</div>
						<div class="text-sm text-slate-400">Microservices</div>
					</div>
					<div class="text-center">
						<div class="text-3xl font-bold text-white">100%</div>
						<div class="text-sm text-slate-400">Cloud Native</div>
					</div>
				</div>
			</div>
		</div>
	</div>

	<div class="mx-auto max-w-6xl px-3 py-12 sm:px-5 lg:px-6">
		<!-- Projects Section -->
		<div>
			<div class="text-center">
				<h2 class="text-3xl font-bold text-foreground">ML Applications</h2>
				<p class="mt-2 text-muted-foreground">Production-ready machine learning use cases</p>
			</div>

			<div class="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
				{#each projects as project}
					{@const Icon = project.icon}
					<a
						href={project.href}
						class="group relative overflow-hidden rounded-2xl border border-border bg-card p-6 transition-all duration-300 hover:shadow-2xl hover:shadow-primary/5 {project.borderColor}"
					>
						<!-- Gradient overlay on hover -->
						<div class="absolute inset-0 bg-gradient-to-br {project.color} opacity-0 transition-opacity group-hover:opacity-5"></div>

						<div class="relative">
							<div class="flex items-start justify-between">
								<div class="rounded-xl p-3 {project.bgColor}">
									<Icon class="h-7 w-7 bg-gradient-to-br {project.color} bg-clip-text text-transparent" style="color: inherit;" />
								</div>
								<ArrowRight
									class="h-5 w-5 text-muted-foreground transition-all group-hover:translate-x-1 group-hover:text-primary"
								/>
							</div>

							<h3 class="mt-5 text-xl font-semibold text-foreground">{project.name}</h3>
							<p class="mt-3 text-sm leading-relaxed text-muted-foreground">{project.description}</p>

							<div class="mt-5 flex flex-wrap items-center gap-2">
								<span class="rounded-full bg-amber-400/10 px-3 py-1 text-xs font-medium text-amber-200">
									Incremental ML
								</span>
								<span class="rounded-full bg-slate-500/10 px-3 py-1 text-xs font-medium text-slate-200">
									Batch ML
								</span>
								<span class="rounded-full bg-stone-500/10 px-3 py-1 text-xs font-medium text-stone-200">
									SQL
								</span>
							</div>
						</div>
					</a>
				{/each}
			</div>
		</div>

		<!-- ML Capabilities Section -->
		<div class="mt-24">
			<div class="text-center">
			<div class="inline-flex items-center gap-2 rounded-full bg-amber-400/10 px-4 py-1.5 text-sm font-medium text-amber-200">
					<Activity class="h-4 w-4" />
					Machine Learning
				</div>
				<h2 class="mt-4 text-3xl font-bold text-foreground">ML Capabilities</h2>
				<p class="mt-2 text-muted-foreground">From streaming predictions to batch training</p>
			</div>

			<div class="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
				{#each mlFeatures as feature}
					{@const Icon = feature.icon}
					<div class="group rounded-2xl border border-border bg-card p-6 transition-all hover:border-primary/30 hover:shadow-lg">
						<div class="rounded-xl p-3 {feature.bgColor} w-fit">
							<Icon class="h-6 w-6 {feature.color}" />
						</div>
						<h3 class="mt-4 font-semibold text-foreground">{feature.name}</h3>
						<p class="mt-2 text-sm leading-relaxed text-muted-foreground">{feature.description}</p>
					</div>
				{/each}
			</div>
		</div>

		<!-- Data Infrastructure Section -->
		<div class="mt-24">
			<div class="text-center">
			<div class="inline-flex items-center gap-2 rounded-full bg-stone-500/10 px-4 py-1.5 text-sm font-medium text-stone-200">
					<Database class="h-4 w-4" />
					Data Layer
				</div>
				<h2 class="mt-4 text-3xl font-bold text-foreground">Data Infrastructure</h2>
				<p class="mt-2 text-muted-foreground">Streaming, storage, and query engines</p>
			</div>

			<div class="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
				{#each dataFeatures as feature}
					{@const Icon = feature.icon}
					<div class="group rounded-2xl border border-border bg-card p-6 transition-all hover:border-primary/30 hover:shadow-lg">
						<div class="flex items-start justify-between">
							<div class="rounded-xl p-3 {feature.bgColor}">
								<Icon class="h-6 w-6 {feature.color}" />
							</div>
							{#if feature.href}
								<a
									href={feature.href}
									target="_blank"
									rel="noopener noreferrer"
									class="rounded-lg p-2 text-muted-foreground opacity-0 transition-all hover:bg-muted hover:text-foreground group-hover:opacity-100"
								>
									<ExternalLink class="h-4 w-4" />
								</a>
							{/if}
						</div>
						<h3 class="mt-4 font-semibold text-foreground">{feature.name}</h3>
						<p class="mt-2 text-sm leading-relaxed text-muted-foreground">{feature.description}</p>
					</div>
				{/each}
			</div>
		</div>

		<!-- Infrastructure Services Section -->
		<div class="mt-24">
			<div class="text-center">
			<div class="inline-flex items-center gap-2 rounded-full bg-slate-500/10 px-4 py-1.5 text-sm font-medium text-slate-200">
					<Server class="h-4 w-4" />
					Infrastructure
				</div>
				<h2 class="mt-4 text-3xl font-bold text-foreground">Platform Services</h2>
				<p class="mt-2 text-muted-foreground">Observability, storage, and orchestration</p>
			</div>

			<div class="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
				{#each infraFeatures as feature}
					{@const Icon = feature.icon}
					<div class="group rounded-2xl border border-border bg-card p-6 transition-all hover:border-primary/30 hover:shadow-lg">
						<div class="flex items-start justify-between">
							<div class="rounded-xl p-3 {feature.bgColor}">
								<Icon class="h-6 w-6 {feature.color}" />
							</div>
							{#if feature.href}
								<a
									href={feature.href}
									target="_blank"
									rel="noopener noreferrer"
									class="rounded-lg p-2 text-muted-foreground opacity-0 transition-all hover:bg-muted hover:text-foreground group-hover:opacity-100"
								>
									<ExternalLink class="h-4 w-4" />
								</a>
							{/if}
						</div>
						<h3 class="mt-4 font-semibold text-foreground">{feature.name}</h3>
						<p class="mt-2 text-sm leading-relaxed text-muted-foreground">{feature.description}</p>
					</div>
				{/each}
			</div>
		</div>

		<!-- Architecture Overview -->
		<div class="mt-24 rounded-3xl border border-border bg-gradient-to-br from-slate-900 to-slate-800 p-8 sm:p-12">
			<div class="text-center">
				<h2 class="text-3xl font-bold text-white">Architecture Highlights</h2>
				<p class="mt-2 text-slate-400">Built for scale, designed for developers</p>
			</div>

			<div class="mt-10 grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
				<div class="flex gap-4">
			<div class="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-amber-400/15">
				<Zap class="h-5 w-5 text-amber-200" />
					</div>
					<div>
						<h3 class="font-semibold text-white">Event-Driven</h3>
						<p class="mt-1 text-sm text-slate-400">Kafka-powered event streaming with exactly-once delivery guarantees</p>
					</div>
				</div>

				<div class="flex gap-4">
			<div class="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-slate-500/15">
				<Cpu class="h-5 w-5 text-slate-200" />
					</div>
					<div>
						<h3 class="font-semibold text-white">Cloud Native</h3>
						<p class="mt-1 text-sm text-slate-400">Kubernetes-native with Helm charts and Skaffold hot-reload</p>
					</div>
				</div>

				<div class="flex gap-4">
			<div class="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-stone-500/15">
				<GitBranch class="h-5 w-5 text-stone-200" />
					</div>
					<div>
						<h3 class="font-semibold text-white">GitOps Ready</h3>
						<p class="mt-1 text-sm text-slate-400">Infrastructure as code with declarative configurations</p>
					</div>
				</div>

				<div class="flex gap-4">
			<div class="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-amber-500/15">
				<BarChart3 class="h-5 w-5 text-amber-200" />
					</div>
					<div>
						<h3 class="font-semibold text-white">Full Observability</h3>
						<p class="mt-1 text-sm text-slate-400">Prometheus metrics, Grafana dashboards, distributed tracing</p>
					</div>
				</div>

				<div class="flex gap-4">
			<div class="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-stone-500/15">
				<Database class="h-5 w-5 text-stone-200" />
					</div>
					<div>
						<h3 class="font-semibold text-white">Lakehouse Architecture</h3>
						<p class="mt-1 text-sm text-slate-400">Delta Lake on MinIO with ACID transactions and time travel</p>
					</div>
				</div>

				<div class="flex gap-4">
			<div class="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-amber-300/15">
				<Activity class="h-5 w-5 text-amber-200" />
					</div>
					<div>
						<h3 class="font-semibold text-white">Dual ML Paradigm</h3>
						<p class="mt-1 text-sm text-slate-400">Incremental learning with River, batch training with Scikit-Learn</p>
					</div>
				</div>
			</div>
		</div>

		<!-- Footer CTA -->
		<div class="mt-24 text-center">
			<p class="text-muted-foreground">
				Built with
				<span class="text-amber-300">&#9829;</span>
				using Python, SvelteKit, Kafka, Spark, and Kubernetes
			</p>
		</div>
	</div>
</div>
