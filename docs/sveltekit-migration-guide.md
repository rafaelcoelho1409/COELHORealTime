# SvelteKit Migration Guide

## From Reflex to SvelteKit: Complete Implementation Reference

**Document Version:** 1.0
**Created:** January 2026
**Purpose:** Comprehensive guide for migrating the COELHO RealTime dashboard from Reflex to SvelteKit while maintaining feature parity and implementing modern frontend improvements.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Analysis](#2-current-architecture-analysis)
3. [Why Migrate to SvelteKit](#3-why-migrate-to-sveltekit)
4. [Technology Stack](#4-technology-stack)
5. [Project Structure](#5-project-structure)
6. [State Management Migration](#6-state-management-migration)
7. [API Client Design](#7-api-client-design)
8. [Component Migration Reference](#8-component-migration-reference)
9. [Page-by-Page Implementation](#9-page-by-page-implementation)
10. [Frontend Improvements](#10-frontend-improvements)
11. [Deployment Configuration](#11-deployment-configuration)
12. [Migration Phases](#12-migration-phases)
13. [Testing Strategy](#13-testing-strategy)
14. [Performance Benchmarks](#14-performance-benchmarks)

---

## 1. Executive Summary

### 1.1 Migration Goals

- **Primary:** Achieve 3-5x faster page load and interaction times
- **Secondary:** Reduce resource consumption (memory: 2Gi → 512Mi, CPU: 2 cores → 0.5 cores)
- **Tertiary:** Modernize UI/UX with current 2026 design patterns

### 1.2 Scope

| Component | Files | Lines of Code | Migration Effort |
|-----------|-------|---------------|------------------|
| States | 4 | ~5,000 | High |
| Components | 4 | ~6,900 | High |
| Pages | 15+ | ~2,000 | Medium |
| Utils | 2 | ~100 | Low |
| **Total** | **25+** | **~14,000** | **~3-4 weeks** |

### 1.3 Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | SvelteKit 2.x | Best performance, smallest bundles |
| Styling | Tailwind CSS + shadcn-svelte | Utility-first, Radix-like components |
| Charts | Plotly.js via svelte-plotly.js | Same library as Reflex, easy migration |
| Maps | Leaflet via sveaflet | Replaces Folium server-side rendering |
| State | Svelte stores | Native, reactive, zero overhead |
| Forms | sveltekit-superforms + zod | Type-safe validation |

---

## 2. Current Architecture Analysis

### 2.1 Reflex Application Structure

```
apps/reflex/coelho_realtime/
├── coelho_realtime.py          # Main app entry (142 lines)
├── rxconfig.py                  # Configuration (26 lines)
├── utils.py                     # HTTP client (34 lines)
│
├── states/
│   ├── __init__.py
│   ├── shared.py               # SharedState base (~1,740 lines)
│   ├── tfd.py                  # TFDState (~1,200 lines)
│   ├── eta.py                  # ETAState (~1,100 lines)
│   └── ecci.py                 # ECCIState (~800 lines)
│
├── components/
│   ├── __init__.py
│   ├── shared.py               # Common components (~1,949 lines)
│   ├── tfd.py                  # TFD components (~1,901 lines)
│   ├── eta.py                  # ETA components (~1,556 lines)
│   └── ecci.py                 # ECCI components (~1,491 lines)
│
├── pages/
│   ├── __init__.py
│   ├── home.py
│   ├── tfd/
│   │   ├── __init__.py
│   │   ├── incremental.py
│   │   ├── sql.py
│   │   └── batch/
│   │       ├── __init__.py
│   │       ├── prediction.py
│   │       └── metrics.py
│   ├── eta/
│   │   └── [same structure]
│   └── ecci/
│       └── [same structure]
│
└── data/
    ├── dropdown_options_tfd.json
    ├── dropdown_options_eta.json
    ├── dropdown_options_ecci.json
    ├── metric_info_tfd.json
    ├── metric_info_eta.json
    ├── metric_info_ecci.json
    ├── yellowbrick_info_tfd.json
    ├── yellowbrick_info_eta.json
    └── yellowbrick_info_ecci.json
```

### 2.2 Routes Structure

| Route | Page | Description |
|-------|------|-------------|
| `/` | Home | Logo and navigation |
| `/tfd/incremental` | TFD Incremental | Real-time fraud detection with River |
| `/tfd/batch/prediction` | TFD Batch Prediction | Batch predictions with CatBoost |
| `/tfd/batch/metrics` | TFD Batch Metrics | YellowBrick visualizations |
| `/tfd/sql` | TFD SQL | Delta Lake queries |
| `/eta/incremental` | ETA Incremental | Real-time ETA with map |
| `/eta/batch/prediction` | ETA Batch Prediction | Batch ETA predictions |
| `/eta/batch/metrics` | ETA Batch Metrics | Regression visualizations |
| `/eta/sql` | ETA SQL | Delta Lake queries |
| `/ecci/incremental` | ECCI Incremental | Real-time clustering |
| `/ecci/batch/prediction` | ECCI Batch Prediction | Batch clustering |
| `/ecci/batch/metrics` | ECCI Batch Metrics | Clustering visualizations |
| `/ecci/sql` | ECCI SQL | Delta Lake queries |

### 2.3 Backend Services (Unchanged)

| Service | Port | Purpose |
|---------|------|---------|
| River | 8002 | Incremental ML (Kafka streams) |
| Sklearn | 8003 | Batch ML (CatBoost training) |

**Key Endpoints - River Service:**
- `POST /predict` - Make prediction
- `POST /switch_model` - Start/stop Kafka streaming
- `GET /mlflow_metrics` - Fetch training metrics
- `GET /model_available` - Check model status
- `POST /sql_query` - Execute Delta Lake SQL
- `GET /table_schema` - Get table metadata
- `GET /cluster_counts` - ECCI cluster distribution
- `GET /cluster_feature_counts` - ECCI feature analysis

**Key Endpoints - Sklearn Service:**
- `POST /predict` - Batch prediction
- `POST /switch_model` - Start batch training
- `GET /batch_status` - Training progress
- `GET /mlflow_runs` - List trained runs
- `GET /mlflow_metrics` - Batch metrics
- `POST /yellowbrick_metric` - Generate visualization

---

## 3. Why Migrate to SvelteKit

### 3.1 Reflex Performance Issues

1. **WebSocket State Synchronization**
   - Every state change requires round-trip to Python backend
   - Latency: 50-200ms per interaction
   - Redis overhead in Kubernetes deployment

2. **Python → React Compilation**
   - Components compiled at build time
   - Full re-render on state changes
   - Large bundle sizes (~500KB+)

3. **Heavy State Objects**
   - SharedState: ~1,740 lines of state variables
   - Entire state tree serialized on updates
   - Computed variables recalculate on every render

### 3.2 SvelteKit Advantages

| Metric | Reflex | SvelteKit | Improvement |
|--------|--------|-----------|-------------|
| Bundle Size | ~500KB | ~65KB | 87% smaller |
| Time to Interactive | 3-5s | 0.5-1s | 3-5x faster |
| State Update Latency | 50-200ms | <10ms | 10-20x faster |
| Memory (Pod) | 2-4Gi | 256-512Mi | 4-8x less |
| CPU (Pod) | 1-2 cores | 0.1-0.5 cores | 4-10x less |

### 3.3 Architecture Comparison

```
REFLEX ARCHITECTURE:
┌─────────────────────────────────────────────────────────┐
│                      Browser                             │
│  ┌─────────────────────────────────────────────────────┐│
│  │              React (Compiled from Python)            ││
│  │                        ↕                             ││
│  │              WebSocket Connection                    ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                           ↕ (Every state change)
┌─────────────────────────────────────────────────────────┐
│                 Reflex Python Backend                    │
│  ┌─────────────────────────────────────────────────────┐│
│  │     State Management + Event Handlers                ││
│  │              (SharedState, TFDState, etc.)           ││
│  └─────────────────────────────────────────────────────┘│
│                           ↕                              │
│                        Redis                             │
└─────────────────────────────────────────────────────────┘
                           ↕
            ┌──────────────┴──────────────┐
            ↓                              ↓
    River (8002)                    Sklearn (8003)


SVELTEKIT ARCHITECTURE:
┌─────────────────────────────────────────────────────────┐
│                      Browser                             │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Svelte Components                       ││
│  │              + Client-Side Stores                    ││
│  │        (State changes: instant, no server)           ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
          ↕ (Only explicit API calls)
┌─────────────────────────────────────────────────────────┐
│               SvelteKit Node.js Server                   │
│         (SSR + API proxy if needed)                      │
└─────────────────────────────────────────────────────────┘
          ↕ (Direct HTTP calls)
     ┌────┴────┐
     ↓         ↓
River (8002)  Sklearn (8003)
```

---

## 4. Technology Stack

### 4.1 Core Dependencies

```json
{
  "name": "coelho-realtime-sveltekit",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite dev",
    "build": "vite build",
    "preview": "vite preview",
    "check": "svelte-kit sync && svelte-check --tsconfig ./tsconfig.json",
    "lint": "eslint .",
    "format": "prettier --write ."
  },
  "devDependencies": {
    "@sveltejs/adapter-node": "^5.2.0",
    "@sveltejs/kit": "^2.6.0",
    "@sveltejs/vite-plugin-svelte": "^4.0.0",
    "@types/leaflet": "^1.9.12",
    "autoprefixer": "^10.4.20",
    "eslint": "^9.0.0",
    "postcss": "^8.4.47",
    "prettier": "^3.3.0",
    "prettier-plugin-svelte": "^3.2.0",
    "svelte": "^5.0.0",
    "svelte-check": "^4.0.0",
    "tailwindcss": "^3.4.13",
    "typescript": "^5.6.3",
    "vite": "^5.4.8"
  },
  "dependencies": {
    "@codemirror/lang-sql": "^6.7.0",
    "bits-ui": "^0.21.0",
    "clsx": "^2.1.1",
    "codemirror": "^6.0.0",
    "leaflet": "^1.9.4",
    "lucide-svelte": "^0.453.0",
    "mode-watcher": "^0.4.0",
    "plotly.js-dist-min": "^2.35.0",
    "svelte-codemirror-editor": "^1.4.0",
    "svelte-plotly.js": "^1.0.0",
    "svelte-sonner": "^0.3.0",
    "sveltekit-superforms": "^2.19.0",
    "tailwind-merge": "^2.5.2",
    "tailwind-variants": "^0.2.1",
    "zod": "^3.23.8"
  }
}
```

### 4.2 Tailwind Configuration

```javascript
// tailwind.config.js
import { fontFamily } from 'tailwindcss/defaultTheme';

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      colors: {
        // Semantic ML metric colors
        metric: {
          excellent: '#22c55e',
          good: '#84cc16',
          fair: '#eab308',
          poor: '#f97316',
          critical: '#ef4444',
        },
        // Background layers
        surface: {
          DEFAULT: '#141414',
          elevated: '#1f1f1f',
          overlay: '#292929',
        },
      },
      fontFamily: {
        sans: ['Inter', ...fontFamily.sans],
        mono: ['JetBrains Mono', ...fontFamily.mono],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
};
```

### 4.3 SvelteKit Configuration

```javascript
// svelte.config.js
import adapter from '@sveltejs/adapter-node';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({
      out: 'build',
      precompress: true,
      envPrefix: 'VITE_',
    }),
    alias: {
      $components: 'src/lib/components',
      $stores: 'src/lib/stores',
      $api: 'src/lib/api',
      $utils: 'src/lib/utils',
    },
  },
};

export default config;
```

### 4.4 Vite Configuration

```typescript
// vite.config.ts
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    port: 3000,
    host: '0.0.0.0',
  },
  preview: {
    port: 3000,
    host: '0.0.0.0',
  },
  // Optimize Plotly.js chunking
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          plotly: ['plotly.js-dist-min'],
          leaflet: ['leaflet'],
        },
      },
    },
  },
});
```

---

## 5. Project Structure

### 5.1 Complete Directory Structure

```
apps/sveltekit/
├── package.json
├── svelte.config.js
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
├── postcss.config.js
├── Dockerfile.sveltekit
│
├── static/
│   ├── favicon.ico
│   └── logo.svg
│
└── src/
    ├── app.html
    ├── app.css
    ├── app.d.ts
    ├── hooks.server.ts
    │
    ├── lib/
    │   ├── index.ts                    # Re-exports
    │   │
    │   ├── api/
    │   │   ├── client.ts               # Base HTTP client
    │   │   ├── river.ts                # River service API
    │   │   ├── sklearn.ts              # Sklearn service API
    │   │   └── types.ts                # API response types
    │   │
    │   ├── stores/
    │   │   ├── index.ts                # Store exports
    │   │   ├── shared.ts               # Common stores
    │   │   ├── tfd.ts                  # TFD-specific stores
    │   │   ├── eta.ts                  # ETA-specific stores
    │   │   └── ecci.ts                 # ECCI-specific stores
    │   │
    │   ├── components/
    │   │   ├── ui/                     # shadcn-svelte components
    │   │   │   ├── button/
    │   │   │   ├── card/
    │   │   │   ├── dialog/
    │   │   │   ├── input/
    │   │   │   ├── select/
    │   │   │   ├── switch/
    │   │   │   ├── tabs/
    │   │   │   ├── badge/
    │   │   │   ├── tooltip/
    │   │   │   ├── collapsible/
    │   │   │   ├── sheet/
    │   │   │   └── skeleton/
    │   │   │
    │   │   ├── shared/
    │   │   │   ├── Navbar.svelte
    │   │   │   ├── PageTabs.svelte
    │   │   │   ├── SubNav.svelte
    │   │   │   ├── MlTrainingSwitch.svelte
    │   │   │   ├── BatchTrainingBox.svelte
    │   │   │   ├── MlflowRunSelector.svelte
    │   │   │   ├── MetricInfoDialog.svelte
    │   │   │   ├── YellowBrickViewer.svelte
    │   │   │   ├── YellowBrickInfoDialog.svelte
    │   │   │   ├── SqlInterface.svelte
    │   │   │   ├── EmptyState.svelte
    │   │   │   ├── ErrorState.svelte
    │   │   │   └── LiveStatusIndicator.svelte
    │   │   │
    │   │   ├── charts/
    │   │   │   ├── PlotlyChart.svelte
    │   │   │   ├── GaugeChart.svelte
    │   │   │   ├── KpiIndicator.svelte
    │   │   │   ├── BulletChart.svelte
    │   │   │   ├── HeatmapChart.svelte
    │   │   │   └── AnimatedNumber.svelte
    │   │   │
    │   │   ├── tfd/
    │   │   │   ├── TfdForm.svelte
    │   │   │   ├── TfdBatchForm.svelte
    │   │   │   ├── TfdMetrics.svelte
    │   │   │   ├── TfdBatchMetrics.svelte
    │   │   │   ├── TfdPredictionResult.svelte
    │   │   │   └── FraudRiskGauge.svelte
    │   │   │
    │   │   ├── eta/
    │   │   │   ├── EtaForm.svelte
    │   │   │   ├── EtaBatchForm.svelte
    │   │   │   ├── EtaMetrics.svelte
    │   │   │   ├── EtaBatchMetrics.svelte
    │   │   │   ├── EtaPredictionResult.svelte
    │   │   │   └── EtaMap.svelte
    │   │   │
    │   │   └── ecci/
    │   │       ├── EcciForm.svelte
    │   │       ├── EcciBatchForm.svelte
    │   │       ├── EcciMetrics.svelte
    │   │       ├── EcciBatchMetrics.svelte
    │   │       ├── EcciPredictionResult.svelte
    │   │       ├── EcciClusterChart.svelte
    │   │       └── EcciMap.svelte
    │   │
    │   ├── utils/
    │   │   ├── formatters.ts           # Number/date formatting
    │   │   ├── validators.ts           # Form validation schemas
    │   │   ├── haversine.ts            # Distance calculation
    │   │   ├── colors.ts               # Metric color helpers
    │   │   └── cn.ts                   # Tailwind class merger
    │   │
    │   └── data/
    │       ├── dropdown-options-tfd.json
    │       ├── dropdown-options-eta.json
    │       ├── dropdown-options-ecci.json
    │       ├── metric-info-tfd.json
    │       ├── metric-info-eta.json
    │       ├── metric-info-ecci.json
    │       ├── yellowbrick-info-tfd.json
    │       ├── yellowbrick-info-eta.json
    │       └── yellowbrick-info-ecci.json
    │
    └── routes/
        ├── +layout.svelte              # Root layout (navbar, theme)
        ├── +layout.ts                  # Root load function
        ├── +page.svelte                # Home page
        ├── +error.svelte               # Error page
        │
        ├── tfd/
        │   ├── +layout.svelte          # TFD layout (page tabs)
        │   ├── +layout.ts              # TFD data loader
        │   │
        │   ├── incremental/
        │   │   ├── +page.svelte
        │   │   └── +page.ts
        │   │
        │   ├── batch/
        │   │   ├── +layout.svelte      # Batch sub-nav
        │   │   ├── prediction/
        │   │   │   ├── +page.svelte
        │   │   │   └── +page.ts
        │   │   └── metrics/
        │   │       ├── +page.svelte
        │   │       └── +page.ts
        │   │
        │   └── sql/
        │       ├── +page.svelte
        │       └── +page.ts
        │
        ├── eta/
        │   └── [same structure as tfd]
        │
        └── ecci/
            └── [same structure as tfd]
```

---

## 6. State Management Migration

### 6.1 Reflex State → Svelte Stores Mapping

#### SharedState Variables

```typescript
// src/lib/stores/shared.ts
import { writable, derived, type Writable, type Readable } from 'svelte/store';

// =============================================================================
// Types
// =============================================================================
export type ProjectKey = 'tfd' | 'eta' | 'ecci';
export type TabName = 'incremental' | 'batch' | 'sql';
export type SqlEngine = 'polars' | 'duckdb';
export type TrainingStage = 'init' | 'data' | 'training' | 'complete' | 'error';

export interface MlflowRunInfo {
  run_id: string;
  status: string;
  start_time: string;
  is_best: boolean;
  is_live: boolean;
}

export interface MlflowMetrics {
  [key: string]: number | string;
}

export interface SqlQueryResults {
  columns: string[];
  data: any[][];
  row_count: number;
}

export interface FormData {
  [key: string]: string | number | boolean;
}

// =============================================================================
// Global Stores
// =============================================================================

// Currently active model (only one can be active across all projects)
export const activatedModel: Writable<string | null> = writable(null);

// Global ML training enabled state
export const mlTrainingEnabled: Writable<boolean> = writable(false);

// =============================================================================
// Per-Project Store Factory
// =============================================================================
export function createProjectStores(project: ProjectKey) {
  return {
    // -------------------------------------------------------------------------
    // Incremental ML Stores
    // -------------------------------------------------------------------------
    incrementalMlState: writable(false),
    incrementalModelAvailable: writable(false),
    mlflowMetrics: writable<MlflowMetrics>({}),
    mlflowRunInfo: writable<MlflowRunInfo>({
      run_id: '',
      status: '',
      start_time: '',
      is_best: false,
      is_live: false,
    }),
    mlflowExperimentUrl: writable(''),

    // -------------------------------------------------------------------------
    // Batch ML Stores
    // -------------------------------------------------------------------------
    batchModelAvailable: writable(false),
    batchMlflowMetrics: writable<MlflowMetrics>({}),
    batchMlflowRuns: writable<MlflowRunInfo[]>([]),
    selectedBatchRun: writable<string | null>(null),
    batchRunsLoading: writable(false),

    // Training progress
    batchTrainingLoading: writable(false),
    batchTrainingStatus: writable(''),
    batchTrainingProgress: writable(0),
    batchTrainingStage: writable<TrainingStage>('init'),
    batchTrainingMetricsPreview: writable<MlflowMetrics>({}),
    batchTrainingTotalRows: writable(0),
    batchTrainingDataPercentage: writable(100),

    // CatBoost training log (for TFD/ETA)
    batchTrainingCatboostLog: writable<string[]>([]),

    // -------------------------------------------------------------------------
    // Form & Prediction Stores
    // -------------------------------------------------------------------------
    formData: writable<FormData>({}),
    predictionResults: writable<any>(null),
    predictionLoading: writable(false),

    // -------------------------------------------------------------------------
    // SQL Interface Stores
    // -------------------------------------------------------------------------
    sqlQueryInput: writable(''),
    sqlQueryResults: writable<SqlQueryResults | null>(null),
    sqlLoading: writable(false),
    sqlError: writable(''),
    sqlExecutionTime: writable(0),
    sqlEngine: writable<SqlEngine>('polars'),
    sqlSearchFilter: writable(''),
    sqlSortColumn: writable(''),
    sqlSortDirection: writable<'asc' | 'desc'>('asc'),
    sqlTableMetadata: writable<{ columns: string[]; row_count: number } | null>(null),

    // -------------------------------------------------------------------------
    // YellowBrick Stores (Batch Metrics)
    // -------------------------------------------------------------------------
    yellowbrickMetricType: writable(''),
    yellowbrickMetricName: writable(''),
    yellowbrickImageBase64: writable(''),
    yellowbrickLoading: writable(false),
  };
}

// =============================================================================
// Create Stores for Each Project
// =============================================================================
export const tfdStores = createProjectStores('tfd');
export const etaStores = createProjectStores('eta');
export const ecciStores = createProjectStores('ecci');

// Helper to get stores by project key
export function getProjectStores(project: ProjectKey) {
  const map = {
    tfd: tfdStores,
    eta: etaStores,
    ecci: ecciStores,
  };
  return map[project];
}

// =============================================================================
// Project Name Mapping (for API calls)
// =============================================================================
export const PROJECT_NAME_MAP: Record<ProjectKey, string> = {
  tfd: 'Transaction Fraud Detection',
  eta: 'Estimated Time of Arrival',
  ecci: 'E-Commerce Customer Interactions',
};

export const MODEL_KEY_MAP: Record<ProjectKey, string> = {
  tfd: 'transaction_fraud_detection',
  eta: 'estimated_time_of_arrival',
  ecci: 'e_commerce_customer_interactions',
};

// =============================================================================
// Incremental ML Model Names
// =============================================================================
export const INCREMENTAL_MODEL_NAMES: Record<ProjectKey, string> = {
  tfd: 'ARFClassifier',
  eta: 'ARFRegressor',
  ecci: 'DBSTREAM',
};
```

#### TFD-Specific Stores

```typescript
// src/lib/stores/tfd.ts
import { derived } from 'svelte/store';
import { tfdStores, type MlflowMetrics } from './shared';

// =============================================================================
// TFD Computed/Derived Stores
// =============================================================================

/**
 * Formatted TFD metrics with percentages
 */
export const tfdFormattedMetrics = derived(
  tfdStores.mlflowMetrics,
  ($metrics): Record<string, string> => {
    const format = (key: string, decimals = 2, suffix = '%') => {
      const value = $metrics[key];
      if (value === undefined || value === null || value === '') return 'N/A';
      const num = typeof value === 'string' ? parseFloat(value) : value;
      if (isNaN(num)) return 'N/A';
      return `${(num * 100).toFixed(decimals)}${suffix}`;
    };

    return {
      fbeta: format('fbeta'),
      rocauc: format('rocauc'),
      precision: format('precision'),
      recall: format('recall'),
      f1: format('f1'),
      accuracy: format('accuracy'),
      balanced_accuracy: format('balanced_accuracy'),
      mcc: format('mcc', 2, ''), // MCC is not a percentage
      cohen_kappa: format('cohen_kappa', 2, ''),
      jaccard: format('jaccard'),
      geometric_mean: format('geometric_mean'),
      logloss: format('logloss', 4, ''),
      rolling_rocauc: format('rolling_rocauc'),
    };
  }
);

/**
 * Fraud prediction result with risk level
 */
export const tfdPredictionWithRisk = derived(
  tfdStores.predictionResults,
  ($results): { probability: number; isFraud: boolean; riskLevel: string; riskColor: string } | null => {
    if (!$results) return null;

    const probability = $results.fraud_probability ?? 0;
    const isFraud = $results.is_fraud ?? false;

    let riskLevel: string;
    let riskColor: string;

    if (probability < 0.3) {
      riskLevel = 'LOW';
      riskColor = 'text-green-400';
    } else if (probability < 0.7) {
      riskLevel = 'MEDIUM';
      riskColor = 'text-yellow-400';
    } else {
      riskLevel = 'HIGH';
      riskColor = 'text-red-400';
    }

    return { probability, isFraud, riskLevel, riskColor };
  }
);

// =============================================================================
// TFD Form Field Options (loaded from JSON)
// =============================================================================
import dropdownOptions from '$lib/data/dropdown-options-tfd.json';

export const tfdDropdownOptions = dropdownOptions;

// =============================================================================
// TFD Metric Info (loaded from JSON)
// =============================================================================
import metricInfo from '$lib/data/metric-info-tfd.json';

export const tfdMetricInfo = metricInfo;

// =============================================================================
// TFD YellowBrick Visualizer Options
// =============================================================================
export const tfdYellowbrickOptions = {
  Classification: [
    'ConfusionMatrix',
    'ClassificationReport',
    'ROCAUC',
    'PrecisionRecallCurve',
    'ClassPredictionError',
    'DiscriminationThreshold',
  ],
  'Feature Analysis': [
    'Rank1D',
    'Rank2D',
    'PCA',
    'Manifold',
    'ParallelCoordinates',
    'RadViz',
    'JointPlot',
  ],
  Target: [
    'ClassBalance',
    'FeatureCorrelation',
    'FeatureCorrelation_Pearson',
    'BalancedBinningReference',
  ],
  'Model Selection': [
    'FeatureImportances',
    'CVScores',
    'ValidationCurve',
    'LearningCurve',
    'RFECV',
    'DroppingCurve',
  ],
};
```

### 6.2 State Update Patterns

#### Reflex Pattern (WebSocket-based)

```python
# Reflex: Every update triggers WebSocket round-trip
class SharedState(rx.State):
    form_data: dict = {}

    def update_field(self, key: str, value: str):
        self.form_data[key] = value  # Triggers WebSocket sync
```

#### SvelteKit Pattern (Client-side)

```typescript
// SvelteKit: Instant client-side updates
import { tfdStores } from '$stores/tfd';

// Update single field (instant)
function updateField(key: string, value: string) {
  tfdStores.formData.update((data) => ({ ...data, [key]: value }));
}

// In component: use bind:value for two-way binding
<Input bind:value={$formData.amount} />
```

---

## 7. API Client Design

### 7.1 Base HTTP Client

```typescript
// src/lib/api/client.ts
import { browser } from '$app/environment';

// =============================================================================
// Configuration
// =============================================================================
const RIVER_URL = browser
  ? (import.meta.env.VITE_RIVER_URL || 'http://localhost:8002')
  : (process.env.RIVER_URL || 'http://coelho-realtime-river:8002');

const SKLEARN_URL = browser
  ? (import.meta.env.VITE_SKLEARN_URL || 'http://localhost:8003')
  : (process.env.SKLEARN_URL || 'http://coelho-realtime-sklearn:8003');

// =============================================================================
// Types
// =============================================================================
export interface ApiError {
  message: string;
  status: number;
  details?: unknown;
}

export interface ApiResponse<T> {
  data: T | null;
  error: ApiError | null;
}

// =============================================================================
// Base Fetch Function
// =============================================================================
async function fetchApi<T>(
  url: string,
  options: RequestInit = {},
  timeout = 10000
): Promise<ApiResponse<T>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      return {
        data: null,
        error: {
          message: `API error: ${response.statusText}`,
          status: response.status,
        },
      };
    }

    const data = await response.json();
    return { data, error: null };
  } catch (err) {
    clearTimeout(timeoutId);

    if (err instanceof Error && err.name === 'AbortError') {
      return {
        data: null,
        error: { message: 'Request timeout', status: 408 },
      };
    }

    return {
      data: null,
      error: {
        message: err instanceof Error ? err.message : 'Unknown error',
        status: 500,
      },
    };
  }
}

// =============================================================================
// Exports
// =============================================================================
export { RIVER_URL, SKLEARN_URL, fetchApi };
```

### 7.2 River Service API

```typescript
// src/lib/api/river.ts
import { fetchApi, RIVER_URL, type ApiResponse } from './client';
import type { MlflowMetrics, SqlQueryResults, FormData } from '$stores/shared';

// =============================================================================
// Types
// =============================================================================
export interface PredictResponse {
  prediction: any;
  model_source: string;
  fraud_probability?: number;
  is_fraud?: boolean;
  eta_seconds?: number;
  cluster?: number;
}

export interface ModelAvailableResponse {
  available: boolean;
}

export interface MlflowMetricsResponse {
  metrics: MlflowMetrics;
  run_info: {
    run_id: string;
    status: string;
    start_time: string;
    is_live: boolean;
  };
  experiment_url: string;
}

export interface TableSchemaResponse {
  columns: string[];
  row_count: number;
}

export interface ClusterCountsResponse {
  counts: Record<string, number>;
}

export interface ClusterFeatureCountsResponse {
  feature: string;
  counts: Record<string, Record<string, number>>;
}

// =============================================================================
// River Service API
// =============================================================================
export const riverApi = {
  /**
   * Make a prediction using the incremental ML model
   */
  predict: async (
    modelKey: string,
    formData: FormData
  ): Promise<ApiResponse<PredictResponse>> => {
    return fetchApi(`${RIVER_URL}/predict`, {
      method: 'POST',
      body: JSON.stringify({ model_key: modelKey, form_data: formData }),
    });
  },

  /**
   * Start or stop the Kafka consumer for incremental training
   */
  switchModel: async (
    modelKey: string,
    enabled: boolean
  ): Promise<ApiResponse<{ status: string }>> => {
    return fetchApi(`${RIVER_URL}/switch_model`, {
      method: 'POST',
      body: JSON.stringify({ model_key: modelKey, enabled }),
    });
  },

  /**
   * Get MLflow metrics for the current training run
   */
  getMlflowMetrics: async (
    projectName: string
  ): Promise<ApiResponse<MlflowMetricsResponse>> => {
    return fetchApi(
      `${RIVER_URL}/mlflow_metrics?project_name=${encodeURIComponent(projectName)}`
    );
  },

  /**
   * Check if a trained model is available
   */
  checkModelAvailable: async (
    projectName: string
  ): Promise<ApiResponse<ModelAvailableResponse>> => {
    return fetchApi(
      `${RIVER_URL}/model_available?project_name=${encodeURIComponent(projectName)}`
    );
  },

  /**
   * Execute a SQL query against Delta Lake
   */
  executeSql: async (
    projectName: string,
    query: string,
    engine: 'polars' | 'duckdb' = 'polars'
  ): Promise<ApiResponse<SqlQueryResults>> => {
    return fetchApi(`${RIVER_URL}/sql_query`, {
      method: 'POST',
      body: JSON.stringify({ project_name: projectName, query, engine }),
    });
  },

  /**
   * Get table schema metadata
   */
  getTableSchema: async (
    projectName: string
  ): Promise<ApiResponse<TableSchemaResponse>> => {
    return fetchApi(
      `${RIVER_URL}/table_schema?project_name=${encodeURIComponent(projectName)}`
    );
  },

  /**
   * Get cluster counts (ECCI only)
   */
  getClusterCounts: async (
    projectName: string
  ): Promise<ApiResponse<ClusterCountsResponse>> => {
    return fetchApi(
      `${RIVER_URL}/cluster_counts?project_name=${encodeURIComponent(projectName)}`
    );
  },

  /**
   * Get cluster feature counts (ECCI only)
   */
  getClusterFeatureCounts: async (
    projectName: string,
    feature: string
  ): Promise<ApiResponse<ClusterFeatureCountsResponse>> => {
    return fetchApi(
      `${RIVER_URL}/cluster_feature_counts?project_name=${encodeURIComponent(projectName)}&feature=${encodeURIComponent(feature)}`
    );
  },

  /**
   * Get report metrics (confusion matrix, classification report - TFD only)
   */
  getReportMetrics: async (
    projectName: string
  ): Promise<ApiResponse<{ confusion_matrix: any; classification_report: any }>> => {
    return fetchApi(
      `${RIVER_URL}/report_metrics?project_name=${encodeURIComponent(projectName)}`
    );
  },

  /**
   * Initialize page data (combined call)
   */
  pageInit: async (
    modelKey: string,
    projectName: string
  ): Promise<ApiResponse<{
    metrics: MlflowMetrics;
    model_available: boolean;
    experiment_url: string;
  }>> => {
    return fetchApi(`${RIVER_URL}/page_init`, {
      method: 'POST',
      body: JSON.stringify({ model_key: modelKey, project_name: projectName }),
    });
  },
};
```

### 7.3 Sklearn Service API

```typescript
// src/lib/api/sklearn.ts
import { fetchApi, SKLEARN_URL, type ApiResponse } from './client';
import type { MlflowMetrics, FormData } from '$stores/shared';

// =============================================================================
// Types
// =============================================================================
export interface MlflowRun {
  run_id: string;
  status: string;
  start_time: string;
  is_best: boolean;
  metrics: Record<string, number>;
}

export interface BatchStatusResponse {
  status: string;
  progress: number;
  stage: string;
  metrics_preview: Record<string, number>;
  catboost_log: string[];
  total_rows: number;
  error?: string;
}

export interface YellowBrickResponse {
  image_base64: string;
  metric_name: string;
}

// =============================================================================
// Sklearn Service API
// =============================================================================
export const sklearnApi = {
  /**
   * Make a batch prediction using trained sklearn model
   */
  predict: async (
    modelKey: string,
    runId: string,
    formData: FormData
  ): Promise<ApiResponse<{
    prediction: any;
    model_source: string;
  }>> => {
    return fetchApi(`${SKLEARN_URL}/predict`, {
      method: 'POST',
      body: JSON.stringify({
        model_key: modelKey,
        run_id: runId,
        form_data: formData,
      }),
    });
  },

  /**
   * Start batch model training
   */
  trainModel: async (
    modelKey: string,
    dataPercentage: number = 100
  ): Promise<ApiResponse<{ status: string }>> => {
    return fetchApi(`${SKLEARN_URL}/switch_model`, {
      method: 'POST',
      body: JSON.stringify({
        model_key: modelKey,
        data_percentage: dataPercentage,
      }),
    });
  },

  /**
   * Stop batch model training
   */
  stopTraining: async (
    projectName: string
  ): Promise<ApiResponse<{ status: string }>> => {
    return fetchApi(`${SKLEARN_URL}/stop_training`, {
      method: 'POST',
      body: JSON.stringify({ project_name: projectName }),
    });
  },

  /**
   * Get batch training status (poll during training)
   */
  getBatchStatus: async (
    projectName: string
  ): Promise<ApiResponse<BatchStatusResponse>> => {
    return fetchApi(
      `${SKLEARN_URL}/batch_status?project_name=${encodeURIComponent(projectName)}`
    );
  },

  /**
   * Get list of all MLflow runs for project
   */
  getMlflowRuns: async (
    projectName: string
  ): Promise<ApiResponse<{ runs: MlflowRun[] }>> => {
    return fetchApi(
      `${SKLEARN_URL}/mlflow_runs?project_name=${encodeURIComponent(projectName)}`
    );
  },

  /**
   * Get metrics for a specific MLflow run
   */
  getMlflowMetrics: async (
    projectName: string,
    runId: string
  ): Promise<ApiResponse<{ metrics: MlflowMetrics }>> => {
    return fetchApi(
      `${SKLEARN_URL}/mlflow_metrics?project_name=${encodeURIComponent(projectName)}&run_id=${encodeURIComponent(runId)}`
    );
  },

  /**
   * Check if batch model is available
   */
  checkModelAvailable: async (
    projectName: string
  ): Promise<ApiResponse<{ available: boolean }>> => {
    return fetchApi(
      `${SKLEARN_URL}/model_available?project_name=${encodeURIComponent(projectName)}`
    );
  },

  /**
   * Generate YellowBrick visualization
   */
  getYellowBrickMetric: async (
    projectName: string,
    runId: string,
    category: string,
    metric: string
  ): Promise<ApiResponse<YellowBrickResponse>> => {
    return fetchApi(
      `${SKLEARN_URL}/yellowbrick_metric`,
      {
        method: 'POST',
        body: JSON.stringify({
          project_name: projectName,
          run_id: runId,
          category,
          metric,
        }),
      },
      300000 // 5 minute timeout for complex visualizations
    );
  },
};
```

---

## 8. Component Migration Reference

### 8.1 Reflex → Svelte Component Mapping

| Reflex Component | Svelte/shadcn Component | Import Path |
|------------------|------------------------|-------------|
| `rx.card()` | `<Card>` | `$components/ui/card` |
| `rx.vstack()` | `<div class="flex flex-col">` | Native |
| `rx.hstack()` | `<div class="flex flex-row">` | Native |
| `rx.grid()` | `<div class="grid">` | Native |
| `rx.input()` | `<Input>` | `$components/ui/input` |
| `rx.select()` | `<Select>` | `$components/ui/select` |
| `rx.button()` | `<Button>` | `$components/ui/button` |
| `rx.switch()` | `<Switch>` | `$components/ui/switch` |
| `rx.checkbox()` | `<Checkbox>` | `$components/ui/checkbox` |
| `rx.badge()` | `<Badge>` | `$components/ui/badge` |
| `rx.divider()` | `<Separator>` | `$components/ui/separator` |
| `rx.dialog.root()` | `<Dialog>` | `$components/ui/dialog` |
| `rx.tabs()` | `<Tabs>` | `$components/ui/tabs` |
| `rx.tooltip()` | `<Tooltip>` | `$components/ui/tooltip` |
| `rx.icon()` | `<Icon>` | `lucide-svelte` |
| `rx.spinner()` | `<Spinner>` | Custom or Lucide |
| `rx.plotly()` | `<Plotly>` | `svelte-plotly.js` |
| `rx.html()` (Folium) | `<LeafletMap>` | Custom Leaflet |
| `rx.cond()` | `{#if}...{/if}` | Native Svelte |
| `rx.foreach()` | `{#each}...{/each}` | Native Svelte |
| `rx.match()` | `{#if}...{:else if}...{/if}` | Native Svelte |

### 8.2 Example Component Migration

#### Reflex: metric_card

```python
def metric_card(label: str, value_var, metric_key: str = None, project_key: str = "tfd") -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.text(label, size="1", weight="medium", color="gray"),
                metric_info_dialog(metric_key, project_key) if metric_key else rx.fragment(),
                spacing="1",
                align="center",
                justify="center"
            ),
            rx.text(value_var, size="4", weight="bold", align="center"),
            spacing="1",
            align_items="center",
            justify="center",
            height="100%"
        ),
        variant="surface",
        size="1"
    )
```

#### SvelteKit: MetricCard.svelte

```svelte
<!-- src/lib/components/shared/MetricCard.svelte -->
<script lang="ts">
  import { Card, CardContent } from '$components/ui/card';
  import MetricInfoDialog from './MetricInfoDialog.svelte';

  export let label: string;
  export let value: string | number;
  export let metricKey: string | null = null;
  export let projectKey: 'tfd' | 'eta' | 'ecci' = 'tfd';
</script>

<Card class="h-full">
  <CardContent class="flex flex-col items-center justify-center h-full p-4 space-y-1">
    <div class="flex items-center justify-center gap-1">
      <span class="text-xs font-medium text-gray-400">{label}</span>
      {#if metricKey}
        <MetricInfoDialog {metricKey} {projectKey} />
      {/if}
    </div>
    <span class="text-2xl font-bold text-center tabular-nums">{value}</span>
  </CardContent>
</Card>
```

### 8.3 Plotly Chart Component

```svelte
<!-- src/lib/components/charts/PlotlyChart.svelte -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';

  export let data: any[];
  export let layout: any = {};
  export let config: any = {};
  export let className: string = '';

  let plotlyContainer: HTMLDivElement;
  let Plotly: any;

  const defaultLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#e5e5e5' },
    margin: { t: 40, b: 40, l: 40, r: 40 },
    ...layout,
  };

  const defaultConfig = {
    responsive: true,
    displayModeBar: false,
    ...config,
  };

  onMount(async () => {
    if (!browser) return;

    // Dynamic import for SSR compatibility
    const plotlyModule = await import('plotly.js-dist-min');
    Plotly = plotlyModule.default;

    Plotly.newPlot(plotlyContainer, data, defaultLayout, defaultConfig);
  });

  // Reactive update when data changes
  $: if (browser && Plotly && plotlyContainer) {
    Plotly.react(plotlyContainer, data, defaultLayout, defaultConfig);
  }

  onDestroy(() => {
    if (browser && Plotly && plotlyContainer) {
      Plotly.purge(plotlyContainer);
    }
  });
</script>

<div bind:this={plotlyContainer} class={className}></div>
```

### 8.4 Leaflet Map Component (ETA)

```svelte
<!-- src/lib/components/eta/EtaMap.svelte -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';

  export let originLat: number;
  export let originLon: number;
  export let destLat: number;
  export let destLon: number;
  export let className: string = 'h-[400px] w-full rounded-lg';

  let mapContainer: HTMLDivElement;
  let map: any;
  let L: any;

  onMount(async () => {
    if (!browser) return;

    // Dynamic imports for SSR
    L = await import('leaflet');
    await import('leaflet/dist/leaflet.css');

    // Fix Leaflet default marker icon issue
    delete (L.Icon.Default.prototype as any)._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
      iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
      shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
    });

    initMap();
  });

  function initMap() {
    const center: [number, number] = [
      (originLat + destLat) / 2,
      (originLon + destLon) / 2,
    ];

    map = L.map(mapContainer).setView(center, 12);

    // Dark theme tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      maxZoom: 19,
    }).addTo(map);

    // Origin marker (green)
    const originIcon = L.divIcon({
      className: 'custom-marker',
      html: '<div class="w-4 h-4 bg-green-500 rounded-full border-2 border-white shadow-lg"></div>',
      iconSize: [16, 16],
      iconAnchor: [8, 8],
    });
    L.marker([originLat, originLon], { icon: originIcon })
      .addTo(map)
      .bindPopup('Origin');

    // Destination marker (red)
    const destIcon = L.divIcon({
      className: 'custom-marker',
      html: '<div class="w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-lg"></div>',
      iconSize: [16, 16],
      iconAnchor: [8, 8],
    });
    L.marker([destLat, destLon], { icon: destIcon })
      .addTo(map)
      .bindPopup('Destination');

    // Route polyline
    L.polyline(
      [
        [originLat, originLon],
        [destLat, destLon],
      ],
      {
        color: '#3b82f6',
        weight: 3,
        dashArray: '10, 10',
      }
    ).addTo(map);

    // Fit bounds
    map.fitBounds([
      [originLat, originLon],
      [destLat, destLon],
    ], { padding: [50, 50] });
  }

  // Update map when coordinates change
  $: if (browser && map && L) {
    map.remove();
    initMap();
  }

  onDestroy(() => {
    if (browser && map) {
      map.remove();
    }
  });
</script>

<div bind:this={mapContainer} class={className}></div>

<style>
  :global(.custom-marker) {
    background: transparent;
    border: none;
  }
</style>
```

---

## 9. Page-by-Page Implementation

### 9.1 Root Layout

```svelte
<!-- src/routes/+layout.svelte -->
<script lang="ts">
  import '../app.css';
  import { page } from '$app/stores';
  import Navbar from '$components/shared/Navbar.svelte';
  import { Toaster } from 'svelte-sonner';

  // Determine current project from URL
  $: currentProject = $page.url.pathname.split('/')[1] || null;
</script>

<div class="min-h-screen bg-gray-950 text-gray-100">
  <Navbar {currentProject} />

  <main class="container mx-auto px-4 py-6">
    <slot />
  </main>
</div>

<Toaster theme="dark" position="bottom-right" />
```

### 9.2 TFD Layout (with Page Tabs)

```svelte
<!-- src/routes/tfd/+layout.svelte -->
<script lang="ts">
  import { page } from '$app/stores';
  import PageTabs from '$components/shared/PageTabs.svelte';

  const tabs = [
    { href: '/tfd/incremental', label: 'Incremental ML', icon: 'activity' },
    { href: '/tfd/batch', label: 'Batch ML', icon: 'layers' },
    { href: '/tfd/sql', label: 'Delta Lake SQL', icon: 'database' },
  ];

  $: activeTab = $page.url.pathname.startsWith('/tfd/batch')
    ? '/tfd/batch'
    : $page.url.pathname.startsWith('/tfd/sql')
      ? '/tfd/sql'
      : '/tfd/incremental';
</script>

<div class="space-y-6">
  <div class="flex items-center justify-between">
    <h1 class="text-2xl font-bold">Transaction Fraud Detection</h1>
    <PageTabs {tabs} {activeTab} />
  </div>

  <slot />
</div>
```

### 9.3 TFD Incremental Page

```svelte
<!-- src/routes/tfd/incremental/+page.svelte -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { tfdStores } from '$stores/tfd';
  import { riverApi } from '$api/river';
  import { PROJECT_NAME_MAP, MODEL_KEY_MAP } from '$stores/shared';

  import MlTrainingSwitch from '$components/shared/MlTrainingSwitch.svelte';
  import TfdForm from '$components/tfd/TfdForm.svelte';
  import TfdMetrics from '$components/tfd/TfdMetrics.svelte';
  import TfdPredictionResult from '$components/tfd/TfdPredictionResult.svelte';

  const projectName = PROJECT_NAME_MAP.tfd;
  const modelKey = MODEL_KEY_MAP.tfd;

  let metricsInterval: ReturnType<typeof setInterval>;

  // Initialize page
  onMount(async () => {
    // Load initial data
    const { data } = await riverApi.pageInit(modelKey, projectName);
    if (data) {
      tfdStores.mlflowMetrics.set(data.metrics);
      tfdStores.incrementalModelAvailable.set(data.model_available);
      tfdStores.mlflowExperimentUrl.set(data.experiment_url);
    }

    // Start metrics polling if training is enabled
    if ($tfdStores.incrementalMlState) {
      startMetricsPolling();
    }
  });

  onDestroy(() => {
    stopMetricsPolling();
  });

  function startMetricsPolling() {
    metricsInterval = setInterval(async () => {
      const { data } = await riverApi.getMlflowMetrics(projectName);
      if (data) {
        tfdStores.mlflowMetrics.set(data.metrics);
        tfdStores.mlflowRunInfo.set(data.run_info);
      }
    }, 5000);
  }

  function stopMetricsPolling() {
    if (metricsInterval) {
      clearInterval(metricsInterval);
    }
  }

  // Toggle training
  async function handleTrainingToggle(enabled: boolean) {
    const { data, error } = await riverApi.switchModel(modelKey, enabled);
    if (data) {
      tfdStores.incrementalMlState.set(enabled);
      if (enabled) {
        startMetricsPolling();
      } else {
        stopMetricsPolling();
      }
    }
  }

  // Make prediction
  async function handlePredict() {
    tfdStores.predictionLoading.set(true);
    const { data, error } = await riverApi.predict(modelKey, $tfdStores.formData);
    if (data) {
      tfdStores.predictionResults.set(data);
    }
    tfdStores.predictionLoading.set(false);
  }
</script>

<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
  <!-- Left Column: Form + Training Switch -->
  <div class="lg:col-span-1 space-y-4">
    <MlTrainingSwitch
      {modelKey}
      {projectName}
      modelName="ARFClassifier"
      enabled={$tfdStores.incrementalMlState}
      onToggle={handleTrainingToggle}
      experimentUrl={$tfdStores.mlflowExperimentUrl}
    />

    <TfdForm
      formData={$tfdStores.formData}
      onUpdate={(key, value) => tfdStores.formData.update(d => ({ ...d, [key]: value }))}
      onPredict={handlePredict}
      onRandomize={() => { /* randomize logic */ }}
      disabled={!$tfdStores.incrementalModelAvailable}
      loading={$tfdStores.predictionLoading}
    />

    {#if $tfdStores.predictionResults}
      <TfdPredictionResult results={$tfdStores.predictionResults} />
    {/if}
  </div>

  <!-- Right Column: Metrics Dashboard -->
  <div class="lg:col-span-2">
    <TfdMetrics
      metrics={$tfdStores.mlflowMetrics}
      runInfo={$tfdStores.mlflowRunInfo}
    />
  </div>
</div>
```

### 9.4 SQL Page

```svelte
<!-- src/routes/tfd/sql/+page.svelte -->
<script lang="ts">
  import { tfdStores } from '$stores/tfd';
  import { riverApi } from '$api/river';
  import { PROJECT_NAME_MAP } from '$stores/shared';
  import SqlInterface from '$components/shared/SqlInterface.svelte';

  const projectName = PROJECT_NAME_MAP.tfd;

  const templates = [
    { name: 'Recent Transactions', query: 'SELECT * FROM transactions ORDER BY timestamp DESC LIMIT 100' },
    { name: 'Fraud Statistics', query: 'SELECT is_fraud, COUNT(*) as count FROM transactions GROUP BY is_fraud' },
    { name: 'By Category', query: 'SELECT product_category, COUNT(*) as count FROM transactions GROUP BY product_category ORDER BY count DESC' },
  ];

  async function handleExecute(query: string, engine: 'polars' | 'duckdb') {
    tfdStores.sqlLoading.set(true);
    tfdStores.sqlError.set('');

    const startTime = performance.now();
    const { data, error } = await riverApi.executeSql(projectName, query, engine);
    const endTime = performance.now();

    tfdStores.sqlExecutionTime.set(Math.round(endTime - startTime));

    if (data) {
      tfdStores.sqlQueryResults.set(data);
    } else if (error) {
      tfdStores.sqlError.set(error.message);
    }

    tfdStores.sqlLoading.set(false);
  }
</script>

<SqlInterface
  query={$tfdStores.sqlQueryInput}
  results={$tfdStores.sqlQueryResults}
  loading={$tfdStores.sqlLoading}
  error={$tfdStores.sqlError}
  executionTime={$tfdStores.sqlExecutionTime}
  engine={$tfdStores.sqlEngine}
  {templates}
  onQueryChange={(q) => tfdStores.sqlQueryInput.set(q)}
  onEngineChange={(e) => tfdStores.sqlEngine.set(e)}
  onExecute={handleExecute}
/>
```

---

## 10. Frontend Improvements

### 10.1 Skeleton Loading States

```svelte
<!-- src/lib/components/shared/MetricsGridSkeleton.svelte -->
<script lang="ts">
  export let columns: number = 5;
  export let rows: number = 1;
</script>

<div class="space-y-3 animate-pulse">
  {#each Array(rows) as _, rowIdx}
    <div class="grid gap-2" style="grid-template-columns: repeat({columns}, minmax(0, 1fr))">
      {#each Array(columns) as _, colIdx}
        <div class="bg-gray-800/50 rounded-lg border border-gray-700/50 p-4">
          <div class="space-y-2">
            <div class="h-3 bg-gray-700 rounded w-2/3"></div>
            <div class="h-6 bg-gray-700 rounded w-1/2"></div>
          </div>
        </div>
      {/each}
    </div>
  {/each}
</div>
```

**Usage with streaming promises:**

```svelte
{#await metricsPromise}
  <MetricsGridSkeleton columns={5} rows={2} />
{:then metrics}
  <MetricsGrid {metrics} />
{:catch error}
  <ErrorState message={error.message} />
{/await}
```

### 10.2 Animated Numbers

```svelte
<!-- src/lib/components/charts/AnimatedNumber.svelte -->
<script lang="ts">
  import { tweened } from 'svelte/motion';
  import { cubicOut } from 'svelte/easing';

  export let value: number;
  export let decimals: number = 2;
  export let suffix: string = '';
  export let prefix: string = '';
  export let duration: number = 800;

  const displayValue = tweened(0, {
    duration,
    easing: cubicOut,
  });

  $: displayValue.set(value);
</script>

<span class="tabular-nums">
  {prefix}{$displayValue.toFixed(decimals)}{suffix}
</span>
```

### 10.3 Metric Card with Microinteractions

```svelte
<!-- src/lib/components/shared/MetricCard.svelte -->
<script lang="ts">
  import { Card } from '$components/ui/card';
  import AnimatedNumber from '$components/charts/AnimatedNumber.svelte';
  import MetricInfoDialog from './MetricInfoDialog.svelte';
  import { TrendingUp, TrendingDown } from 'lucide-svelte';

  export let label: string;
  export let value: number;
  export let previousValue: number | null = null;
  export let metricKey: string | null = null;
  export let projectKey: 'tfd' | 'eta' | 'ecci' = 'tfd';
  export let suffix: string = '%';
  export let decimals: number = 2;
  export let higherIsBetter: boolean = true;

  $: trend = previousValue !== null ? value - previousValue : null;
  $: trendPositive = trend !== null && trend > 0;
  $: trendColor = trend === null
    ? ''
    : (higherIsBetter ? (trendPositive ? 'text-green-400' : 'text-red-400')
                      : (trendPositive ? 'text-red-400' : 'text-green-400'));
</script>

<Card
  class="group relative overflow-hidden transition-all duration-200
         hover:border-blue-500/50 hover:shadow-lg hover:shadow-blue-500/10"
>
  <!-- Subtle gradient overlay on hover -->
  <div
    class="absolute inset-0 bg-gradient-to-br from-blue-500/0 to-purple-500/0
           group-hover:from-blue-500/5 group-hover:to-purple-500/5
           transition-all duration-300"
  />

  <div class="relative p-4 space-y-1">
    <div class="flex items-center justify-center gap-1">
      <span class="text-xs font-medium text-gray-400">{label}</span>
      {#if metricKey}
        <MetricInfoDialog {metricKey} {projectKey} />
      {/if}
    </div>

    <div class="flex items-baseline justify-center gap-2">
      <span class="text-2xl font-bold">
        <AnimatedNumber {value} {decimals} {suffix} />
      </span>

      {#if trend !== null && trend !== 0}
        <span class="flex items-center gap-0.5 text-sm {trendColor} transition-colors">
          {#if trendPositive}
            <TrendingUp size={14} />
          {:else}
            <TrendingDown size={14} />
          {/if}
          {Math.abs(trend).toFixed(1)}
        </span>
      {/if}
    </div>
  </div>
</Card>
```

### 10.4 Progressive Disclosure Form

```svelte
<!-- src/lib/components/tfd/TfdForm.svelte -->
<script lang="ts">
  import { Card, CardHeader, CardTitle, CardContent } from '$components/ui/card';
  import { Button } from '$components/ui/button';
  import { Input } from '$components/ui/input';
  import { Select } from '$components/ui/select';
  import { Checkbox } from '$components/ui/checkbox';
  import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '$components/ui/collapsible';
  import { Badge } from '$components/ui/badge';
  import { CreditCard, Smartphone, MapPin, ChevronDown, Shuffle, Loader2 } from 'lucide-svelte';
  import { tfdDropdownOptions } from '$stores/tfd';

  export let formData: Record<string, any>;
  export let onUpdate: (key: string, value: any) => void;
  export let onPredict: () => void;
  export let onRandomize: () => void;
  export let disabled: boolean = false;
  export let loading: boolean = false;

  let showDeviceInfo = false;
  let showLocationInfo = false;
</script>

<form on:submit|preventDefault={onPredict} class="space-y-4">
  <!-- Essential Fields (Always Visible) -->
  <Card>
    <CardHeader class="pb-3">
      <CardTitle class="flex items-center gap-2 text-lg">
        <CreditCard size={18} class="text-blue-400" />
        Transaction Details
      </CardTitle>
    </CardHeader>
    <CardContent class="grid grid-cols-3 gap-3">
      <div class="space-y-1">
        <label class="text-xs text-gray-400">Amount</label>
        <Input
          type="number"
          step="0.01"
          value={formData.amount || ''}
          on:input={(e) => onUpdate('amount', e.currentTarget.value)}
        />
      </div>
      <div class="space-y-1">
        <label class="text-xs text-gray-400">Currency</label>
        <Select
          value={formData.currency || ''}
          onValueChange={(v) => onUpdate('currency', v)}
          options={tfdDropdownOptions.currency}
        />
      </div>
      <div class="space-y-1">
        <label class="text-xs text-gray-400">Category</label>
        <Select
          value={formData.product_category || ''}
          onValueChange={(v) => onUpdate('product_category', v)}
          options={tfdDropdownOptions.product_category}
        />
      </div>
      <div class="space-y-1">
        <label class="text-xs text-gray-400">Transaction Type</label>
        <Select
          value={formData.transaction_type || ''}
          onValueChange={(v) => onUpdate('transaction_type', v)}
          options={tfdDropdownOptions.transaction_type}
        />
      </div>
      <div class="space-y-1">
        <label class="text-xs text-gray-400">Payment Method</label>
        <Select
          value={formData.payment_method || ''}
          onValueChange={(v) => onUpdate('payment_method', v)}
          options={tfdDropdownOptions.payment_method}
        />
      </div>
      <div class="space-y-1">
        <label class="text-xs text-gray-400">Account Age (days)</label>
        <Input
          type="number"
          min="0"
          value={formData.account_age_days || ''}
          on:input={(e) => onUpdate('account_age_days', e.currentTarget.value)}
        />
      </div>
    </CardContent>
  </Card>

  <!-- Device Information (Collapsible) -->
  <Collapsible bind:open={showDeviceInfo}>
    <Card>
      <CollapsibleTrigger asChild let:builder>
        <CardHeader
          class="cursor-pointer hover:bg-gray-800/50 transition-colors pb-3"
          {...builder}
        >
          <div class="flex items-center justify-between">
            <CardTitle class="flex items-center gap-2 text-lg">
              <Smartphone size={18} class="text-gray-400" />
              Device Information
              <Badge variant="outline" class="text-xs">Optional</Badge>
            </CardTitle>
            <ChevronDown
              size={18}
              class="text-gray-400 transition-transform duration-200 {showDeviceInfo ? 'rotate-180' : ''}"
            />
          </div>
        </CardHeader>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <CardContent class="grid grid-cols-3 gap-3 pt-0">
          <div class="space-y-1">
            <label class="text-xs text-gray-400">Browser</label>
            <Select
              value={formData.browser || ''}
              onValueChange={(v) => onUpdate('browser', v)}
              options={tfdDropdownOptions.browser}
            />
          </div>
          <div class="space-y-1">
            <label class="text-xs text-gray-400">OS</label>
            <Select
              value={formData.os || ''}
              onValueChange={(v) => onUpdate('os', v)}
              options={tfdDropdownOptions.os}
            />
          </div>
          <div class="space-y-1">
            <label class="text-xs text-gray-400">IP Address</label>
            <Input
              value={formData.ip_address || ''}
              on:input={(e) => onUpdate('ip_address', e.currentTarget.value)}
              placeholder="192.168.1.1"
            />
          </div>
        </CardContent>
      </CollapsibleContent>
    </Card>
  </Collapsible>

  <!-- Location Information (Collapsible) -->
  <Collapsible bind:open={showLocationInfo}>
    <Card>
      <CollapsibleTrigger asChild let:builder>
        <CardHeader
          class="cursor-pointer hover:bg-gray-800/50 transition-colors pb-3"
          {...builder}
        >
          <div class="flex items-center justify-between">
            <CardTitle class="flex items-center gap-2 text-lg">
              <MapPin size={18} class="text-gray-400" />
              Location
              <Badge variant="outline" class="text-xs">Optional</Badge>
            </CardTitle>
            <ChevronDown
              size={18}
              class="text-gray-400 transition-transform duration-200 {showLocationInfo ? 'rotate-180' : ''}"
            />
          </div>
        </CardHeader>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <CardContent class="grid grid-cols-2 gap-3 pt-0">
          <div class="space-y-1">
            <label class="text-xs text-gray-400">Latitude</label>
            <Input
              type="number"
              step="0.0001"
              min="-90"
              max="90"
              value={formData.lat || ''}
              on:input={(e) => onUpdate('lat', e.currentTarget.value)}
            />
          </div>
          <div class="space-y-1">
            <label class="text-xs text-gray-400">Longitude</label>
            <Input
              type="number"
              step="0.0001"
              min="-180"
              max="180"
              value={formData.lon || ''}
              on:input={(e) => onUpdate('lon', e.currentTarget.value)}
            />
          </div>
        </CardContent>
      </CollapsibleContent>
    </Card>
  </Collapsible>

  <!-- Security Flags -->
  <Card>
    <CardContent class="flex items-center gap-6 py-4">
      <Checkbox
        checked={formData.cvv_provided || false}
        onCheckedChange={(v) => onUpdate('cvv_provided', v)}
        label="CVV Provided"
      />
      <Checkbox
        checked={formData.billing_address_match || false}
        onCheckedChange={(v) => onUpdate('billing_address_match', v)}
        label="Billing Address Match"
      />
    </CardContent>
  </Card>

  <!-- Action Buttons -->
  <div class="flex gap-2">
    <Button type="submit" class="flex-1" {disabled}>
      {#if loading}
        <Loader2 size={16} class="mr-2 animate-spin" />
        Predicting...
      {:else}
        Predict Fraud Risk
      {/if}
    </Button>
    <Button type="button" variant="outline" on:click={onRandomize} title="Randomize (⌘⇧R)">
      <Shuffle size={16} />
    </Button>
  </div>
</form>
```

### 10.5 Live Status Indicator

```svelte
<!-- src/lib/components/shared/LiveStatusIndicator.svelte -->
<script lang="ts">
  import { Activity } from 'lucide-svelte';

  export let isConnected: boolean = false;
  export let isTraining: boolean = false;
  export let samplesProcessed: number = 0;
</script>

<div class="flex items-center gap-3 px-3 py-2 bg-gray-800/50 rounded-lg border border-gray-700/50">
  <!-- Connection status with animated pulse -->
  <div class="flex items-center gap-2">
    <div class="relative">
      <div
        class="w-2 h-2 rounded-full transition-colors"
        class:bg-green-500={isConnected}
        class:bg-red-500={!isConnected}
      />
      {#if isConnected && isTraining}
        <div class="absolute inset-0 w-2 h-2 rounded-full bg-green-500 animate-ping" />
      {/if}
    </div>
    <span
      class="text-sm transition-colors"
      class:text-green-400={isConnected}
      class:text-red-400={!isConnected}
    >
      {isConnected ? 'Connected' : 'Disconnected'}
    </span>
  </div>

  <!-- Live counter -->
  {#if isTraining && samplesProcessed > 0}
    <div class="flex items-center gap-1.5 text-sm text-gray-400">
      <Activity size={14} class="text-blue-400" />
      <span class="tabular-nums font-mono">
        {samplesProcessed.toLocaleString()}
      </span>
      <span>samples</span>
    </div>
  {/if}
</div>
```

### 10.6 Empty State Component

```svelte
<!-- src/lib/components/shared/EmptyState.svelte -->
<script lang="ts">
  import { Button } from '$components/ui/button';
  import type { ComponentType } from 'svelte';

  export let icon: ComponentType;
  export let title: string;
  export let description: string;
  export let action: { label: string; onClick: () => void } | null = null;
</script>

<div class="flex flex-col items-center justify-center py-12 text-center animate-fade-in">
  <div class="w-12 h-12 rounded-full bg-gray-800 flex items-center justify-center mb-4">
    <svelte:component this={icon} size={24} class="text-gray-500" />
  </div>
  <h3 class="text-lg font-medium text-gray-200 mb-1">{title}</h3>
  <p class="text-sm text-gray-400 max-w-sm mb-4">{description}</p>
  {#if action}
    <Button variant="outline" on:click={action.onClick}>
      {action.label}
    </Button>
  {/if}
</div>
```

### 10.7 SQL Interface with Syntax Highlighting

```svelte
<!-- src/lib/components/shared/SqlInterface.svelte -->
<script lang="ts">
  import { Card, CardHeader, CardTitle, CardContent } from '$components/ui/card';
  import { Button } from '$components/ui/button';
  import { Select } from '$components/ui/select';
  import { Badge } from '$components/ui/badge';
  import { Database, Play, Loader2 } from 'lucide-svelte';
  import CodeMirror from 'svelte-codemirror-editor';
  import { sql } from '@codemirror/lang-sql';
  import { oneDark } from '@codemirror/theme-one-dark';

  export let query: string;
  export let results: { columns: string[]; data: any[][]; row_count: number } | null;
  export let loading: boolean;
  export let error: string;
  export let executionTime: number;
  export let engine: 'polars' | 'duckdb';
  export let templates: { name: string; query: string }[];
  export let onQueryChange: (query: string) => void;
  export let onEngineChange: (engine: 'polars' | 'duckdb') => void;
  export let onExecute: (query: string, engine: 'polars' | 'duckdb') => void;
</script>

<div class="space-y-4">
  <!-- Query Editor -->
  <Card>
    <CardHeader class="flex flex-row items-center justify-between py-3">
      <CardTitle class="flex items-center gap-2 text-lg">
        <Database size={18} class="text-blue-400" />
        SQL Query
      </CardTitle>
      <div class="flex items-center gap-2">
        <Select
          value=""
          onValueChange={(v) => onQueryChange(v)}
          placeholder="Templates..."
        >
          {#each templates as template}
            <option value={template.query}>{template.name}</option>
          {/each}
        </Select>
        <Select
          value={engine}
          onValueChange={(v) => onEngineChange(v)}
        >
          <option value="polars">Polars</option>
          <option value="duckdb">DuckDB</option>
        </Select>
        <Button
          size="sm"
          on:click={() => onExecute(query, engine)}
          disabled={loading || !query.trim()}
        >
          {#if loading}
            <Loader2 size={14} class="mr-1 animate-spin" />
          {:else}
            <Play size={14} class="mr-1" />
          {/if}
          Execute
        </Button>
      </div>
    </CardHeader>
    <CardContent class="p-0 border-t border-gray-700">
      <CodeMirror
        bind:value={query}
        lang={sql()}
        theme={oneDark}
        styles={{
          '&': {
            fontSize: '14px',
            minHeight: '150px',
          },
        }}
      />
    </CardContent>
  </Card>

  <!-- Error Display -->
  {#if error}
    <Card class="border-red-500/50 bg-red-500/10">
      <CardContent class="py-3 text-red-400 text-sm">
        {error}
      </CardContent>
    </Card>
  {/if}

  <!-- Results Table -->
  {#if results}
    <Card>
      <CardHeader class="flex flex-row items-center justify-between py-3">
        <span class="text-sm text-gray-400">
          {results.row_count.toLocaleString()} rows
        </span>
        <Badge variant="outline">
          {executionTime}ms
        </Badge>
      </CardHeader>
      <CardContent class="p-0 max-h-96 overflow-auto">
        <table class="w-full text-sm">
          <thead class="bg-gray-800 sticky top-0">
            <tr>
              {#each results.columns as column}
                <th class="px-4 py-2 text-left font-medium text-gray-300 border-b border-gray-700">
                  {column}
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each results.data as row}
              <tr class="hover:bg-gray-800/50">
                {#each row as cell}
                  <td class="px-4 py-2 border-b border-gray-800 text-gray-300">
                    {cell ?? '-'}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </CardContent>
    </Card>
  {/if}
</div>
```

---

## 11. Deployment Configuration

### 11.1 Dockerfile

```dockerfile
# apps/sveltekit/Dockerfile.sveltekit
FROM node:22-slim AS builder

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build

# Production stage
FROM node:22-slim AS runner

WORKDIR /app

# Copy built application
COPY --from=builder /app/build ./build
COPY --from=builder /app/package*.json ./

# Install production dependencies only
RUN npm ci --omit=dev

# Set environment
ENV NODE_ENV=production
ENV PORT=3000

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# Start server
CMD ["node", "build"]
```

### 11.2 Helm Values Addition

```yaml
# k3d/helm/values.yaml (add to existing file)

sveltekit:
  enabled: true
  image: coelho-realtime-sveltekit:latest
  imagePullPolicy: IfNotPresent
  replicas: 1
  resources:
    requests:
      memory: "256Mi"
      cpu: "100m"
    limits:
      memory: "512Mi"
      cpu: "500m"
  portsSettings:
    ports:
      - name: http
        port: 3000
        targetPort: 3000
        nodePort: 30010
        protocol: TCP
    type: LoadBalancer
  env:
    - name: VITE_RIVER_URL
      value: "http://coelho-realtime-river:8002"
    - name: VITE_SKLEARN_URL
      value: "http://coelho-realtime-sklearn:8003"
    - name: NODE_ENV
      value: "production"
  livenessProbeSettings:
    livenessProbe:
      httpGet:
        path: /health
        port: 3000
      initialDelaySeconds: 30
      periodSeconds: 10
      failureThreshold: 3
  readinessProbeSettings:
    readinessProbe:
      httpGet:
        path: /health
        port: 3000
      initialDelaySeconds: 10
      periodSeconds: 5
      failureThreshold: 3
```

### 11.3 Skaffold Configuration Addition

```yaml
# skaffold.yaml (add to artifacts section)

- image: coelho-realtime-sveltekit
  context: ./apps/sveltekit
  docker:
    dockerfile: Dockerfile.sveltekit
  sync:
    manual:
      - src: "src/**/*.svelte"
        dest: /app
      - src: "src/**/*.ts"
        dest: /app
      - src: "src/**/*.css"
        dest: /app
```

### 11.4 Port Forward Addition

```yaml
# skaffold.yaml (add to portForward section)

# SvelteKit (new frontend)
- resourceType: service
  resourceName: coelho-realtime-sveltekit
  namespace: coelho-realtime
  port: 3000
  localPort: 3010  # Different port during migration
  address: 0.0.0.0
```

---

## 12. Migration Phases

### Phase 1: Setup (Week 1)

**Tasks:**
- [ ] Create `apps/sveltekit/` directory structure
- [ ] Initialize SvelteKit project with TypeScript
- [ ] Install and configure dependencies
- [ ] Set up Tailwind CSS and shadcn-svelte
- [ ] Copy JSON data files from Reflex
- [ ] Create base API client
- [ ] Create shared stores structure
- [ ] Implement root layout with Navbar

**Deliverable:** Empty SvelteKit app running on port 3010

### Phase 2: Home + TFD Pages (Week 2)

**Tasks:**
- [ ] Implement Home page
- [ ] Implement TFD layout with tabs
- [ ] Implement TFD Incremental page
  - [ ] ML Training Switch
  - [ ] TFD Form component
  - [ ] TFD Metrics component
  - [ ] Prediction result display
- [ ] Implement TFD Batch pages
  - [ ] Batch training box
  - [ ] MLflow run selector
  - [ ] Batch metrics with YellowBrick
- [ ] Implement TFD SQL page

**Deliverable:** Fully functional TFD section

### Phase 3: ETA + ECCI Pages (Week 3)

**Tasks:**
- [ ] Implement ETA layout and pages
  - [ ] Leaflet map component
  - [ ] ETA-specific forms
  - [ ] Regression metrics
- [ ] Implement ECCI layout and pages
  - [ ] Cluster visualization
  - [ ] Clustering metrics
  - [ ] ECCI-specific features

**Deliverable:** All pages functional

### Phase 4: Polish + Testing (Week 4)

**Tasks:**
- [ ] Add skeleton loading states
- [ ] Implement microinteractions
- [ ] Add error boundaries and empty states
- [ ] Keyboard shortcuts
- [ ] Accessibility audit
- [ ] Performance testing
- [ ] Visual comparison with Reflex
- [ ] Fix any discrepancies

**Deliverable:** Production-ready SvelteKit app

### Phase 5: Cutover

**Tasks:**
- [ ] Update Helm values to make SvelteKit primary (port 3000)
- [ ] Move Reflex to backup port (3005)
- [ ] Update any documentation
- [ ] Monitor for issues
- [ ] After 2 weeks stability: remove Reflex service

---

## 13. Testing Strategy

### 13.1 Unit Tests (Vitest)

```typescript
// src/lib/stores/tfd.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { get } from 'svelte/store';
import { tfdStores } from './tfd';

describe('TFD Stores', () => {
  beforeEach(() => {
    // Reset stores before each test
    tfdStores.mlflowMetrics.set({});
    tfdStores.predictionResults.set(null);
  });

  it('should update metrics correctly', () => {
    tfdStores.mlflowMetrics.set({ rocauc: 0.85, recall: 0.90 });
    expect(get(tfdStores.mlflowMetrics)).toEqual({ rocauc: 0.85, recall: 0.90 });
  });

  it('should calculate formatted metrics', () => {
    tfdStores.mlflowMetrics.set({ rocauc: 0.85 });
    // Test derived store
  });
});
```

### 13.2 Component Tests (Testing Library)

```typescript
// src/lib/components/shared/MetricCard.test.ts
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect } from 'vitest';
import MetricCard from './MetricCard.svelte';

describe('MetricCard', () => {
  it('renders label and value', () => {
    render(MetricCard, { props: { label: 'ROC-AUC', value: 85.5 } });
    expect(screen.getByText('ROC-AUC')).toBeInTheDocument();
    expect(screen.getByText(/85.5/)).toBeInTheDocument();
  });

  it('shows trend indicator when previousValue provided', () => {
    render(MetricCard, {
      props: { label: 'F1', value: 90, previousValue: 85 }
    });
    // Should show positive trend
  });
});
```

### 13.3 E2E Tests (Playwright)

```typescript
// tests/tfd.spec.ts
import { test, expect } from '@playwright/test';

test.describe('TFD Incremental Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/tfd/incremental');
  });

  test('should display metrics dashboard', async ({ page }) => {
    await expect(page.locator('text=Transaction Fraud Detection')).toBeVisible();
    await expect(page.locator('[data-testid="metrics-grid"]')).toBeVisible();
  });

  test('should enable ML training toggle', async ({ page }) => {
    const toggle = page.locator('[data-testid="ml-training-switch"]');
    await toggle.click();
    await expect(toggle).toBeChecked();
  });

  test('should make prediction with form data', async ({ page }) => {
    // Fill form
    await page.fill('[name="amount"]', '150.00');
    await page.selectOption('[name="currency"]', 'USD');

    // Submit
    await page.click('button:has-text("Predict")');

    // Wait for result
    await expect(page.locator('[data-testid="prediction-result"]')).toBeVisible();
  });
});
```

---

## 14. Performance Benchmarks

### 14.1 Target Metrics

| Metric | Reflex Baseline | SvelteKit Target | Measurement |
|--------|-----------------|------------------|-------------|
| Lighthouse Performance | 45-55 | 90+ | Chrome DevTools |
| First Contentful Paint | 2.5s | 0.8s | Lighthouse |
| Time to Interactive | 4.0s | 1.2s | Lighthouse |
| Total Blocking Time | 800ms | 50ms | Lighthouse |
| Cumulative Layout Shift | 0.15 | 0.05 | Lighthouse |
| Bundle Size (gzipped) | 180KB | 25KB | `npm run build` |
| State Update Latency | 100ms | 5ms | Custom timing |
| API Response (cached) | 150ms | 10ms | Network tab |

### 14.2 Monitoring Setup

```typescript
// src/lib/utils/performance.ts
export function measureInteraction(name: string, fn: () => void | Promise<void>) {
  const start = performance.now();
  const result = fn();

  if (result instanceof Promise) {
    return result.finally(() => {
      const duration = performance.now() - start;
      console.log(`[PERF] ${name}: ${duration.toFixed(2)}ms`);
      // Send to analytics if needed
    });
  }

  const duration = performance.now() - start;
  console.log(`[PERF] ${name}: ${duration.toFixed(2)}ms`);
}
```

---

## Appendix A: JSON Data Files

Copy these files from `apps/reflex/coelho_realtime/data/` to `apps/sveltekit/src/lib/data/`:

- `dropdown_options_tfd.json` → `dropdown-options-tfd.json`
- `dropdown_options_eta.json` → `dropdown-options-eta.json`
- `dropdown_options_ecci.json` → `dropdown-options-ecci.json`
- `metric_info_tfd.json` → `metric-info-tfd.json`
- `metric_info_eta.json` → `metric-info-eta.json`
- `metric_info_ecci.json` → `metric-info-ecci.json`
- `yellowbrick_info_tfd.json` → `yellowbrick-info-tfd.json`
- `yellowbrick_info_eta.json` → `yellowbrick-info-eta.json`
- `yellowbrick_info_ecci.json` → `yellowbrick-info-ecci.json`

---

## Appendix B: shadcn-svelte Components to Install

```bash
# Initialize shadcn-svelte
npx shadcn-svelte@latest init

# Install required components
npx shadcn-svelte@latest add button
npx shadcn-svelte@latest add card
npx shadcn-svelte@latest add input
npx shadcn-svelte@latest add select
npx shadcn-svelte@latest add switch
npx shadcn-svelte@latest add checkbox
npx shadcn-svelte@latest add badge
npx shadcn-svelte@latest add dialog
npx shadcn-svelte@latest add tabs
npx shadcn-svelte@latest add tooltip
npx shadcn-svelte@latest add collapsible
npx shadcn-svelte@latest add separator
npx shadcn-svelte@latest add sheet
npx shadcn-svelte@latest add skeleton
npx shadcn-svelte@latest add alert
```

---

## Appendix C: Environment Variables

```bash
# .env (development)
VITE_RIVER_URL=http://localhost:8002
VITE_SKLEARN_URL=http://localhost:8003

# .env.production (Kubernetes)
VITE_RIVER_URL=http://coelho-realtime-river:8002
VITE_SKLEARN_URL=http://coelho-realtime-sklearn:8003
```

---

## References

- [SvelteKit Documentation](https://kit.svelte.dev/docs)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [shadcn-svelte](https://www.shadcn-svelte.com/)
- [svelte-plotly.js](https://github.com/cshaa/svelte-plotly.js)
- [Leaflet](https://leafletjs.com/reference.html)
- [Dashboard UI Design Principles 2026](https://www.designstudiouiux.com/blog/dashboard-ui-design-guide/)
