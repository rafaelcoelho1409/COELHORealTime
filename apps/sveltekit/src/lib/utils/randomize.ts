/**
 * Form Randomization Utilities
 *
 * Local randomization functions that match Reflex implementation.
 * Uses pre-loaded dropdown options for categorical fields.
 * No API calls needed - instant randomization.
 */

import type { DropdownOptions, FormData } from '$types';

// =============================================================================
// Field Configuration (numerical fields with min/max from Kafka producers)
// =============================================================================

export interface FieldConfig {
	min: number;
	max: number;
	step?: number;
	decimals?: number;
	prefix?: string; // For ID fields that need prefix construction
}

export const FIELD_CONFIG: Record<string, FieldConfig> = {
	// =========================
	// TFD - Transaction Fraud Detection
	// =========================
	merchant_id: { min: 1, max: 200, prefix: 'merchant_' },
	amount: { min: 1, max: 6000, step: 0.01, decimals: 2 },
	account_age_days: { min: 0, max: 1825 }, // 0 to 5 years
	// TFD/ECCI location (Houston metro - shared)
	lat: { min: 29.5, max: 30.1, step: 0.000001, decimals: 6 },
	lon: { min: -95.8, max: -95.0, step: 0.000001, decimals: 6 },

	// =========================
	// ETA - Estimated Time of Arrival
	// =========================
	driver_id: { min: 1000, max: 5000, prefix: 'driver_' },
	vehicle_id: { min: 100, max: 999, prefix: 'vehicle_' },
	// ETA coordinates (Houston metro)
	origin_lat: { min: 29.5, max: 30.1, step: 0.000001, decimals: 6 },
	origin_lon: { min: -95.8, max: -95.0, step: 0.000001, decimals: 6 },
	destination_lat: { min: 29.5, max: 30.1, step: 0.000001, decimals: 6 },
	destination_lon: { min: -95.8, max: -95.0, step: 0.000001, decimals: 6 },
	estimated_distance_km: { min: 0.1, max: 100, step: 0.01, decimals: 2 },
	temperature_celsius: { min: 15, max: 35, step: 0.1, decimals: 1 },
	driver_rating: { min: 3.5, max: 5.0, step: 0.1, decimals: 1 },
	hour_of_day: { min: 0, max: 23 },
	day_of_week: { min: 0, max: 6 },
	initial_estimated_travel_time_seconds: { min: 60, max: 36000 }, // 1 min to 10 hours
	debug_traffic_factor: { min: 0.8, max: 2.0, step: 0.01, decimals: 2 },
	debug_weather_factor: { min: 0.9, max: 2.0, step: 0.01, decimals: 2 },
	debug_incident_delay_seconds: { min: 0, max: 1800 }, // 0 to 30 min
	debug_driver_factor: { min: 0.9, max: 1.1, step: 0.01, decimals: 2 },

	// =========================
	// ECCI - E-Commerce Customer Interactions
	// =========================
	product_id: { min: 1000, max: 1100, prefix: 'prod_' },
	price: { min: 5, max: 2500, step: 0.01, decimals: 2 },
	quantity: { min: 1, max: 10 },
	time_on_page_seconds: { min: 5, max: 300 },
	session_event_sequence: { min: 1, max: 100 }
};

// Legacy alias for ID fields (backward compatibility)
export const ID_FIELD_CONFIG = FIELD_CONFIG;

/**
 * Build full ID string from numeric value (for prefixed ID fields)
 */
export function buildIdString(field: string, numericValue: number): string {
	const config = FIELD_CONFIG[field];
	if (!config?.prefix) return String(numericValue);
	return `${config.prefix}${numericValue}`;
}

/**
 * Extract numeric value from full ID string
 */
export function extractIdNumber(field: string, fullValue: string): number {
	const config = FIELD_CONFIG[field];
	if (!config?.prefix) return parseInt(fullValue) || config?.min || 1;
	const numStr = fullValue.replace(config.prefix, '');
	const num = parseInt(numStr);
	return isNaN(num) ? config.min : num;
}

/**
 * Generate random number within field's valid range
 */
export function randomIdNumber(field: string): number {
	const config = FIELD_CONFIG[field];
	if (!config) return 1;
	return randomInt(config.min, config.max);
}

/**
 * Clamp a value to the field's valid range
 */
export function clampFieldValue(field: string, value: number): number {
	const config = FIELD_CONFIG[field];
	if (!config) return value;
	const clamped = Math.max(config.min, Math.min(config.max, value));
	if (config.decimals !== undefined) {
		return Number(clamped.toFixed(config.decimals));
	}
	return clamped;
}

/**
 * Get field config or return default config
 */
export function getFieldConfig(field: string): FieldConfig {
	return FIELD_CONFIG[field] || { min: 0, max: 999999 };
}

// =============================================================================
// Utility Functions
// =============================================================================

function randomChoice<T>(array: T[]): T {
	return array[Math.floor(Math.random() * array.length)];
}

function pickOption<T>(options: T[] | undefined, fallback: T[]): T {
	const list = options && options.length ? options : fallback;
	return randomChoice(list);
}

function randomInt(min: number, max: number): number {
	return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randomFloat(min: number, max: number, decimals: number = 2): number {
	const value = Math.random() * (max - min) + min;
	return Number(value.toFixed(decimals));
}

function randomHex(length: number): string {
	return [...Array(length)].map(() => Math.floor(Math.random() * 16).toString(16)).join('');
}

function formatDate(date: Date): string {
	return date.toISOString().split('T')[0];
}

function formatTime(date: Date): string {
	return date.toTimeString().split(' ')[0].substring(0, 5);
}

function formatTimestamp(datePart?: string, timePart?: string): string {
	// Handle undefined, null, and empty strings
	const safeDate = datePart && datePart.trim() ? datePart.trim() : formatDate(new Date());
	const safeTime = timePart && timePart.trim() ? timePart.trim() : formatTime(new Date());
	return `${safeDate}T${safeTime}:00.000000+00:00`;
}

// =============================================================================
// TFD - Transaction Fraud Detection Randomization
// =============================================================================
export function randomizeTFDForm(opts: DropdownOptions): FormData {
	const now = new Date();

	// User agents for variety
	const userAgents = [
		'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0',
		'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/17.0',
		'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15',
		'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/120.0.0.0 Mobile',
		'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0'
	];

	return {
		// Categorical fields from dropdown options
		currency: pickOption(opts.currency, ['USD']),
		merchant_id: randomIdNumber('merchant_id'), // Numeric ID (1-200)
		product_category: pickOption(opts.product_category, ['electronics']),
		transaction_type: pickOption(opts.transaction_type, ['purchase']),
		payment_method: pickOption(opts.payment_method, ['credit_card']),
		browser: pickOption(opts.browser, ['Chrome']),
		os: pickOption(opts.os, ['Windows']),

		// Numeric fields
		amount: randomFloat(10.0, 5000.0, 2),
		account_age_days: randomInt(1, 3650),

		// Location (Houston metro area)
		lat: randomFloat(29.5, 30.1, 6),
		lon: randomFloat(-95.8, -95.0, 6),

		// Timestamp
		timestamp_date: formatDate(now),
		timestamp_time: formatTime(now),

		// Boolean fields
		cvv_provided: randomChoice([true, false]),
		billing_address_match: randomChoice([true, false]),

		// Generated IDs
		transaction_id: `txn_${randomHex(12)}`,
		user_id: `user_${randomInt(1000, 9999)}`,
		ip_address: `${randomInt(1, 255)}.${randomInt(0, 255)}.${randomInt(0, 255)}.${randomInt(1, 255)}`,
		user_agent: randomChoice(userAgents)
	};
}

// =============================================================================
// ETA - Estimated Time of Arrival Randomization
// =============================================================================
export function randomizeETAForm(opts: DropdownOptions): FormData {
	const now = new Date();

	// Generate origin coordinates (Houston metro area)
	const originLat = randomFloat(29.5, 30.1, 6);
	const originLon = randomFloat(-95.8, -95.0, 6);

	// Generate destination that's different from origin
	let destLat = randomFloat(29.5, 30.1, 6);
	let destLon = randomFloat(-95.8, -95.0, 6);

	// Ensure destination is at least 0.01 degrees away
	while (Math.abs(destLat - originLat) < 0.01 && Math.abs(destLon - originLon) < 0.01) {
		destLat = randomFloat(29.5, 30.1, 6);
		destLon = randomFloat(-95.8, -95.0, 6);
	}

	// Calculate estimated distance (simplified formula)
	const latDiff = Math.abs(destLat - originLat);
	const lonDiff = Math.abs(destLon - originLon);
	const distanceKm = latDiff * 111 + lonDiff * 85; // Approximate km per degree

	// Calculate initial travel time (60 seconds per km baseline)
	const initialTravelTimeSeconds = Math.round(distanceKm * 60);

	return {
		// Categorical fields from dropdown options
		driver_id: randomIdNumber('driver_id'), // Numeric ID (1000-5000)
		vehicle_id: randomIdNumber('vehicle_id'), // Numeric ID (100-999)
		weather: pickOption(opts.weather, ['Clear']),
		vehicle_type: pickOption(opts.vehicle_type, ['Sedan']),

		// Coordinates
		origin_lat: originLat,
		origin_lon: originLon,
		destination_lat: destLat,
		destination_lon: destLon,

		// Calculated fields
		estimated_distance_km: Number(distanceKm.toFixed(2)),
		initial_estimated_travel_time_seconds: initialTravelTimeSeconds,

		// Timestamp
		timestamp_date: formatDate(now),
		timestamp_time: formatTime(now),

		// Time parameters
		hour_of_day: randomInt(0, 23),
		day_of_week: randomInt(0, 6),

		// Numeric fields
		driver_rating: randomFloat(3.5, 5.0, 1),
		temperature_celsius: randomFloat(15.0, 35.0, 1),

		// Debug factors
		debug_traffic_factor: randomFloat(0.8, 1.5, 2),
		debug_weather_factor: randomFloat(0.9, 1.3, 2),
		debug_incident_delay_seconds: randomChoice([0, 0, 0, 60, 120, 300]),
		debug_driver_factor: randomFloat(0.9, 1.1, 2),

		// Generated ID
		trip_id: `trip_${randomHex(12)}`
	};
}

// =============================================================================
// ECCI - E-Commerce Customer Interactions Randomization
// =============================================================================
export function randomizeECCIForm(opts: DropdownOptions): FormData {
	const now = new Date();

	// Search queries (empty is common)
	const searchQueries = ['', '', '', 'laptop', 'phone', 'headphones', 'shoes', 'dress', 'camera'];

	return {
		// Categorical fields from dropdown options
		browser: pickOption(opts.browser, ['Chrome']),
		device_type: pickOption(opts.device_type, ['Desktop']),
		os: pickOption(opts.os, ['Windows']),
		event_type: pickOption(opts.event_type, ['page_view']),
		product_category: pickOption(opts.product_category, ['Electronics']),
		product_id: randomIdNumber('product_id'), // Numeric ID (1000-1100)
		referrer_url: pickOption(opts.referrer_url, ['direct']),

		// Location
		lat: randomFloat(29.5, 30.1, 3),
		lon: randomFloat(-95.8, -95.0, 3),

		// Product info
		price: randomFloat(9.99, 999.99, 2),
		quantity: randomInt(1, 5),

		// Session info
		session_event_sequence: randomInt(1, 20),
		time_on_page_seconds: randomInt(5, 300),

		// Timestamp
		timestamp_date: formatDate(now),
		timestamp_time: formatTime(now),

		// Generated IDs
		customer_id: `cust_${randomHex(8)}`,
		event_id: `evt_${randomHex(12)}`,
		session_id: `sess_${randomHex(10)}`,

		// Generated URLs
		page_url: `https://shop.example.com/products/${randomInt(1000, 9999)}`,
		search_query: randomChoice(searchQueries)
	};
}

// =============================================================================
// Build Prediction Payloads (matching FastAPI expected format)
// =============================================================================

export function buildTFDPredictPayload(form: FormData): Record<string, unknown> {
	return {
		transaction_id: form.transaction_id || `txn_${randomHex(12)}`,
		user_id: form.user_id || `user_${randomInt(1000, 9999)}`,
		timestamp: formatTimestamp(form.timestamp_date as string, form.timestamp_time as string),
		amount: Number(form.amount),
		currency: form.currency,
		merchant_id: buildIdString('merchant_id', Number(form.merchant_id)),
		product_category: form.product_category,
		transaction_type: form.transaction_type,
		payment_method: form.payment_method,
		location: {
			lat: Number(form.lat),
			lon: Number(form.lon)
		},
		ip_address: form.ip_address || '192.168.1.1',
		device_info: {
			os: form.os,
			browser: form.browser
		},
		user_agent: form.user_agent || 'Mozilla/5.0',
		account_age_days: Number(form.account_age_days),
		cvv_provided: Boolean(form.cvv_provided),
		billing_address_match: Boolean(form.billing_address_match)
	};
}

export function buildETAPredictPayload(form: FormData): Record<string, unknown> {
	return {
		trip_id: form.trip_id || `trip_${randomHex(12)}`,
		driver_id: buildIdString('driver_id', Number(form.driver_id)),
		vehicle_id: buildIdString('vehicle_id', Number(form.vehicle_id)),
		timestamp: formatTimestamp(form.timestamp_date as string, form.timestamp_time as string),
		origin: {
			lat: Number(form.origin_lat),
			lon: Number(form.origin_lon)
		},
		destination: {
			lat: Number(form.destination_lat),
			lon: Number(form.destination_lon)
		},
		estimated_distance_km: Number(form.estimated_distance_km),
		weather: form.weather,
		temperature_celsius: Number(form.temperature_celsius),
		day_of_week: Number(form.day_of_week),
		hour_of_day: Number(form.hour_of_day),
		driver_rating: Number(form.driver_rating),
		vehicle_type: form.vehicle_type,
		initial_estimated_travel_time_seconds: Number(form.initial_estimated_travel_time_seconds),
		debug_traffic_factor: Number(form.debug_traffic_factor),
		debug_weather_factor: Number(form.debug_weather_factor),
		debug_incident_delay_seconds: Number(form.debug_incident_delay_seconds),
		debug_driver_factor: Number(form.debug_driver_factor)
	};
}

export function buildECCIPredictPayload(form: FormData): Record<string, unknown> {
	const quantityValue =
		form.quantity !== undefined && form.quantity !== ''
			? Number(form.quantity)
			: null;
	return {
		customer_id: form.customer_id || `cust_${randomHex(8)}`,
		event_id: form.event_id || `evt_${randomHex(12)}`,
		session_id: form.session_id || `sess_${randomHex(10)}`,
		timestamp: formatTimestamp(form.timestamp_date as string, form.timestamp_time as string),
		event_type: form.event_type,
		device_info: {
			device_type: form.device_type,
			browser: form.browser,
			os: form.os
		},
		product_id: buildIdString('product_id', Number(form.product_id)),
		product_category: form.product_category,
		price: Number(form.price),
		location: {
			lat: Number(form.lat),
			lon: Number(form.lon)
		},
		page_url: form.page_url || 'https://shop.example.com/products/1000',
		referrer_url: form.referrer_url,
		time_on_page_seconds: Number(form.time_on_page_seconds),
		quantity: quantityValue,
		search_query: form.search_query || '',
		session_event_sequence: Number(form.session_event_sequence)
	};
}
