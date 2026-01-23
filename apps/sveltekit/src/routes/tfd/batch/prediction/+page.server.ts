import { redirect } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	// Prediction functionality is now integrated into the metrics page
	redirect(307, '/tfd/batch/metrics');
};
