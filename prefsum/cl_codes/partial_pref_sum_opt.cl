#define BANKS_COUNT 32
size_t padded(int k) {
	return k + k / BANKS_COUNT;
}
kernel void kmain (global const float *inp, unsigned int inp_size, global float *output, global float *sums) {
	float temp[PER_THREAD + 2];
	temp[0] = 0;
	temp[PER_THREAD / 2 + 1] = 0;
	local float part_sum[2 * GR_SIZE + 2 * GR_SIZE / BANKS_COUNT];
	size_t gr_id = get_group_id(0);
	size_t th_id = get_local_id(0);
	
	for(size_t i = 0; i < PER_THREAD / 2; i++) {
		size_t elem = gr_id * GR_SIZE * PER_THREAD + PER_THREAD * th_id + i;
		float to_add = (elem >= inp_size)? 0 : inp[elem];
		temp[i + 1] = temp[i] + to_add;
	}
	for(size_t i = PER_THREAD / 2; i < PER_THREAD; i++) {
		size_t elem = gr_id * GR_SIZE * PER_THREAD + PER_THREAD * th_id + i;
		float to_add = (elem >= inp_size)? 0 : inp[elem];
		temp[i + 2] = temp[i + 1] + to_add;
	}
	
	part_sum[padded(2 * th_id)] = temp[PER_THREAD / 2];
	part_sum[padded(2 * th_id + 1)] = temp[PER_THREAD + 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (size_t d = 0; d <= log2((float)GR_SIZE); d++) {
		size_t temp_pow = pow((float)2, (float)d + 1);
		if(th_id * temp_pow <= 2 * GR_SIZE - 1) {
			part_sum[padded((th_id + 1) * temp_pow - 1)] += part_sum[padded(th_id * temp_pow + temp_pow / 2 - 1)];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (th_id == 0) {
		sums[gr_id] = part_sum[padded(2 * GR_SIZE - 1)];
		part_sum[padded(2 * GR_SIZE - 1)] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int d = log2((float)GR_SIZE); d >= 0; d--) {
		size_t temp_pow = pow((float)2, (float)d + 1);
		if (th_id * temp_pow <= 2 * GR_SIZE - 1) {
			size_t elem = padded(th_id * temp_pow + temp_pow / 2 - 1);
			float t = part_sum[elem];
			part_sum[elem] = part_sum[elem + temp_pow / 2 + temp_pow / (2 * BANKS_COUNT)];
			part_sum[elem + temp_pow / 2 + temp_pow / (2 * BANKS_COUNT)] += t;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}	 
	
	for (size_t i = 0; i < PER_THREAD / 2; i++) {
		size_t elem = gr_id * GR_SIZE * PER_THREAD + th_id * PER_THREAD + i;
		temp[i + 1] += part_sum[padded(2 * th_id)]; 
		temp[i + PER_THREAD / 2 + 2] += part_sum[padded(2 * th_id + 1)];
		output[elem] = temp[i + 1];
		output[elem + PER_THREAD / 2] = temp[i + PER_THREAD / 2 + 2];
	}
}