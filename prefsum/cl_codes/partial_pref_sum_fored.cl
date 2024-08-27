#define BANKS_COUNT 32
int padded(int k) {
	return k + k / BANKS_COUNT;
}
int calc_elem(size_t th_id, int i) {
	return th_id * PER_THREAD / 2 + i;
}
kernel void kmain (global const float *inp, unsigned int inp_size, global float *output, global float *sums) {
	local float temp[PER_THREAD * GR_SIZE + GR_SIZE * PER_THREAD / BANKS_COUNT];
	size_t gr_id = get_group_id(0);
	size_t th_id = get_local_id(0);
	for (size_t i = 0;  i < PER_THREAD; i++) {
		temp[padded(PER_THREAD * th_id + i)] = (gr_id * GR_SIZE * PER_THREAD + PER_THREAD * th_id + i >= inp_size)? 0 : inp[gr_id * GR_SIZE * PER_THREAD + PER_THREAD * th_id + i]; 
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for (size_t d = 0; d <= log2((float)GR_SIZE * PER_THREAD) - 1; d++) {
		size_t temp_pow = pow((float)2, (float)d + 1);
		for (size_t i = 0; i < PER_THREAD / 2; i++) {
			if(calc_elem(th_id, i) * temp_pow <= PER_THREAD * GR_SIZE - 1) {
				temp[padded((calc_elem(th_id, i) + 1) * temp_pow - 1)] += temp[padded(calc_elem(th_id, i) * temp_pow + temp_pow / 2 - 1)];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (th_id == 0) { // корявенько это выглядит
		sums[gr_id] = temp[padded(PER_THREAD * GR_SIZE - 1)];
		temp[padded(PER_THREAD * GR_SIZE - 1)] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int d = log2((float)GR_SIZE * PER_THREAD) - 1; d >= 0; d--) {
		size_t temp_pow = pow((float)2, (float)d + 1);
		for (size_t i = 0; i < PER_THREAD / 2; i++) {
			if (calc_elem(th_id, i) * temp_pow <= PER_THREAD * GR_SIZE - 1) {
				float t = temp[padded(calc_elem(th_id, i) * temp_pow + temp_pow / 2 - 1)];
				temp[padded(calc_elem(th_id, i) * temp_pow + temp_pow / 2 - 1)] = temp[padded((calc_elem(th_id, i) + 1) * temp_pow - 1)];
				temp[padded((calc_elem(th_id, i) + 1) * temp_pow - 1)] += t;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	for (size_t i = 0; i < PER_THREAD; i++) {
		output[gr_id * GR_SIZE * PER_THREAD + PER_THREAD * th_id + i] = (PER_THREAD * th_id + 1 + i >= PER_THREAD * GR_SIZE)? sums[gr_id]:temp[padded(PER_THREAD * th_id + 1 + i)];
	}
	
}