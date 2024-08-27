#define BANKS_COUNT 32
int padded(int k) {
	return k + k / BANKS_COUNT;
}
kernel void kmain (global const float *inp, unsigned int inp_size, global float *output, global float *sums) {
	local float temp[PER_THREAD * GR_SIZE + GR_SIZE * PER_THREAD / BANKS_COUNT];
	size_t gr_id = get_group_id(0);
	size_t th_id = get_local_id(0);
	temp[padded(PER_THREAD * th_id)] = (gr_id * GR_SIZE * PER_THREAD + PER_THREAD * th_id >= inp_size)? 0 : inp[gr_id * GR_SIZE * PER_THREAD + PER_THREAD * th_id]; 
	temp[padded(PER_THREAD * th_id + 1)] = (gr_id * GR_SIZE * PER_THREAD + PER_THREAD * th_id + 1 >= inp_size)? 0 : inp[gr_id * GR_SIZE * PER_THREAD + PER_THREAD * th_id + 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (size_t d = 0; d <= log2((float)GR_SIZE * PER_THREAD) - 1; d++) {
		size_t temp_pow = pow((float)2, (float)d + 1);
		if(th_id * temp_pow <= PER_THREAD * GR_SIZE - 1) {
			temp[padded((th_id + 1) * temp_pow - 1)] += temp[padded(th_id * temp_pow + temp_pow / 2 - 1)];
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
		if (th_id * temp_pow <= PER_THREAD * GR_SIZE - 1) {
			float t = temp[padded(th_id * temp_pow + temp_pow / 2 - 1)];
			temp[padded(th_id * temp_pow + temp_pow / 2 - 1)] = temp[padded((th_id + 1) * temp_pow - 1)];
			temp[padded((th_id + 1) * temp_pow - 1)] += t;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}	
	output[gr_id * GR_SIZE * PER_THREAD + PER_THREAD * th_id] = temp[padded(PER_THREAD * th_id + 1)]; 
	output[gr_id * GR_SIZE * PER_THREAD + PER_THREAD * th_id + 1] = (PER_THREAD * th_id + 2 >= PER_THREAD * GR_SIZE)? sums[gr_id]:temp[padded(PER_THREAD * th_id + 2)];
}