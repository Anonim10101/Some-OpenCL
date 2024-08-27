kernel void mul (global const float *a, global const float *b, global float *c, unsigned int c_w, unsigned int k) {
	local float la [TILE][TILE];
	local float lb [TILE][TILE];
	size_t x = get_group_id(0);
	size_t y = get_group_id(1);
	size_t loc_x = get_local_id(0);
	size_t loc_y = get_local_id(1);
	float sum = 0;
	for (size_t i = 0; i < k / TILE; i++) {
		la[loc_y][loc_x] = a[TILE * TILE * (y * (k / TILE) + i) + loc_y *  TILE + loc_x];
		lb[loc_y][loc_x] = b[TILE * TILE * (i * (c_w / TILE) + x) + loc_y *  TILE + loc_x];
		barrier(CLK_LOCAL_MEM_FENCE);
		for (size_t j = 0; j < TILE; j++) {
			sum += la[loc_y][j] * lb[j][loc_x];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	c[TILE * TILE * (y * (c_w / TILE) + x) + loc_y *  TILE + loc_x] = sum;
}
/*

	local float temp[GR_SIZE]; //GR_SIZE- размер обрабатываемого участка
	size_t gr_id = get_group_id(0);
	size_t th_id = get_local_id(0);
	temp[2 * th_id] = inp[gr_id * GR_SIZE + th_id]; 
	temp[2 * th_id + 1] = inp[gr_id * GR_SIZE + th_id + 1];
	// а можно и даже нужно, наверное, векторными - так легче регулировать число вычислений в одном потоке
	for (size_t d = 0; d < log2((float)GR_SIZE); d++) { //-1?
		size_t temp_pow = pow((float)2, (float)d + 1);
		if(th_id <= (GR_SIZE-1) / temp_pow) {
			temp[gr_id * GR_SIZE + th_id + temp_pow - 1] = temp[th_id + temp_pow - 1] + temp[th_id + temp_pow];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	output[gr_id * GR_SIZE + th_id] = inp[2 * th_id]; 
	output[gr_id * GR_SIZE + th_id + 1] = temp[2 * th_id + 1];
*/