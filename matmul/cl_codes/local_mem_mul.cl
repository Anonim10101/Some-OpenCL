kernel void mul (global const float *a, global const float *b, global float *c, unsigned int c_w, unsigned int k) {
	local float la [TILE][TILE];
	local float lb [TILE][TILE];
	size_t x = get_group_id(0);
	size_t y = get_group_id(1);
	size_t loc_x = get_local_id(0);
	size_t loc_y = get_local_id(1);
	float sum = 0;
	for (size_t i = 0; i < k; i += TILE) {
		la[loc_y][loc_x] = a[(y*TILE+loc_y)*k+i+loc_x];
		lb[loc_y][loc_x] = b[(i+loc_y)*c_w+x*TILE+loc_x];
		barrier(CLK_LOCAL_MEM_FENCE);
		for (size_t j = 0; j < TILE; j++) {
			sum += la[loc_y][j] * lb[j][loc_x];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	c[(y * TILE + loc_y) * c_w + x * TILE + loc_x] = sum;
}