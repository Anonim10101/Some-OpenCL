#define float(X) expand(X)
#define expand(X) float##X
typedef float(WIDTH) floatX;
typedef union v_float{
	floatX vect;
	float mas[WIDTH];
} v_float;
kernel void mul (global const floatX *a, global const floatX *b, global floatX *c, unsigned int c_w, unsigned int k) {
	c_w /= WIDTH;
	size_t loc_x = get_local_id(0);
	size_t loc_y = get_local_id(1);
	size_t x = get_group_id(0);
	size_t y = get_group_id(1);
	local v_float la [TILE][TILE / WIDTH];
	local v_float lb [TILE][TILE / WIDTH];
	
	v_float sum[HEIGHT];
	for (size_t i = 0; i < HEIGHT; i++) {
		sum[i].vect = 0;
	}
	
	for (size_t i = 0; i < k / TILE; i++) {
		for (size_t j = 0; j < HEIGHT; j++) {
			la[loc_y * HEIGHT + j][loc_x].vect = a[(y * TILE + loc_y * HEIGHT + j) * (k / WIDTH) + i * TILE / WIDTH + loc_x];
			lb[loc_y * HEIGHT + j][loc_x].vect = b[(i * TILE + loc_y * HEIGHT + j) * c_w + x * TILE / WIDTH + loc_x];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		v_float a_part;
		for (size_t k = 0; k < HEIGHT; k++) {
			for (size_t j = 0; j < TILE / WIDTH; j++) {
				a_part = la[loc_y * HEIGHT + k][j];
				for (size_t num = 0; num < WIDTH; num++) {
					sum[k].vect += lb[WIDTH*j + num][loc_x].vect * a_part.mas[num];
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	for (size_t i = 0; i < HEIGHT; i++) {
		c[(y * TILE + loc_y * HEIGHT + i) * c_w + x * TILE / WIDTH + loc_x] = sum[i].vect;
	}
}