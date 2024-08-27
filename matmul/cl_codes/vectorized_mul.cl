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
	
	v_float sum;
	
	for (size_t i = 0; i < k / TILE; i++) {
		la[loc_y][loc_x].vect = a[(y * TILE + loc_y) *(k / WIDTH) + i * TILE / WIDTH + loc_x];
		lb[loc_y][loc_x].vect = b[(i * TILE + loc_y) * c_w + x * TILE / WIDTH + loc_x];
		barrier(CLK_LOCAL_MEM_FENCE);
		v_float a_part, b_part;
		float multiplier;
		for (size_t j = 0; j < TILE / WIDTH; j++) {
			a_part = la[loc_y][j];
			for (size_t num = 0; num < WIDTH; num++) {
				b_part = lb[WIDTH*j + num][loc_x];
				multiplier = a_part.mas[num];
				sum.vect += b_part.vect * multiplier;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	c[(y * TILE + loc_y) * c_w + x * TILE / WIDTH + loc_x] = sum.vect;
}