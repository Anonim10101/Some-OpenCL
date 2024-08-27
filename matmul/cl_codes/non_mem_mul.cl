kernel void mul (global const float *a, global const float *b, global float *c, unsigned int c_w, unsigned int k) {
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	float sum = 0;
	for (size_t i = 0; i < k; i++) {
		sum += a[y * k + i] * b[c_w * i + x];
	}
	c[y * c_w + x] = sum;
}