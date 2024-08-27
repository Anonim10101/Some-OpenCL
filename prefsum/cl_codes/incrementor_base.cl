#define float(X) expand(X)
#define expand(X) float##X
typedef float(PER_THREAD_INC) floatX;
kernel void kmain (global floatX *mas, global float *block_sums) {
	size_t gr_id = get_group_id(0);
	size_t th_id = get_local_id(0);
	local float to_add;
	if (th_id == 0) {
		to_add = gr_id > 0? block_sums[gr_id - 1]:0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	mas[gr_id * GR_SIZE * PER_THREAD / PER_THREAD_INC + th_id] += to_add;
}