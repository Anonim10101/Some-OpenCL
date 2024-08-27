#define float(X) expand(X)
#define expand(X) float##X
typedef float(PER_THREAD_INC) floatX;
kernel void kmain (global floatX *mas, global float *block_sums) {
	size_t th_id = get_global_id(0);
	floatX temp[GR_SIZE * PER_THREAD];
	temp[i] += ((int)(th_id * PER_THREAD_INC / (2 * GR_SIZE)) > 0)? block_sums[th_id * PER_THREAD_INC / (2 * GR_SIZE) - 1]:0;
	//float to_add = ((int)(th_id * PER_THREAD_INC / (2 * GR_SIZE)) > 0)? block_sums[th_id * PER_THREAD_INC / (2 * GR_SIZE) - 1]:0;
	mas[th_id] = temp[i];
}