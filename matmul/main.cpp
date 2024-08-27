#include <iostream>
#include <fstream>
#include <string>
#include <optional>
#include <vector>
#include <errno.h>
#include <algorithm>
#include <optional>
#include <iomanip>
#include <stdexcept>
#include <thread>
#include <omp.h>
#include <ctime>
#include <functional>
#include <CL/cl.h>
#pragma warning(disable:4996)
#define TILE 32
#define WIDTH 4
#define HEIGHT 1

bool check_errno(const std::string& err_message) {
    if (errno != 0) {
        std::cout << err_message << std::endl;
        return true;
    }
    return false;
}

char* readFile(const char* fname) {
    FILE* f = fopen(fname, "rb");
    if (f == NULL) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    size_t fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* contents = static_cast<char*>(malloc(fsize + 1));
    fread(contents, fsize, 1, f);
    contents[fsize] = 0;
    if (ferror(f)) {
        free(contents);
        return NULL;
    }
    fclose(f);
    return contents;
}

namespace cmd_options {
    void print_help_message() {
        std::cout << "--input name – имя входного файла" << std::endl
            << "--output name – имя выходного файла" << std::endl
            << "--output name - имя выходного файла" << std::endl
            << "--device-type { dgpu | igpu | gpu | cpu | all } – указание типа девайса (дискретная видеокарта, интегрированная видеокарта, видеокарты, процессор, все)" << std::endl
            << "--device-index – указание порядкового номера девайса" << std::endl
            << "--realization вариант реализации умножения (0 - на хосте, 1"
            << " - с исопльзованием openCL без использования локальной памяти, "
            << "2 - с исопльзованием openCL и локальной памяти, "
            << "3 - с исопльзованием openCL, локальной памяти и векторным вычислением нескольких элементов в каждом потоке";
    }
    char* get_option(char** begin, char** end, const std::string& option)
    {
        char** itr = std::find(begin, end, option);
        if (itr != end && ++itr != end)
        {
            return *itr;
        }
        return nullptr;
    }

    std::string get_obligatory(char** begin, char** end, const std::string& arg_name) {
        char* arg = get_option(begin, end, arg_name);
        if (arg == nullptr || std::string(arg) == "") {
            throw std::invalid_argument("One of obligatory argument: " + arg_name + " was not provided\n");
        }
        return std::string(arg);
    }

    std::optional<std::string> get_optional(char** begin, char** end, const std::string& inp) {
        char* arg_pointer = get_option(begin, end, inp);
        std::optional<std::string> res;
        if (arg_pointer == nullptr) {
            return res;
        }
        std::string arg(arg_pointer);
        if (arg == "") {
            std::cout << "key to option " << inp << " was setted but argument was not provided";
            return res;
        }
        return std::optional(arg);
    }

    bool check_if_present(char** begin, char** end, const std::string& inp) {
        if (get_option(begin, end, inp) == nullptr) {
            return false;
        }
        return true;
    }
}

namespace matrix_multiply {
    enum class mem_order { base, transposed, partioned, partioned_transposed };

    template <typename T>
    struct matrix {
        matrix(size_t rows, size_t columns, mem_order type = mem_order::base, size_t part_size = 1)
            : type(type), part_size(part_size), row_length(columns), data(rows* columns, 0) {}

        matrix(
            std::fstream& inp,
            size_t rows,
            size_t columns, 
            mem_order type = mem_order::base,
            size_t r_pad = 0,
            size_t c_pad = 0,
            size_t part_size = 1) :
            type(type), part_size(part_size), row_length(columns + c_pad), data((rows + r_pad)* (columns + c_pad), 0) {
            for (size_t i = 0; i < rows * columns; i++) {
                size_t row = i / (columns_num() - c_pad);
                size_t column = i % (columns_num() - c_pad);
                if (inp.peek() == EOF) {
                    std::cout << "input has not sufficient number of elements\n";
                    inp.close();
                    exit(1);
                }
                inp >> data[calc_pos(row, column)];
            }
        }

        size_t size() const {
            return data.size();
        }

        size_t columns_num() const {
            return row_length;
        }

        size_t rows_num() const {
            return data.size() / row_length;
        }

        T get(size_t row, size_t column) const {
            return data[calc_pos(row, column)];
        }

        void set(size_t row, size_t column, T new_dec) {
            data[calc_pos(row, column)] = new_dec;
        }

        T* raw_begin() {
            return &data[0];
        }

        void print() {
            for (size_t i = 0; i < rows_num(); i++) {
                for (size_t j = 0; j < columns_num(); j++) {
                    std::cout << get(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }

    private:
        size_t calc_pos(size_t row, size_t col) const {
            if (type == mem_order::transposed) {
                return col * rows_num() + row;
            }
            else if (type == mem_order::partioned) {
                size_t block_col = col / part_size;
                size_t block_row = row / part_size;
                size_t row_in_block = row % part_size;
                size_t col_in_block = col % part_size;
                return (block_row * (row_length / part_size) + block_col) * part_size * part_size + part_size * row_in_block + col_in_block;
            }
            else if (type == mem_order::partioned_transposed) {
                size_t block_col = col / part_size;
                size_t block_row = row / part_size;
                size_t row_in_block = row % part_size;
                size_t col_in_block = col % part_size;
                return (block_col * (rows_num() / part_size) + block_row) * part_size * part_size + part_size * col_in_block + row_in_block;
            }
            return columns_num() * row + col;
        }
        mem_order type;
        size_t part_size = 1;
        size_t row_length;
        std::vector<T> data;
    };

    template <typename T>
    void print_part_to_file(const matrix<T>& in, std::fstream& file, int32_t row, int32_t col) {
        file << col << " " << row << std::endl;
        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                file << in.get(i, j) << " ";
            }
            file << std::endl;
        }
    }

    template <typename T>
    matrix<T> operator*(const matrix<T>& a, const matrix<T>& b) {
        matrix<T> res(a.rows_num(), b.columns_num());
        std::vector<std::thread> threads(b.columns_num());
        for (size_t i = 0; i < b.columns_num(); i++) {
            threads[i] = std::thread([i, &res, &a, &b]() {
                size_t k = a.columns_num();
                size_t num = a.rows_num();
                #pragma omp simd
                for (size_t j = 0; j < num; j++) {
                    T sum = 0;
                    for (size_t l = 0; l < k; l++) {
                        sum += a.get(j, l) * b.get(l, i);
                    }
                    res.set(j, i, sum);
                }
            });
        }
        for (size_t i = 0; i < threads.size(); i++) {
            threads[i].join();
        }
        return res;
    }
}

namespace openCL_operation {
    cl_int CL_err = CL_SUCCESS;

    template <typename T>
    concept callable = requires(T a) {
        a();
    };

    template <callable T>
    void check_err(T release) {
        if (CL_err != CL_SUCCESS) {
            std::cout << "OpenCL error code" << CL_err << std::endl;
            release();
            exit(1);
        }
    }

    void check_err() {
        check_err([]() {});
    }

    template <callable T>
    bool check_ret_code(cl_int errcode, T& release) {
        if (errcode != 0) {
            std::cout << "Error occured " << errcode << std::endl;
            release();
            exit(1);
        }
        return false;
    }

    bool device_comp(std::pair<cl_device_id, cl_device_type> a, std::pair<cl_device_id, cl_device_type> b) {
        if (a.second == CL_DEVICE_TYPE_ACCELERATOR) return false;
        if (b.second == CL_DEVICE_TYPE_ACCELERATOR) return true;
        if (b.second == CL_DEVICE_TYPE_GPU) {
            cl_bool a_mem, b_mem;
            clGetDeviceInfo(a.first, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(a_mem), &a_mem, NULL);
            clGetDeviceInfo(b.first, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(b_mem), &b_mem, NULL);
            return a.second == CL_DEVICE_TYPE_GPU && (b_mem && !a_mem);
        }
        return true;
    }

    cl_device_type  resolve_tag(std::optional<std::string> inp) {
        if (inp && (inp.value() == "gpu" || inp.value() == "igpu" || inp.value() == "dgpu")) {
            return CL_DEVICE_TYPE_GPU;
        }
        if (inp && inp.value() == "cpu") {
            return CL_DEVICE_TYPE_CPU;
        }
        return CL_DEVICE_TYPE_ALL;
    }


    void print_device_info(cl_device_id device) {
        size_t valueSize;
        cl_platform_id platform;
        clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
        char* dev_name = (char*)malloc(valueSize);
        clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, dev_name, NULL);
        clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, NULL);

        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &valueSize);
        char* plat_name = (char*)malloc(valueSize);
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, valueSize, plat_name, NULL);

        printf("Device: %s\tPlatform : %s\n", dev_name, plat_name);
        free(dev_name);
        free(plat_name);
    }

    cl_device_id resolve_device(std::optional<std::string> inp, const int32_t index) {
        cl_uint platforms_num = 0;
        CL_err = clGetPlatformIDs(NULL, NULL, &platforms_num);
        check_err();
        std::vector <cl_platform_id> platforms = std::vector<cl_platform_id>(platforms_num);
        CL_err = clGetPlatformIDs(platforms_num, &platforms[0], NULL);
        check_err();
        std::vector<std::pair<cl_device_id, cl_device_type>> devices;
        for (auto platform : platforms) {
            cl_uint devices_num = 0;
            CL_err = clGetDeviceIDs(platform, resolve_tag(inp), NULL, NULL, &devices_num);
            check_err();
            std::vector<cl_device_id> temp_devices{ devices_num };
            CL_err = clGetDeviceIDs(platform, resolve_tag(inp), devices_num, &temp_devices[0], NULL);
            check_err();
            cl_device_type type;
            for (auto device : temp_devices) {
                cl_bool test = false;
                if (inp == "dgpu" || inp == "igpu") {
                    clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &test, NULL);
                }
                clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
                if ((inp == "dgpu" && test) || (inp == "igpu" && !test));
                else devices.push_back({ device, type });
            }
        }
        std::sort(devices.begin(), devices.end(), device_comp);
        return devices[index >= devices.size() ? 0 : index].first;
    }

    std::pair<cl_kernel, cl_program> load_and_build(const std::string name, const cl_context& context, const cl_device_id& device) {
        std::string temp = name + ".cl";
        const char* src = readFile(&temp[0]);
        if (src == NULL) {
            std::cout << "Error occured during reading code source file\n";
            exit(1);
        }
        size_t length = strlen(src);
        cl_program matrix_mul = clCreateProgramWithSource(context, 1, (const char**)&src, &length, NULL);
        std::string args = "-D WIDTH=" + std::to_string(WIDTH);
        args += " -D TILE=" + std::to_string(TILE);
        args += " -D HEIGHT=" + std::to_string(HEIGHT);
        if (clBuildProgram(matrix_mul, 1, &device, &args[0], NULL, NULL) != CL_SUCCESS) {
            printf_s("Programm build failed \n");
            size_t val_size;
            clGetProgramBuildInfo(matrix_mul, device, CL_PROGRAM_BUILD_LOG, NULL, NULL, &val_size);
            char* debug_info = (char*)malloc(val_size);
            clGetProgramBuildInfo(matrix_mul, device, CL_PROGRAM_BUILD_LOG, val_size, debug_info, NULL);
            std::cout << debug_info;
            free(debug_info);
            clReleaseContext(context);
            clReleaseProgram(matrix_mul);
            exit(1);
        }
        cl_kernel matrix_kernel = clCreateKernel(matrix_mul, "mul", &CL_err);
        check_err();
        return { matrix_kernel,matrix_mul };
    }

    void release(cl_mem* args[], size_t size, cl_command_queue command_queue, cl_program pr, cl_kernel ker, cl_context cont) {
        for (size_t i = 0; i < size; i++) {
            clReleaseMemObject(*args[i]);
        }
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(pr);
        clReleaseKernel(ker);
        clReleaseContext(cont);
    }

    cl_ulong getEventTime(cl_event& event) {
        cl_ulong starttime, endtime;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL);
        return endtime - starttime;
    }

    template <typename T>
    void execute_mul(
        cl_device_id device,
        int32_t realization,
        std::string program_path,
        matrix_multiply::matrix<T> &a,
        matrix_multiply::matrix<T> &b,
        matrix_multiply::matrix<T> &res
    ) {
        cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &CL_err);
        check_err([&]() {clReleaseContext(context); });
        std::pair<cl_kernel, cl_program> temp = load_and_build(program_path, context, device);
        cl_kernel matrix_kernel = temp.first;
        cl_command_queue command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &CL_err);
        check_err([&]() {clReleaseContext(context); clReleaseCommandQueue(command_queue); });

        int32_t n = b.columns_num();
        int32_t k = a.columns_num();
        cl_mem arg0 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * a.size(), NULL, NULL);
        cl_mem arg1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * b.size(), NULL, NULL);
        cl_mem ret = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T) * res.size(), NULL, NULL);

        cl_event wr0, wr1;
        cl_mem* to_release[] = { &arg1, &arg0, &ret };
        auto release_op = std::bind(release, to_release, 3, command_queue, temp.second, matrix_kernel, context);
        check_ret_code(clSetKernelArg(matrix_kernel, 0, sizeof(cl_mem), &arg0), release_op);
        check_ret_code(clSetKernelArg(matrix_kernel, 1, sizeof(cl_mem), &arg1), release_op);
        check_ret_code(clSetKernelArg(matrix_kernel, 2, sizeof(cl_mem), &ret), release_op);
        check_ret_code(clSetKernelArg(matrix_kernel, 3, sizeof(cl_uint), &n), release_op);
        check_ret_code(clSetKernelArg(matrix_kernel, 4, sizeof(cl_uint), &k), release_op);
        check_ret_code(clEnqueueWriteBuffer(command_queue, arg0, 0, 0, a.size() * sizeof(T), a.raw_begin(), 0, NULL, &wr0), release_op);
        check_ret_code(clEnqueueWriteBuffer(command_queue, arg1, 0, 0, b.size() * sizeof(T), b.raw_begin(), 0, NULL, &wr1), release_op);
        
        size_t gl_work[] = { b.columns_num(), a.rows_num() };
        size_t lc_work[] = { TILE, TILE };
        cl_event event;
        switch (realization)
        {
        case 1:
            clEnqueueNDRangeKernel(command_queue, matrix_kernel, 2, NULL, gl_work, NULL, 0, NULL, &event);
            break;
        case 2:
            clEnqueueNDRangeKernel(command_queue, matrix_kernel, 2, NULL, gl_work, lc_work, 0, NULL, &event);
            break;
        default:
            gl_work[0] /= WIDTH;
            gl_work[1] /= HEIGHT;
            lc_work[0] /= WIDTH;
            lc_work[1] /= HEIGHT;
            clEnqueueNDRangeKernel(command_queue, matrix_kernel, 2, NULL, gl_work, lc_work, 0, NULL, &event);
            break;
        }

        cl_event read;
        check_ret_code(clEnqueueReadBuffer(command_queue, ret, 1, 0, res.size() * sizeof(T), res.raw_begin(), 0, NULL, &read), release_op);
        cl_event times[] = { wr0, wr1, event, read };
        clWaitForEvents(4, &times[0]);
        double kernel_time = getEventTime(event);
        double write_time = getEventTime(wr0) + getEventTime(wr1);
        clFinish(command_queue);
        double all_arg_time = write_time + getEventTime(read);
        release_op();
        printf("Time: %g\t%g\n", kernel_time / 1000000.0, (kernel_time + all_arg_time) / 1000000.0);
        if (realization > 1) {
            printf("LOCAL_WORK_SIZE [%i, %i]\n", lc_work[0], lc_work[1]);
        }
        if (realization == 3) {
            printf("WI_WORK %i\n", WIDTH * HEIGHT);
        }
    }
}

int main(int all_argc, char* all_argv[])
{
    char** argv = all_argv++;
    int argc = all_argc--;
    if (cmd_options::check_if_present(argv, argv + argc, "--help")) {
        cmd_options::print_help_message();
        return 0;
    }
    std::string inp_file_name,  out_file_name;
    int realization;
    try {
        inp_file_name  = cmd_options::get_obligatory(argv, argv + argc, "--input");
        out_file_name = cmd_options::get_obligatory(argv, argv + argc, "--output");
        realization = std::stoi(cmd_options::get_obligatory(argv, argv + argc, "--realization"));
    }
    catch (const std::invalid_argument& e) {
        std::cout << e.what();
        return 1;
    }
    std::optional<std::string> device_type = cmd_options::get_optional(argv, argv + argc, "--device-type");
    std::optional<std::string> device_index_s = cmd_options::get_optional(argv, argv + argc, "--device-index");
    if (realization == 2) TILE = 16;
    int32_t device_index = std::stoi(device_index_s.value_or("0"));
    if (realization > 3 || realization < 0) {
        std::cout << "Unknown realization" << std::endl;
        return 1;
    }
    std::fstream inp(inp_file_name, std::ios_base::in);
    if (check_errno("input file does not exist or can not be opened\n")) return 0;
    int32_t n, k, m;
    inp >> n >> k >> m;
    if (check_errno("can not get matrix dimensions from input file\n")) {
        inp.close();
        return 1;
    }
    matrix_multiply::matrix<cl_float> a(inp, m, k,
        matrix_multiply::mem_order::base,
        realization <= 1 ? 0 : (TILE - k % TILE) % TILE, realization <= 1 ? 0 : (TILE- n % TILE) % TILE, TILE);
    matrix_multiply::matrix<cl_float> b(inp, k, n,
        realization == 0 ? matrix_multiply::mem_order::transposed : matrix_multiply::mem_order::base,
        realization <= 1 ? 0 : (TILE - k % TILE) % TILE, realization <= 1 ? 0 : (TILE - n % TILE) % TILE, TILE);
    matrix_multiply::matrix<cl_float> res(
        a.rows_num(),
        b.columns_num(),
        matrix_multiply::mem_order::base, TILE);
    inp.close();
    if (realization == 0) {
        double t_start = std::clock();
        res = a * b;
        double t_end = std::clock();
        printf("Time: %g", t_end - t_start);
    }
    else {
        cl_device_id device = openCL_operation::resolve_device(device_type, device_index);
        openCL_operation::print_device_info(device);
        std::string program_path = ((realization == 1) ?
            "cl_codes/non_mem_mul" :
            ((realization == 2) ? "cl_codes/local_mem_mul" : "cl_codes/vectorized_mul"));
        openCL_operation::execute_mul(device, realization, program_path, a, b, res);
    }
    std::fstream out(out_file_name, std::ios_base::out);
    if (check_errno("output file does not exist or cannot be opened\n")) return 0;
    out << std::fixed << std::setprecision(6);
    matrix_multiply::print_part_to_file(res, out, m, n);
    out.close();
}
