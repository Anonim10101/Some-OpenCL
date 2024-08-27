#include <optional>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <iomanip>
#include <functional>
#include <CL/cl.h>
#pragma warning(disable:4996)
#define GR_SIZE 256
#define PER_THREAD 16
#define PER_THREAD_INC 16

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
            << "--output name - имя выходного файла" << std::endl
            << "--device-type { dgpu | igpu | gpu | cpu | all } – указание типа девайса (дискретная видеокарта, интегрированная видеокарта, видеокарты, процессор, все)" << std::endl
            << "--device-index – указание порядкового номера девайса" << std::endl;
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

    std::pair<cl_kernel, cl_program> load_and_build(const std::string name, const cl_context& context, const cl_device_id& device, std::string args) {
        std::string temp = name + ".cl";
        const char* src = readFile(&temp[0]);
        if (src == NULL) {
            std::cout << "Error occured during reading code source file\n";
            exit(1);
        }
        size_t length = strlen(src);
        cl_program prog = clCreateProgramWithSource(context, 1, (const char**)&src, &length, NULL);
        if (clBuildProgram(prog, 1, &device, &args[0], NULL, NULL) != CL_SUCCESS) {
            printf_s("Programm build failed \n");
            size_t val_size;
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, NULL, NULL, &val_size);
            char* debug_info = (char*)malloc(val_size);
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, val_size, debug_info, NULL);
            std::cout << debug_info;
            free(debug_info);
            clReleaseContext(context);
            clReleaseProgram(prog);
            exit(1);
        }
        cl_kernel matrix_kernel = clCreateKernel(prog, "kmain", &CL_err);
        check_err();
        return { matrix_kernel,prog };
    }

    void release(
        cl_mem* args,
        size_t size,
        cl_command_queue command_queue,
        cl_program* pr[], size_t pr_num,
        cl_kernel* ker[], size_t ker_num,
        cl_context cont) {
        for (size_t i = 0; i < size; i++) {
            clReleaseMemObject(*(args + i));
        }
        clReleaseCommandQueue(command_queue);
        for (size_t i = 0; i < pr_num; i++) {
            clReleaseProgram(*pr[i]);
        }
        for (size_t i = 0; i < ker_num; i++) {
            clReleaseKernel(*ker[i]);
        }
        clReleaseContext(cont);
    }

    cl_ulong getEventTime(cl_event& event) {
        cl_ulong starttime, endtime;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL);
        return endtime - starttime;
    }

    std::vector<size_t> calc_sizes(size_t size) {
        std::vector<size_t> res;
        while (size > GR_SIZE * PER_THREAD) {
            size = (size + (PER_THREAD * GR_SIZE - size % (PER_THREAD * GR_SIZE)) % (PER_THREAD * GR_SIZE));
            res.push_back(size);
            size /= (PER_THREAD * GR_SIZE);
        }
        res.push_back(PER_THREAD * GR_SIZE);
        return res;
    }

    template <typename T>
    void execute_sum(
        cl_device_id device,
        std::string pref_program_path,
        std::string add_program_path,
        std::vector<T>& inp,
        std::vector<T>& res
    ) {
        cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &CL_err);
        check_err([&]() {clReleaseContext(context); });
        std::string args = "-D GR_SIZE=" + std::to_string(GR_SIZE);
        args += " -D PER_THREAD=" + std::to_string(PER_THREAD);
        std::pair<cl_kernel, cl_program> pref_ker = load_and_build(pref_program_path, context, device, args);
        args += " -D PER_THREAD_INC=" + std::to_string(PER_THREAD_INC);
        std::pair<cl_kernel, cl_program> add_ker = load_and_build(add_program_path, context, device, args);
        cl_kernel sum_kernel = pref_ker.first;
        cl_kernel add_kernel = add_ker.first;
        cl_command_queue command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &CL_err);
        check_err([&]() {clReleaseContext(context); clReleaseCommandQueue(command_queue); });

        std::vector<size_t> buf_sizes = calc_sizes(inp.size());
        std::vector<cl_mem> buffers(buf_sizes.size());
        for (size_t i = 0; i < buf_sizes.size(); i++) {
           buffers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T) * buf_sizes[i], NULL, NULL);
        }
        buffers.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T) * res.size(), NULL, NULL));
        buffers.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T), NULL, NULL)); 
        size_t ret = buffers.size() - 2;
        size_t sum_check = buffers.size() - 1;

        cl_event wr;
        std::vector<cl_event> events(2 * buf_sizes.size() - 1);
        cl_program* programs[] = { &pref_ker.second, &add_ker.second };
        cl_kernel* kernels[] = { &sum_kernel, &add_kernel };
        auto release_op = std::bind(
            release,
            &buffers[0], buffers.size(), 
            command_queue,
            programs, 2,
            kernels, 2,
            context
        );
        for (size_t i = 0; i < buf_sizes.size(); i++) {
            size_t gl_work[] = { (size_t)std::max(GR_SIZE, (int)(buf_sizes[i] / PER_THREAD)) };
            size_t lc_work[] = { GR_SIZE };
            size_t n = i == 0 ? inp.size() : (buf_sizes[i - 1] / (PER_THREAD * GR_SIZE));
            check_ret_code(clSetKernelArg(sum_kernel, 0, sizeof(cl_mem), &buffers[i]), release_op);
            check_ret_code(clSetKernelArg(sum_kernel, 1, sizeof(cl_uint), &n), release_op);
            check_ret_code(clSetKernelArg(sum_kernel, 2, sizeof(cl_mem), i == 0 ? &buffers[ret] : &buffers[i]), release_op);
            check_ret_code(clSetKernelArg(sum_kernel, 3, sizeof(cl_mem), i + 1 == buf_sizes.size() ? &buffers[sum_check] : &buffers[i + 1]), release_op);
            if (i == 0) {
                check_ret_code(clEnqueueWriteBuffer(command_queue, buffers[0], 0, 0, inp.size() * sizeof(T), &inp[0], 0, NULL, &wr), release_op);
            }
            clEnqueueNDRangeKernel(command_queue, sum_kernel, 1, NULL, gl_work, lc_work, 0, NULL, &events[i]);
        }
        for (int i = buf_sizes.size() - 1; i > 0; i--) {
            size_t gl_work[] = { 
                (size_t)std::max(GR_SIZE * PER_THREAD / PER_THREAD_INC, (int)(buf_sizes[i - 1] / PER_THREAD_INC))
            };
            size_t lc_work[] = { PER_THREAD * GR_SIZE / PER_THREAD_INC};
            check_ret_code(clSetKernelArg(add_kernel, 0, sizeof(cl_mem), i == 1? &buffers[ret] : &buffers[i - 1]), release_op);
            check_ret_code(clSetKernelArg(add_kernel, 1, sizeof(cl_mem), &buffers[i]), release_op);
            clEnqueueNDRangeKernel(command_queue, add_kernel, 1, NULL, gl_work, lc_work, 0, NULL, &events[buf_sizes.size() - 1 + i]);
        }

        cl_event read;
        check_ret_code(clEnqueueReadBuffer(command_queue, buffers[ret], 1, 0, res.size() * sizeof(T), &res[0], 0, NULL, &read), release_op);
        double kernel_time_sum = 0, kernel_time_inc = 0;
        for (size_t i = 0; i < events.size(); i++) {
            if (i < buf_sizes.size()) {
                kernel_time_sum += getEventTime(events[i]);
            }
            else {
                kernel_time_inc += getEventTime(events[i]);
            }
        }
        double write_time = getEventTime(wr);
        clFinish(command_queue);
        double all_arg_time = write_time + getEventTime(read);
        release_op();
        printf("Time: %g\t%g\n",
            (kernel_time_sum + kernel_time_inc) / 1000000.0,
            (kernel_time_inc + kernel_time_sum + all_arg_time) / 1000000.0
        );
        printf("LOCAL_WORK_SIZE [%i, %i]\n", GR_SIZE, 1);
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
    std::string inp_file_name, out_file_name;
    try {
        inp_file_name = cmd_options::get_obligatory(argv, argv + argc, "--input");
        out_file_name = cmd_options::get_obligatory(argv, argv + argc, "--output");
    }
    catch (const std::invalid_argument& e) {
        std::cout << e.what();
        return 1;
    }
    std::optional<std::string> device_type = cmd_options::get_optional(argv, argv + argc, "--device-type");
    std::optional<std::string> device_index_s = cmd_options::get_optional(argv, argv + argc, "--device-index");
    int32_t device_index = std::stoi(device_index_s.value_or("0"));

    int n;
    std::fstream inp(inp_file_name, std::ios_base::in);
    if (check_errno("input file does not exist or cannot be opened\n")) return 0;
    inp >> n;
    std::vector<float> data(n + ((PER_THREAD * GR_SIZE) - n % (PER_THREAD * GR_SIZE)) % (PER_THREAD * GR_SIZE), 0);
    std::vector<float> res(n + ((PER_THREAD * GR_SIZE) - n % (PER_THREAD * GR_SIZE)) % (PER_THREAD * GR_SIZE), 0);
    for (size_t i = 0; i < n; i++) {
        inp >> data[i];
    }

    cl_device_id device = openCL_operation::resolve_device(device_type, device_index);
    openCL_operation::print_device_info(device);

    openCL_operation::execute_sum(
        device,
        "cl_codes/partial_pref_sum_opt",
        "cl_codes/incrementor_base",
        data,
        res
    );
    std::fstream out(out_file_name, std::ios_base::out);
    if (check_errno("output file does not exist or cannot be opened\n")) return 0;
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < n; i++) {
        out << res[i] << " ";
    }
    out.close();
}
