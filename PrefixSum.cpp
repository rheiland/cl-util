#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <sys/time.h>
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

#define LW 256
#define ELEMENTS_PER_THREAD 2
#define ELEMENTS_PER_WORKGROUP (ELEMENTS_PER_THREAD*LW)

#define MAX_SOURCE_SIZE (0x100000)

#define ITERATIONS 1000

#define PROFILE_GPU TRUE

int pow2gt(int x) {
	int i = 1;

	while (i < x)
		i <<= 1;

	return i;
}

class PrefixSum {
	int capacity;
	cl_context context;
	cl_command_queue queue;
	cl_mem *d_parts;
	size_t local;

	cl_kernel kern_scan_pad_to_pow2;
	cl_kernel kern_scan_subarrays;
	cl_kernel kern_scan_inc_subarrays;

	public:
		double elapsed;

		PrefixSum(cl_context context, cl_device_id device_id, int n_devices, int capacity);

		cl_mem factory(int len);
		cl_mem factory();

		int scan(cl_mem d_array, cl_mem d_total);
		int scan(cl_mem d_array, cl_mem d_total, int len);
};

int PrefixSum::scan(cl_mem d_array, cl_mem d_total) {
	PrefixSum::scan(d_array, d_total, capacity);
}

int PrefixSum::scan(cl_mem d_array, cl_mem d_total, int len) {
	cl_event event;
#if PROFILE_GPU == TRUE
	cl_ulong time_start, time_end;
#endif

	int i;
	int k = (len + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP;
	size_t global = k*LW;
	cl_mem d_part;
	cl_int ret;

	if (k == 1) {
		ret  = clSetKernelArg(kern_scan_pad_to_pow2, 0, sizeof(cl_mem), &d_array);
		ret  = clSetKernelArg(kern_scan_pad_to_pow2, 1, ELEMENTS_PER_WORKGROUP*sizeof(int), NULL);
		ret |= clSetKernelArg(kern_scan_pad_to_pow2, 2, sizeof(int), &len);
		ret  = clSetKernelArg(kern_scan_pad_to_pow2, 3, sizeof(cl_mem), &d_total);
		if (ret != CL_SUCCESS)
		{
			printf("Error: Failed to set kernel arguments! %d\n", ret);
			exit(1);
		}

#if PROFILE_GPU == TRUE
		ret = clEnqueueNDRangeKernel(queue, kern_scan_pad_to_pow2, 1, NULL, &global, &local, 0, NULL, &event);
		clWaitForEvents(1, &event);
		if (ret) {
			printf("Error: Failed to execute kernel!\n");
			return EXIT_FAILURE;
		}

		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		elapsed += time_end - time_start;
#endif
	}
	else {
		d_part = d_parts[(int) (log(len)/log(ELEMENTS_PER_WORKGROUP)) - 1];

		ret  = clSetKernelArg(kern_scan_subarrays, 0, sizeof(cl_mem), &d_array);
		ret |= clSetKernelArg(kern_scan_subarrays, 1, ELEMENTS_PER_WORKGROUP*sizeof(int), NULL);
		ret |= clSetKernelArg(kern_scan_subarrays, 2, sizeof(cl_mem), &d_part);
		ret |= clSetKernelArg(kern_scan_subarrays, 3, sizeof(int), &len);
		if (ret != CL_SUCCESS)
		{
			printf("Error: Failed to set kernel arguments! %d\n", ret);
			exit(1);
		}
		
		ret = clEnqueueNDRangeKernel(queue, kern_scan_subarrays, 1, NULL, &global, &local, 0, NULL, &event);
		clWaitForEvents(1, &event);
		if (ret) {
			printf("Error: Failed to execute kernel!\n");
			return EXIT_FAILURE;
		}

#if PROFILE_GPU == TRUE
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		elapsed += time_end - time_start;
#endif

		ret  = clSetKernelArg(kern_scan_inc_subarrays, 0, sizeof(cl_mem), &d_array);
		ret |= clSetKernelArg(kern_scan_inc_subarrays, 1, ELEMENTS_PER_WORKGROUP*sizeof(int), NULL);
		ret |= clSetKernelArg(kern_scan_inc_subarrays, 2, sizeof(cl_mem), &d_part);
		ret |= clSetKernelArg(kern_scan_inc_subarrays, 3, sizeof(int), &len);
		if (ret != CL_SUCCESS)
		{
			printf("Error: Failed to set kernel arguments! %d\n", ret);
			exit(1);
		}

		ret = clEnqueueNDRangeKernel(queue, kern_scan_inc_subarrays, 1, NULL, &global, &local, 0, NULL, &event);
		clWaitForEvents(1, &event);
		if (ret) {
			printf("Error: Failed to execute kernel!\n");
			return EXIT_FAILURE;
		}	

#if PROFILE_GPU == TRUE
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		elapsed += time_end - time_start;
#endif
	}

	clFinish(queue);
}

cl_mem PrefixSum::factory() {
	return factory(capacity);
}

cl_mem PrefixSum::factory(int len) {
	cl_int ret;

	len = pow2gt(len);

	cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE, len*sizeof(int), NULL, &ret);

	if (ret != CL_SUCCESS)
		cout << "error" << endl;

	return buf;
}

PrefixSum::PrefixSum(cl_context p_context, cl_device_id device_id, int n_devices, int p_capacity) {
	capacity = p_capacity;
	context = p_context;

	FILE *fp;
	char fileName[] = "./prefixsum.cl";
	char *source_str;
	size_t source_size;

	cl_program program = NULL;
	cl_int ret;
	 
	/* Load the source code containing the kernel*/
	fp = fopen(fileName, "r");
	if (!fp) {
	fprintf(stderr, "Failed to load kernel.\n");
	exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	/* Create Command Queue */
#if PROFILE_GPU == TRUE
	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
#else
	queue = clCreateCommandQueue(context, device_id, 0, &ret);
#endif

	/* Create Kernel Program from the source */
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	 
	/* Build Kernel Program */
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	 
	/* Create OpenCL Kernel */
	kern_scan_pad_to_pow2 = clCreateKernel(program, "scan_pad_to_pow2", &ret);
	kern_scan_subarrays = clCreateKernel(program, "scan_subarrays", &ret);
	kern_scan_inc_subarrays = clCreateKernel(program, "scan_inc_subarrays", &ret);

	local = LW;
	elapsed = 0;

	int len = capacity/ELEMENTS_PER_WORKGROUP;
	int n = (int) ceil(log((float) capacity)/log((float) ELEMENTS_PER_WORKGROUP));

	d_parts = (cl_mem*) malloc(n*sizeof(cl_mem));

	for (int i=0; i<n; i++) {
		d_parts[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, len*sizeof(int), NULL, NULL);

		len /= ELEMENTS_PER_WORKGROUP;

		i++;
	}
}

typedef struct dim2d {
	int w;
	int h;
} dim2d_t;

vector<dim2d_t> tile_dims;
vector<dim2d_t> img_dims;

dim2d_t tile_dims0 = {16, 16};
dim2d_t tile_dims1 = {32, 32};

dim2d_t img_dims0 = {400, 300};
dim2d_t img_dims1 = {640, 480};
dim2d_t img_dims2 = {800, 600};
dim2d_t img_dims3 = {1024, 768};
dim2d_t img_dims4 = {1600, 1200};
dim2d_t img_dims5 = {1920, 1080};
dim2d_t img_dims6 = {2560, 1440};
dim2d_t img_dims7 = {2048, 2048};
dim2d_t img_dims8 = {3600, 2400};
dim2d_t img_dims9 = {4096, 4096};
dim2d_t img_dims10 = {8192, 8192};


int main() {
	cl_context context = NULL;
	cl_device_id device_id = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;

	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	cl_command_queue queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
	 
	tile_dims.push_back(tile_dims0);
	tile_dims.push_back(tile_dims1);

	img_dims.push_back(img_dims0);
	img_dims.push_back(img_dims1);
	img_dims.push_back(img_dims2);
	img_dims.push_back(img_dims3);
	img_dims.push_back(img_dims4);
	img_dims.push_back(img_dims5);
	img_dims.push_back(img_dims6);
	img_dims.push_back(img_dims7);
	img_dims.push_back(img_dims8);
	img_dims.push_back(img_dims9);
	img_dims.push_back(img_dims10);

	cout << "img dim,mp,tile dim,num tiles,total ms,kernels ms" << endl;

	for(vector<dim2d_t>::iterator it = img_dims.begin(); it != img_dims.end(); ++it) {
		int img_w = (*it).w;
		int img_h = (*it).h;

		float mp = (float) (img_w*img_h)/(1024*1024);

		for(vector<dim2d_t>::iterator it = tile_dims.begin(); it != tile_dims.end(); ++it) {
			int tile_w = (*it).w;
			int tile_h = (*it).h;

			int n_tiles = (img_w/tile_w)*(img_h/tile_h);

			PrefixSum ps = PrefixSum(context, device_id, 1, n_tiles);

			int h_total = 0;
			cl_mem d_total = clCreateBuffer(context, CL_MEM_READ_WRITE, 1*sizeof(int), NULL, &ret);
			cl_mem d_list = ps.factory();

			int data[n_tiles];
			for (int i=0; i<n_tiles; i++) {
				data[i] = 1; 
			}
			ret = clEnqueueWriteBuffer(queue, d_list, CL_TRUE, 0, n_tiles*sizeof(int), data, 0, NULL, NULL);

			struct timeval t, t2;
			double elapsed;

			gettimeofday(&t, NULL);
			for (int i=0; i<ITERATIONS; i++) {
				ps.scan(d_list, d_total, n_tiles);
			}
			gettimeofday(&t2, NULL);

			double seconds = t2.tv_sec - t.tv_sec;
			double microseconds = t2.tv_usec - t.tv_usec;
			elapsed = (seconds * 1.0e6 + microseconds);

/*
			cout << mp << "mp, " << n_tiles << " (" << tile_w << "x" << tile_h << ") tiles" << endl;
			cout << "total: " << (1e-3 * elapsed) / ITERATIONS << "ms" << endl;
#if PROFILE_GPU == TRUE
			cout << "kernels: " << (1e-6 * ps.elapsed) / ITERATIONS << "ms" << endl;
#endif
			cout << endl;
*/
			cout << "(" << img_w << "x" << img_h << "),";
			cout << mp << ",";
			cout << "(" << tile_w << "x" << tile_h << "),";
			cout << n_tiles << ",";
			cout << (1e-3 * elapsed) / ITERATIONS << ",";
			cout << (1e-6 * ps.elapsed) / ITERATIONS;
			cout << endl;

			ret = clEnqueueReadBuffer(queue, d_total, CL_TRUE, 0, 1*sizeof(int), &h_total, 0, NULL, NULL);
			ret = clEnqueueReadBuffer(queue, d_list, CL_TRUE, 0, n_tiles*sizeof(int), &data, 0, NULL, NULL);
		}
	}

}
