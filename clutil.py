import pyopencl as cl

def pow2lt(x):
	i = 1;
	while(i <= x>>1):
		i = i<<1;

	return i

def pow2gt(x):
	i = 1;
	while(i < x):
		i = i<<1;

	return i

def isPow2(x):
	return (x != 0) and ((x & (x - 1)) == 0)

def ceil_divi(dividend, divisor):
	return (dividend + divisor - 1) / divisor

def print_deviceAttr(device):
	attrs = [
			'GLOBAL_MEM_SIZE',
			'DRIVER_VERSION',
			'GLOBAL_MEM_CACHE_SIZE',
			'GLOBAL_MEM_CACHELINE_SIZE',
			'GLOBAL_MEM_SIZE',
			'GLOBAL_MEM_CACHE_TYPE',
			'IMAGE_SUPPORT',
			'IMAGE2D_MAX_HEIGHT',
			'IMAGE2D_MAX_WIDTH',
			'LOCAL_MEM_SIZE',
			'LOCAL_MEM_TYPE',
			'MAX_COMPUTE_UNITS',
			'MAX_WORK_GROUP_SIZE',
			'MAX_WORK_ITEM_SIZES',
			'MIN_DATA_TYPE_ALIGN_SIZE',
			'PREFERRED_VECTOR_WIDTH_FLOAT',
			'PREFERRED_VECTOR_WIDTH_CHAR',
			'PREFERRED_VECTOR_WIDTH_DOUBLE',
			'PREFERRED_VECTOR_WIDTH_HALF',
			'MAX_CONSTANT_BUFFER_SIZE',
			'MAX_MEM_ALLOC_SIZE',
			]

	for attr in attrs:
		tmp = getattr(cl.device_info, attr)
		print '\t' + attr +':\t' + str(device.get_info(tmp))

def kernelInfo(kernel, device):
	attrs = [
			'LOCAL_MEM_SIZE',
			'WORK_GROUP_SIZE',
			'PREFERRED_WORK_GROUP_SIZE_MULTIPLE',
			'COMPILE_WORK_GROUP_SIZE',
			]

	for attr in attrs:
		tmp = getattr(cl.kernel_work_group_info, attr)
		print attr +':\t' + str(kernel.get_work_group_info(tmp, device))


def platformInfo():
	platforms = cl.get_platforms();
	if len(platforms) == 0:
		print "Failed to find any OpenCL platforms."
		return None

	for platform in platforms:
		print platform

		for device in platform.get_devices():
			print cl.deviceInfo(device)

#
#  Create an OpenCL program from the kernel source file
#
def createProgram(context, devices, options, fileName):
	kernelFile = open(fileName, 'r')
	kernelStr = kernelFile.read()
	
	# Load the program source
	program = cl.Program(context, kernelStr)
	
	# Build the program and check for errors   
	program.build(options, devices)
	
	return program

def localToGlobalWorkgroup(size, lWorkGroup):
	if len(size) != len(lWorkGroup):
		raise TypeError('dimensions so not match: {0}, {1}'.format(len(size), len(lWorkGroup)))

	return tuple([roundUp(l, d) for l, d in zip(lWorkGroup, size)])

def roundUp(size, multiple):
	if type(size) == int:
		if type(multiple) != int:
			raise TypeError('types do not match: {0}, {1}'.format(type(size), type(multiple)))

		r = size % multiple;

		if r == 0:
			return size
		else:
			return size + multiple - r;

	elif type(size) == tuple:
		if len(size) != len(multiple):
			raise TypeError('dimensions do not match: {0}, {1}'.format(len(size), len(multiple)))

		out = [0]*len(size)

		for i, (s, m) in enumerate(zip(size, multiple)):
			r = s % m;
			if r == 0:
				out[i] = s
			else:
				out[i] = s + m - r;

		return tuple(out)

def padArray2D(arr, shape, mode):
	from numpy import pad

	padding = [(0, shape[0]-arr.shape[0]), (0, shape[1]-arr.shape[1])]
	return pad(arr, padding, mode)