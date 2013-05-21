#define NDIM 3
#define MAX_NCOMPONENT 8
//width of matrix a = (2*NDIM + 1) and round to 8
#define AW 8
#define MIN_COVAR 0.001f
#define MIN_COVAR4 (float4) (MIN_COVAR, MIN_COVAR, MIN_COVAR, MIN_COVAR)

#define rgba2float4(c) (float4) (c & 0x000000FF, (c & 0x0000FF00) >> 8, (c & 0x00FF0000) >> 16, 0)
uint pseudocolorf(float val, float m, float M);
float4 HSVtoRGB(float4 HSV);

//#define M2_PI_POW_D_OVER_2 pow(M_PI+M_PI, float(NDIM)/2)
#define M2_PI_POW_D_OVER_2 pow((float)(M_PI+M_PI), 1.5f)

__kernel void get_samples(
    int n_points,
    __global int* points,
	sampler_t sampler,
	__read_only image2d_t src,
//    __global uint* src,
    __global uint* samples
) {
    int gi = get_global_id(0);

    if (gi > n_points-1)
        return;

    int point = points[gi];
    int px = point%get_image_width(src);
    int py = point/get_image_width(src);
	uint4 rgba = read_imageui(src, sampler, (int2) (px, py));
	samples[gi] = 0xFF000000 | rgba.x | (rgba.y << 8) | (rgba.z << 16);
//	samples[gi] = (float4) (rgba.x, rgba.y, rgba.z, rgba.w);
//	samples[gi] = (float4) (255, 0, 0, 0);

//    samples[gi] = src[point];
}

__kernel void score(
	__global uint* src,
	__constant float4* gA,
	__local float4* sA,
	int gw,
	int gh,
	int nComponents,
	__global float* out
	//__global uint* out,
	//__global uint* samples
) {
	int gs = gw*gh;
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*gw + gx;

	int wx = get_group_id(0); //workgroup id
	int wy = get_group_id(1);
	int ww = get_num_groups(0);
	int wh = get_num_groups(1);
	int ws = ww*wh;

	int lw = get_local_size(0);
	int lh = get_local_size(1);
	int ls = lw*lh; //local size
	int lx = get_local_id(0);
	int ly = get_local_id(1);
	int lxy = ly*lw + lx;

	if (lxy < nComponents*2) {
		sA[lxy] = gA[lxy];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (gxy > gs-1)
		return;

	uint s = src[gxy];

	float4 x = (float4) (1, s & 0x000000FF, (s & 0x0000FF00) >> 8, (s & 0x00FF0000) >> 16);
	//float4 x2 = pown(x, 2); <-- bug in OpenCL

	float4 x2 = x*x;
	x2[0] = 0;

	float e[MAX_NCOMPONENT];

	for (int i=0; i<nComponents; i++) {
		int is = i*2;
		float4 a = sA[is];//(float4) (sA[is+0], sA[is+1], sA[is+2], sA[is+3]);

		e[i] = dot(a, x);
		a = sA[is+1];// (float4) (sA[is+4], sA[is+5], sA[is+6], sA[is+7]);

		e[i] += dot(a, x2);
	}

	float eMax = e[0];

	for (int i=1; i<nComponents; i++) {
		if (e[i] > eMax)
			eMax = e[i];
	}

	float sum = 0;
	for (int i=0; i<nComponents; i++) {
		sum += exp(e[i] - eMax);
	}

	//out[gxy] = pseudocolorf(p, 0.0f, 255.0f);
	out[gxy] = -(eMax + log(sum));
	//out[gxy] = 0xFF000000 | (int) (-(eMax + log(sum)));
}

__kernel void sampleFg(
	__global uint *src,
	__global uint* tri,
	uint fg,
	int minX,
	int maxX,
	int minY,
	int maxY
)
{
	int gi = get_global_id(0);
	int gs = get_global_size(1);
	
	int nSamples = (maxX-minX)*(maxY-minY);

	if (gi > nSamples-1)
		return;
}
/*
__kernel void sampleBg(
    __global uchar* labels,
    int label,
	sampler_t sampler,
	__read_only image2d_t src,
    __global float4* samples
)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*IMAGEW + gx;

	int ws = get_num_groups(0)*get_num_groups(1);
	int wx = get_group_id(0); //workgroup id
	int wy = get_group_id(1);
	int wxy = wy*get_num_groups(0) + wx;

	int lx = get_local_id(0);
	int ly = get_local_id(1);

	if (gxy > gs-1)
		return;

	int i;
	if (lx == 0 && ly == 0)
		i = 0;
	if (lx == 7 && ly == 0)
		i = 1;
	if (lx == 0 && ly == 7)
		i = 2;
	if (lx == 7 && ly == 7)
		i = 3;

	if (tri[gxy] != fg) {
		samples[4*wxy+i] = src[gxy];
	}
	else {
		samples[4*wxy+i] = 0x00000000;
	}
}
*/