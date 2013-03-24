#define NEIGHBOURHOOD_VON_NEUMANN 0
#define NEIGHBOURHOOD_MOORE 1

#ifndef NEIGHBOURHOOD
#define NEIGHBOURHOOD NEIGHBOURHOOD_VON_NEUMANN
#endif

//max(norm_f(c)) = sqrt(3*255*255) = 441.67295593006372
//max(norm_ui(c)) = sqrt(3*1.0*1.0) = 1.7320508075688772
#define g_f(x) (1.0 - (x/1.7320508075688772))
#define g_ui(x) (1.0 - (x/441.67295593006372f))

#define rgba2f4(c) (float4) (c & 0x000000FF, (c & 0x0000FF00) >> 8, (c & 0x00FF0000) >> 16, 0)
#define rgba_f2_to_uint(c) (uint) (0xFF << 24 | ((int) (255*c.z)) << 16 | ((int) (255*c.y)) << 8 | (int) (255*c.x))

__kernel void evolve(
	__global int* labels_in,
	__global int* labels_out,
	__global float* strength_in,
	__global float* strength_out,
	__global int* has_converge,
	__local int* s_labels_in,
	__local float* s_strength_in,
	__local float4* s_img,
	//int imgW,
	//int imgH,
	//__global uint* img,
	__read_only image2d_t img,
	sampler_t sampler
)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gw = get_global_size(0);
	int gh = get_global_size(1);
	int gxy = gy*gw + gx;
	
	int lx = get_local_id(0);
	int ly = get_local_id(1);
	int lw = get_local_size(0);
	int lh = get_local_size(1);

	int sx = 1 + lx;
	int sy = 1 + ly;
	int sw = lw + 2;
	int sxy = sy*sw + sx;

	int imgW = get_image_width(img);

	s_labels_in[sxy]   = labels_in[gxy];
	s_strength_in[sxy] = strength_in[gxy];
	//s_img[sxy]         = img[gxy];
	s_img[sxy]         = read_imagef(img, sampler, (int2) (gx, gy));

	//load padding
	if (ly == 0) { //top
		s_strength_in[sxy-sw] = (gy != 0) ? strength_in[gxy-imgW] : 0;
		s_labels_in[sxy-sw]   = labels_in[gxy-imgW];
		//s_img[sxy-sw]         = img[gxy-imgW];
		s_img[sxy-sw]         = read_imagef(img, sampler, (int2) (gx, gy-1));
	}
	if (ly == lh-1) { //bottom
		s_strength_in[sxy+sw] = (gy != gh-1) ? strength_in[gxy+imgW] : 0;
		s_labels_in[sxy+sw]   = labels_in[gxy+imgW];
		//s_img[sxy+sw]         = img[gxy+imgW];
		s_img[sxy+sw]         = read_imagef(img, sampler, (int2) (gx, gy+1));
	}
	if (lx == 0) { //left
		s_strength_in[sxy-1] = (gx != 0) ? strength_in[gxy-1] : 0;
		s_labels_in[sxy-1]   = labels_in[gxy-1];
		//s_img[sxy-1]         = img[gxy-1];
		s_img[sxy-1]         = read_imagef(img, sampler, (int2) (gx-1, gy));
	}
	if (lx == lw-1) { //right
		s_strength_in[sxy+1] = (gx != gw-1) ? strength_in[gxy+1] : 0;
		s_labels_in[sxy+1]   = labels_in[gxy+1];
		//s_img[sxy+1]         = img[gxy+1];
		s_img[sxy+1]         = read_imagef(img, sampler, (int2) (gx+1, gy));
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	float4 c4 = s_img[sxy];
	int label = s_labels_in[sxy];
	float strength = s_strength_in[sxy];
	float strength_new;

	float4 norm2 = c4 - s_img[sxy-sw];
	norm2 *= norm2;
	strength_new = g_ui(sqrt(norm2.x + norm2.y + norm2.z))*s_strength_in[sxy-sw];
	if (strength_new > strength) {
		label = s_labels_in[sxy-sw];
		strength = strength_new;
	}

	norm2 = c4 - s_img[sxy+sw];
	norm2 *= norm2;
	strength_new = g_ui(sqrt(norm2.x + norm2.y + norm2.z))*s_strength_in[sxy+sw];
	if (strength_new > strength) {
		label = s_labels_in[sxy+sw];
		strength = strength_new;
	}

	norm2 = c4 - s_img[sxy-1];
	norm2 *= norm2;
	strength_new = g_ui(sqrt(norm2.x + norm2.y + norm2.z))*s_strength_in[sxy-1];
	if (strength_new > strength) {
		label = s_labels_in[sxy-1];
		strength = strength_new;
	}

	norm2 = c4 - s_img[sxy+1];
	norm2 *= norm2;
	strength_new = g_ui(sqrt(norm2.x + norm2.y + norm2.z))*s_strength_in[sxy+1];
	if (strength_new > strength) {
		label = s_labels_in[sxy+1];
		strength = strength_new;
	}

	labels_out[gxy] = label;
	strength_out[gxy] = strength;
}
