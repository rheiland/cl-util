#define FALSE 0
#define TRUE 1

#define CL_TRUE -1
#define CL_FALSE 0

#define CL_TRUE_2_TRUE -1

#define NEIGHBOURHOOD_VON_NEUMANN 0
#define NEIGHBOURHOOD_MOORE 1

#ifndef NEIGHBOURHOOD
#define NEIGHBOURHOOD NEIGHBOURHOOD_VON_NEUMANN
#endif

//max(norm_f(c)) = sqrt(3*255*255) = 441.67295593006372
//max(norm_ui(c)) = sqrt(3*1.0*1.0) = 1.7320508075688772
#define g_f(x) (1.0f - (x/1.7320508075688772f))
#define g_ui(x) (1.0f - (x/441.67295593006372f))

#define rgba2f4(c) (float4) (c & 0x000000FF, (c & 0x0000FF00) >> 8, (c & 0x00FF0000) >> 16, 0)
#define rgba_f2_to_uint(c) (uint) (0xFF << 24 | ((int) (255*c.z)) << 16 | ((int) (255*c.y)) << 8 | (int) (255*c.x))

#define CAN_ATTACK_THRESHOLD 6
#define OVER_PROWER_THRESHOLD 6

__kernel void countEnemies(
	__global int* labels,
	__local int* s_labels,
	__global int* g_enemies
){
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

	s_labels[sxy] = labels[gxy];

	if (ly == 0)    s_labels[sxy-sw] = (gy != 0)    ? labels[gxy-gw] : -1;
	if (ly == lh-1) s_labels[sxy+sw] = (gy != gh-1) ? labels[gxy+gw] : -1;
	if (lx == 0)    s_labels[sxy-1]  = (gx != 0)    ? labels[gxy-1]  : -1;
	if (lx == lw-1) s_labels[sxy+1]  = (gx != gw-1) ? labels[gxy+1]  : -1;

	barrier(CLK_LOCAL_MEM_FENCE);

	int label = s_labels[sxy];
	int4 neighbours = (int4) (s_labels[sxy-sw], s_labels[sxy+sw], s_labels[sxy-1], s_labels[sxy+1]);	
	int4 enemies = neighbours != label;

	g_enemies[gxy] = CL_TRUE_2_TRUE*(enemies.s0 + enemies.s1 + enemies.s2 + enemies.s3);
}

__kernel void evolve(
	__global int* labels_in,
	__global int* labels_out,
	__global float* strength_in,
	__global float* strength_out,
	__global int* g_enemies,
	__global int* has_converge,
	__local int* s_labels_in,
	__local float* s_strength_in,
	__local float4* s_img,
	__local int* s_enemies,
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
	s_img[sxy]         = read_imagef(img, sampler, (int2) (gx, gy));
	s_enemies[sxy]     = g_enemies[gxy];

	int isxy, igxy;

	//load padding
	if (ly == 0) { //top
		isxy = sxy - sw;
		igxy = gxy - imgW;
		s_strength_in[isxy] = (gy != 0) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx, gy-1));
		s_enemies[isxy]     = g_enemies[igxy];
	}
	else if (ly == lh-1) { //bottom
		isxy = sxy + sw;
		igxy = gxy + imgW;
		s_strength_in[isxy] = (gy != gh-1) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx, gy+1));
		s_enemies[isxy]     = g_enemies[igxy];
	}
	if (lx == 0) { //left
		isxy = sxy - 1;
		igxy = gxy - 1;
		s_strength_in[isxy] = (gx != 0) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx-1, gy));
		s_enemies[isxy]     = g_enemies[igxy];
	}
	else if (lx == lw-1) { //right
		isxy = sxy + 1;
		igxy = gxy + 1;
		s_strength_in[isxy] = (gx != gw-1) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx+1, gy));
		s_enemies[isxy]     = g_enemies[igxy];
	}

	if (lx == 0 && ly == 0) { //top
		isxy = sxy - sw - 1;
		igxy = gxy - imgW - 1;
		s_strength_in[isxy] = (gx != 0 && gy != 0) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx-1, gy-1));
		s_enemies[isxy]     = g_enemies[igxy];
	}
	else if (lx == lw-1 && ly == lh-1) { //bottom
		isxy = sxy + sw + 1;
		igxy = gxy + imgW + 1;
		s_strength_in[isxy] = (gx != gw-1 && gy != gh-1) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx+1, gy+1));
		s_enemies[isxy]     = g_enemies[igxy];
	}
	if (lx == 0 && ly == lh-1) { //left
		isxy = sxy - 1 + sw;
		igxy = gxy - 1 + imgW;
		s_strength_in[isxy] = (gx != 0 && gy != gh-1) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx-1, gy+1));
		s_enemies[isxy]     = g_enemies[igxy];
	}
	else if (lx == lw-1 && ly == 0) { //right
		isxy = sxy + 1 - sw;
		igxy = gxy + 1 - imgW;
		s_strength_in[isxy] = 0;//(gx != gw-1 && gy != 0) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx+1, gy-1));
		s_enemies[isxy]     = g_enemies[igxy];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 c = s_img[sxy];
	int label = s_labels_in[sxy];
	float defence = s_strength_in[sxy];

	float8 attack = (float8) ( 
		g_f(length(c - s_img[sxy-sw])) * s_strength_in[sxy-sw],
		g_f(length(c - s_img[sxy+sw])) * s_strength_in[sxy+sw],
		g_f(length(c - s_img[sxy-1])) * s_strength_in[sxy-1],
		g_f(length(c - s_img[sxy+1])) * s_strength_in[sxy+1],
		g_f(length(c - s_img[sxy-sw-1])) * s_strength_in[sxy-sw-1],
		g_f(length(c - s_img[sxy+sw+1])) * s_strength_in[sxy+sw+1],
		g_f(length(c - s_img[sxy-1+sw])) * s_strength_in[sxy-1+sw],
		g_f(length(c - s_img[sxy+1-sw])) * s_strength_in[sxy+1-sw]
	);

	if (s_enemies[sxy] > OVER_PROWER_THRESHOLD) {
		defence = FLT_MAX;

		int8 enemies = (int8) (
			s_labels_in[sxy-sw], s_labels_in[sxy+sw], s_labels_in[sxy-1], s_labels_in[sxy+1],
			s_labels_in[sxy-sw-1], s_labels_in[sxy+sw+1], s_labels_in[sxy-1+sw], s_labels_in[sxy+1-sw]
			) != label;

		if (gy != 0 && attack.s0 < defence && enemies.s0) {
			defence = attack.s0;
			label = s_labels_in[sxy-sw];
		}

		if (gy != gh-1 && attack.s1 < defence && enemies.s1) {
			defence = attack.s1;
			label = s_labels_in[sxy+sw];
		}

		if (gx != 0 && attack.s2 < defence && enemies.s2) {
			defence = attack.s2;
			label = s_labels_in[sxy-1];
		}

		if (gx != gw-1 && attack.s3 < defence && enemies.s3) {
			defence = attack.s3;
			label = s_labels_in[sxy+1];
		}

		if (gy != 0 && gx != 0 && attack.s4 < defence && enemies.s4) {
			defence = attack.s4;
			label = s_labels_in[sxy-sw-1];
		}

		if (gy != gh-1 && gx != gw-1 && attack.s5 < defence && enemies.s5) {
			defence = attack.s5;
			label = s_labels_in[sxy+sw+1];
		}

		if (gy != gh-1 && gx != 0 && attack.s6 < defence && enemies.s6) {
			defence = attack.s6;
			label = s_labels_in[sxy-1+sw];
		}

		if (gy != 0 && gx != gw-1 && attack.s7 < defence && enemies.s7) {
			defence = attack.s7;
			label = s_labels_in[sxy+1-sw];
		}
	}
	else { 
		int8 can_attack = (int8) (
			s_enemies[sxy-sw], s_enemies[sxy+sw], s_enemies[sxy-1], s_enemies[sxy+1],
			s_enemies[sxy-sw-1], s_enemies[sxy+sw+1], s_enemies[sxy-1+sw], s_enemies[sxy+1-sw]
			) < CAN_ATTACK_THRESHOLD;

		if (attack.s0 > defence && can_attack.s0) {
			defence = attack.s0;
			label = s_labels_in[sxy-sw];
		}

		if (attack.s1 > defence && can_attack.s1) {
			defence = attack.s1;
			label = s_labels_in[sxy+sw];
		}

		if (attack.s2 > defence && can_attack.s2) {
			defence = attack.s2;
			label = s_labels_in[sxy-1];
		}

		if (attack.s3 > defence && can_attack.s3) {
			defence = attack.s3;
			label = s_labels_in[sxy+1];
		}

		if (attack.s4 > defence && can_attack.s4) {
			defence = attack.s4;
			label = s_labels_in[sxy-sw-1];
		}

		if (attack.s5 > defence && can_attack.s5) {
			defence = attack.s5;
			label = s_labels_in[sxy+sw+1];
		}

		if (attack.s6 > defence && can_attack.s6) {
			defence = attack.s6;
			label = s_labels_in[sxy-1+sw];
		}

		if (attack.s7 > defence && can_attack.s7) {
			defence = attack.s7;
			label = s_labels_in[sxy+1-sw];
		}
	}

	strength_out[gxy] = defence;
	labels_out[gxy] = label;
}


__kernel void evolveVonNeumann(
	__global int* labels_in,
	__global int* labels_out,
	__global float* strength_in,
	__global float* strength_out,
	__global int* g_enemies,
	__global int* has_converge,
	__local int* s_labels_in,
	__local float* s_strength_in,
	__local float4* s_img,
	__local int* s_enemies,
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
	s_img[sxy]         = read_imagef(img, sampler, (int2) (gx, gy));
	s_enemies[sxy]     = g_enemies[gxy];

	int isxy, igxy;

	//load padding
	if (ly == 0) { //top
		isxy = sxy - sw;
		igxy = gxy - imgW;
		s_strength_in[isxy] = (gy != 0) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx, gy-1));
		s_enemies[isxy]     = g_enemies[igxy];
	}
	else if (ly == lh-1) { //bottom
		isxy = sxy + sw;
		igxy = gxy + imgW;
		s_strength_in[isxy] = (gy != gh-1) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx, gy+1));
		s_enemies[isxy]     = g_enemies[igxy];
	}
	if (lx == 0) { //left
		isxy = sxy - 1;
		igxy = gxy - 1;
		s_strength_in[isxy] = (gx != 0) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx-1, gy));
		s_enemies[isxy]     = g_enemies[igxy];
	}
	else if (lx == lw-1) { //right
		isxy = sxy + 1;
		igxy = gxy + 1;
		s_strength_in[isxy] = (gx != gw-1) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx+1, gy));
		s_enemies[isxy]     = g_enemies[igxy];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	float4 c = s_img[sxy];
	int label = s_labels_in[sxy];
	float defence = s_strength_in[sxy];

	float4 attack = (float4) ( 
		g_f(length(c - s_img[sxy-sw])) * s_strength_in[sxy-sw],
		g_f(length(c - s_img[sxy+sw])) * s_strength_in[sxy+sw],
		g_f(length(c - s_img[sxy-1])) * s_strength_in[sxy-1],
		g_f(length(c - s_img[sxy+1])) * s_strength_in[sxy+1]
	);

	if (s_enemies[sxy] > OVER_PROWER_THRESHOLD) {
		defence = FLT_MAX;

		if (attack.x < defence && s_labels_in[sxy-sw] != label) {
			defence = attack.x;
			label = s_labels_in[sxy-sw];
		}

		if (attack.y < defence && s_labels_in[sxy+sw] != label) {
			defence = attack.y;
			label = s_labels_in[sxy+sw];
		}

		if (attack.z < defence && s_labels_in[sxy-1] != label) {
			defence = attack.z;
			label = s_labels_in[sxy-1];
		}

		if (attack.w < defence && s_labels_in[sxy+1] != label) {
			defence = attack.w;
			label = s_labels_in[sxy+1];
		}
	}
	else {
		if (attack.x > defence && s_enemies[sxy-sw] < CAN_ATTACK_THRESHOLD) {
			defence = attack.x;
			label = s_labels_in[sxy-sw];
		}

		if (attack.y > defence && s_enemies[sxy+sw] < CAN_ATTACK_THRESHOLD) {
			defence = attack.y;
			label = s_labels_in[sxy+sw];
		}

		if (attack.z > defence && s_enemies[sxy-1] < CAN_ATTACK_THRESHOLD) {
			defence = attack.z;
			label = s_labels_in[sxy-1];
		}

		if (attack.w > defence && s_enemies[sxy+1] < CAN_ATTACK_THRESHOLD) {
			defence = attack.w;
			label = s_labels_in[sxy+1];
		}
	}

	strength_out[gxy] = defence;
	labels_out[gxy] = label;
}
