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
#define OVER_PROWER_THRESHOLD 1

__kernel void evolve(
	__global int* labels_in,
	__global int* labels_out,
	__global float* strength_in,
	__global float* strength_out,
	__global int* has_converge,
	__local int* s_labels_in,
	__local float* s_strength_in,
	__local float4* s_img,
	__local int* s_canAttack,
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

	int sx = 2 + lx;
	int sy = 2 + ly;
	int sw = lw + 4;
	int sxy = sy*sw + sx;

	//int imgW = get_image_width(img);
	int imgW = 800;

	s_labels_in[sxy]   = labels_in[gxy];
	s_strength_in[sxy] = strength_in[gxy];
	s_img[sxy]         = read_imagef(img, sampler, (int2) (gx, gy));
	//s_canAttack[sxy]   = FALSE;

	int isxy, igxy;

	//load padding
	if (ly < 2) { //top
		isxy = sxy - 2*sw;
		igxy = gxy - 2*imgW;
		s_strength_in[isxy] = (gy >= 2) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = (gy >= 2) ? labels_in[igxy] : -1;
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx, gy-2));
		s_canAttack[isxy]   = FALSE;
	}
	else if (ly >= lh-2) { //bottom
		isxy = sxy + 2*sw;
		igxy = gxy + 2*imgW;
		s_strength_in[isxy] = (gy < gh-2) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = (gy < gh-2) ? labels_in[igxy] : -1;
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx, gy+2));
		s_canAttack[isxy]   = FALSE;
	}
	if (lx < 2) { //left
		isxy = sxy - 2;
		igxy = gxy - 2;
		s_strength_in[isxy] = (gx >= 2) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = (gx >= 2) ? labels_in[igxy] : -1;
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx-2, gy));
		s_canAttack[isxy]   = FALSE;
	}
	else if (lx >= lw-2) { //right
		isxy = sxy + 2;
		igxy = gxy + 2;
		s_strength_in[isxy] = (gx < gw-2) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = (gx < gw-2) ? labels_in[igxy] : -1;
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx+2, gy));
		s_canAttack[isxy]   = FALSE;
	}
	if (lx < 2 && ly < 2) {
		isxy = sxy - 2*sw - 2;
		igxy = gxy - 2*imgW - 2;
		s_strength_in[isxy] = (gx >= 2 && gy >= 2) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = (gx >= 2 && gy >= 2) ? labels_in[igxy] : -1;
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx-2, gy-2));
		s_canAttack[isxy]   = FALSE;
	}
	else if (lx >= lw-2 && ly < 2) {
		isxy = sxy - 2*sw + 2;
		igxy = gxy - 2*imgW + 2;
		s_strength_in[isxy] = (gx < gw-2 && gy >= 2) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = (gx < gw-2 && gy >= 2) ? labels_in[igxy] : -1;
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx+2, gy-2));
		s_canAttack[isxy]   = FALSE;
	}
	else if (lx < 2 && ly >= lh-2) {
		isxy = sxy + 2*sw - 2;
		igxy = gxy + 2*imgW - 2;
		s_strength_in[isxy] = (gx >= 2 && gy < gh-2) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = (gx >= 2 && gy < gh-2) ? labels_in[igxy] : -1;
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx-2, gy+2));
		s_canAttack[isxy]   = FALSE;
	}
	else if (lx >= lw-2 && ly >= lh-2) {
		isxy = sxy + 2*sw + 2;
		igxy = gxy + 2*imgW + 2;
		s_strength_in[isxy] = (gx < gw-2 && gy < gh-2) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = (gx < gw-2 && gy < gh-2) ? labels_in[igxy] : -1;
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx+2, gy+2));
		s_canAttack[isxy]   = FALSE;
	}	

	barrier(CLK_LOCAL_MEM_FENCE);

	int label;
	int8 neighbourLabels;
	int8 enemies;

	if (ly == 0) {
		isxy = sxy-sw;
		label = s_labels_in[isxy];
		neighbourLabels = (int8)
			(s_labels_in[isxy-sw], s_labels_in[isxy+sw], s_labels_in[isxy-1], s_labels_in[isxy+1],
			s_labels_in[isxy-sw-1], s_labels_in[isxy+sw+1], s_labels_in[isxy-1+sw], s_labels_in[isxy+1-sw]
		);
		enemies = neighbourLabels != label;
		s_canAttack[isxy] = CL_TRUE_2_TRUE*(
			enemies.s0 + enemies.s1 + enemies.s2 + enemies.s3 + 
			enemies.s4 + enemies.s5 + enemies.s6 + enemies.s7) < CAN_ATTACK_THRESHOLD;
	}
	else if (ly == lh-1) {
		isxy = sxy+sw;
		label = s_labels_in[isxy];
		neighbourLabels = (int8)
			(s_labels_in[isxy-sw], s_labels_in[isxy+sw], s_labels_in[isxy-1], s_labels_in[isxy+1],
			s_labels_in[isxy-sw-1], s_labels_in[isxy+sw+1], s_labels_in[isxy-1+sw], s_labels_in[isxy+1-sw]
		);
		enemies = neighbourLabels != label;
		s_canAttack[isxy] = CL_TRUE_2_TRUE*(
			enemies.s0 + enemies.s1 + enemies.s2 + enemies.s3 + 
			enemies.s4 + enemies.s5 + enemies.s6 + enemies.s7) < CAN_ATTACK_THRESHOLD;
	}
	if (lx == 0) {
		isxy = sxy-1;
		label = s_labels_in[isxy];
		neighbourLabels = (int8)
			(s_labels_in[isxy-sw], s_labels_in[isxy+sw], s_labels_in[isxy-1], s_labels_in[isxy+1],
			s_labels_in[isxy-sw-1], s_labels_in[isxy+sw+1], s_labels_in[isxy-1+sw], s_labels_in[isxy+1-sw]
		);
		enemies = neighbourLabels != label;
		s_canAttack[isxy] = CL_TRUE_2_TRUE*(
			enemies.s0 + enemies.s1 + enemies.s2 + enemies.s3 + 
			enemies.s4 + enemies.s5 + enemies.s6 + enemies.s7) < CAN_ATTACK_THRESHOLD;
	}
	else if (lx == lw-1) {
		isxy = sxy+1;
		label = s_labels_in[isxy];
		neighbourLabels = (int8)
			(s_labels_in[isxy-sw], s_labels_in[isxy+sw], s_labels_in[isxy-1], s_labels_in[isxy+1],
			s_labels_in[isxy-sw-1], s_labels_in[isxy+sw+1], s_labels_in[isxy-1+sw], s_labels_in[isxy+1-sw]
		);
		enemies = neighbourLabels != label;
		s_canAttack[isxy] = CL_TRUE_2_TRUE*(
			enemies.s0 + enemies.s1 + enemies.s2 + enemies.s3 + 
			enemies.s4 + enemies.s5 + enemies.s6 + enemies.s7) < CAN_ATTACK_THRESHOLD;
	}

	if (lx == 0 && ly == 0) {
		isxy = sxy-sw-1;
		label = s_labels_in[isxy];
		neighbourLabels = (int8)
			(s_labels_in[isxy-sw], s_labels_in[isxy+sw], s_labels_in[isxy-1], s_labels_in[isxy+1],
			s_labels_in[isxy-sw-1], s_labels_in[isxy+sw+1], s_labels_in[isxy-1+sw], s_labels_in[isxy+1-sw]
		);
		enemies = neighbourLabels != label;
		s_canAttack[isxy] = CL_TRUE_2_TRUE*(
			enemies.s0 + enemies.s1 + enemies.s2 + enemies.s3 + 
			enemies.s4 + enemies.s5 + enemies.s6 + enemies.s7) < CAN_ATTACK_THRESHOLD;
	}

	if (lx == lw-1 && ly == 0) {
		isxy = sxy-sw+1;
		label = s_labels_in[isxy];
		neighbourLabels = (int8)
			(s_labels_in[isxy-sw], s_labels_in[isxy+sw], s_labels_in[isxy-1], s_labels_in[isxy+1],
			s_labels_in[isxy-sw-1], s_labels_in[isxy+sw+1], s_labels_in[isxy-1+sw], s_labels_in[isxy+1-sw]
		);
		enemies = neighbourLabels != label;
		s_canAttack[isxy] = CL_TRUE_2_TRUE*(
			enemies.s0 + enemies.s1 + enemies.s2 + enemies.s3 + 
			enemies.s4 + enemies.s5 + enemies.s6 + enemies.s7) < CAN_ATTACK_THRESHOLD;
	}

	if (lx == 0 && ly == lh-1) {
		isxy =  sxy+sw-1;
		label = s_labels_in[isxy];
		neighbourLabels = (int8)
			(s_labels_in[isxy-sw], s_labels_in[isxy+sw], s_labels_in[isxy-1], s_labels_in[isxy+1],
			s_labels_in[isxy-sw-1], s_labels_in[isxy+sw+1], s_labels_in[isxy-1+sw], s_labels_in[isxy+1-sw]
		);
		enemies = neighbourLabels != label;
		s_canAttack[isxy] = CL_TRUE_2_TRUE*(
			enemies.s0 + enemies.s1 + enemies.s2 + enemies.s3 + 
			enemies.s4 + enemies.s5 + enemies.s6 + enemies.s7) < CAN_ATTACK_THRESHOLD;
	}

	if (lx == lw-1 && ly == lh-1) {
		isxy = sxy+sw+1;
		label = s_labels_in[isxy];
		neighbourLabels = (int8)
			(s_labels_in[isxy-sw], s_labels_in[isxy+sw], s_labels_in[isxy-1], s_labels_in[isxy+1],
			s_labels_in[isxy-sw-1], s_labels_in[isxy+sw+1], s_labels_in[isxy-1+sw], s_labels_in[isxy+1-sw]
		);
		enemies = neighbourLabels != label;
		s_canAttack[isxy] = CL_TRUE_2_TRUE*(
			enemies.s0 + enemies.s1 + enemies.s2 + enemies.s3 + 
			enemies.s4 + enemies.s5 + enemies.s6 + enemies.s7) < CAN_ATTACK_THRESHOLD;
	}


	label = s_labels_in[sxy];
	neighbourLabels = (int8)
		(s_labels_in[sxy-sw], s_labels_in[sxy+sw], s_labels_in[sxy-1], s_labels_in[sxy+1],
		s_labels_in[sxy-sw-1], s_labels_in[sxy+sw+1], s_labels_in[sxy-1+sw], s_labels_in[sxy+1-sw]
	);
	enemies = neighbourLabels != label;
	int nEnemies = CL_TRUE_2_TRUE*(
		enemies.s0 + enemies.s1 + enemies.s2 + enemies.s3 + 
		enemies.s4 + enemies.s5 + enemies.s6 + enemies.s7) < CAN_ATTACK_THRESHOLD;

	s_canAttack[sxy] = nEnemies;

	barrier(CLK_LOCAL_MEM_FENCE);

	float4 c = s_img[sxy];
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

	if (nEnemies > OVER_PROWER_THRESHOLD) {
		defence = FLT_MAX;

		if (gy != 0 && attack.s0 < defence && enemies.s0) {
			defence = attack.s0;
			label = neighbourLabels.s0;
		}

		if (gy != gh-1 && attack.s1 < defence && enemies.s1) {
			defence = attack.s1;
			label = neighbourLabels.s1;
		}

		if (gx != 0 && attack.s2 < defence && enemies.s2) {
			defence = attack.s2;
			label = neighbourLabels.s2;
		}

		if (gx != gw-1 && attack.s3 < defence && enemies.s3) {
			defence = attack.s3;
			label = neighbourLabels.s3;
		}

		if (gy != 0 && gx != 0 && attack.s4 < defence && enemies.s4) {
			defence = attack.s4;
			label = neighbourLabels.s4;
		}

		if (gy != gh-1 && gx != gw-1 && attack.s5 < defence && enemies.s5) {
			defence = attack.s5;
			label = neighbourLabels.s5;
		}

		if (gy != gh-1 && gx != 0 && attack.s6 < defence && enemies.s6) {
			defence = attack.s6;
			label = neighbourLabels.s6;
		}

		if (gy != 0 && gx != gw-1 && attack.s7 < defence && enemies.s7) {
			defence = attack.s7;
			label = neighbourLabels.s7;
		}
	}
	else { 
		if (attack.s0 > defence && s_canAttack[sxy-sw]) {
			defence = attack.s0;
			label = neighbourLabels.s0;
		}

		if (attack.s1 > defence && s_canAttack[sxy+sw]) {
			defence = attack.s1;
			label = neighbourLabels.s1;
		}

		if (attack.s2 > defence && s_canAttack[sxy-1]) {
			defence = attack.s2;
			label = neighbourLabels.s2;
		}

		if (attack.s3 > defence && s_canAttack[sxy+1]) {
			defence = attack.s3;
			label = neighbourLabels.s3;
		}

		if (attack.s4 > defence && s_canAttack[sxy-sw-1]) {
			defence = attack.s4;
			label = neighbourLabels.s4;
		}

		if (attack.s5 > defence && s_canAttack[sxy+sw+1]) {
			defence = attack.s5;
			label = neighbourLabels.s5;
		}

		if (attack.s6 > defence && s_canAttack[sxy-1+sw]) {
			defence = attack.s6;
			label = neighbourLabels.s6;
		}

		if (attack.s7 > defence && s_canAttack[sxy+1-sw]) {
			defence = attack.s7;
			label = neighbourLabels.s7;
		}

	}

	strength_out[gxy] = defence;
	labels_out[gxy] = label;
}





__kernel void evolveMoore(
	__global int* labels_in,
	__global int* labels_out,
	__global float* strength_in,
	__global float* strength_out,
	__global int* has_converge,
	__local int* s_labels_in,
	__local float* s_strength_in,
	__local float4* s_img,
	__local int* s_canAttack,
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

	int sx = 2 + lx;
	int sy = 2 + ly;
	int sw = lw + 4;
	int sxy = sy*sw + sx;

	//int imgW = get_image_width(img);
	int imgW = 800;

	s_labels_in[sxy]   = labels_in[gxy];
	s_strength_in[sxy] = strength_in[gxy];
	s_img[sxy]         = read_imagef(img, sampler, (int2) (gx, gy));
	s_canAttack[sxy]   = FALSE;

	int isxy, igxy;

	//load padding
	if (ly < 2) { //top
		isxy = sxy - 2*sw;
		igxy = gxy - 2*imgW;
		s_strength_in[isxy] = (gy >= 2) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx, gy-2));
		s_canAttack[isxy]   = FALSE;
	}
	else if (ly >= lh-2) { //bottom
		isxy = sxy + 2*sw;
		igxy = gxy + 2*imgW;
		s_strength_in[isxy] = (gy < gh-2) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx, gy+2));
		s_canAttack[isxy]   = FALSE;
	}
	if (lx < 2) { //left
		isxy = sxy - 2;
		igxy = gxy - 2;
		s_strength_in[isxy] = (gx >= 2) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx-2, gy));
		s_canAttack[isxy]   = FALSE;
	}
	else if (lx >= lw-2) { //right
		isxy = sxy + 2;
		igxy = gxy + 2;
		s_strength_in[isxy] = (gx < gw-2) ? strength_in[igxy] : 0;
		s_labels_in[isxy]   = labels_in[igxy];
		s_img[isxy]         = read_imagef(img, sampler, (int2) (gx+2, gy));
		s_canAttack[isxy]   = FALSE;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int label;
	int4 neighbourLabels;
	int4 enemies;

	if (ly == 0) {
		isxy = sxy-sw;
		label = s_labels_in[isxy];
		neighbourLabels = (int4) (s_labels_in[isxy-sw], s_labels_in[isxy+sw], s_labels_in[isxy-1], s_labels_in[isxy+1]);
		enemies = neighbourLabels != label;
		s_canAttack[isxy] = CL_TRUE_2_TRUE*(enemies.x + enemies.y + enemies.z + enemies.w) < CAN_ATTACK_THRESHOLD;
	}
	else if (ly == lh-1) {
		isxy = sxy+sw;
		label = s_labels_in[isxy];
		neighbourLabels = (int4) (s_labels_in[isxy-sw], s_labels_in[isxy+sw], s_labels_in[isxy-1], s_labels_in[isxy+1]);
		enemies = neighbourLabels != label;
		s_canAttack[isxy] = CL_TRUE_2_TRUE*(enemies.x + enemies.y + enemies.z + enemies.w) < CAN_ATTACK_THRESHOLD;
	}
	if (lx == 0) {
		isxy = sxy-1;
		label = s_labels_in[isxy];
		neighbourLabels = (int4) (s_labels_in[isxy-sw], s_labels_in[isxy+sw], s_labels_in[isxy-1], s_labels_in[isxy+1]);
		enemies = neighbourLabels != label;
		s_canAttack[isxy] = CL_TRUE_2_TRUE*(enemies.x + enemies.y + enemies.z + enemies.w) < CAN_ATTACK_THRESHOLD;
	}
	else if (lx == lw-1) {
		isxy = sxy+1;
		label = s_labels_in[isxy];
		neighbourLabels = (int4) (s_labels_in[isxy-sw], s_labels_in[isxy+sw], s_labels_in[isxy-1], s_labels_in[isxy+1]);
		enemies = neighbourLabels != label;
		s_canAttack[isxy] = CL_TRUE_2_TRUE*(enemies.x + enemies.y + enemies.z + enemies.w) < CAN_ATTACK_THRESHOLD;
	}

	label = s_labels_in[sxy];
	neighbourLabels = (int4) (s_labels_in[sxy-sw], s_labels_in[sxy+sw], s_labels_in[sxy-1], s_labels_in[sxy+1]);
	enemies = neighbourLabels != label;
	int nEnemies = CL_TRUE_2_TRUE*(enemies.x + enemies.y + enemies.z + enemies.w);

	s_canAttack[sxy] = nEnemies < CAN_ATTACK_THRESHOLD;

	barrier(CLK_LOCAL_MEM_FENCE);

	float4 c = s_img[sxy];
	float defence = s_strength_in[sxy];

	float4 attack = (float4) ( 
		g_f(length(c - s_img[sxy-sw])) * s_strength_in[sxy-sw],
		g_f(length(c - s_img[sxy+sw])) * s_strength_in[sxy+sw],
		g_f(length(c - s_img[sxy-1])) * s_strength_in[sxy-1],
		g_f(length(c - s_img[sxy+1])) * s_strength_in[sxy+1]
	);

	if (nEnemies > OVER_PROWER_THRESHOLD) {
		defence = FLT_MAX;

		if (attack.x < defence && enemies.x) {
			defence = attack.x;
			label = neighbourLabels.x;
		}

		if (attack.y < defence && enemies.y) {
			defence = attack.y;
			label = neighbourLabels.y;
		}

		if (attack.z < defence && enemies.z) {
			defence = attack.z;
			label = neighbourLabels.z;
		}

		if (attack.w < defence && enemies.w) {
			defence = attack.w;
			label = neighbourLabels.w;
		}
	}
	else {
		if (attack.x > defence && s_canAttack[sxy-sw] == TRUE) {
			defence = attack.x;
			label = neighbourLabels.x;
		}

		if (attack.y > defence && s_canAttack[sxy+sw] == TRUE) {
			defence = attack.y;
			label = neighbourLabels.y;
		}

		if (attack.z > defence && s_canAttack[sxy-1] == TRUE) {
			defence = attack.z;
			label = neighbourLabels.z;
		}

		if (attack.w > defence && s_canAttack[sxy+1] == TRUE) {
			defence = attack.w;
			label = neighbourLabels.w;
		}
	}

	strength_out[gxy] = defence;
	labels_out[gxy] = label;
}
