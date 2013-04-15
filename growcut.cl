#define FALSE 0
#define TRUE 1

#define CL_TRUE -1
#define CL_FALSE 0

#define CL_TRUE_2_TRUE -1

//sqrt(3*255*255) = 441.67295593006372
//sqrt(3*1.0*1.0) = 1.7320508075688772
#ifndef G_NORM(X)
#define G_NORM(X) (1.0f - X/1.7320508075688772f)
#endif

#define rgba2f4(c) (float4) (c & 0x000000FF, (c & 0x0000FF00) >> 8, (c & 0x00FF0000) >> 16, 0)
#define rgba_f2_to_uint(c) (uint) (0xFF << 24 | ((int) (255*c.z)) << 16 | ((int) (255*c.y)) << 8 | (int) (255*c.x))

//#define CAN_ATTACK_THRESHOLD 6
//#define OVER_PROWER_THRESHOLD 6

float norm_length_ui(uint4 vector) {
	float4 f = (float4) (((float) vector.x)/255, ((float) vector.y)/255, ((float) vector.z)/255, ((float) vector.w)/255);

	return length(f);
}

__kernel void countEnemies(
	__global int* labels,
	__local int* s_labels,
	__global int* g_enemies
){
	int ix = get_global_id(0);
	int iy = get_global_id(1);
	int gw = get_global_size(0);
	int gh = get_global_size(1);
	int ixy = iy*gw + ix;
	
	int lx = get_local_id(0);
	int ly = get_local_id(1);

	int sx = 1 + lx;
	int sy = 1 + ly;
	int sw = TILEW + 2;
	int sxy = sy*sw + sx;

	s_labels[sxy] = labels[ixy];

	if (ly == 0)    s_labels[sxy-sw] = (iy != 0)    ? labels[ixy-gw] : -1;
	if (ly == TILEH-1) s_labels[sxy+sw] = (iy != gh-1) ? labels[ixy+gw] : -1;
	if (lx == 0)    s_labels[sxy-1]  = (ix != 0)    ? labels[ixy-1]  : -1;
	if (lx == TILEW-1) s_labels[sxy+1]  = (ix != gw-1) ? labels[ixy+1]  : -1;

	barrier(CLK_LOCAL_MEM_FENCE);

	int label = s_labels[sxy];
	int4 neighbours = (int4) (s_labels[sxy-sw], s_labels[sxy+sw], s_labels[sxy-1], s_labels[sxy+1]);
	int4 enemies = neighbours != label;

	g_enemies[ixy] = CL_TRUE_2_TRUE*(enemies.s0 + enemies.s1 + enemies.s2 + enemies.s3);
}

float4 norm_rgba_ui4(uint4 rgba) {
	return (float4) (((float) rgba.x)/255, ((float) rgba.y)/255, ((float) rgba.z)/255, ((float) rgba.w)/255);
}

__kernel void evolveMoore(
	__global int* tiles_list,
	__global int* labels_in,
	__global int* labels_out,
	__global float* strength_in,
	__global float* strength_out,
	__global int* g_enemies,
	__global int* has_converge,
	int iteration,
	__global int* tiles,
	__local int* tile_flags, //true if any updates
	__local int* s_labels_in,
	__local float* s_strength_in,
	__local float4* s_img,
	__local int* s_enemies,
	__read_only image2d_t img,
	sampler_t sampler
)
{
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%TILESW;
	int ty = txy/TILESW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);

	//image coordinates
	int ix = tx*TILEW + lx;
	int iy = ty*TILEH + ly;
	int ixy = iy*IMAGEW + ix;

	int sw = TILEW + 2;
	int sx = 1 + lx;
	int sy = 1 + ly;
	int sxy = sy*sw + sx;

	s_labels_in[sxy]   = labels_in[ixy];
	s_strength_in[sxy] = strength_in[ixy];
	s_img[sxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix, iy)));
	s_enemies[sxy]     = g_enemies[ixy];

	int i_sxy, i_ixy;

	//load padding
	if (ly == 0) { //top
		i_sxy = sxy - sw;
		i_ixy = ixy - IMAGEW;
		s_strength_in[i_sxy] = (iy != 0) ? strength_in[i_ixy] : 0;
		s_labels_in[i_sxy]   = labels_in[i_ixy];
		s_img[i_sxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix, iy-1)));
		s_enemies[i_sxy]     = g_enemies[i_ixy];
	}
	else if (ly == TILEH-1) { //bottom
		i_sxy = sxy + sw;
		i_ixy = ixy + IMAGEW;
		s_strength_in[i_sxy] = (iy != IMAGEH-1) ? strength_in[i_ixy] : 0;
		s_labels_in[i_sxy]   = labels_in[i_ixy];
		s_img[i_sxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix, iy+1)));
		s_enemies[i_sxy]     = g_enemies[i_ixy];
	}
	if (lx == 0) { //left
		i_sxy = sxy - 1;
		i_ixy = ixy - 1;
		s_strength_in[i_sxy] = (ix != 0) ? strength_in[i_ixy] : 0;
		s_labels_in[i_sxy]   = labels_in[i_ixy];
		s_img[i_sxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix-1, iy)));
		s_enemies[i_sxy]     = g_enemies[i_ixy];
	}
	else if (lx == TILEW-1) { //right
		i_sxy = sxy + 1;
		i_ixy = ixy + 1;
		s_strength_in[i_sxy] = (ix != IMAGEW-1) ? strength_in[i_ixy] : 0;
		s_labels_in[i_sxy]   = labels_in[i_ixy];
		s_img[i_sxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix+1, iy)));
		s_enemies[i_sxy]     = g_enemies[i_ixy];
	}

	if (lx == 0 && ly == 0) { //top
		i_sxy = sxy - sw - 1;
		i_ixy = ixy - IMAGEW - 1;
		s_strength_in[i_sxy] = (ix != 0 && iy != 0) ? strength_in[i_ixy] : 0;
		s_labels_in[i_sxy]   = labels_in[i_ixy];
		s_img[i_sxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix-1, iy-1)));
		s_enemies[i_sxy]     = g_enemies[i_ixy];
	}
	else if (lx == TILEW-1 && ly == TILEH-1) { //bottom
		i_sxy = sxy + sw + 1;
		i_ixy = ixy + IMAGEW + 1;
		s_strength_in[i_sxy] = (ix != IMAGEW-1 && iy != IMAGEH-1) ? strength_in[i_ixy] : 0;
		s_labels_in[i_sxy]   = labels_in[i_ixy];
		s_img[i_sxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix+1, iy+1)));
		s_enemies[i_sxy]     = g_enemies[i_ixy];
	}
	if (lx == 0 && ly == TILEH-1) { //left
		i_sxy = sxy - 1 + sw;
		i_ixy = ixy - 1 + IMAGEW;
		s_strength_in[i_sxy] = (ix != 0 && iy != IMAGEH-1) ? strength_in[i_ixy] : 0;
		s_labels_in[i_sxy]   = labels_in[i_ixy];
		s_img[i_sxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix-1, iy+1)));
		s_enemies[i_sxy]     = g_enemies[i_ixy];
	}
	else if (lx == TILEW-1 && ly == 0) { //right
		i_sxy = sxy + 1 - sw;
		i_ixy = ixy + 1 - IMAGEW;
		s_strength_in[i_sxy] = (ix != IMAGEW-1 && iy != 0) ? strength_in[i_ixy] : 0;
		s_labels_in[i_sxy]   = labels_in[i_ixy];
		s_img[i_sxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix+1, iy-1)));
		s_enemies[i_sxy]     = g_enemies[i_ixy];
	}

	if (lx < 9 && ly == 0)
		tile_flags[lx] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);

	float4 c = s_img[sxy];
	int label = s_labels_in[sxy];
	float defence = s_strength_in[sxy];

	float8 attack = (float8) ( 
		G_NORM(length(c - s_img[sxy-sw])) * s_strength_in[sxy-sw],
		G_NORM(length(c - s_img[sxy+sw])) * s_strength_in[sxy+sw],
		G_NORM(length(c - s_img[sxy-1])) * s_strength_in[sxy-1],
		G_NORM(length(c - s_img[sxy+1])) * s_strength_in[sxy+1],
		G_NORM(length(c - s_img[sxy-sw-1])) * s_strength_in[sxy-sw-1],
		G_NORM(length(c - s_img[sxy+sw+1])) * s_strength_in[sxy+sw+1],
		G_NORM(length(c - s_img[sxy-1+sw])) * s_strength_in[sxy-1+sw],
		G_NORM(length(c - s_img[sxy+1-sw])) * s_strength_in[sxy+1-sw]
	);

	if (s_enemies[sxy] > OVER_PROWER_THRESHOLD) {
		defence = FLT_MAX;

		int8 enemies = (int8) (
			s_labels_in[sxy-sw], s_labels_in[sxy+sw], s_labels_in[sxy-1], s_labels_in[sxy+1],
			s_labels_in[sxy-sw-1], s_labels_in[sxy+sw+1], s_labels_in[sxy-1+sw], s_labels_in[sxy+1-sw]
			) != label;

		if (attack.s0 < defence && enemies.s0) {
			defence = attack.s0;
			label = s_labels_in[sxy-sw];
		}

		if (attack.s1 < defence && enemies.s1) {
			defence = attack.s1;
			label = s_labels_in[sxy+sw];
		}

		if (attack.s2 < defence && enemies.s2) {
			defence = attack.s2;
			label = s_labels_in[sxy-1];
		}

		if (attack.s3 < defence && enemies.s3) {
			defence = attack.s3;
			label = s_labels_in[sxy+1];
		}

		if (attack.s4 < defence && enemies.s4) {
			defence = attack.s4;
			label = s_labels_in[sxy-sw-1];
		}

		if (attack.s5 < defence && enemies.s5) {
			defence = attack.s5;
			label = s_labels_in[sxy+sw+1];
		}

		if (attack.s6 < defence && enemies.s6) {
			defence = attack.s6;
			label = s_labels_in[sxy-1+sw];
		}

		if (attack.s7 < defence && enemies.s7) {
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

	strength_out[ixy] = defence;
	labels_out[ixy] = label;

	if (defence != s_strength_in[sxy] || label != s_labels_in[sxy]) {
		tile_flags[0] = TRUE;

		if (ly == 0)       tile_flags[1] = TRUE;
		if (ly == TILEH-1) tile_flags[2] = TRUE;
		if (lx == 0)       tile_flags[3] = TRUE;
		if (lx == TILEW-1) tile_flags[4] = TRUE;
		if (ly == 0 && lx == 0)             tile_flags[5] = TRUE;
		if (ly == TILEH-1 && lx == TILEW-1) tile_flags[6] = TRUE;
		if (lx == 0 && ly == TILEH-1)       tile_flags[7] = TRUE;
		if (lx == TILEW-1 && ly == 0)       tile_flags[8] = TRUE;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lx == 0 && ly == 0 && tile_flags[0] == TRUE)
		tiles[txy] = iteration;

	if (tile_flags[1]) tiles[txy-TILESW] = iteration;
	if (tile_flags[2]) tiles[txy+TILESW] = iteration;
	if (tile_flags[3]) tiles[txy-1] = iteration;
	if (tile_flags[4]) tiles[txy+1] = iteration;
	if (tile_flags[5]) tiles[txy-TILESW-1] = iteration;
	if (tile_flags[6]) tiles[txy+TILESW+1] = iteration;
	if (tile_flags[7]) tiles[txy-1+TILESW] = iteration;
	if (tile_flags[8]) tiles[txy+1-TILESW] = iteration;

}

#define IMAGEW 800
#define IMAGEH 608

__kernel void evolveVonNeumann(
	__global int* tiles_list,
	__global int* labels_in,
	__global int* labels_out,
	__global float* strength_in,
	__global float* strength_out,
	__global int* g_enemies,
	__global int* has_converge,
	int iteration,
	__global int* tiles,
	__local int* tile_flags, //true if any updates
	__local int* s_labels_in,
	__local float* s_strength_in,
	__local float4* s_img,
	__local int* s_enemies,
	__read_only image2d_t img,
	sampler_t sampler
)
{
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%TILESW;
	int ty = txy/TILESW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);

	//image coordinates
	int ix = tx*TILEW + lx;
	int iy = ty*TILEH + ly;
	int ixy = iy*IMAGEW + ix;

	int sw = TILEW + 2;
	int sx = 1 + lx;
	int sy = 1 + ly;
	int sxy = sy*sw + sx;

	s_labels_in[sxy]   = labels_in[ixy];
	s_strength_in[sxy] = strength_in[ixy];
	s_img[sxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix, iy)));
	s_enemies[sxy]     = g_enemies[ixy];

	int isxy, i_ixy;

	//load padding
	if (ly == 0) { //top
		isxy = sxy - sw;
		i_ixy = ixy - IMAGEW;
		s_strength_in[isxy] = (iy != 0) ? strength_in[i_ixy] : 0;
		s_labels_in[isxy]   = labels_in[i_ixy];
		s_img[isxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix, iy-1)));
		s_enemies[isxy]     = g_enemies[i_ixy];
	}
	else if (ly == TILEH-1) { //bottom
		isxy = sxy + sw;
		i_ixy = ixy + IMAGEW;
		s_strength_in[isxy] = (iy != IMAGEH-1) ? strength_in[i_ixy] : 0;
		s_labels_in[isxy]   = labels_in[i_ixy];
		s_img[isxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix, iy+1)));
		s_enemies[isxy]     = g_enemies[i_ixy];
	}
	if (lx == 0) { //left
		isxy = sxy - 1;
		i_ixy = ixy - 1;
		s_strength_in[isxy] = (ix != 0) ? strength_in[i_ixy] : 0;
		s_labels_in[isxy]   = labels_in[i_ixy];
		s_img[isxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix-1, iy)));
		s_enemies[isxy]     = g_enemies[i_ixy];
	}
	else if (lx == TILEW-1) { //right
		isxy = sxy + 1;
		i_ixy = ixy + 1;
		s_strength_in[isxy] = (ix != IMAGEW-1) ? strength_in[i_ixy] : 0;
		s_labels_in[isxy]   = labels_in[i_ixy];
		s_img[isxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix+1, iy)));
		s_enemies[isxy]     = g_enemies[i_ixy];
	}

	if (lx < 5 && ly == 0)
		tile_flags[lx] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);

	float4 c = s_img[sxy];
	int label = s_labels_in[sxy];
	float defence = s_strength_in[sxy];

	float4 attack = (float4) (
		G_NORM(length(c - s_img[sxy-sw])) * s_strength_in[sxy-sw],
		G_NORM(length(c - s_img[sxy+sw])) * s_strength_in[sxy+sw],
		G_NORM(length(c - s_img[sxy-1])) * s_strength_in[sxy-1],
		G_NORM(length(c - s_img[sxy+1])) * s_strength_in[sxy+1]
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

	strength_out[ixy] = defence;
	labels_out[ixy] = label;

	if (defence != s_strength_in[sxy] || label != s_labels_in[sxy]) {
		tile_flags[0] = TRUE;

		if (iy != 0 && ly == 0)            tile_flags[1] = TRUE;
		if (iy != IMAGEH && ly == TILEH-1) tile_flags[2] = TRUE;
		if (ix != 0 && lx == 0)            tile_flags[3] = TRUE;
		if (ix != IMAGEW && lx == TILEW-1) tile_flags[4] = TRUE;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lx == 0 && ly == 0 && tile_flags[0] == TRUE)
		tiles[txy] = iteration;

	if (tile_flags[1]) tiles[txy-TILESW] = iteration;
	if (tile_flags[2]) tiles[txy+TILESW] = iteration;
	if (tile_flags[3]) tiles[txy-1] = iteration;
	if (tile_flags[4]) tiles[txy+1] = iteration;
}