#define TRUE 1
#define FALSE 0

#define imin(a, b) (a < b) ? a : b

float weight(uint c1, uint c2);

#define WAVE_BREDTH 32
#define WAVE_LENGTH 8
#define WAVES_PER_WORKGROUP 4
#define WORKGROUP_LENGTH (WAVES_PER_WORKGROUP * WAVE_LENGTH)

#define BETA 1.0f/10

#define IMAGEW 800
#define IMAGEH 608

#define NWAVES (IMAGEH/WAVE_LENGTH)

__kernel void check_completion(
	__global int* tiles_list,
	__global float* excess,
	__global int* bfs,
	__local int* tile_has_active_nodes,
	__global int* isCompleted
)
{
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%TILESW;
	int ty = txy/TILESW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);

	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	int ixy = iy*IMAGEW + ix;

	if (lx == 0 && ly == 0)
		tile_has_active_nodes[0] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i=0; i<WAVE_LENGTH; i++) {
		if (excess[ixy] > 0 && bfs[ixy] < MAX_HEIGHT) { //looking for active nodes
			tile_has_active_nodes[0] = TRUE;
		}

		ixy += IMAGEW;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lx == 0 && ly == 0) {
		if (tile_has_active_nodes[0]) //tile has active nodes
			isCompleted[0] = FALSE;
	}
}

//work straight from the excess image - this is just a place holder for now
__kernel void init_gc(
	__global float* up,
	__global float* down,
	__global float* left,
	__global float* right,
	__global float* excesses,
	__local int* tile_flags, //tile has nodes with excess >= 0
	__global int* tilesLoad,
	int iteration
) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*IMAGEW + gx;

	int lx = get_local_id(0);
	int ly = get_local_id(0);

	if (ly == 0) tile_flags[0] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);

	int ixy = WAVE_LENGTH*gy*IMAGEW + gx;

	float excess;

	for (int i=0; i<WAVE_LENGTH; i++) {
		up[ixy] = 0;
		down[ixy] = 0;
		left[ixy] = 0;
		right[ixy] = 0;

		//bfs needs all tiles with excess >= 0
		if (excesses[ixy] >= 0)
			tile_flags[0] = TRUE;

		ixy += IMAGEW;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lx == 0 && ly == 0) {
		int tx = get_group_id(0);
		int ty = get_group_id(1);
		int txy = ty*TILESW + tx;

		if (tile_flags[0]) {
			tilesLoad[txy] = iteration;
		}
	}
}

__kernel void add_border(
	__global int* tiles_list,
	__global float* border,
	__global float* excess,
	int direction
)
{
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%TILESW;
	int ty = txy/TILESW;

	int lx = get_local_id(0);

	int ix = tx*WAVE_BREDTH + lx;
	int iy = ty;
	int ixy = iy*IMAGEW + ix;

	int ix2, iy2, ixy2;

	switch(direction) {
		case 0:
			ix2 = ix;
			iy2 = ty*WORKGROUP_LENGTH;
		break;
		case 1:
			ix2 = ix;
			iy2 = (ty+1)*WORKGROUP_LENGTH - 1;
		break;
		case 2:
			ix2 = tx*WORKGROUP_LENGTH;
			iy2 = ty*WAVE_BREDTH + lx;
		break;
		case 3: //flow received from tile to right
			ix2 = (tx+1)*WORKGROUP_LENGTH - 1;
			iy2 = ty*WAVE_BREDTH + lx;
		break;
	}

	ixy2 = iy2*IMAGEW + ix2;

	excess[ixy2] += border[ixy];
}

__kernel void push_left(
	__global int* tiles_list,
	__global float* excess,
	__local float* excess_s,
	__global int* height,
	__local int* height_s,
	__global float* right,
	__global float* left,
	__global float* border,
	__local int* flags,
	__global int* border_tiles,
	int iteration
)
{
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%TILESW;
	int ty = txy/TILESW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	int lw = get_local_size(0);
	int lh = get_local_size(1);
	
	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	int ixy = iy*IMAGEW + ix;

	int iwS = lw + 1;
	int ixS = lx;
	int iyS = ly*WAVE_LENGTH;
	int ixyS = iyS*iwS + ixS; 

	if (lx == 0 && ly == 0)
		flags[0] = FALSE;

	for (int i=0; i<WAVE_LENGTH; i++) {
		excess_s[ixyS] = excess[ixy];
		height_s[ixyS + 1] = height[ixy];

		ixy += IMAGEW;
		ixyS += iwS; //padding to prevent bank conflicts
	}

	//load first row of heights from next tile
	if (ly == lh-1) {
		int ix2 = tx*WAVE_BREDTH - 1;
		int iy2 = ty*WORKGROUP_LENGTH + lx;
		ixy = iy2*IMAGEW + ix2;

		height_s[lx*iwS + 0] = height[ixy];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	iy = (WAVES_PER_WORKGROUP*ty + ly+1)*WAVE_LENGTH - 1;
	ixy = iy*IMAGEW + ix;

	ixS = (ly+1)*WAVE_LENGTH - 1;
	iyS = lx;
	ixyS = iyS*iwS + ixS; 
	
	float ef = 0;
	float flow = 0;
	int h = height_s[ixyS + 1];
	int hNext;

	flow = 0;

	for (int i=0; i<WAVE_LENGTH; i++) {
		ef = excess_s[ixyS] + flow;
		hNext = height_s[ixyS +1 - 1]; //+1 for offset, -1 for going left

		if (ef > 0 && h < MAX_HEIGHT && h == hNext + 1) {
			flow = min(ef, left[ixy]);
		}
		else
			flow = 0;

		left[ixy] -= flow;
		excess_s[ixyS] = ef-flow;

		if (ly == 0 && i == WAVE_LENGTH-1)
			right[ixy - 32 + 31*IMAGEW] += flow;
		else
			right[ixy-IMAGEW] += flow;

		h = hNext;
		ixy -= IMAGEW;
		ixyS -= 1;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ly != 0)
		excess_s[ixyS] += flow;
	else if (flow > 0) // does the tile-overspilling wave have excess flow?
		flags[0] = TRUE;

	barrier(CLK_LOCAL_MEM_FENCE);

	iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	ixy = iy*IMAGEW + ix;

	iwS = lw + 1;
	ixS = lx;
	iyS = ly*WAVE_LENGTH;
	ixyS = iyS*iwS + ixS; 

	for (int i=0; i<WAVE_LENGTH; i++) {
		excess[ixy] = excess_s[ixyS];

		ixy += IMAGEW;
		ixyS += iwS;
	}

	if (ly == 0 && flags[0] == TRUE) {
		border[ty*IMAGEW + ix - WAVE_BREDTH] = flow;
		border_tiles[txy - 1] = iteration;
	}
}


__kernel void push_right(
	__global int* tiles_list,
	__global float* excess,
	__local float* excess_s,
	__global int* height,
	__local int* height_s,
	__global float* right,
	__global float* left,
	__global float* border,
	__local int* flags,
	__global int* border_tiles,
	int iteration
)
{
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%TILESW;
	int ty = txy/TILESW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	int lw = get_local_size(0);
	int lh = get_local_size(1);
	
	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	int ixy = iy*IMAGEW + ix;

	int iwS = lw + 1;
	int ixS = lx;
	int iyS = ly*WAVE_LENGTH;
	int ixyS = iyS*iwS + ixS; 

	if (lx == 0 && ly == 0)
		flags[0] = FALSE;

	for (int i=0; i<WAVE_LENGTH; i++) {
		excess_s[ixyS] = excess[ixy];
		height_s[ixyS] = height[ixy];

		ixy += IMAGEW;
		ixyS += iwS; //padding to prevent bank conflicts
	}

	//load first row of heights from next tile
	if (ly == lh-1) {
		int ix2 = (tx+1)*WAVE_BREDTH;
		int iy2 = ty*WORKGROUP_LENGTH + lx;
		ixy = iy2*IMAGEW + ix2;

		height_s[lx*iwS + lw] = height[ixy];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	ixy = iy*IMAGEW + ix;

	ixS = ly*WAVE_LENGTH;
	iyS = lx;
	ixyS = iyS*iwS + ixS; 
	
	float ef = 0;
	float flow = 0;
	int h = height_s[ixyS];
	int hNext;

	flow = 0;

	for (int i=0; i<WAVE_LENGTH; i++) {
		ef = excess_s[ixyS] + flow;
		hNext = height_s[ixyS+1];

		if (ef > 0 && h < MAX_HEIGHT && h == hNext + 1) {
			flow = min(ef, right[ixy]);
		}
		else
			flow = 0;
		
		right[ixy] -= flow;
		excess_s[ixyS] = ef-flow;

		if (ly == 3 && i == WAVE_LENGTH-1)
			left[ixy + 32 - 31*IMAGEW] += flow;
		else
			left[ixy+IMAGEW] += flow;

		h = hNext;
		ixy += IMAGEW;
		ixyS += 1;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ly != lh-1)
		excess_s[ixyS] += flow;
	else if (flow > 0) // does the tile-overspilling wave have excess flow?
		flags[0] = TRUE;

	barrier(CLK_LOCAL_MEM_FENCE);

	ixy = iy*IMAGEW + ix;

	iwS = lw + 1;
	ixS = lx;
	iyS = ly*WAVE_LENGTH;
	ixyS = iyS*iwS + ixS; 

	for (int i=0; i<WAVE_LENGTH; i++) {
		excess[ixy] = excess_s[ixyS];

		ixy += IMAGEW;
		ixyS += iwS;
	}

	if (ly == lh-1 && flags[0] == TRUE) {
		border[ty*IMAGEW + ix + WAVE_BREDTH] = flow;
		border_tiles[txy + 1] = iteration;
	}
}

__kernel void push_down(
	__global int* tiles_list,
	__global float* down,
	__global float* up,
	__global int* height,
	__global float* excess,
	__global float* border,
	__local int* flags,
	__global int* border_tiles,
	int iteration
)
{
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%TILESW;
	int ty = txy/TILESW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	int lh = get_local_size(1);

	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	int ixy = iy*IMAGEW + ix;

	if (lx == 0 && ly == 0)
		flags[0] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);
	
	float ef = 0;
	float flow = 0;
	int h = height[ixy];
	int hNext;
	
	for (int i=0; i<WAVE_LENGTH; i++) {
		ef = excess[ixy] + flow;
		hNext = height[ixy+IMAGEW];

		if (ef > 0 && h < MAX_HEIGHT && h == hNext + 1) {
			flow = min(ef, down[ixy]);
		}
		else
			flow = 0;
		
		down[ixy] -= flow;
		excess[ixy] = ef-flow;
		up[ixy+IMAGEW] += flow;

		h = hNext;
		ixy += IMAGEW;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ly != lh-1)
		excess[ixy] += flow;
	else if (flow > 0) // does the tile-overspilling wave have excess flow?
		flags[0] = TRUE;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ly == lh-1 && flags[0] == TRUE) {
		border[ty*IMAGEW + ix + IMAGEW] = flow;
		border_tiles[txy + TILESW] = iteration;
	}
}

__kernel void push_up(
	__global int* tiles_list,
	__global float* down,
	__global float* up,
	__global int* height,
	__global float* excess,
	__global float* border,
	__local int* flags,
	__global int* border_tiles,
	int iteration
)
{
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%TILESW;
	int ty = txy/TILESW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	
	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly+1)*WAVE_LENGTH - 1;
	int ixy = iy*IMAGEW + ix;
	
	if (lx == 0 && ly == 0)
		flags[0] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);
	
	float ef = 0;
	float flow = 0;
	int h = height[ixy];
	int hNext;

	for (int i=WAVE_LENGTH-1; i>=0; i--) {
		ef = excess[ixy] + flow;
		hNext = height[ixy-IMAGEW];

		if (ef > 0 && h < MAX_HEIGHT && h == hNext + 1)
			flow = min(ef, up[ixy]);
		else
			flow = 0;
		
		up[ixy] -= flow;
		excess[ixy] = ef-flow;
		down[ixy-IMAGEW] += flow;

		h = hNext;
		ixy -= IMAGEW;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ly != 0)
		excess[ixy] += flow;
	else if (flow > 0) // does the tile-overspilling wave have excess flow?
		flags[0] = TRUE;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ly == 0 && flags[0] == TRUE) {
		border[ty*IMAGEW + ix - IMAGEW] = flow;
		border_tiles[txy - TILESW] = iteration;
	}
}

__kernel void relabel(
	__global float* down,
	__global float* right,
	__global float* up,
	__global float* left,
	__global float* excess,
	__global int* height,
	__global int* height2
)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*IMAGEW + gx;

	int gxT = (get_group_id(0)/2)*WAVE_BREDTH + gy%32;
	int gyT = (get_group_id(1)/2)*WORKGROUP_LENGTH + gx%32;
	int gxyT = gyT*IMAGEW + gxT;

	int h = height[gxy];
	int newHeight = h;

	if (excess[gxy] > 0 && h < MAX_HEIGHT) {
		newHeight = MAX_HEIGHT;

		if (gy < IMAGEH-1 && down[gxy] > 0)
			newHeight = imin(newHeight, height[gxy+IMAGEW]+1);

		if (gy > 0 && up[gxy] > 0)
			newHeight = imin(newHeight, height[gxy-IMAGEW]+1);

		if (gx < IMAGEW-1 && right[gxyT] > 0)
			newHeight = imin(newHeight, height[gxy+1]+1);

		if (gx > 0 && left[gxyT] > 0)
			newHeight = imin(newHeight, height[gxy-1]+1);
	}

	height2[gxy] = newHeight;
	return;
}

//compact up,down,left,right residual weights
//handle intertile gaps, flag tiles with discrepancies
//flag tiles with intratile gaps
__kernel void init_bfs(
	__global int* tiles_list,
	__global float* excess,
	__local float* excess_s,
	__global int* bfs,
	__local int* tile_flags, //0:has nodes with excess < 0, 1: nodes with excess >= 0
	__global float* down,
	__global float* up,
	__global float* right,
	__global float* left,
	__global uint* can_downs,
	__global uint* can_ups,
	__global uint* can_rights,
	__global uint* can_lefts,
	__local uchar* can_downs_s,
	__local uchar* can_ups_s,
	__local uchar* can_rights_s,
	__local uchar* can_lefts_s,
	__global int* bfs_tiles,
	int iteration
)
{
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%TILESW;
	int ty = txy/TILESW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	int lw = get_local_size(0);
	
	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	int ixy = iy*IMAGEW + ix;

	int iwS = lw;
	int ixS = lx;
	int iyS = ly*WAVE_LENGTH;
	int ixyS = iyS*iwS + ixS; 

	for (int i=0; i<WAVE_LENGTH; i++) {
		excess_s[ixyS] = excess[ixy];

		ixy += IMAGEW;
		ixyS += iwS;
	}

	int ix2, iy2, ixy2;

	if (lx < 6) tile_flags[lx] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);
	
	uchar can_down = 0;
	uchar can_up = 0;
	uchar can_right = 0;
	uchar can_left = 0;
	int dist;

	ixy = iy*IMAGEW + ix;
	ixyS = iyS*iwS + ixS;

	for (int i=0; i<WAVE_LENGTH; i++) {
		if (excess_s[ixyS] >= 0) {
			dist = MAX_HEIGHT;
			tile_flags[1] = TRUE;
		}
		else {
			dist = 0;	
			tile_flags[0] = TRUE;
		}

		bfs[ixy] = dist;	

		if (down[ixy]  > 0) can_down  |= 128; //10000000 in binary 
		if (up[ixy]    > 0) can_up    |= 128;
		if (right[ixy] > 0) can_right |= 128;
		if (left[ixy]  > 0) can_left  |= 128;

		if (i != WAVE_LENGTH-1) {
			can_down  >>= 1;
			can_up    >>= 1;
			can_right >>= 1;
			can_left  >>= 1;
		}

		ixy += IMAGEW;
		ixyS += iwS;
	}

	can_downs_s[ly*WAVE_BREDTH + lx] = can_down;
	can_ups_s[ly*WAVE_BREDTH + lx] = can_up;
	can_rights_s[ly*WAVE_BREDTH + lx] = can_right;
	can_lefts_s[ly*WAVE_BREDTH + lx] = can_left;

	barrier(CLK_LOCAL_MEM_FENCE);

	float e1, e2;

	switch(ly) {
		case 0: //up
			if (ty == 0) break;

			ix2 = ix;
			iy2 = ty*WORKGROUP_LENGTH;
			ixy2 = iy2*IMAGEW + ix2;

			ixS = lx;
			iyS = 0;
			ixyS = iyS*iwS + ixS;

			e1 = excess_s[ixyS];
			e2 = excess[ixy2-IMAGEW];

			// no need to check the reverse i.e e2 >= 0 and e1 < 0
			// if e2 >= 0 was true the tile above would be processed
			if ((can_ups_s[lx] & 1) && e1 >= 0 && e2 < 0) { // & 1 -> check first bit to see if possible to push up
				bfs[ixy2] = 1; // dist of 1 away from sink
				bfs_tiles[txy] = iteration;
			}
			break;

		case 1: //down
			if (ty == TILESH-1) break;

			ix2 = ix;
			iy2 = (ty+1)*WORKGROUP_LENGTH - 1;
			ixy2 = iy2*IMAGEW + ix2;

			ixS = lx;
			iyS = WORKGROUP_LENGTH-1;
			ixyS = iyS*iwS + ixS;

			e1 = excess_s[ixyS];
			e2 = excess[ixy2+IMAGEW];

			if (((can_downs_s[3*WAVE_BREDTH + lx] & 128) == 128) && e1 >= 0 && e2 < 0) {
				bfs[ixy2] = 1;
				bfs_tiles[txy] = iteration;
			}
			break;

		case 2: //right
			if (tx == TILESW-1) break;

			ix2 = (tx+1)*WAVE_BREDTH - 1;
			iy2 = ty*WORKGROUP_LENGTH + lx;
			ixy2 = iy2*IMAGEW + ix2;

			ixS = WAVE_BREDTH-1;
			iyS = lx;
			ixyS = iyS*iwS + ixS;

			e1 = excess_s[ixyS];
			e2 = excess[ixy2 + 1];

			if (((can_rights_s[3*WAVE_BREDTH + lx] & 128) == 128) && e1 >= 0 && e2 < 0) {
				bfs[ixy2] = 1;
				bfs_tiles[txy] = iteration;
			}
			break;
	
		case 3: //left
			if (ty == 0) break;

			ix2 = tx*WAVE_BREDTH;
			iy2 = ty*WORKGROUP_LENGTH + lx;
			ixy2 = iy2*IMAGEW + ix2;

			ixS = 0;
			iyS = lx;
			ixyS = iyS*iwS + ixS;

			e1 = excess_s[ixyS];
			e2 = excess[ixy2 - 1];

			if ((can_lefts_s[lx] & 1) && e1 >= 0 && e2 < 0) {
				bfs[ixy2] = 1;
				bfs_tiles[txy] = iteration;
			}
			break;
	} 

	if (lx == 0 && ly == 0) {
		if (tile_flags[0] && tile_flags[1]) //has intratile gaps
			bfs_tiles[txy] = iteration;
	}

	if (ly == 0) {
		can_downs[ty*IMAGEW + ix] =
			(can_downs_s[3*WAVE_BREDTH + lx] << 24) |
			(can_downs_s[2*WAVE_BREDTH + lx] << 16) |
			(can_downs_s[1*WAVE_BREDTH + lx] << 8 ) |
			can_downs_s[lx];
	}

	if (ly == 1) {
		can_ups[ty*IMAGEW + ix] =
			(can_ups_s[3*WAVE_BREDTH + lx] << 24) |
			(can_ups_s[2*WAVE_BREDTH + lx] << 16) |
			(can_ups_s[1*WAVE_BREDTH + lx] << 8 ) |
			can_ups_s[lx];
	}

	if (ly == 2) {
		can_rights[ty*IMAGEW + ix] =
			(can_rights_s[3*WAVE_BREDTH + lx] << 24) |
			(can_rights_s[2*WAVE_BREDTH + lx] << 16) |
			(can_rights_s[1*WAVE_BREDTH + lx] << 8 ) |
			can_rights_s[lx];
	}

	if (ly == 3) {
		can_lefts[ty*IMAGEW + ix] =
			(can_lefts_s[3*WAVE_BREDTH + lx] << 24) |
			(can_lefts_s[2*WAVE_BREDTH + lx] << 16) |
			(can_lefts_s[1*WAVE_BREDTH + lx] << 8 ) |
			can_lefts_s[lx];
	}
}

__kernel void bfs_intertile(
	__global int* tiles_list,
	__local int* flags,
	__global int* bfs,
	__global uint* can_downs,
	__global uint* can_ups,
	__global uint* can_rights,
	__global uint* can_lefts,
	int iteration,
	__global int* bfs_tiles
) {
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%TILESW;
	int ty = txy/TILESW;

	int lx = get_local_id(0);
	int lw = get_local_size(0);

	if (lx < 3) flags[lx] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);

	//intertile gap with tile above
	int ixy = (ty*WORKGROUP_LENGTH)*IMAGEW + (tx*WAVE_BREDTH + lx); //index into image
	int ixy2 = ty*IMAGEW + (tx*WAVE_BREDTH + lx); //index into compressed edges

	bool can = can_ups[ixy2] & 0x1;
	bool can2 = (can_downs[ixy2-IMAGEW] >> WORKGROUP_LENGTH-1) & 0x1;

	int dist = bfs[ixy];
	int dist2 = bfs[ixy-IMAGEW];

	if(can && dist2+1 < dist) {
		bfs[ixy] = dist2+1;
		flags[0] = TRUE;
	}
	if(can2 && dist+1 < dist2) {
		bfs[ixy-IMAGEW] = dist+1;
		flags[1] = TRUE;
	}

	//intertile gap with tile to left
	can = can_lefts[ixy2] & 0x1;
	can2 = (can_rights[ixy2-WAVE_BREDTH] >> WORKGROUP_LENGTH-1) & 0x1;

	ixy = (ty*WORKGROUP_LENGTH + lx)*IMAGEW + tx*WAVE_BREDTH;

	dist = bfs[ixy];
	dist2 = bfs[ixy-1];

	if(can && dist2+1 < dist) { //up and left checked by same block
		bfs[ixy] = dist2+1;
		flags[0] = TRUE;
	}
	if(can2 && dist+1 < dist2) {
		bfs[ixy-1] = dist+1;
		flags[2] = TRUE;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if      (lx == 0 && flags[0] == TRUE) bfs_tiles[txy]        = iteration;
	else if (lx == 1 && flags[1] == TRUE) bfs_tiles[txy-TILESW] = iteration;
	else if (lx == 2 && flags[2] == TRUE) bfs_tiles[txy-1]      = iteration;
}


__kernel void bfs_intratile(
	__global int* active_tiles,
	__global int* bfs,
	__local int* bfs_s,
	__global uint* can_downs,
	__global uint* can_ups,
	__global uint* can_rights,
	__global uint* can_lefts,
	__local int* _l_didChange,
	__global int* bfs_edges,
	int iteration
)
{
	int txy = active_tiles[get_group_id(0)];
	int tx = txy%TILESW;
	int ty = txy/TILESW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	int lw = get_local_size(0);

	int sw = lw + 2;

	int ixy = (ty*WORKGROUP_LENGTH + ly*WAVE_LENGTH)*IMAGEW + tx*WAVE_BREDTH + lx;
	int sxy = (ly*WAVE_LENGTH + 1)*sw + lx + 1;
	
	//load bfs distances into shared mem
	//MAX_HEIGHT filled padding used to not worry about boundary issues later on
	//and eliminate bank conflicts
	for (int i=0; i<WAVE_LENGTH; i++) {
		bfs_s[sxy] = bfs[ixy]; //height is offeset by 1 in shared mem

		ixy += IMAGEW;
		sxy += sw;
	}

	switch(ly) {
		case 0: bfs_s[1+lx] = MAX_HEIGHT; break; //fill top padding
		case 1: bfs_s[(1+WORKGROUP_LENGTH)*sw + 1+lx] = MAX_HEIGHT; break; //bottom
		case 2: bfs_s[(1+lx)*sw] = MAX_HEIGHT; break; //left
		case 3: bfs_s[(1+lx)*sw + sw-1] = MAX_HEIGHT; break; //right
		//dont need corners - only using 4 neightbourhood
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int bxy = ty*IMAGEW + tx*WAVE_BREDTH + lx;
	int bxyT = tx*IMAGEH + ty*WAVE_BREDTH + lx;

	uint can_down  = (can_downs[bxy]  >> ly*WAVE_LENGTH);
	uint can_up    = (can_ups[bxy]    >> ly*WAVE_LENGTH);
	uint can_right = (can_rights[bxy] >> ly*WAVE_LENGTH);
	uint can_left  = (can_lefts[bxy]  >> ly*WAVE_LENGTH);
	
	int _p_didChange = FALSE;
	_l_didChange[0] = TRUE;
	int i = 0;
	while(_l_didChange[0] && i < 30) {
		_p_didChange = FALSE;
		_l_didChange[0] = FALSE;

		sxy = (1 + ly*WAVE_LENGTH)*sw + lx + 1;

		uint mask = 1; //00000001 in binary

		for (int i=0; i<WAVE_LENGTH; i++) {
			if ((can_up & mask) == mask && bfs_s[sxy-sw]+1 < bfs_s[sxy]) {
				bfs_s[sxy] = bfs_s[sxy-sw]+1;
				_p_didChange = TRUE;
			}

			mask <<= 1;

			sxy += sw;
		}

		sxy -= sw;

		//note that the bit sequence is in reverse for down and right
		mask = 128; //10000000 in binary

		for (int i=0; i<WAVE_LENGTH; i++) {
			if ((can_down & mask) == mask && bfs_s[sxy+sw]+1 < bfs_s[sxy]) {
				bfs_s[sxy] = bfs_s[sxy+sw]+1;
				_p_didChange = TRUE;
			}

			mask >>= 1;

			sxy -= sw;
		}

		sxy = (1 + lx)*sw + (1 + ly*WAVE_LENGTH);
		mask = 1;
		for (int i=0; i<WAVE_LENGTH; i++) {
			if ((can_left & mask) == mask && bfs_s[sxy-1]+1 < bfs_s[sxy]) {
				bfs_s[sxy] = bfs_s[sxy-1]+1;
				_p_didChange = TRUE;
			}

			mask <<= 1;

			sxy += 1;
		}

		sxy -= 1;

		mask = 128;

		for (int i=0; i<WAVE_LENGTH; i++) {
			if ((can_right & mask) == mask && bfs_s[sxy+1]+1 < bfs_s[sxy]) {
				bfs_s[sxy] = bfs_s[sxy+1]+1;
				_p_didChange = TRUE;
			}

			mask >>= 1;

			sxy -= 1;
		}
		
		if (_p_didChange) _l_didChange[0] = TRUE;

		barrier(CLK_LOCAL_MEM_FENCE);

		i++;
	}
	i--;

	if (lx ==0 && ly == 0) {
		if (i>0) {
			bfs_edges[txy] = iteration;
			bfs_edges[txy + TILESW] = iteration;
			bfs_edges[txy + 1] = iteration;
		}
	}

	ixy = (ty*WORKGROUP_LENGTH + ly*WAVE_LENGTH)*IMAGEW + tx*WAVE_BREDTH + lx;
	sxy = (ly*WAVE_LENGTH + 1)*sw + lx + 1;

	for (int i=0; i<WAVE_LENGTH; i++) {
		bfs[ixy] = bfs_s[sxy];

		ixy += IMAGEW;
		sxy += sw;
	}
}

float weight(uint c1, uint c2) {
	uint r1, r2, g1, g2, b1, b2;
		
	r1 = (c1 & 0x000000FF);
	g1 = (c1 & 0x0000FF00) >> 8;
	b1 = (c1 & 0x00FF0000) >> 16;

	r2 = (c2 & 0x000000FF);
	g2 = (c2 & 0x0000FF00) >> 8;
	b2 = (c2 & 0x00FF0000) >> 16;
	
	//return LAMBA*exp(WEIGHT_NORM*((r2-r1)*(r2-r1) + (g2-g1)*(g2-g1) + (b2-b1)*(b2-b1)));	
	return 1.0f / (BETA * sqrt((float) ((r2-r1)*(r2-r1) + (g2-g1)*(g2-g1) + (b2-b1)*(b2-b1))) + EPSILON);
}

__kernel void load_tiles(
	__global int* tiles_list,
	__global uint* img,
	__global float* up,
	__global float* down,
	__global float* left,
	__global float* right,
	__local uint* _l_img
) {
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%TILESW;
	int ty = txy/TILESW;

	int lw = get_local_size(0);
	int lh = get_local_size(1);
	int lx = get_local_id(0);
	int ly = get_local_id(1);

	int _g_img_w = IMAGEW;
	int _g_img_x = tx*WAVE_BREDTH + lx;
	int _g_img_y = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	int _g_img_xy = _g_img_y*_g_img_w + _g_img_x;

	//load 1 pixel thick top and left padded tile of img data,
	int _l_img_w = lw+2;
	int _l_img_x = 1 + lx;
	int _l_img_y = 1 + ly*WAVE_LENGTH;
	int _l_img_xy = _l_img_y*_l_img_w + _l_img_x;
	
	for (int i=0; i<WAVE_LENGTH; i++) {
		_l_img[_l_img_xy] = img[_g_img_xy];

		_l_img_xy += _l_img_w;
		_g_img_xy += _g_img_w;
	}

	//dont check for boundaries, padding can be filled with garbage, we'll replace it later anyways
	switch(ly) {
		case 0: //fill top padding
			_l_img[(1 + lx)] = img[(ty*WORKGROUP_LENGTH-1)*_g_img_w + _g_img_x];
			break;
		case 1: //bottom
			_l_img[(1 + WORKGROUP_LENGTH)*_l_img_w + (1 + lx)] = img[((ty+1)*WORKGROUP_LENGTH)*_g_img_w + _g_img_x];
			break;
		case 2: //left
			_l_img[(1 + lx)*_l_img_w]	= img[(ty*WORKGROUP_LENGTH+lx)*_g_img_w + tx*WAVE_BREDTH-1];
			break;
		case 3: //right
			_l_img[(1 + lx)*_l_img_w + (1 + WAVE_BREDTH)]	= img[(ty*WORKGROUP_LENGTH+lx)*_g_img_w + (tx+1)*WAVE_BREDTH];
			break;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	float term;
	
	//up and down weights
	_g_img_xy = _g_img_y*_g_img_w + _g_img_x;
	_l_img_xy = _l_img_y*_l_img_w + _l_img_x;

	for (int i=0; i<WAVE_LENGTH; i++, _l_img_xy += _l_img_w, _g_img_xy += _g_img_w) {
		term = LAMBDA*weight(_l_img[_l_img_xy], _l_img[_l_img_xy-_l_img_w]);

		up[_g_img_xy] += term;
		if (ly == 0 && i == 0) continue;
		down[_g_img_xy-_g_img_w] += term;
	}

	if (ly == lh-1) {
		term = LAMBDA*weight(_l_img[_l_img_xy], _l_img[_l_img_xy-_l_img_w]);
		down[_g_img_xy-_g_img_w] = term;
	}

	if (ty == 0 && ly == 0) up[_g_img_x] = 0;
	if (ty == TILESH-1 && ly == lh-1) down[_g_img_xy-_g_img_w] = 0;

	//left and right weights
	_g_img_xy = _g_img_y*_g_img_w + _g_img_x;
	_l_img_xy = _l_img_x*_l_img_w + _l_img_y;

	for (int i=0; i<WAVE_LENGTH; i++, _l_img_xy += 1, _g_img_xy += _g_img_w) {
		if (tx == 0 && ly == 0 && i == 0)
			term = 0;
		else
			term = LAMBDA*weight(_l_img[_l_img_xy], _l_img[_l_img_xy-1]);

		left[_g_img_xy] += term;
		if (ly == 0 && i == 0) continue;
		right[_g_img_xy-_g_img_w] += term;
	}

	if (tx == TILESW-1 && ty == TILESH-1) {
		right[_g_img_xy-_g_img_w] = 0;
	}
	else if (ly == lh-1) {
		term = LAMBDA*weight(_l_img[_l_img_xy], _l_img[_l_img_xy-1]);
		right[_g_img_xy-_g_img_w] = term;
	}
}
