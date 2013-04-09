#define TRUE 1
#define FALSE 0

#define imin(a, b) (a < b) ? a : b

float weight(uint c1, uint c2);

#define WAVE_BREDTH 32
#define WAVE_LENGTH 8
#define WAVES_PER_WORKGROUP 4
#define WORKGROUP_LENGTH (WAVES_PER_WORKGROUP * WAVE_LENGTH)

#define BETA 1.0f/10

#define EXCESS_ZERO 0
#define EXCESS_LARGER_THAN_ZERO 2
#define EXCESS_LESS_THAN_ZERO 3

__kernel void checkCompletion(
	__global float* excesses,
	__global int* height,
	__local int* tile_has_active_nodes, //tile has active nodes
	__global int* isCompleted
){
	int gw = get_global_size(0);
	int gh = get_global_size(1);
	int gs = gw*gh;
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*gw + gx;

	int lx = get_local_id(0);
	int ly = get_local_id(0);

	if (ly == 0 && lx == 0)
		tile_has_active_nodes[0] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);

	int ixy = WAVE_LENGTH*gy*gw + gx;

	for (int i=0; i<WAVE_LENGTH; i++) {
		if (excesses[ixy] > 0 && height[ixy] < MAX_HEIGHT)
			tile_has_active_nodes[0] = TRUE;

		ixy += gw;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lx == 0 && ly == 0) {
		if (tile_has_active_nodes[0])
			isCompleted[0] = FALSE;
	}
}


//work straight from the excess image - this is just a place holder for now
__kernel void init_gc(
	__global float* src,
	__global float* sink,
	__local int* tile_flags, //tile has nodes with excess >= 0
	__global int* gc_tiles,
	int2 gc_tilesPad,
	int iteration,
	__global float* excesses
) {
	int gw = get_global_size(0);
	int gh = get_global_size(1);
	int gs = gw*gh;
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*gw + gx;

	int lx = get_local_id(0);
	int ly = get_local_id(0);

	if (ly == 0) tile_flags[0] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);

	int ixy = WAVE_LENGTH*gy*gw + gx;

	float excess;
	
	for (int i=0; i<WAVE_LENGTH; i++) {
		excess = sink[ixy] - src[ixy];

		if (excess >= 0) tile_flags[0] = TRUE;

		excesses[ixy] = excess;

		ixy += gw;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lx == 0 && ly == 0) {
		int tx = get_group_id(0);
		int ty = get_group_id(1);
		int txy = ty*gc_tilesPad.x + tx;

		if (tile_flags[0]) {
			gc_tiles[txy] = iteration;
		}
	}
}


__kernel void initNeighbourhood(
	__global int* active_tiles_list,
	int active_tilesW,
	int active_tilesH,
	__global uint* img,
	__global float* up,
	__global float* down,
	__global float* left,
	__global float* right,
	__local uint* _l_img
) {
	//get tile x,y offset
	int txy = active_tiles_list[get_group_id(0)];
	int tx = txy%active_tilesW;
	int ty = txy/active_tilesW;

	int lw = get_local_size(0);
	int lh = get_local_size(1);
	int lx = get_local_id(0);
	int ly = get_local_id(1);

	int _g_img_w = 800;
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

		up[_g_img_xy] = term;
		if (ly == 0 && i == 0) continue;
		down[_g_img_xy-_g_img_w] = term;
	}

	if (ly == lh-1) {
		term = LAMBDA*weight(_l_img[_l_img_xy], _l_img[_l_img_xy-_l_img_w]);
		down[_g_img_xy-_g_img_w] = term;
	}

	if (ty == 0 && ly == 0) up[_g_img_x] = 0;
	if (ty == active_tilesH-1 && ly == lh-1) down[_g_img_xy-_g_img_w] = 0;

	//left and right weights
	_g_img_xy = _g_img_y*_g_img_w + _g_img_x;
	_l_img_xy = _l_img_x*_l_img_w + _l_img_y;

	for (int i=0; i<WAVE_LENGTH; i++, _l_img_xy += 1, _g_img_xy += _g_img_w) {
		if (tx == 0 && ly == 0 && i == 0)
			term = 0;
		else
			term = LAMBDA*weight(_l_img[_l_img_xy], _l_img[_l_img_xy-1]);

		left[_g_img_xy] = term;
		if (ly == 0 && i == 0) continue;
		right[_g_img_xy-_g_img_w] = term;
	}

	if (tx == active_tilesW-1 && ty == active_tilesH-1) {
		right[_g_img_xy-_g_img_w] = 0;
	}
	else if (ly == lh-1) {
		term = LAMBDA*weight(_l_img[_l_img_xy], _l_img[_l_img_xy-1]);
		right[_g_img_xy-_g_img_w] = term;
	}
}

#define NUM_NEIGHBOURS 4
#define NUM_ITERATIONS 5

__kernel void addBorder(
	__global int* tiles_list,
	int tilesW,
	__global float* border,
	__global float* excess,
	int direction
)
{
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%tilesW;
	int ty = txy/tilesW;

	int lx = get_local_id(0);

	int iw = 800;
	int ix = tx*WAVE_BREDTH + lx;
	int iy = ty;
	int ixy = iy*iw + ix;

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

	ixy2 = iy2*iw + ix2;

	excess[ixy2] += border[ixy];
}

__kernel void testL(
	__global int* tiles_list,
	int tilesW,
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
	int tx = txy%tilesW;
	int ty = txy/tilesW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	int lw = get_local_size(0);
	int lh = get_local_size(1);
	
	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	int iw = 800;
	int ixy = iy*iw + ix;

	int iwS = lw + 1;
	int ixS = lx;
	int iyS = ly*WAVE_LENGTH;
	int ixyS = iyS*iwS + ixS; 

	if (lx == 0 && ly == 0)
		flags[0] = FALSE;

	for (int i=0; i<WAVE_LENGTH; i++) {
		excess_s[ixyS] = excess[ixy];
		height_s[ixyS + 1] = height[ixy];

		ixy += iw;
		ixyS += iwS; //padding to prevent bank conflicts
	}

	//load first row of heights from next tile
	if (ly == lh-1) {
		int ix2 = tx*WAVE_BREDTH - 1;
		int iy2 = ty*WORKGROUP_LENGTH + lx;
		ixy = iy2*iw + ix2;

		height_s[lx*iwS + 0] = height[ixy];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	iy = (WAVES_PER_WORKGROUP*ty + ly+1)*WAVE_LENGTH - 1;
	ixy = iy*iw + ix;

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
			right[ixy - 32 + 31*iw] += flow;	
		else
			right[ixy-iw] += flow;	

		h = hNext;
		ixy -= iw;
		ixyS -= 1;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ly != 0)
		excess_s[ixyS] += flow;
	else if (flow > 0) // does the tile-overspilling wave have excess flow?
		flags[0] = TRUE;

	barrier(CLK_LOCAL_MEM_FENCE);

	 iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	ixy = iy*iw + ix;

	iwS = lw + 1;
	ixS = lx;
	iyS = ly*WAVE_LENGTH;
	ixyS = iyS*iwS + ixS; 

	for (int i=0; i<WAVE_LENGTH; i++) {
		excess[ixy] = excess_s[ixyS];

		ixy += iw;
		ixyS += iwS;
	}

	if (ly == 0 && flags[0] == TRUE) {
		border[ty*iw + ix - WAVE_BREDTH] = flow;
		border_tiles[txy - 1] = iteration;
	}
}


__kernel void test(
	__global int* tiles_list,
	int tilesW,
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
	int tx = txy%tilesW;
	int ty = txy/tilesW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	int lw = get_local_size(0);
	int lh = get_local_size(1);
	
	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	int iw = 800;
	int ixy = iy*iw + ix;

	int iwS = lw + 1;
	int ixS = lx;
	int iyS = ly*WAVE_LENGTH;
	int ixyS = iyS*iwS + ixS; 

	if (lx == 0 && ly == 0)
		flags[0] = FALSE;

	for (int i=0; i<WAVE_LENGTH; i++) {
		excess_s[ixyS] = excess[ixy];
		height_s[ixyS] = height[ixy];

		ixy += iw;
		ixyS += iwS; //padding to prevent bank conflicts
	}

	//load first row of heights from next tile
	if (ly == lh-1) {
		int ix2 = (tx+1)*WAVE_BREDTH;
		int iy2 = ty*WORKGROUP_LENGTH + lx;
		ixy = iy2*iw + ix2;

		height_s[lx*iwS + lw] = height[ixy];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	ixy = iy*iw + ix;

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
			left[ixy + 32 - 31*iw] += flow;	
		else
			left[ixy+iw] += flow;	

		h = hNext;
		ixy += iw;
		ixyS += 1;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ly != lh-1)
		excess_s[ixyS] += flow;
	else if (flow > 0) // does the tile-overspilling wave have excess flow?
		flags[0] = TRUE;

	barrier(CLK_LOCAL_MEM_FENCE);

	ixy = iy*iw + ix;

	iwS = lw + 1;
	ixS = lx;
	iyS = ly*WAVE_LENGTH;
	ixyS = iyS*iwS + ixS; 

	for (int i=0; i<WAVE_LENGTH; i++) {
		excess[ixy] = excess_s[ixyS];

		ixy += iw;
		ixyS += iwS;
	}

	if (ly == lh-1 && flags[0] == TRUE) {
		border[ty*iw + ix + WAVE_BREDTH] = flow;
		border_tiles[txy + 1] = iteration;
	}
}

__kernel void pushDown(
	__global int* tiles_list,
	int tilesW,
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
	int tx = txy%tilesW;
	int ty = txy/tilesW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	int lh = get_local_size(1);

	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	int iw = 800;
	int ixy = iy*iw + ix;

	if (lx == 0 && ly == 0)
		flags[0] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);
	
	float ef = 0;
	float flow = 0;
	int h = height[ixy];
	int hNext;
	
	for (int i=0; i<WAVE_LENGTH; i++) {
		ef = excess[ixy] + flow;
		hNext = height[ixy+iw];

		if (ef > 0 && h < MAX_HEIGHT && h == hNext + 1) {
			flow = min(ef, down[ixy]);
		}
		else
			flow = 0;
		
		down[ixy] -= flow;
		excess[ixy] = ef-flow;
		up[ixy+iw] += flow;	

		h = hNext;
		ixy += iw;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ly != lh-1)
		excess[ixy] += flow;
	else if (flow > 0) // does the tile-overspilling wave have excess flow?
		flags[0] = TRUE;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ly == lh-1 && flags[0] == TRUE) {
		border[ty*iw + ix + iw] = flow;
		border_tiles[txy + tilesW] = iteration;
	}
}

__kernel void pushUp(
	__global int* tiles_list,
	int tilesW,
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
	int tx = txy%tilesW;
	int ty = txy/tilesW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	
	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly+1)*WAVE_LENGTH - 1;
	int iw = 800;
	int ixy = iy*iw + ix;
	
	if (lx == 0 && ly == 0)
		flags[0] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);
	
	float ef = 0;
	float flow = 0;
	int h = height[ixy];
	int hNext;

	for (int i=WAVE_LENGTH-1; i>=0; i--) {
		ef = excess[ixy] + flow;
		hNext = height[ixy-iw];

		if (ef > 0 && h < MAX_HEIGHT && h == hNext + 1)
			flow = min(ef, up[ixy]);
		else
			flow = 0;
		
		up[ixy] -= flow;
		excess[ixy] = ef-flow;
		down[ixy-iw] += flow;

		h = hNext;
		ixy -= iw;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ly != 0)
		excess[ixy] += flow;
	else if (flow > 0) // does the tile-overspilling wave have excess flow?
		flags[0] = TRUE;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ly == 0 && flags[0] == TRUE) {
		border[ty*iw + ix - iw] = flow;
		border_tiles[txy - tilesW] = iteration;
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
	int gw = get_global_size(0);
	int gh = get_global_size(1);
	int gs = gw*gh;
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*gw + gx;

	int gxT = (get_group_id(0)/2)*WAVE_BREDTH + gy%32;
	int gyT = (get_group_id(1)/2)*WORKGROUP_LENGTH + gx%32;
	int gxyT = gyT*gw + gxT;

	int h = height[gxy];
	int newHeight = h;

	if (excess[gxy] > 0 && h < MAX_HEIGHT) {
		newHeight = MAX_HEIGHT;

		if (gy < gh-1 && down[gxy] > 0)
			newHeight = imin(newHeight, height[gxy+gw]+1);

		if (gy > 0 && up[gxy] > 0)
			newHeight = imin(newHeight, height[gxy-gw]+1);

		if (gx < gw-1 && right[gxyT] > 0)
			newHeight = imin(newHeight, height[gxy+1]+1);

		if (gx > 0 && left[gxyT] > 0)
			newHeight = imin(newHeight, height[gxy-1]+1);
	}

	height2[gxy] = newHeight;
	return;
}

#define BFS_ACTIVE_TILE 0x10000000

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
	int tilesW,
	int tilesH,
	__global int* bfs_tiles,
	int iteration
)
{
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%tilesW;
	int ty = txy/tilesW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	int lw = get_local_size(0);
	
	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	int iw = 800;
	int ixy = iy*iw + ix;

	int iwS = lw;
	int ixS = lx;
	int iyS = ly*WAVE_LENGTH;
	int ixyS = iyS*iwS + ixS; 

	for (int i=0; i<WAVE_LENGTH; i++) {
		excess_s[ixyS] = excess[ixy];

		ixy += iw;
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

	ixy = iy*iw + ix;
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

		ixy += iw;
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
			ixy2 = iy2*iw + ix2;

			ixS = lx;
			iyS = 0;
			ixyS = iyS*iwS + ixS;

			e1 = excess_s[ixyS];
			e2 = excess[ixy2-iw];

			// no need to check the reverse i.e e2 >= 0 and e1 < 0
			// if e2 >= 0 was true the tile above would be processed
			if ((can_ups_s[lx] & 1) && e1 >= 0 && e2 < 0) { // & 1 -> check first bit to see if possible to push up
				bfs[ixy2] = 1; // dist of 1 away from sink
				bfs_tiles[txy] = iteration;
			}
			break;

		case 1: //down
			if (ty == tilesH-1) break;

			ix2 = ix;
			iy2 = (ty+1)*WORKGROUP_LENGTH - 1;
			ixy2 = iy2*iw + ix2;

			ixS = lx;
			iyS = WORKGROUP_LENGTH-1;
			ixyS = iyS*iwS + ixS;

			e1 = excess_s[ixyS];
			e2 = excess[ixy2+iw];

			if (((can_downs_s[3*WAVE_BREDTH + lx] & 128) == 128) && e1 >= 0 && e2 < 0) {
				bfs[ixy2] = 1;
				bfs_tiles[txy] = iteration;
			}
			break;

		case 2: //right
			if (tx == tilesW-1) break;

			ix2 = (tx+1)*WAVE_BREDTH - 1;
			iy2 = ty*WORKGROUP_LENGTH + lx;
			ixy2 = iy2*iw + ix2;

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
			ixy2 = iy2*iw + ix2;

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
		can_downs[ty*iw + ix] =
			(can_downs_s[3*WAVE_BREDTH + lx] << 24) |
			(can_downs_s[2*WAVE_BREDTH + lx] << 16) |
			(can_downs_s[1*WAVE_BREDTH + lx] << 8 ) |
			can_downs_s[lx];
	}

	if (ly == 1) {
		can_ups[ty*iw + ix] =
			(can_ups_s[3*WAVE_BREDTH + lx] << 24) |
			(can_ups_s[2*WAVE_BREDTH + lx] << 16) |
			(can_ups_s[1*WAVE_BREDTH + lx] << 8 ) |
			can_ups_s[lx];
	}

	if (ly == 2) {
		can_rights[ty*iw + ix] =
			(can_rights_s[3*WAVE_BREDTH + lx] << 24) |
			(can_rights_s[2*WAVE_BREDTH + lx] << 16) |
			(can_rights_s[1*WAVE_BREDTH + lx] << 8 ) |
			can_rights_s[lx];
	}

	if (ly == 3) {
		can_lefts[ty*iw + ix] =
			(can_lefts_s[3*WAVE_BREDTH + lx] << 24) |
			(can_lefts_s[2*WAVE_BREDTH + lx] << 16) |
			(can_lefts_s[1*WAVE_BREDTH + lx] << 8 ) |
			can_lefts_s[lx];
	}
}

__kernel void bfs_compact(
	__global int* active_tiles_list,
	__global int* num_active_tiles,
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
	int active_tilesW
)
{
	//get tile x,y offset
	int txy = active_tiles_list[get_group_id(0)];
	int tx = txy%active_tilesW;
	int ty = txy/active_tilesW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	
	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	int iw = 800;
	int ixy = iy*iw + ix;

	//transposed image coordinates
	int ixT = ty*WORKGROUP_LENGTH + lx;
	int iyT = tx*WAVE_BREDTH + ly*WAVE_LENGTH;
	int ixyT = iyT*608 + ixT;

	//int gwT = gh*WAVE_LENGTH;
	int iwT = 608;
	
	uchar can_down = 0;
	uchar can_up = 0;
	uchar can_right = 0;
	uchar can_left = 0;
	for (int i=0; i<WAVE_LENGTH; i++) {
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

		ixy += iw;
		ixyT += iwT;
	}

	can_downs_s[ly*WAVE_BREDTH + lx] = can_down;
	can_ups_s[ly*WAVE_BREDTH + lx] = can_up;
	can_rights_s[ly*WAVE_BREDTH + lx] = can_right;
	can_lefts_s[ly*WAVE_BREDTH + lx] = can_left;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (ly == 0) {
		can_downs[ty*iw + ix] =
			(can_downs_s[3*WAVE_BREDTH + lx] << 24) |
			(can_downs_s[2*WAVE_BREDTH + lx] << 16) |
			(can_downs_s[1*WAVE_BREDTH + lx] << 8 ) |
			can_downs_s[lx];
	}

	if (ly == 1) {
		can_ups[ty*iw + ix] =
			(can_ups_s[3*WAVE_BREDTH + lx] << 24) |
			(can_ups_s[2*WAVE_BREDTH + lx] << 16) |
			(can_ups_s[1*WAVE_BREDTH + lx] << 8 ) |
			can_ups_s[lx];
	}

	if (ly == 2) {
		can_rights[ty*iw + ix] =
			(can_rights_s[3*WAVE_BREDTH + lx] << 24) |
			(can_rights_s[2*WAVE_BREDTH + lx] << 16) |
			(can_rights_s[1*WAVE_BREDTH + lx] << 8 ) |
			can_rights_s[lx];
	}

	if (ly == 3) {
		can_lefts[ty*iw + ix] =
			(can_lefts_s[3*WAVE_BREDTH + lx] << 24) |
			(can_lefts_s[2*WAVE_BREDTH + lx] << 16) |
			(can_lefts_s[1*WAVE_BREDTH + lx] << 8 ) |
			can_lefts_s[lx];
	}
}

#define BLOCK_SIZE 16
#define TILE_SIZE 32
#define BLOCKS_PER_TILE_DIM 2



__kernel void bfs_intertile(
	__global int* tiles_list,
	__global int* num_tiles,
	__local int* flags,
	__global int* bfs,
	__global uint* can_downs,
	__global uint* can_ups,
	__global uint* can_rights,
	__global uint* can_lefts,
	int tilesW,
	int imgW,
	int iteration,
	__global int* bfs_tiles
) {
	int lx = get_local_id(0);
	int lw = get_local_size(0);
	
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%tilesW;
	int ty = txy/tilesW;

	if (lx < 3) flags[lx] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);

	//intertile gap with tile above
	int ixy = (ty*WORKGROUP_LENGTH)*imgW + (tx*WAVE_BREDTH + lx); //index into image
	int ixy2 = ty*imgW + (tx*WAVE_BREDTH + lx); //index into compressed edges

	bool can = can_ups[ixy2] & 0x1;
	bool can2 = (can_downs[ixy2-imgW] >> WORKGROUP_LENGTH-1) & 0x1;

	int dist = bfs[ixy];
	int dist2 = bfs[ixy-imgW];

	if(can && dist2+1 < dist) {
		bfs[ixy] = dist2+1;
		flags[0] = TRUE;
	}
	if(can2 && dist+1 < dist2) {
		bfs[ixy-imgW] = dist+1;
		flags[1] = TRUE;
	}

	//intertile gap with tile to left
	can = can_lefts[ixy2] & 0x1;
	can2 = (can_rights[ixy2-WAVE_BREDTH] >> WORKGROUP_LENGTH-1) & 0x1;

	ixy = (ty*WORKGROUP_LENGTH + lx)*imgW + tx*WAVE_BREDTH;

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
	else if (lx == 1 && flags[1] == TRUE) bfs_tiles[txy-tilesW] = iteration;
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
	//__global int* bfs2,
	int active_tilesW,
	int imgW,
	__local int* _l_didChange,
	__global int* num_iterations,
	__global int* bfs_edges,
	int iteration
)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gw = get_global_size(0);

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	int lw = get_local_size(0);
	
	//get tile x,y offset
	int txy = active_tiles[get_group_id(0)];
	int tx = txy%active_tilesW;
	int ty = txy/active_tilesW;

	int sw = lw + 2;

	int ixy = (ty*WORKGROUP_LENGTH + ly*WAVE_LENGTH)*imgW + tx*WAVE_BREDTH + lx;
	int sxy = (ly*WAVE_LENGTH + 1)*sw + lx + 1;
	
	//load bfs distances into shared mem
	//MAX_HEIGHT filled padding used to not worry about boundary issues later on
	//and eliminate bank conflicts
	for (int i=0; i<WAVE_LENGTH; i++) {
		bfs_s[sxy] = bfs[ixy]; //height is offeset by 1 in shared mem

		ixy += imgW;
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

	int bxy = ty*imgW + tx*WAVE_BREDTH + lx;
	int bxyT = tx*608 + ty*WAVE_BREDTH + lx;

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
	num_iterations[0] = i;

	if (lx ==0 && ly == 0) {
		if (i>0) {
			bfs_edges[txy] = iteration;
			bfs_edges[txy + active_tilesW] = iteration;
			bfs_edges[txy + 1] = iteration;
		}
	}

	ixy = (ty*WORKGROUP_LENGTH + ly*WAVE_LENGTH)*imgW + tx*WAVE_BREDTH + lx;
	sxy = (ly*WAVE_LENGTH + 1)*sw + lx + 1;

	for (int i=0; i<WAVE_LENGTH; i++) {
		bfs[ixy] = bfs_s[sxy];

		ixy += imgW;
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


__kernel void clear(
	__global uint* view
)
{
	int gw = get_global_size(0);
	int gh = get_global_size(1);
	int gs = gw*gh;
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*gw + gx;

	barrier(CLK_LOCAL_MEM_FENCE);

	int ixy = WAVE_LENGTH*gy*gw + gx;

	for (int i=0; i<WAVE_LENGTH; i++) {
		view[ixy] = 0xFF000000;

		ixy += gw;
	}
}
__kernel void mapTileList(
	__global int* tiles_list,
	int tilesW,
	int iteration,
	__global uint* view
)
{
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%tilesW;
	int ty = txy/tilesW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);

	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	int iw = 800;
	int ixy = iy*iw + ix;

	for (int i=0; i<WAVE_LENGTH; i++) {
		view[ixy] = 0xFF0000FF;	

		ixy += iw;
	}
}

__kernel void init_push(
	__global int* tiles_list,
	int tilesW,
	__global float* excess,
	__global int* bfs,
	__local int* tile_flags,
	__global int* gc_tiles,
	int iteration
)
{
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%tilesW;
	int ty = txy/tilesW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);
	
	//image coordinates
	int ix = tx*WAVE_BREDTH + lx;
	int iy = (WAVES_PER_WORKGROUP*ty + ly)*WAVE_LENGTH;
	int iw = 800;
	int ixy = iy*iw + ix;

	if (lx == 0 && ly == 0) tile_flags[0] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int i=0; i<WAVE_LENGTH; i++) {
		if (excess[ixy] > 0 && bfs[ixy] < MAX_HEIGHT) { //looking for active nodes
			tile_flags[0] = TRUE;
		}

		ixy += iw;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lx == 0 && ly == 0) {
		if (tile_flags[0]) //tile has active nodes
			gc_tiles[txy] = iteration;
		else
			//tiles having no nodes with excess are needed when initializing the next round of bfs
			gc_tiles[txy] = iteration+1; 
	}
}

__kernel void viewBorder(
	__global float* border,
	__global float* view,
	uint direction
)
{
	int gw = get_global_size(0);
	int gh = get_global_size(1);
	int gs = gw*gh;
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*gw + gx;

	if (direction == 0 && gy < gh-1)
		view[(gy+1)*WORKGROUP_LENGTH*gw + gx] = border[gxy];

	else if (direction == 1 && gy > 0)
		view[gy*WORKGROUP_LENGTH*gw + gx - gw] = border[gxy];

	int gxT = gy;
	int gyT = 608-gx;

	if (direction == 2 && gy < gh-1)
		view[(gyT+1)*WORKGROUP_LENGTH*800 + gxT] = border[gxy];

	else if (direction == 3 && gy > 0)
		view[gyT*WORKGROUP_LENGTH*800 -1] = border[gxy];
}

__kernel void viewActive(
	__global int* height,
	__global float* excess,
	__global float* down,
	__global float* right,
	__global float* up,
	__global float* left,
	__global uint* view
)
{
	int gw = get_global_size(0);
	int gh = get_global_size(1);
	int gs = gw*gh;
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*gw + gx;

	int gxT = (get_group_id(0)/2)*WAVE_BREDTH + gy%32;
	int gyT = (get_group_id(1)/2)*WORKGROUP_LENGTH + gx%32;
	int gxyT = gyT*gw + gxT;

	

	uint color = 0xFF000000;

	int h = height[gxy];

	if (excess[gxy] > 0)
		color |= 0x0000FF00;

	if (h < MAX_HEIGHT)
		color |= 0x00FF0000;

	if (down[gxy] > 0 && up[gxy] > 0 && left[gxyT] > 0 && right[gxyT] > 0)
		color |= 0x000000FF;

	view[gxy] = color;
}


__kernel void load_tiles(
	__global int* tiles_list,
	int tilesW,
	int tilesH,
	__global uint* img,
	__global float* up,
	__global float* down,
	__global float* left,
	__global float* right,
	__local uint* _l_img,
	__global int* tilesLoaded
) {
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%tilesW;
	int ty = txy/tilesW;

	int lw = get_local_size(0);
	int lh = get_local_size(1);
	int lx = get_local_id(0);
	int ly = get_local_id(1);

	int _g_img_w = 800;
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
	if (ty == tilesH-1 && ly == lh-1) down[_g_img_xy-_g_img_w] = 0;

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

	if (tx == tilesW-1 && ty == tilesH-1) {
		right[_g_img_xy-_g_img_w] = 0;
	}
	else if (ly == lh-1) {
		term = LAMBDA*weight(_l_img[_l_img_xy], _l_img[_l_img_xy-1]);
		right[_g_img_xy-_g_img_w] = term;
	}

	tilesLoaded[txy] = TRUE;
}

#define NORM 0.00392156862745098f
#define uint42f4n(c) (float4) (NORM*c.x, NORM*c.y, NORM*c.z, NORM*c.w)
#define rgba2f4(c) (float4) (c & 0x000000FF, (c & 0x0000FF00) >> 8, (c & 0x00FF0000) >> 16, (c & 0x00FF0000) >> 24)

float4 HSVtoRGB(float4 HSV)
{
        float4 RGB = (float4)0;
        if (HSV.z != 0)
        {
                float var_h = HSV.x * 6;
                float var_i = (float) ((int) (var_h-0.000001));
                float var_1 = HSV.z * (1.0 - HSV.y);
                float var_2 = HSV.z * (1.0 - HSV.y * (var_h-var_i));
                float var_3 = HSV.z * (1.0 - HSV.y * (1-(var_h-var_i)));
                switch((int)(var_i))
                {
                        case 0: RGB = (float4)(HSV.z, var_3, var_1, HSV.w); break;
                        case 1: RGB = (float4)(var_2, HSV.z, var_1, HSV.w); break;
                        case 2: RGB = (float4)(var_1, HSV.z, var_3, HSV.w); break;
                        case 3: RGB = (float4)(var_1, var_2, HSV.z, HSV.w); break;
                        case 4: RGB = (float4)(HSV.z, var_1, var_2, HSV.w); break;
                        default: RGB = (float4)(HSV.z, var_1, var_2, HSV.w); break;
                }
        }
        RGB.w = HSV.w;
        return (RGB);
}

float4 colorizef(float val, float2 range, int2 hues) {
	if (val < range[0])
		return (float4) (0, 0, 0, 0);
	else if (val > range[1])
		return (float4) (1, 1, 1, 1);

	float normalized = (val-range[0])/(range[1]-range[0]);
	float hue = hues[0] + normalized*(hues[1]-hues[0]);

	float4 hsv = (float4) (hue/360, 1.0, 1.0, 0.0);

	return HSVtoRGB(hsv);
}

float4 colorizei(int val, int2 range, int2 hues) {
	if (val < range[0])
		return (float4) (0, 0, 0, 0);
	else if (val > range[1])
		return (float4) (1, 1, 1, 1);

	float normalized = (float) (val-range[0])/(range[1]-range[0]);
	float hue = hues[0] + normalized*(hues[1]-hues[0]);

	float4 hsv = (float4) (hue/360, 1.0, 1.0, 0.0);

	return HSVtoRGB(hsv);
}

__kernel void tranpose(
//	__global float* n_link,
//	__global float* n_linkT
	float2 range,
	int2 hues,
	sampler_t sampler,
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	float opacity,
	__global float* input,
	int2 inputDim
) {
	int gw = get_global_size(0);
	int gh = get_global_size(1);
	int gs = gw*gh;
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*gw + gx;

	int lw = get_local_size(0);
	int lh = get_local_size(1);
	int lx = get_local_id(0);
	int ly = get_local_id(1);

	int wx = get_group_id(0);
	int wy = get_group_id(1);

	int iw = gw;
	int ix = gx;
	int iy = gy*WAVE_LENGTH;
	int ixy = iy*iw + ix;

	//transposed coordinates, threads are enumerated vertically within the tile
	int iwT = gw;
	int ixT = wx*WAVE_BREDTH + ly*WAVE_LENGTH;
	int iyT = wy*WORKGROUP_LENGTH + lx;
//	int ixyT = iyT*iwT + ixT;
	int2 ixyT = (int2) (ixT, iyT);

	for (int i=0; i<WAVE_LENGTH; i++) {
//		n_linkT[ixyT] = n_link[ixy];
		float in = input[ixy];
		float4 out = colorizef(in, range, hues);

		float4 read = read_imagef(rbo_read, sampler, ixyT);

		write_imagef(rbo_write, ixyT, read + (out-read)*opacity);

		ixy += iw;
//		ixyT += 1;
		ixyT.x += 1;
	}
}

__kernel void tilelist(
	int2 range,
	int2 hues,
	sampler_t sampler,
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	float opacity,
	__global int* input,
	int2 inputDim
) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int2 gxy = (int2) (gx, gy);

	int bxy = (gy/TILE_SIZE)*inputDim.x + gx/TILE_SIZE;
	int in = input[bxy];
	float4 out = colorizei(in, range, hues);

	float4 read = read_imagef(rbo_read, sampler, gxy);

	write_imagef(rbo_write, gxy, read + (out-read)*opacity);
}