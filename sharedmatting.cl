#define ENUM_TRI_FG 0
#define ENUM_TRI_BG 1
#define ENUM_TRI_UK 2

#define COLOR_F4_TRI_FG (float4) (1, 0, 0, 1)
#define COLOR_F4_TRI_BG (float4) (0, 0, 1, 1)
#define COLOR_F4_TRI_UK (float4) (0, 1, 0, 1)

#define TRI_UK 0xFF00FF00
#define TRI_BG 0xFFFF0000
#define TRI_FG 0xFF0000FF

#define TRUE 1
#define FALSE 0

#define K_G 4
#define K_S 6
#define K_SM 600
#define K_WX 3
#define K_WY 3
#define K_WS 9
#define K_Q 1
#define K_V 2
#define K_R 7

#define THETA0 (1.0/K_WS)*(2*M_PI/K_G)
#define THETA_INC 2*M_PI/K_G

//#define K_LAMDA 0.00000001
#define K_LAMDA 0.1
//3*(10^2) diff in r,g,b
#define K_ALT_LCV 300

int cDist2(uint c1, uint c2);
int iDist2(int x1, int y1, int x2, int y2);

float alpha_est(uint c, uint fg, uint bg) {

    int c_minus_bg_r  = ((c >> 16) & 0x000000FF) - ((bg >> 16) & 0x000000FF);
    int c_minus_bg_g  = ((c >> 8 ) & 0x000000FF) - ((bg >> 8)  & 0x000000FF);
    int c_minus_bg_b  = ( c        & 0x000000FF) - ( bg        & 0x000000FF);

    int fg_minus_bg_r = ((fg    >> 16) & 0x000000FF) - ((bg >> 16) & 0x000000FF);
    int fg_minus_bg_g = ((fg    >> 8 ) & 0x000000FF) - ((bg >> 8)  & 0x000000FF);
    int fg_minus_bg_b = ( fg           & 0x000000FF) - ( bg        & 0x000000FF);

    int num =  c_minus_bg_r*fg_minus_bg_r +  c_minus_bg_g*fg_minus_bg_g +  c_minus_bg_b*fg_minus_bg_b;
    int den = fg_minus_bg_r*fg_minus_bg_r + fg_minus_bg_g*fg_minus_bg_g + fg_minus_bg_b*fg_minus_bg_b; // ||fg_minus_bg||^2
    
    float res = (float) num / den;

	if (res < 0) res = 0;
	else if (res > 1) res = 1;
	
    return res;
}

float cd2(uint c, uint fg, uint bg) {
    float alpha = alpha_est(c, fg, bg);

    float tmp_b = ((c >> 16) & 0x000000FF) - (alpha*((fg >> 16) & 0x000000FF) + (1 - alpha)*((bg >> 16) & 0x000000FF));
    float tmp_g = ((c >> 8)  & 0x000000FF) - (alpha*((fg >> 8)  & 0x000000FF) + (1 - alpha)*((bg >> 8)  & 0x000000FF));
    float tmp_r = ( c        & 0x000000FF) - (alpha*( fg        & 0x000000FF) + (1 - alpha)*( bg        & 0x000000FF));
	
	return tmp_r*tmp_r + tmp_g*tmp_g + tmp_b*tmp_b;
}

//int cDist2(uint c1, uint c2);
//int iDist2(int x1, int y1, int x2, int y2);

__kernel void local_color_variation(
	__global uint* src,
	__global float* lcv,
	__global float* alpha
	)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*IMAGEW + gx;

	int sum = 0;
	int n = 0;

	uint c = src[gxy];

	for(int iy=gy-K_V; iy<gy+K_V; iy++) {
		for(int ix=gx-K_V; ix<gx+K_V; ix++) {
			if (ix < 0 || ix > IMAGEW-1 || iy < 0 || iy > IMAGEH-1)
				continue;

			int ixy = iy*IMAGEW + ix;

			sum += cDist2(c, src[ixy]);
			n++;
		}
	}

	lcv[gxy] = (float) sum/n;
	alpha[gxy] = 1.0;
}

#define DILATE 2
#define TILEW 16
#define TILEH 16

__kernel void process_trimap(
	__global uchar* triOut,
	__global uchar* triIn,
	__global float* strength,
	float threshold,
	__local uchar* s_triIn
) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*IMAGEW + gx;

	int lx = get_local_id(0);
	int ly = get_local_id(1);

	int sx = DILATE + lx;
	int sy = DILATE + ly;
	int sw = DILATE + TILEW + DILATE;
	int sxy = sy*sw + sx;

	if (strength[gxy] < threshold)
		s_triIn[sxy] = ENUM_TRI_UK;
	else
        s_triIn[sxy] = triIn[gxy];

	if (lx < DILATE && gx >= DILATE)
		s_triIn[sxy - DILATE] = triIn[gxy - DILATE];
	if (lx >= TILEW-DILATE && gx < IMAGEW-DILATE)
		s_triIn[sxy + DILATE] = triIn[gxy + DILATE];

	if (ly < DILATE && gy >= DILATE)
		s_triIn[sxy - DILATE*sw] = triIn[gxy - DILATE*IMAGEW];
	if (ly >= TILEH-DILATE && gy < IMAGEH-DILATE)
		s_triIn[sxy + DILATE*sw] = triIn[gxy + DILATE*IMAGEW];

	//corner padding
	if (lx < DILATE && gx >= DILATE && ly < DILATE && gy >= DILATE)
		s_triIn[sxy - DILATE - DILATE*sw] = triIn[gxy - DILATE - DILATE*IMAGEW];

	if (lx < DILATE && gx >= DILATE && ly >= TILEH-DILATE && gy < IMAGEH-DILATE)
		s_triIn[sxy - DILATE + DILATE*sw] = triIn[gxy - DILATE + DILATE*IMAGEW];

	if (lx >= TILEW-DILATE && gx < IMAGEW-DILATE && ly < DILATE && gy >= DILATE)
		s_triIn[sxy + DILATE - DILATE*sw] = triIn[gxy + DILATE - DILATE*IMAGEW];

	if (lx >= TILEW-DILATE && gx < IMAGEW-DILATE && ly >= TILEH-DILATE && gy < IMAGEH-DILATE)
		s_triIn[sxy + DILATE + DILATE*sw] = triIn[gxy + DILATE + DILATE*IMAGEW];

	barrier(CLK_LOCAL_MEM_FENCE);

	uchar p_triOut = s_triIn[sxy];

    if (p_triOut != ENUM_TRI_UK) {
        for (int i=-DILATE; i<= DILATE; i++) {
            for (int j=-DILATE; j<= DILATE; j++) {
                if (p_triOut != s_triIn[sxy + j*sw + i] && s_triIn[sxy + j*sw + i] != ENUM_TRI_UK) {
                    p_triOut = ENUM_TRI_UK;

                    break;
                }
            }
        }
    }

	triOut[gxy] = p_triOut;
}

__kernel void gather(
	__global uint* src,
	__global uchar* tri,
	__global uint* fg,
	__global uint* bg
) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*IMAGEW + gx;

	uint t = tri[gxy];
	if (t == ENUM_TRI_FG) {
		fg[gxy] = src[gxy];
		return;
	}
	else if (t == ENUM_TRI_BG) {
		fg[gxy] = 0x00000000;
		bg[gxy] = src[gxy];
		return;
	}

	uint c = src[gxy];

	int wx = gx%K_WX;
	int wy = gy%K_WY;
	int wi = wy*K_WX + wx;

	float theta = wi*THETA0 + THETA_INC;

	uint cFg[K_G];
	uint cBg[K_G];

	int eFg[K_G];
	int eBg[K_G];

	int iFg[K_G];
	int iBg[K_G];

	int ix, iy, ixy;
	for(int ik_g=0; ik_g<K_G; ik_g++) {
		cFg[ik_g] = 0x00000000;
		cBg[ik_g] = 0x00000000;

		eFg[ik_g] = 0;
		eBg[ik_g] = 0;

		iFg[ik_g] = -1;
		iBg[ik_g] = -1;

		uint c_prev = c;
		int e = 0;

		for (int ik_s=K_S; ik_s<K_SM; ik_s++) {
			ix = (int) (ik_s*cos(theta)) + gx;		
			iy = (int) (ik_s*sin(theta)) + gy;

			if (ix < 0 || ix > IMAGEW-1 || iy < 0 || iy > IMAGEH-1)
				continue;

			ixy = iy*IMAGEW + ix;

			t = tri[ixy];
			uint c_this = src[ixy];

			e += cDist2(c_this, c_prev);

			if (t == ENUM_TRI_FG && cFg[ik_g] == 0x00000000) {
				cFg[ik_g] = c_this;
				eFg[ik_g] = e;
				iFg[ik_g] = ixy;

				if (cBg[ik_g] != 0x00000000) break;
			}
			if (t == ENUM_TRI_BG && cBg[ik_g] == 0x00000000) {
				cBg[ik_g] = c_this;
				eBg[ik_g] = e;
				iBg[ik_g] = ixy;

				if (cFg[ik_g] != 0x00000000) break;
			}

			ik_s += K_S;
			c_prev = c_this;
		}
	
		theta += THETA_INC;
	}

	float min_of = FLT_MAX;
	int min_ibg = -1;
	int min_ifg = -1;

	for(int ifg=0; ifg<K_G; ifg++) {
		if (cFg[ifg] == 0x00000000) continue;

		for(int ibg=0; ibg<K_G; ibg++) {
			if (cBg[ibg] == 0x00000000) continue;

//WHY IS THIS?
			if (ibg < 0 || ibg > 3) return;

			float cd2s = 0;

			for(iy=gy-K_Q; iy<gy+K_Q; iy++) {
				for(ix=gx-K_Q; ix<gx+K_Q; ix++) {
					if (ix < 0 || ix > IMAGEW-1 || iy < 0 || iy > IMAGEH-1)
						continue;

					ixy = iy*IMAGEW + ix;

					cd2s += cd2(src[ixy], cFg[ifg], cBg[ibg]);
				}
			}

			float alpha = alpha_est(c, cFg[ifg], cBg[ibg]);
			float pFg = (float) (eBg[ibg])/(eFg[ifg]+eBg[ibg]);
				  pFg = pFg + (1 - 2*pFg)*alpha;
			//pFg = pFg - alpha;
			//if (pFg < 0) pFg *= -1;
			//pFg = 1 - pFg;
			
			float this_of = pow(cd2s, 3);
				  this_of *= pow(pFg, 2);
				  this_of *= sqrt((float)(iDist2(gx, gy, iFg[ifg]%IMAGEW, iFg[ifg]/IMAGEW)));
				  this_of *= sqrt((float)(iDist2(gx, gy, iBg[ibg]%IMAGEW, iBg[ibg]/IMAGEW)));

			if (this_of < min_of) {
				min_of = this_of;
				min_ibg = ibg;
				min_ifg = ifg;
			}
		}
	}

	int ifg = min_ifg;
	int ibg = min_ibg;

	if (min_of == FLT_MAX) {
		//tri[gxy] = 0xFF000000;
		return;
	}

	fg[gxy] = cFg[ifg];
	bg[gxy] = cBg[ibg];
	
	return;
}

__kernel void refine(
	__global uint* src,
	__global uchar* tri,
	__global uint* fg,
	__global uint* bg,
	__global float* alpha,
	__global float* lcv
	)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*IMAGEW + gx;

	uint t = tri[gxy];
	if (t != ENUM_TRI_UK) {
		//alpha[gxy] = (t == TRI_FG) ? src[gxy] : 0xFF000000;
		uint a = (t == ENUM_TRI_FG) ? 1.0 : 0.0;
		alpha[gxy] = a;

		return;
	}

	uint fg1, fg2, fg3, bg1, bg2, bg3;
	float cd21, cd22, cd23;

    fg1 = fg2 = fg3 = 0x00000000;
    bg1 = bg2 = bg3 = 0x00000000;

    cd21 = cd22 = cd23 = FLT_MAX;
	
	uint c = src[gxy];

	for(int iy=gy-K_R; iy<gy+K_R; iy++) {
		for(int ix=gx-K_R; ix<gx+K_R; ix++) {
			if (ix < 0 || ix > IMAGEW-1 || iy < 0 || iy > IMAGEH-1)
				continue;

			int ixy = iy*IMAGEW + ix;
			uint this_fg = fg[ixy];
			uint this_bg = bg[ixy];

			if (!this_fg || !this_bg)
				continue;

			float this_cd2 = cd2(c, this_fg, this_bg);

			if (this_cd2 < cd21) { //found better top best pair
				cd23 = cd22;
				bg3 = bg2;
				fg3 = fg2;

				cd22 = cd21;
				bg2 = bg1;
				fg2 = fg1;

				cd21 = this_cd2;
				bg1 = this_bg;
				fg1 = this_fg;
			}
			else if (this_cd2 < cd22) { //found 2nd top best pair
				cd22 = cd21;
				bg2 = bg1;
				fg2 = fg1;

				cd21 = this_cd2;
				bg1 = this_bg;
				fg1 = this_fg;
			}
			else if (this_cd2 < cd23) { // found 3d top best pair
				cd21 = this_cd2;
				bg1 = this_bg;
				fg1 = this_fg;
			}
		}
	}

	int n = 0;
		 if (cd23 != FLT_MAX) n = 3;
	else if (cd22 != FLT_MAX) n = 2;
	else if (cd21 != FLT_MAX) n = 1;
	//else					n = 0;
	
	if (n > 0) {
		int fgR = ((fg1       & 0xFF) + (fg2      & 0xFF)  + (fg3       & 0xFF))/n; 
		int fgG = ((fg1 >> 8  & 0xFF) + (fg2 >> 8  & 0xFF) + (fg3 >> 8  & 0xFF))/n; 
		int fgB = ((fg1 >> 16 & 0xFF) + (fg2 >> 16 & 0xFF) + (fg3 >> 16 & 0xFF))/n; 

		int bgR = ((bg1       & 0xFF) + (bg2      & 0xFF)  + (bg3       & 0xFF))/n; 
		int bgG = ((bg1 >> 8  & 0xFF) + (bg2 >> 8  & 0xFF) + (bg3 >> 8  & 0xFF))/n; 
		int bgB = ((bg1 >> 16 & 0xFF) + (bg2 >> 16 & 0xFF) + (bg3 >> 16 & 0xFF))/n; 

		fg1 = 0xFF000000 | (fgB << 16) | (fgG << 8) | fgR; 	
		bg1 = 0xFF000000 | bgB << 16 | bgG << 8 | bgR;
		
		if (cDist2(c, fg1) < K_ALT_LCV) fg1 = c;
		if (cDist2(c, bg1) < K_ALT_LCV) bg1 = c;

		float a, p;
		if (fg1 == bg1) {
			a = 0.5;
			p = 0.0001;
		}
		else if (fg1 == c) {
			a = 1;
			p = 1;
		}
		else if (bg1 == c) {
			a = 0;
			p = 1;
		}
		else {
			a = alpha_est(c, fg1, bg1);
			p = -K_LAMDA*sqrt(cd2(c, fg1, bg1));
			p = exp(p);
		}

		alpha[gxy] = a;
		fg[gxy] = (fg1 & 0x00FFFFFF) | (int) (a*255) << 24;
		bg[gxy] = bg1;
	}
}

int cDist2(uint c1, uint c2) {
	uint r1, r2, g1, g2, b1, b2;
		
	r1 = (c1 & 0x000000FF);
	g1 = (c1 & 0x0000FF00) >> 8;
	b1 = (c1 & 0x00FF0000) >> 16;

	r2 = (c2 & 0x000000FF);
	g2 = (c2 & 0x0000FF00) >> 8;
	b2 = (c2 & 0x00FF0000) >> 16;
	
	return (r2-r1)*(r2-r1) + (g2-g1)*(g2-g1) + (b2-b1)*(b2-b1);	
}

int iDist2(int x1, int y1, int x2, int y2) {
	return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
}


__kernel void trimap_filter(
	sampler_t sampler,
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	float opacity,
	__global char* input,
	int2 inputDim
) {
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > inputDim.x-1 || gxy.y > inputDim.y-1)
		return;

	char in = input[gxy.y*inputDim.x + gxy.x];
	float4 out;
	switch(in) {
		case ENUM_TRI_FG: out = COLOR_F4_TRI_FG; break;
		case ENUM_TRI_BG: out = COLOR_F4_TRI_BG; break;
		case ENUM_TRI_UK: out = COLOR_F4_TRI_UK; break;
	}

	float4 read = read_imagef(rbo_read, sampler, gxy);

	write_imagef(rbo_write, gxy, read + (out-read)*opacity);
}
