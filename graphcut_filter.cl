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

float4 colorizef(float x, float2 range, float2 hues, float2 sats, float2 vals) {
	float4 hsv;

	if (x < range[0])
		hsv = (float4) (0, 0, 0, 0);
	else if (x > range[1])
		hsv = (float4) (0, 0, 1, 0);
	else {
		float normalized = (x-range[0])/(range[1]-range[0]);

		hsv = (float4) (
			hues[0] + normalized*(hues[1]-hues[0]),
			sats[0] + normalized*(sats[1]-sats[0]),
			vals[0] + normalized*(vals[1]-vals[0]),
			0
			);
	}

	return HSVtoRGB(hsv);
}

#define WAVE_BREDTH 32
#define WAVE_LENGTH 8
#define WAVES_PER_WORKGROUP 4
#define WORKGROUP_LENGTH (WAVES_PER_WORKGROUP * WAVE_LENGTH)

#define TILE_SIZE 32

__kernel void tranpose(
	sampler_t sampler,
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	float opacity,
	__global float* input,
	int2 inputDim,
	float2 range,
	float2 hues,
	float2 sats,
	float2 vals
) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*IMAGEW + gx;

	int lw = get_local_size(0);
	int lh = get_local_size(1);
	int lx = get_local_id(0);
	int ly = get_local_id(1);

	int wx = get_group_id(0);
	int wy = get_group_id(1);

	int ix = gx;
	int iy = gy*WAVE_LENGTH;
	int ixy = iy*IMAGEW + ix;

	//transposed coordinates, threads are enumerated vertically within the tile
	int iwT = IMAGEW;
	int ixT = wx*WAVE_BREDTH + ly*WAVE_LENGTH;
	int iyT = wy*WORKGROUP_LENGTH + lx;
	int2 ixyT = (int2) (ixT, iyT);

	for (int i=0; i<WAVE_LENGTH; i++) {
		float in = input[ixy];
		float4 out = colorizef(in, range, hues, (float2) (1, 1), (float2) (1, 1));

		float4 read = read_imagef(rbo_read, sampler, ixyT);

		write_imagef(rbo_write, ixyT, read + (out-read)*opacity);

		ixy += IMAGEW;
		ixyT.x += 1;
	}
}

__kernel void tilelist(
	sampler_t sampler,
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	float opacity,
	__global int* input,
	int2 inputDim,
	int2 range,
	float2 hues,
	float2 sats,
	float2 vals
) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int2 gxy = (int2) (gx, gy);

	int bxy = (gy/WORKGROUP_LENGTH)*TILESW + gx/WAVE_BREDTH;
	int in = input[bxy];
	float4 out = colorizef((float) in, (float2) range, hues, sats, vals);

	float4 read = read_imagef(rbo_read, sampler, gxy);

	write_imagef(rbo_write, gxy, read + (out-read)*opacity);
}

#define TILEW 16
#define TILEH 16

__kernel void tilelist_growcut(
	sampler_t sampler,
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	float opacity,
	__global int* input,
	int2 inputDim,
	int2 range,
	float2 hues,
	float2 sats,
	float2 vals
) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int2 gxy = (int2) (gx, gy);

	int bxy = (gy/TILEH)*TILESW + gx/TILEW;
	int in = input[bxy];
	float4 out = colorizef((float) in, (float2) range, hues, sats, vals);

	float4 read = read_imagef(rbo_read, sampler, gxy);

	write_imagef(rbo_write, gxy, read + (out-read)*opacity);
}