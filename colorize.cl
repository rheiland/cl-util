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

uint pseudocolorf(float val, float m, float M, float hue1, float hue2) {
	if (val < m)
		return 0xFF000000;
	if (val > M)
		return 0xFFFFFFFF;

	float4 hsv = (float4) ((hue1 + (val-m)/(M-m)*(hue2-hue1))/360, 1.0, 1.0, 0.0);
	hsv = HSVtoRGB(hsv);

	return (int) (255*hsv.z) << 16 | (int) (255*hsv.y) << 8 | (int) (255*hsv.x);
}

uint pseudocolorf_sat(float val, float m, float M, float hue) {
	if (val < m)
		return 0xFF000000;
	if (val > M)
		return 0xFFFFFFFF;

	float4 hsv = (float4) (hue, 1.0, (val-m)/(M-m), 0.0);
	hsv = HSVtoRGB(hsv);

	return (int) (255*hsv.z) << 16 | (int) (255*hsv.y) << 8 | (int) (255*hsv.x);
}

__kernel void colorizef(
	__global float* in,
	float m,
	float M,
	int gs,
	__global uint* out,
	float hue1,
	float hue2
) {
	int gi = get_global_id(0);

	if (gi > gs-1)
		return;

	out[gi] = 0xFF000000 | pseudocolorf(in[gi], m, M, hue1, hue2);
}

__kernel void colorizei(
	__global int* in,
	int m,
	int M,
	int gs,
	__global uint* out,
	float hue1,
	float hue2
) {
	int gi = get_global_id(0);

	if (gi > gs-1)
		return;

	out[gi] =  0xFF000000 | pseudocolorf((float) (in[gi]), (float) m, (float) M, hue1, hue2);
}

__kernel void colorizef_sat(
	__global float* in,
	float m,
	float M,
	int gs,
	__global uint* out,
	float hue
) {
	int gi = get_global_id(0);

	if (gi > gs-1)
		return;

	out[gi] = 0xFF000000 | pseudocolorf_sat(in[gi], m, M, hue);
}

__kernel void colorizei_sat(
	__global int* in,
	int m,
	int M,
	int gs,
	__global uint* out,
	float hue
) {
	int gi = get_global_id(0);

	if (gi > gs-1)
		return;

	out[gi] = 0xFF000000 | pseudocolorf_sat(in[gi], m, M, hue);
}
