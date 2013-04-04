#define NORM 0.00392156862745098f
#define uint42f4n(c) (float4) (NORM*c.x, NORM*c.y, NORM*c.z, NORM*c.w)
#define rgba2f4(c) (float4) (c & 0x000000FF, (c & 0x0000FF00) >> 8, (c & 0x00FF0000) >> 16, (c & 0x00FF0000) >> 24)
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



__global kernel void colorize_i32(
	int2 range,
	int2 hues,
	__global int* input,
	int2 inputDim,
	__write_only image2d_t output
) {
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > inputDim.x-1 || gxy.y > inputDim.y-1)
		return;

	int in = input[gxy.y*inputDim.x + gxy.x];
	float4 out = colorizei(in, range, hues);

	write_imagef(output, gxy, out);
}

__global kernel void colorize_f32(
	float2 range,
	int2 hues,
	__global float* input,
	int2 inputDim,
	__write_only image2d_t output
) {
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > inputDim.x-1 || gxy.y > inputDim.y-1)
		return;

	float in = input[gxy.y*inputDim.x + gxy.x];
	float4 out = colorizef(in, range, hues);

	write_imagef(output, gxy, out);
}