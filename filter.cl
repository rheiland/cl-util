#define NORM 0.00392156862745098f

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
	float normalized = (val-range[0])/(range[1]-range[0]);
	float hue = hues[0] + normalized*(hues[1]-hues[0]);

	float4 hsv = (float4) (hue/360, 1.0, 1.0, 0.0);

	return HSVtoRGB(hsv);
}

float4 colorizei(int val, int2 range, int2 hues) {
	float normalized = (float) (val-range[0])/(range[1]-range[0]);
	float hue = hues[0] + normalized*(hues[1]-hues[0]);

	float4 hsv = (float4) (hue/360, 1.0, 1.0, 0.0);

	return HSVtoRGB(hsv);
}

#define uint42f4n(c) (float4) (NORM*c.x, NORM*c.y, NORM*c.z, NORM*c.w)

__global kernel void buffer_i32(
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	sampler_t sampler,
	float opacity,
	int2 range,
	int2 hues,
	__global int* input,
	int2 inputDim
) {
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > inputDim[0]-1 || gxy.y > inputDim[1]-1)
		return;

	float4 read = read_imagef(rbo_read, sampler, gxy);

	int in = input[gxy.y*inputDim[0] + gxy.x];
	float4 out = colorizei(in, range, hues);

	write_imagef(rbo_write, gxy, read + (out-read)*opacity);
}

__global kernel void buffer_f32(
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	sampler_t sampler,
	float opacity,
	float2 range,
	int2 hues,
	__global float* input,
	int2 inputDim
) {
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > inputDim[0]-1 || gxy.y > inputDim[1]-1)
		return;

	float4 read = read_imagef(rbo_read, sampler, gxy);

	float in = input[gxy.y*inputDim[0] + gxy.x];
	float4 out = colorizef(in, range, hues);

	write_imagef(rbo_write, gxy, read + (out-read)*opacity);
}

__global kernel void blend_ui(
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	sampler_t sampler,
	float opacity,
	__read_only image2d_t input
) {
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > get_image_width(input)-1 || gxy.y > get_image_height(input)-1)
		return;

	float4 read = read_imagef(rbo_read, sampler, gxy);

	uint4 in = read_imageui(input, sampler, gxy);
	float4 out = uint42f4n(in);

	write_imagef(rbo_write, gxy, read + (out-read)*opacity);
}

__global kernel void flip(
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	sampler_t sampler
){
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > get_image_width(rbo_read)-1 || gxy.y > get_image_height(rbo_read)-1)
		return;

	float4 read = read_imagef(rbo_read, sampler, gxy);

	gxy.y = get_image_height(rbo_write) - gxy.y;
	write_imagef(rbo_write, gxy, read);
}