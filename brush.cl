#define rgba2f4(c) (float4) (c & 0x000000FF, (c & 0x0000FF00) >> 8, (c & 0x00FF0000) >> 16, 255)


__kernel void point(
	int2 point,
	int radius,
	uint color
	OUTPUT_ARGS
	//__write_only image2d_t canvas,
	//global uint* canvas,
	//int2 shape
) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);

	if (gx > 2*radius || gy > 2*radius)
		return;

	int2 gcoord = (int2) (gx, gy);
	gcoord += point - radius;
	
	CODE
	//write_imagef(canvas, gcoord, rgba2f4(color)/255.0f);
}

__kernel void stroke(
	float dx_dy,
	int2 point,
	int rows,
	int row,
	uint color
	OUTPUT_ARGS
	//__write_only image2d_t canvas,
)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);

	if (gx > row-1 || gy > rows-1)
		return;

	int2 gcoord = (int2) (gy*dx_dy + gx, gy);

	gcoord += point;

	CODE
	//write_imagef(canvas, gcoord, rgba2f4(color)/255.0f);
}
