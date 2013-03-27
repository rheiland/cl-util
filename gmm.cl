#define NDIM 3
#define AW 8 //width of matrix a = (2*NDIM + 1) and round to 8

#ifndef MAX_NCOMPONENT
#define MAX_NCOMPONENT 8
#endif

#ifndef MIN_COVAR
#define MIN_COVAR 0.001f
#endif

#ifndef INIT_COVAR
#define INIT_COVAR 10.0f
#endif

#define MIN_COVAR4 (float4) (MIN_COVAR, MIN_COVAR, MIN_COVAR, MIN_COVAR)
#define INIT_COVAR4 (float4) (MIN_COVAR, MIN_COVAR, MIN_COVAR, MIN_COVAR)
#define INIT_V2 (float4) (0, -1/(2*MIN_COVAR), -1/(2*MIN_COVAR), -1/(2*MIN_COVAR))

#define rgba2float4(c) (float4) (c & 0x000000FF, (c & 0x0000FF00) >> 8, (c & 0x00FF0000) >> 16, 0)

//utility functions
uint pseudocolorf(float val, float m, float M);
float4 HSVtoRGB(float4 HSV);

#define M2_PI_POW_D_OVER_2 pow((float)(M_PI+M_PI), 1.5f)
//#define INIT_V1_TERM1 -0.5f*(NDIM*log((float) 2.0*M_PI) + NDIM*log(INIT_COVAR))

__kernel void initA(
	__global uint* samples,
	int nComps,
	int nSamples,
	__global float4* A
)
{
	int gi = get_global_id(0);
	int gs = get_global_size(0);

	int li = get_local_id(0);

	if (li > 2*nComps-1)
		return;
	
	float4 out;

	if (li%2 == 0) {
		uint mean = samples[li];
		out.y = mean & 0x000000FF;
		out.z = (mean & 0x0000FF00) >> 8;
		out.w = (mean & 0x00FF0000) >> 16;
		float4 mean2 = out*out;

		out *= 1.0f/INIT_COVAR;
		mean2 *= 1.0f/INIT_COVAR;

		out.x = log(1.0f/nComps) - 0.5f*(NDIM*log((float) (2.0*M_PI)) + NDIM*log(INIT_COVAR) + (mean2.y + mean2.z + mean2.w));
		//out.x = (mean2.y + mean2.z + mean2.w);
	}
	else {
		out.x = 0;
		out.y = -1.0/(2*INIT_COVAR);
		out.z = -1.0/(2*INIT_COVAR);
		out.w = -1.0/(2*INIT_COVAR);
	}

	A[li] = out;
}


__kernel void em1(
	__global uint* samples,
	__global float4* gA,
	__local float4* sA,
	int m,
	int nSamples,
	__global float* resp,
	__global float4* resp_x,
	__global float4* resp_x2
) {
	int gi = get_global_id(0);
	int gs = get_global_size(0);

	int li = get_local_id(0);

	if (li < m*2) {
		sA[li] = gA[li];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (gi > nSamples-1)
		return;

	uint s = samples[gi];
	float4 x = (float4) (1, s & 0x000000FF, (s & 0x0000FF00) >> 8, (s & 0x00FF0000) >> 16);
	//float4 x2 = pown(x, 2); <-- bug in OpenCL

	float4 x2 = x*x;
	x2[0] = 0;

	float ln_wgs[MAX_NCOMPONENT];

	for (int i=0; i<m; i++) {
		int is = i*2;

		float4 a = sA[is];//(float4) (sA[is+0], sA[is+1], sA[is+2], sA[is+3]);
		ln_wgs[i] = dot(a, x);

		a = sA[is+1];// (float4) (sA[is+4], sA[is+5], sA[is+6], sA[is+7]);
		ln_wgs[i] += dot(a, x2);
	}

	float aMax = ln_wgs[0];

	for (int i=1; i<m; i++) {
		if (ln_wgs[i] > aMax)
			aMax = ln_wgs[i];
	}

	float sum = 0;
	for (int i=0; i<m; i++) {
		sum += exp(ln_wgs[i] - aMax);
	}

	float lpr = aMax + log(sum);

	x = rgba2float4(s);

	for (int k=0; k<m; k++) {
		float r = exp(ln_wgs[k] - lpr);

		resp[k*nSamples + gi] = r;
		resp_x[k*nSamples + gi] = r*x;
		resp_x2[k*nSamples + gi] = r*x*x;
	}
}

__kernel void check_converge(
	__global float* eval,
	__local float* s_eval,
	int nCurrent,
	int nReduced,
	__global float* eval_back
) {
	int li = get_local_id(0);
	int ls = get_local_size(0);

	int wi = get_group_id(0);

	int gi = li + 2*wi*ls;

	//load shared mem
	if (gi < nCurrent) {
		s_eval[li] = eval[gi];
		
		if (gi + ls < nCurrent) {
			s_eval[li] += eval[gi + ls];
		}
	}
	else {
		s_eval[li] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	//reduce in shared mem
	for (int s=ls/2; s>0; s/=2) {
		if (li < s) {
			s_eval[li] += s_eval[li + s];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (li == 0)
		eval_back[wi] = s_eval[0];
}

__kernel void em2(
	__global float* resp,
	__global float4* resp_x,
	__global float4* resp_x2,
	__local float* s_resp,
	__local float4* s_resp_x,
	__local float4* s_resp_x2,
	int k,
	int nSamples,
	int nSamplesCurrent,
	int nSamplesReduced,
	__global float* resp_back,
	__global float4* resp_x_back,
	__global float4* resp_x2_back,
	__global float4* g_A,
	__local float4* s_A
) {
	int li = get_local_id(0);
	int ls = get_local_size(0);

	int wi = get_group_id(0);

	int gi = li + 2*wi*ls;

	float ksum_resps = 0;

	for (int c=0; c<k; c++) {
		//load shared mem
		if (gi < nSamplesCurrent) {
			s_resp[li] = resp[c*nSamplesCurrent + gi];
			s_resp_x[li] = resp_x[c*nSamplesCurrent + gi];
			s_resp_x2[li] = resp_x2[c*nSamplesCurrent + gi];
			
			if (gi + ls < nSamplesCurrent) {
				s_resp[li] += resp[c*nSamplesCurrent + gi + ls];
				s_resp_x[li] += resp_x[c*nSamplesCurrent + gi + ls];
				s_resp_x2[li] += resp_x2[c*nSamplesCurrent + gi + ls];
			}
		}
		else {
			s_resp[li] = 0;
			s_resp_x[li] = 0;
			s_resp_x2[li] = 0;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		//reduce in shared mem
		for (int s=ls/2; s>0; s/=2) {
			if (li < s) {
				s_resp[li] += s_resp[li + s];
				s_resp_x[li] += s_resp_x[li + s];
				s_resp_x2[li] += s_resp_x2[li + s];
			}

			barrier(CLK_LOCAL_MEM_FENCE);
		}

		float h = NDIM*log((float) (2*M_PI));
		//float log_nSamples = log((float) nSamples);

		//write result to global mem
		if (nSamplesReduced == 1) {
			if (li == 0) {
				float one_over_resp_sum = (1.0f/(s_resp[0]) + 10*FLT_MIN);
				float weight = s_resp[0]/nSamples;
				float4 mean = one_over_resp_sum * s_resp_x[0];
				float4 covar = one_over_resp_sum * (s_resp_x2[0] - 2.0f*mean*s_resp_x[0]) + mean*mean + MIN_COVAR4;
				
				//resp_back[c] = weight; 
				//resp_x_back[c] = mean;
				//resp_x2_back[c] = covar;
				
				//float one_over_resp_sum = (1.0f/s_resp[0]);
				//float4 mean = one_over_resp_sum * s_resp_x[0];
				float4 mean2 = mean*mean;
				float4 one_over_covar = pow((one_over_resp_sum * (s_resp_x2[0] - 2.0f*mean*s_resp_x[0]) + mean2 + MIN_COVAR4), -1);

				mean.w = mean.z;
				mean.z = mean.y;
				mean.y = mean.x;
				mean.x = 0;

				mean2.w = mean2.z;
				mean2.z = mean2.y;
				mean2.y = mean2.x;
				mean2.x = 0;

				one_over_covar.w = one_over_covar.z;
				one_over_covar.z = one_over_covar.y;
				one_over_covar.y = one_over_covar.x;
				one_over_covar.x = 0;

				s_A[c*2+1] = -0.5f*one_over_covar;

				mean *= one_over_covar; // <- mean/covar
				mean2 *= one_over_covar; // <- mean2/covar
				mean2.y += mean2.z + mean2.w; // <- sum(mean2/covar)
				one_over_covar = -log(one_over_covar); // <- log(covar) or -log(1/covar)
				one_over_covar.y += one_over_covar.z + one_over_covar.w; // <- sum(log(covar))
				//mean.x = log(s_resp[0]) - log_nSamples - 0.5f*(h + one_over_covar.y + mean2.y);
				mean.x = log(s_resp[0]) - 0.5f*(h + one_over_covar.y + mean2.y);

				ksum_resps += s_resp[0];

				s_A[c*2] = mean;
			}
		}
		else {
			if (li == 0) {
				resp_back[c*nSamplesReduced + wi] = s_resp[0];
				resp_x_back[c*nSamplesReduced + wi] = s_resp_x[0];
				resp_x2_back[c*nSamplesReduced + wi] = s_resp_x2[0];
			}
		}
	}

	if (nSamplesReduced == 1) {
		if (li == 0) {
			for (int c=0; c<k; c++) {
				s_A[2*c].x -= log(ksum_resps);
			}
		}		

		if (li < 2*k) {
			g_A[li] = s_A[li];
		}
	}
}




__kernel void calc_weights(
	__global uint* samples,
	__constant float4* gA,
	__local float4* sA,
	int m,
	int nSamples,
	__global float* responsibilities,
	__global float4* means
) {
	int gi = get_global_id(0);
	int gs = get_global_size(0);

	int li = get_local_id(0);

	if (li < m*2) {
		sA[li] = gA[li];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (gi > nSamples-1)
		return;

	uint s = samples[gi];
	float4 x = (float4) (1, s & 0x000000FF, (s & 0x0000FF00) >> 8, (s & 0x00FF0000) >> 16);
	//float4 x2 = pown(x, 2); <-- bug in OpenCL

	float4 x2 = x*x;
	x2[0] = 0;

	float e[MAX_NCOMPONENT];

	for (int i=0; i<m; i++) {
		int is = i*2;

		float4 a = sA[is];//(float4) (sA[is+0], sA[is+1], sA[is+2], sA[is+3]);
		e[i] = dot(a, x);

		a = sA[is+1];// (float4) (sA[is+4], sA[is+5], sA[is+6], sA[is+7]);
		e[i] += dot(a, x2);
	}

	float eMax = e[0];

	for (int i=1; i<m; i++) {
		if (e[i] > eMax)
			eMax = e[i];
	}

	float sum = 0;
	for (int i=0; i<m; i++) {
		sum += exp(e[i] - eMax);
	}

	float lpr = eMax + log(sum);

	x = rgba2float4(s);

	for (int k=0; k<m; k++) {
		float r = exp(e[k] - lpr);

		responsibilities[k*nSamples + gi] = r;
		means[k*nSamples + gi] = r*x;
	}
}

__kernel void m_step2(
	__global float* responsibilities,
	__global uint* samples,
	__global float4* means,
	__local float4* sMeans,
	int k,
	int nSamples,
	__global float4* covars
)
{
	int gi = get_global_id(0);
	int gs = get_global_size(0);

	int li = get_local_id(0);

	if (li < k) {
		sMeans[li] = means[li];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (gi > nSamples-1)
		return;

	uint s = samples[gi];
	float4 x = rgba2float4(s);

	for (int c=0; c<k; c++) {
		float4 diff = x - sMeans[c];
		diff *= diff;
		diff *= responsibilities[c*nSamples + gi];

		covars[c*nSamples + gi] = diff;
	}
}

__kernel void m_step1(
	__global float* responsibilies,
	__global float4* means,
	__local float* sResponsibilies,
	__local float4* sMeans,
	int k,
	int nSamples,
	int nSamplesCurrent,
	int nSamplesReduced,
	__global float* responsibilies2,
	__global float4* means2,
	__global float* weights
) {
	int li = get_local_id(0);
	int ls = get_local_size(0);

	int wi = get_group_id(0);

	int gi = li + 2*wi*ls;

	for (int c=0; c<k; c++) {
		//load shared mem
		if (gi < nSamplesCurrent) {
			sResponsibilies[li] = responsibilies[c*nSamplesCurrent + gi];
			sMeans[li] = means[c*nSamplesCurrent + gi];
			
			if (gi + ls < nSamplesCurrent) {
				sResponsibilies[li] += responsibilies[c*nSamplesCurrent + gi + ls];
				sMeans[li] += means[c*nSamplesCurrent + gi + ls];
			}
		}
		else {
			sResponsibilies[li] = 0;
			sMeans[li] = 0;
		}

		//reduce in shared mem
		for (int s=ls/2; s>0; s/=2) {
			if (li < s) {
				sResponsibilies[li] += sResponsibilies[li + s];
				sMeans[li] += sMeans[li + s];
			}

			barrier(CLK_LOCAL_MEM_FENCE);
		}	

		//write result to global mem
		if (li == 0) {
			responsibilies2[c*nSamplesReduced + wi] = sResponsibilies[0];

			if (nSamplesReduced == 1) {
				means2[c*nSamplesReduced + wi] = (1.0f/sResponsibilies[0]) * sMeans[0];
				weights[c] = sResponsibilies[0]/nSamples;
			}
			else
				means2[c*nSamplesReduced + wi] = sMeans[0];
		}
	}
}

__kernel void m_step1_0(
	__global float* responsibilies,
	__global float4* means,
	__local float* sResponsibilies,
	__local float4* sMeans,
	int m,
	int nSamples,
	int nSamplesReduced,
	__global float* responsibilies2,
	__global float4* means2
) {
	int gi = get_global_id(0);
	int gs = get_global_size(0);

	int li = get_local_id(0);
	int ls = get_local_size(0);

	int wi = get_group_id(0);

	float r_sum;
	float4 m_sum;

	for (int k=0; k<m; k++) {
		sResponsibilies[li] = responsibilies[k*nSamples + gi];
		sMeans[li] = means[k*nSamples + gi];

		barrier(CLK_LOCAL_MEM_FENCE);

		if (li == 0) {
			int sumto = min(ls, nSamples - gi);

			r_sum = 0;
			m_sum = (float4) (0, 0, 0, 0);

			for (int i=0; i<sumto; i++) {
				r_sum += sResponsibilies[i];
				m_sum += sMeans[i];
			}

			responsibilies2[k*nSamplesReduced + wi] = r_sum;
			means2[k*nSamplesReduced + wi] = m_sum;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void eval(
	__global uint* samples,
	__constant float4* gA,
	__local float4* sA,
	int nComponents,
	int nSamples,
	__global float* out
) {
	int gi = get_global_id(0);
	int gs = get_global_size(0);

	int wi = get_group_id(0); //workgroup id
	int ws = get_num_groups(0);

	int li = get_local_id(0);
	int ls = get_local_size(0);

	if (li < nComponents*2) {
		sA[li] = gA[li];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (gi > nSamples-1)
		return;

	uint s = samples[gi];

	float4 x = (float4) (1, s & 0x000000FF, (s & 0x0000FF00) >> 8, (s & 0x00FF0000) >> 16);
	//float4 x2 = pown(x, 2); <-- bug in OpenCL

	float4 x2 = x*x;
	x2[0] = 0;

	float e[MAX_NCOMPONENT];

	for (int i=0; i<nComponents; i++) {
		int is = i*2;
		float4 a = sA[is];//(float4) (sA[is+0], sA[is+1], sA[is+2], sA[is+3]);

		e[i] = dot(a, x);
		a = sA[is+1];// (float4) (sA[is+4], sA[is+5], sA[is+6], sA[is+7]);

		e[i] += dot(a, x2);
	}

	float eMax = e[0];

	for (int i=1; i<nComponents; i++) {
		if (e[i] > eMax)
			eMax = e[i];
	}

	float sum = 0;
	for (int i=0; i<nComponents; i++) {
		sum += exp(e[i] - eMax);
	}

	out[gi] = -(eMax + log(sum));
}

__kernel void score(
	__global uint* src,
	__constant float4* gA,
	__local float4* sA,
	int nPopulation,
	int nComponents,
	__global float* out
) {
	int gi = get_global_id(0);

	int wi = get_group_id(0); //workgroup id
	int ws = get_num_groups(0);

	int ls = get_local_size(0);
	int li = get_local_id(0);

	if (li < nComponents*2) {
		sA[li] = gA[li];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (gi > nPopulation-1)
		return;

	uint s = src[gi];

	float4 x = (float4) (1, s & 0x000000FF, (s & 0x0000FF00) >> 8, (s & 0x00FF0000) >> 16);
	//float4 x2 = pown(x, 2); <-- bug in OpenCL

	float4 x2 = x*x;
	x2[0] = 0;

	float e[MAX_NCOMPONENT];

	for (int i=0; i<nComponents; i++) {
		int is = i*2;
		float4 a = sA[is];//(float4) (sA[is+0], sA[is+1], sA[is+2], sA[is+3]);

		e[i] = dot(a, x);
		a = sA[is+1];// (float4) (sA[is+4], sA[is+5], sA[is+6], sA[is+7]);

		e[i] += dot(a, x2);
	}

	float eMax = e[0];

	for (int i=1; i<nComponents; i++) {
		if (e[i] > eMax)
			eMax = e[i];
	}

	float sum = 0;
	for (int i=0; i<nComponents; i++) {
		sum += exp(e[i] - eMax);
	}

	out[gi] = -(eMax + log(sum));
}

__kernel void sampleBg(
	__global uint* src,
	__global uint* tri,
	uint fg,
	int gw,
	int gh,
	__global uint* samples
)
{
	int gs = gw*gh;
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gxy = gy*gw + gx;

	int ws = get_num_groups(0)*get_num_groups(1);
	int wx = get_group_id(0); //workgroup id
	int wy = get_group_id(1);
	int wxy = wy*get_num_groups(0) + wx;

	int lx = get_local_id(0);
	int ly = get_local_id(1);

	if (gxy > gs-1)
		return;

	int i;
	if (lx == 0 && ly == 0)
		i = 0;
	if (lx == 7 && ly == 0)
		i = 1;
	if (lx == 0 && ly == 7)
		i = 2;
	if (lx == 7 && ly == 7)
		i = 3;

	if (tri[gxy] != fg) {
		samples[4*wxy+i] = src[gxy];
	}
	else {
		samples[4*wxy+i] = 0x00000000;
	}
}