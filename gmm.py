__author__ = 'Marc de Klerk'

import pyopencl as cl
import numpy as np
from clutil import createProgram, cl, roundUp, ceil_divi, padArray2D
import os

NDIM_MAX = 3
INIT_COVAR = 30.0
MIN_COVAR = 0.001
A_W = roundUp(2*NDIM_MAX + 1, 4)

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

class GMM:
	def __init__(self, context, nIter, nComps, capSamples=None, min_covar=None, init_covar=None):
		self.context = context
		self.nIter = nIter
		self.nComps = nComps
		self.capSamples = capSamples
		self.lWorksize = (256, )

		self.init_covar = init_covar if (init_covar) else INIT_COVAR
		self.min_covar = min_covar if (min_covar) else MIN_COVAR

		options = [
			'-D INIT_COVAR='+str(self.init_covar)+'f',
			'-D MIN_COVAR='+str(self.min_covar)+'f'
		]

		filename = os.path.join(os.path.dirname(__file__), 'gmm.cl')
		program = createProgram(self.context, self.context.devices, options, filename)

		self.kernEM1 = cl.Kernel(program, 'em1')
		self.kernEM2 = cl.Kernel(program, 'em2')
		self.kernEval = cl.Kernel(program, 'eval')
		self.kernCheckConverge = cl.Kernel(program, 'check_converge')
		self.kernScore = cl.Kernel(program, 'score')
		self.kernInitA = cl.Kernel(program, 'initA')

		if capSamples != None:
			self.initClMem()

		self.logLiklihoods = [None]*nIter
		self.converged = False

	def initClMem(self):
		capSamples = self.capSamples
		nComps = self.nComps

		self.hA = np.empty((nComps, A_W), np.float32)
		self.dA = cl.Buffer(self.context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hA)

		shpFront1 = (capSamples)
		shpBack1 = ceil_divi(capSamples, 2*self.lWorksize[0])
		szFront1 = int(np.prod(shpFront1))
		szBack1 = int(np.prod(shpBack1))

		shpFront2 = (nComps, capSamples)
		shpBack2 = (nComps, ceil_divi(capSamples, 2*self.lWorksize[0]))
		szFront2 = int(np.prod(shpFront2))
		szBack2 = int(np.prod(shpBack2))

		shpFront3 = (nComps, capSamples, 4)
		shpBack3 = (nComps, ceil_divi(capSamples, 2*self.lWorksize[0]), 4)
		szFront3 = int(np.prod(shpFront3))
		szBack3 = int(np.prod(shpBack3))

		self.hResp = np.empty(shpFront2, np.float32)
		self.hEval = np.empty(shpBack1, np.float32)
		self.dSamples = cl.Buffer(self.context, cm.READ_ONLY, szInt*szFront1)
		self.dEval = cl.Buffer(self.context, cm.READ_WRITE, szFloat*szFront1)
		self.dEval_back = cl.Buffer(self.context, cm.READ_WRITE, szFloat*szBack1)
		self.dResp = cl.Buffer(self.context, cm.READ_ONLY, szFloat*szFront2)
		self.dResp_back = cl.Buffer(self.context, cm.READ_ONLY, szFloat*szBack2)
		self.dResp_x = cl.Buffer(self.context, cm.READ_ONLY, szFloat*szFront3)
		self.dResp_x_back = cl.Buffer(self.context, cm.READ_ONLY, szFloat*szBack3)
		self.dResp_x2 = cl.Buffer(self.context, cm.READ_ONLY, szFloat*szFront3)
		self.dResp_x2_back = cl.Buffer(self.context, cm.READ_ONLY, szFloat*szBack3)

		self.argsEM1 = [
			self.dSamples,
			self.dA,
			cl.LocalMemory(4*self.hA.size),
			np.int32(nComps),
			None,
			self.dResp,
			self.dResp_x,
			self.dResp_x2
			]

		self.argsEM2 = [
			None,
			None,
			None,
			cl.LocalMemory(4*self.lWorksize[0]),
			cl.LocalMemory(4*4*self.lWorksize[0]),
			cl.LocalMemory(4*4*self.lWorksize[0]),
			np.int32(nComps),
			None,
			None,
			None,
			None,
			None,
			None,
			self.dA,
			cl.LocalMemory(4*self.hA.size),
			]

		self.argsEval = [
			self.dSamples,
			self.dA,
			cl.LocalMemory(4*self.hA.size),
			np.int32(nComps),
			None,
			self.dEval
		]

		self.argsCheckEval = [
			None,
			cl.LocalMemory(4*self.lWorksize[0]),
			None,
			None,
			None,
			]

		self.argsScore = [
			None,
			self.dA,
			cl.LocalMemory(4*self.hA.size),
			None,
			None,
			None,
		]

		self.argsInitA = [
			self.dSamples,
			np.int32(nComps),
			None,
			self.dA,
			]

	def score(self, queue, population, dScore=None):
		if type(population) == np.ndarray:
			nPopulation = population.size
			dPopulation = cl.Buffer(self.context, cm.READ_ONLY, szInt*nPopulation)
			cl.enqueue_copy(queue, dPopulation, population).wait()
		elif type(population) == cl.Buffer or type(population) == cl.GLBuffer:
			nPopulation = population.size/szFloat
			dPopulation = population
		else:
			raise TypeError("Invalid input array, must be either numpy.ndarray or pyopencl.Buffer")

		if dScore == None:
			hScore = np.empty((nPopulation, ), np.float32)
			dScore = cl.Buffer(self.context, cm.WRITE_ONLY, szFloat*nPopulation)
			ret = True
		elif type(dScore) == cl.Buffer or type(dScore) == cl.GLBuffer:
			ret = False
		else:
			raise TypeError("dScore must be pyopencl.Buffer")

		self.argsScore[0] = dPopulation
		self.argsScore[3] = np.int32(nPopulation)
		self.argsScore[4] = np.int32(self.nComps)
		self.argsScore[5] = dScore

		gWorksize = roundUp((nPopulation, ), self.lWorksize)

		event = self.kernScore(queue, gWorksize, self.lWorksize, *(self.argsScore))
		event.wait()

		if ret:
			cl.enqueue_copy(queue, hScore, dScore).wait()
			return hScore

	def fit(self, queue, samples, nSamples=None, weights=None, means=None, covars=None, retParams=False):
		nComp = self.nComps
		#on cpu
		if type(samples) == np.ndarray:
			if nSamples == None:
				nSamples = samples.shape[0]

			if samples.shape[-1] == 3 and samples.dtype == np.float32:
				samples = np.pad(samples.astype(np.uint8), [(0, 0), (0, 1)], 'constant')
				samples = samples.view(np.uint32)

			cl.enqueue_copy(queue, self.dSamples, samples).wait()
		else:
			if nSamples == None:
				nSamples = samples.size/szInt

		if nSamples < nComp:
			raise ValueError('nSamples < nComp: {0}, {1}'.format(nSamples, nComp))

		if self.capSamples == None or nSamples > self.capSamples:
			self.capSamples = nSamples
			self.initClMem()

		self.argsEM1[4] = np.int32(nSamples)
		self.argsEM2[7] = np.int32(nSamples)
		self.argsEval[4] = np.int32(nSamples)
		self.argsInitA[2] = np.int32(nSamples)

		if (weights != None and means != None and covars != None):
			if weights == None:
				weights = np.tile(1.0 / nComp, nComp).astype(np.float32)

			if means == None:
				means = samples[0:4]
				means = means.view(np.uint8).reshape(4, 4)[:, 0:3].astype(np.float32)

			if covars == None:
				covars = np.empty((nComp, 3), np.float32)
				covars[:] = 10

			calcA_cpu(weights, means, covars, self.hA)
			cl.enqueue_copy(queue, self.dA, self.hA).wait()
		elif self.converged == False:
			self.kernInitA(queue, roundUp((nComp, ), (16, )), (16, ), *(self.argsInitA)).wait()

		self.converged = False

		for i in xrange(self.nIter):
			gWorksize = roundUp((nSamples, ), self.lWorksize)

			self.kernEM1(queue, gWorksize, self.lWorksize, *(self.argsEM1)).wait()

			nSamplesCurrent = nSamples

			dRespIn = self.dResp
			dResp_xIn = self.dResp_x
			dResp_x2In = self.dResp_x2

			dRespOut = self.dResp_back
			dResp_xOut = self.dResp_x_back
			dResp_x2Out = self.dResp_x2_back

			while nSamplesCurrent != 1:
				nSamplesReduced = ceil_divi(nSamplesCurrent, 2*self.lWorksize[0]);

				self.argsEM2[0] = dRespIn
				self.argsEM2[1] = dResp_xIn
				self.argsEM2[2] = dResp_x2In
				self.argsEM2[8] = np.int32(nSamplesCurrent)
				self.argsEM2[9] = np.int32(nSamplesReduced)
				self.argsEM2[10] = dRespOut
				self.argsEM2[11] = dResp_xOut
				self.argsEM2[12] = dResp_x2Out

				#lWorksize = (256, )
				gWorksize = roundUp((ceil_divi(nSamplesCurrent, 2), ), self.lWorksize)

				self.kernEM2(queue, gWorksize, self.lWorksize, *(self.argsEM2)).wait()

				if nSamplesReduced == 1:
					break;

				dTmp = dRespIn
				dRespIn = dRespOut
				dRespOut = dTmp

				dTmp = dResp_xIn
				dResp_xIn = dResp_xOut
				dResp_xOut = dTmp

				dTmp = dResp_x2In
				dResp_x2In = dResp_x2Out
				dResp_x2Out = dTmp

				nSamplesCurrent = nSamplesReduced

			gWorksize = roundUp((nSamples, ), self.lWorksize)

			self.kernEval(queue, gWorksize, self.lWorksize, *(self.argsEval)).wait()

			nSamplesCurrent = nSamples

			dEvalIn = self.dEval
			dEvalOut = self.dEval_back

			while nSamplesCurrent != 1:
				nSamplesReduced = ceil_divi(nSamplesCurrent, 2*self.lWorksize[0]);

				self.argsCheckEval[0] = dEvalIn
				self.argsCheckEval[2] = np.int32(nSamplesCurrent)
				self.argsCheckEval[3] = np.int32(nSamplesReduced)
				self.argsCheckEval[4] = dEvalOut

				gWorksize = roundUp((nSamplesCurrent, ), self.lWorksize)

				self.kernCheckConverge(queue, gWorksize, self.lWorksize, *(self.argsCheckEval)).wait()

				if nSamplesReduced == 1:
					break;

				dTmp = dEvalIn
				dEvalIn = dEvalOut
				dEvalOut = dTmp

				nSamplesCurrent = nSamplesReduced

			cl.enqueue_copy(queue, self.hEval, dEvalOut).wait()

			self.logLiklihoods[i] = self.hEval[0]

			if i > 1:
				diff = np.abs(self.logLiklihoods[i] - self.logLiklihoods[i-1])
				#print i, self.logLiklihoods[i], diff

				if diff < 0.01:
					self.converged = True
					break
			#else:
			#	print i, self.logLiklihoods[i]

		#print 'converged: {0} ({1})'.format(converged, i)

		if retParams:
			nSamplesCurrent = nSamples

			while True:
				nSamplesReduced = ceil_divi(nSamplesCurrent, 2*self.lWorksize[0]);

				if nSamplesReduced == 1:
					break;

				nSamplesCurrent = nSamplesReduced

			hResp = np.empty(dRespIn.size/szFloat, np.float32)
			hResp_x = np.empty(dResp_xIn.size/szFloat, np.float32)
			hResp_x2 = np.empty(dResp_x2In.size/szFloat, np.float32)

			cl.enqueue_copy(queue, hResp, dRespIn).wait()
			cl.enqueue_copy(queue, hResp_x, dResp_xIn).wait()
			cl.enqueue_copy(queue, hResp_x2, dResp_x2In).wait()

			resps = np.sum(hResp.ravel()[0:nComp*nSamplesCurrent].reshape(nComp, nSamplesCurrent), axis=1)
			resps_x = np.sum(hResp_x.ravel()[0:4* nComp *nSamplesCurrent].reshape(nComp, nSamplesCurrent, 4), axis=1)
			resps_x2 = np.sum(hResp_x2.ravel()[0:4* nComp *nSamplesCurrent].reshape(nComp, nSamplesCurrent, 4), axis=1)
			one_over_resps = (1.0/resps).reshape(nComp, 1)

			weights = resps/np.sum(resps)
			means = one_over_resps*resps_x
			covars = one_over_resps*(resps_x2 -2*means*resps_x) + means*means

			return weights, means, covars

def calcA_cpu(weights, means, covars, A=None):
	nComp = weights.shape[0]

	if A == None:
		ret = True
		A = np.empty((nComp, 8), np.float32)
	else:
		ret = False

	for i in xrange(0, nComp):
		w = np.log(weights[i])
		h = -0.5*( 3*np.log(2*np.pi) + np.sum(np.log(covars[i])) + np.sum((means[i]**2) / (covars[i])) )

		A[i][0] = w + h
		A[i][1:4] = (means[i]/covars[i])
		A[i][4] = 0
		A[i][5:8] = -1/(2*covars[i])

	if ret:
		return A

def calcResp(A):
	resp = np.empty(hResp.shape, np.float32)

	for i in xrange(len(s)):
		p = s[i]

		v = np.array([1, p[0], p[1], p[2], 0, p[0]*p[0], p[1]*p[1], p[2]*p[2]])
		d = np.dot(hA, v)
		lpr = logsumexp(d)
		resp[:, i] = np.exp(d - lpr)

	return resp

if __name__ == "__main__":
	import sys
	import Image
	import time
	from sklearn import mixture

	platforms = cl.get_platforms()

	devices = platforms[0].get_devices()
	device = devices[1]
	context = cl.Context([device])
	queue = cl.CommandQueue(context,properties=cl.command_queue_properties.PROFILING_ENABLE)

	np.set_printoptions(suppress=True)

	src = Image.open("/Users/marcdeklerk/msc/code/dataset/processed/source/800x600/GT04.png")
	if src.mode != 'RGBA':
		src = src.convert('RGBA')

	width = src.size[0]
	height = src.size[1]
	shape = src.size
	shapeNp = (src.size[1], src.size[0])
	size = width * height

	lWorksize = (16, 16)

	hSrc = padArray2D(np.array(src).view(np.uint32).squeeze(), roundUp(shapeNp, lWorksize), 'edge')
	dSrc = cl.Buffer(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=hSrc)
	rgb = hSrc.reshape(-1, 1).view(np.uint8).astype(np.float32)[:, 0:3]

	x0 = 0
	x1 = 500
	y0 = 200
	y1 = 220

	rect = (y1-y0, x1-x0)

	nSamples = rect[0]*rect[1]
	nIter = 65
	nComp = 4
	nDim = 3

	samples = hSrc[y0:y1, x0:x1].ravel()
	dSamples = cl.Buffer(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=samples)
	samples = samples.reshape(nSamples, 1).view(np.uint8).astype(np.float32)[:, 0:3]

	weights = np.tile(1.0 / nComp, nComp).astype(np.float32)

	i = np.random.randint(0, nSamples-1, nComp)
	means = samples[i]

	covars = np.empty((nComp, 3), np.float32)
	covars[:] = 10

	gmm_cpu = mixture.GMM(nComp)
	gmm_cpu.dtype = np.float32
	gmm_cpu.init_params = ''
	gmm_cpu.means_ = means
	gmm_cpu.weights_ = weights
	gmm_cpu.covars_ = covars
	gmm_cpu.fit(samples)

	gmm = GMM(context, nIter, nComp, nSamples)
	w,m,c = gmm.fit(queue, samples, nSamples, weights, means, covars, retParams=True)
	#weights, means, covars = gmm.fit(samples, retParams=True)
	print 'converged: {0}'.format(gmm.converged)
	#gmm.fit(samples, 8, weights, means, covars)

	print gmm_cpu.weights_
	print w
	print
	print gmm_cpu.means_
	print m
	print
	print gmm_cpu.covars_
	print c

	#profile.run("gmm.fit(samples)")

	gmm_cpu.init_params = 'wmc'
	iter = 10

	#to estimate wmc on cpu
	elapsed = 0
	t1 = t2 = 0
	for i in xrange(iter):
		t1 = time.time()
		gmm_cpu.fit(samples)
		#a = calcA_cpu(gmm_cpu.weights_, gmm_cpu.means_, gmm_cpu.covars_)
		#cl.enqueue_copy(clQueue, gmm.dA, a).wait()
		#gmm.score(dSrc, dOut)
		elapsed += time.time()-t1
	print elapsed/iter

	hOut = np.empty((hSrc.shape), np.float32)
	dOut = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hOut)

	elapsed = 0
	t1 = t2 = 0
	for i in xrange(iter):
		t1 = time.time()
		gmm.converged = False
		#gmm.fit(samples)
		w,m,c = gmm.fit(queue, samples, nSamples, weights, means, covars, retParams=True)
		#gmm.fit(dOut, nSamples)
		#gmm.score(dSrc, dOut)
		elapsed += time.time()-t1
	print elapsed/iter

	#to estimate wmc for data already on gpu
	elapsed = 0
	t1 = t2 = 0
	for i in xrange(iter):
		t1 = time.time()
		gmm_cpu.fit(samples)
		#a = calcA_cpu(gmm_cpu.weights_, gmm_cpu.means_, gmm_cpu.covars_)
		#cl.enqueue_copy(clQueue, gmm.dA, a).wait()
		#gmm.score(dSrc, dOut)
		elapsed += time.time()-t1
	print 'cpu fit: ', elapsed/iter

	elapsed = 0
	t1 = t2 = 0
	for i in xrange(iter):
		t1 = time.time()
		gmm_cpu.fit(samples)
		#a = calcA_cpu(gmm_cpu.weights_, gmm_cpu.means_, gmm_cpu.covars_)
		#cl.enqueue_copy(clQueue, gmm.dA, a).wait()
		#gmm.score(dSrc, dOut)
		elapsed += time.time()-t1
	print 'cpu fit: ', elapsed/iter

	hOut = np.empty((hSrc.shape), np.float32)
	dOut = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hOut)


	elapsed = 0
	t1 = t2 = 0
	for i in xrange(iter):
		t1 = time.time()
		gmm.converged = False
		#gmm.fit(samples)
#		w,m,c = gmm.fit(queue, samples, nSamples, weights, means, covars, retParams=True)
		gmm.fit(queue, dSamples)
		#gmm.fit(dOut, nSamples)
		#gmm.score(dSrc, dOut)
		elapsed += time.time()-t1
	print 'gpu fit: ', elapsed/iter

True