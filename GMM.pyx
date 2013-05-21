import pyopencl as cl
import numpy as np
from clutil import createProgram, ceil_divi, roundUp
import os
from Image2D import Image2D
cimport numpy as cnp

DEF NDIM = 3
DEF INIT_COVAR = 30.0
DEF MIN_COVAR = 0.001
#cdef int A_W = roundUp(2 * NDIM + 1, 4)
DEF A_W = 8

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

cdef class GMM:
    cdef:
        int n_iterations
        int n_components
        int cap_samples
        tuple local_worksize

        float init_covar
        float min_covar

        list logLiklihoods
        public int has_converged
        public int has_preset_wmc

        object context
        object queue

        object kernEM1
        object kernEM2
        object kernEval
        object kernCheckConverge
        object kernScore_buf
        object kernScore_img2d
        object kernInitA

        cnp.ndarray hResp
        cnp.ndarray hEval
        cnp.ndarray hA

        readonly object dA
        object dEval
        object dEval_back
        object dResp
        object dResp_back
        object dResp_x
        object dResp_x_back
        object dResp_x2
        object dResp_x2_back



    def __init__(self, context, n_iterations, n_components, cap_samples=None, min_covar=None, init_covar=None):
        self.context = context
        self.n_iterations = n_iterations
        self.n_components = n_components
        self.cap_samples = cap_samples
        self.local_worksize = (256, )

        self.init_covar = init_covar if (init_covar) else INIT_COVAR
        self.min_covar = min_covar if (min_covar) else MIN_COVAR

        options = [
            '-D INIT_COVAR={0:.1}f'.format(self.init_covar),
            '-D MIN_COVAR={0:.1}f'.format(self.min_covar)
        ]

        self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)

        filename = os.path.join(os.path.dirname(__file__), 'gmm.cl')
        program = createProgram(self.context, self.context.devices, options, filename)

        self.kernEM1 = cl.Kernel(program, 'em1')
        self.kernEM2 = cl.Kernel(program, 'em2')
        self.kernEval = cl.Kernel(program, 'eval')
        self.kernCheckConverge = cl.Kernel(program, 'check_converge')
        self.kernScore_buf = cl.Kernel(program, 'score_buf')
        self.kernScore_img2d = cl.Kernel(program, 'score_img2d')
        self.kernInitA = cl.Kernel(program, 'initA')

        if cap_samples != None:
            self.initClMem()

        self.logLiklihoods = [None] * n_iterations
        self.has_converged = False
        self.has_preset_wmc = False


    def initClMem(self):
        capSamples = self.cap_samples
        nComps = self.n_components

        self.hA = np.empty((nComps, A_W), np.float32)
        self.dA = cl.Buffer(self.context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hA)

        shpFront1 = (capSamples)
        shpBack1 = ceil_divi(capSamples, 2 * self.local_worksize[0])
        szFront1 = int(np.prod(shpFront1))
        szBack1 = int(np.prod(shpBack1))

        shpFront2 = (nComps, capSamples)
        shpBack2 = (nComps, ceil_divi(capSamples, 2 * self.local_worksize[0]))
        szFront2 = int(np.prod(shpFront2))
        szBack2 = int(np.prod(shpBack2))

        shpFront3 = (nComps, capSamples, 4)
        shpBack3 = (nComps, ceil_divi(capSamples, 2 * self.local_worksize[0]), 4)
        szFront3 = int(np.prod(shpFront3))
        szBack3 = int(np.prod(shpBack3))

        self.hResp = np.empty(shpFront2, np.float32)
        self.hEval = np.empty(shpBack1, np.float32)
#        self.dSamples = cl.Buffer(self.context, cm.READ_ONLY, szInt * szFront1)
        self.dEval = cl.Buffer(self.context, cm.READ_WRITE, szFloat * szFront1)
        self.dEval_back = cl.Buffer(self.context, cm.READ_WRITE, szFloat * szBack1)
        self.dResp = cl.Buffer(self.context, cm.READ_ONLY, szFloat * szFront2)
        self.dResp_back = cl.Buffer(self.context, cm.READ_ONLY, szFloat * szBack2)
        self.dResp_x = cl.Buffer(self.context, cm.READ_ONLY, szFloat * szFront3)
        self.dResp_x_back = cl.Buffer(self.context, cm.READ_ONLY, szFloat * szBack3)
        self.dResp_x2 = cl.Buffer(self.context, cm.READ_ONLY, szFloat * szFront3)
        self.dResp_x2_back = cl.Buffer(self.context, cm.READ_ONLY, szFloat * szBack3)


    def score(self, d_population, dScore):
        args = [
            self.dA,
            cl.LocalMemory(4 * self.hA.size),
            np.int32(self.n_components),
            dScore,
        ]

        if isinstance(d_population, cl.Buffer):
            n_population = d_population.size / szFloat
            args += [
                d_population,
                np.int32(n_population),
            ]
            kern = self.kernScore_buf

            gWorksize = roundUp((n_population, ), self.local_worksize)
            kern(self.queue, gWorksize, self.local_worksize, *args).wait()

        elif type(d_population) == Image2D:
            n_population = d_population.size / szFloat
            args += [
                d_population,
                cl.Sampler(self.context, False, cl.addressing_mode.NONE,
                    cl.filter_mode.NEAREST),
                np.int32(n_population),
                ]
            kern = self.kernScore_img2d

            gWorksize = roundUp(d_population.dim, (16, 16))
            kern(self.queue, gWorksize, (16, 16), *args).wait()

        else:
            raise NotImplementedError()

    def fit(self, d_samples, nSamples=None, retParams=None):
        if nSamples == None:
            nSamples = d_samples.size / szInt

        if nSamples < self.n_components:
            raise ValueError('nSamples < nComp: {0}, {1}'.format(nSamples,
                self.n_components))

        if self.cap_samples == None or nSamples > self.cap_samples:
            self.cap_samples = nSamples
            self.initClMem()

        argsEM1 = [
            d_samples,
            self.dA,
            cl.LocalMemory(4 * self.hA.size),
            np.int32(self.n_components),
            np.int32(nSamples),
            self.dResp,
            self.dResp_x,
            self.dResp_x2
        ]

        argsEM2 = [
            None,
            None,
            None,
            cl.LocalMemory(4 * self.local_worksize[0]),
            cl.LocalMemory(4 * 4 * self.local_worksize[0]),
            cl.LocalMemory(4 * 4 * self.local_worksize[0]),
            np.int32(self.n_components),
            np.int32(nSamples),
            None,
            None,
            None,
            None,
            None,
            self.dA,
            cl.LocalMemory(4 * self.hA.size),
        ]

        argsEval = [
            d_samples,
            self.dA,
            cl.LocalMemory(4 * self.hA.size),
            np.int32(self.n_components),
            np.int32(nSamples),
            self.dEval
        ]

        argsCheckEval = [
            None,
            cl.LocalMemory(4 * self.local_worksize[0]),
            None,
            None,
            None,
        ]

        argsInitA = [
            d_samples,
            np.int32(self.n_components),
            np.int32(nSamples),
            self.dA,
        ]

        if self.has_converged == False and self.has_preset_wmc == False:
            self.kernInitA(self.queue, roundUp((self.n_components, ), (16, )), (16, ), *(argsInitA)).wait()

        self.has_converged = False
        self.has_preset_wmc = False

        cdef int nSamplesCurrent
        cdef int nSamplesReduced
        cdef tuple gWorksize
        cdef object dTmp
        cdef int i

        for i in xrange(self.n_iterations):
            gWorksize = roundUp((nSamples, ), self.local_worksize)

            self.kernEM1(self.queue, gWorksize, self.local_worksize, *(argsEM1)).wait()

            nSamplesCurrent = nSamples

            dRespIn = self.dResp
            dResp_xIn = self.dResp_x
            dResp_x2In = self.dResp_x2

            dRespOut = self.dResp_back
            dResp_xOut = self.dResp_x_back
            dResp_x2Out = self.dResp_x2_back

            while nSamplesCurrent != 1:
                nSamplesReduced = ceil_divi(nSamplesCurrent, 2 * self.local_worksize[0]);

                argsEM2[0] = dRespIn
                argsEM2[1] = dResp_xIn
                argsEM2[2] = dResp_x2In
                argsEM2[8] = np.int32(nSamplesCurrent)
                argsEM2[9] = np.int32(nSamplesReduced)
                argsEM2[10] = dRespOut
                argsEM2[11] = dResp_xOut
                argsEM2[12] = dResp_x2Out

                #lWorksize = (256, )
                gWorksize = roundUp((ceil_divi(nSamplesCurrent, 2), ), self.local_worksize)

                self.kernEM2(self.queue, gWorksize, self.local_worksize, *(argsEM2)).wait()

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

            gWorksize = roundUp((nSamples, ), self.local_worksize)

            self.kernEval(self.queue, gWorksize, self.local_worksize, *(argsEval)).wait()

            nSamplesCurrent = nSamples

            dEvalIn = self.dEval
            dEvalOut = self.dEval_back

            while nSamplesCurrent != 1:
                nSamplesReduced = ceil_divi(nSamplesCurrent, 2 * self.local_worksize[0]);

                argsCheckEval[0] = dEvalIn
                argsCheckEval[2] = np.int32(nSamplesCurrent)
                argsCheckEval[3] = np.int32(nSamplesReduced)
                argsCheckEval[4] = dEvalOut

                gWorksize = roundUp((nSamplesCurrent, ), self.local_worksize)

                self.kernCheckConverge(self.queue, gWorksize, self.local_worksize, *(argsCheckEval)).wait()

                if nSamplesReduced == 1:
                    break;

                dTmp = dEvalIn
                dEvalIn = dEvalOut
                dEvalOut = dTmp

                nSamplesCurrent = nSamplesReduced

            cl.enqueue_copy(self.queue, self.hEval, dEvalOut).wait()

            self.logLiklihoods[i] = self.hEval[0]

            if i > 1:
                diff = np.abs(self.logLiklihoods[i] - self.logLiklihoods[i - 1])
#                print i, self.logLiklihoods[i], diff

                if diff < 0.01:
                    self.has_converged = True
                    break
                #else:
                #	print i, self.logLiklihoods[i]

#        print 'converged: {0} ({1})'.format(self.converged, i)

        if retParams:
            nSamplesCurrent = nSamples

            while True:
                nSamplesReduced = ceil_divi(nSamplesCurrent, 2 * self.local_worksize[0]);

                if nSamplesReduced == 1:
                    break;

                nSamplesCurrent = nSamplesReduced

            hResp = np.empty(dRespIn.size / szFloat, np.float32)
            hResp_x = np.empty(dResp_xIn.size / szFloat, np.float32)
            hResp_x2 = np.empty(dResp_x2In.size / szFloat, np.float32)

            cl.enqueue_copy(self.queue, hResp, dRespIn).wait()
            cl.enqueue_copy(self.queue, hResp_x, dResp_xIn).wait()
            cl.enqueue_copy(self.queue, hResp_x2, dResp_x2In).wait()

            resps = np.sum(hResp.ravel()[0:self.n_components * nSamplesCurrent]
            .reshape(self.n_components,
                nSamplesCurrent), axis=1)
            resps_x = np.sum(
                hResp_x.ravel()[0:4 * self.n_components * nSamplesCurrent].reshape(self.n_components,
                    nSamplesCurrent, 4), axis=1)
            resps_x2 = np.sum(
                hResp_x2.ravel()[0:4 * self.n_components * nSamplesCurrent].reshape(self.n_components,
                    nSamplesCurrent, 4), axis=1)
            one_over_resps = (1.0 / resps).reshape(self.n_components, 1)

            weights = resps / np.sum(resps)
            means = one_over_resps * resps_x
            covars = one_over_resps * (
            resps_x2 - 2 * means * resps_x) + means * means

            return weights, means, covars

    @staticmethod
    def calcA_cpu(weights, means, covars, A=None):
        nComp = weights.shape[0]

        if A == None:
            ret = True
            A = np.empty((nComp, 8), np.float32)
        else:
            ret = False

        for i in xrange(0, nComp):
            w = np.log(weights[i])
            h = -0.5 * (3 * np.log(2 * np.pi) + np.sum(np.log(covars[i])) + np.sum(
                (means[i] ** 2) / (covars[i])) )

            A[i][0] = w + h
            A[i][1:4] = (means[i] / covars[i])
            A[i][4] = 0
            A[i][5:8] = -1 / (2 * covars[i])

        if ret:
            return A