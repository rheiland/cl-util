from Brush import Brush
import numpy as np
from clutil import createProgram, roundUp
from Buffer2D import Buffer2D
from GMM import GMM
import os, sys
import pyopencl as cl

DILATE = 5

class QuickBrush(Brush):
    lWorksize = (16, 16)

    def __init__(self, context, devices, d_img, d_labels):
        Brush.__init__(self, context, devices, d_labels)

        self.context = context
        self.queue = cl.CommandQueue(context,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

        nComponentsFg = 4
        nComponentsBg = 4
        self.nDim = 3

        self.dim = d_img.dim

        filename = os.path.join(os.path.dirname(__file__), 'quick.cl')
        program = createProgram(context, context.devices, [], filename)
        #        self.kernSampleBg = cl.Kernel(program, 'sampleBg')
        self.kern_get_samples = cl.Kernel(program, 'get_samples')

        self.lWorksize = (16, 16)
        self.gWorksize = roundUp(self.dim, self.lWorksize)

        nSamples = 4 * (self.gWorksize[0] / self.lWorksize[0]) * (
            self.gWorksize[1] / self.lWorksize[1])

        #		self.gmmFg_cpu = mixture.GMM(4)

        self.gmmFg = GMM(context, 65, nComponentsFg, 10240)
        self.gmmBg = GMM(context, 65, nComponentsBg, nSamples)

        self.hScore = np.empty(self.dim, np.float32)
        self.hSampleFg = np.empty((10240, ), np.uint32)
        self.hSampleBg = np.empty((12000, ), np.uint32)
        self.hA = np.empty((max(nComponentsFg, nComponentsBg), 8), np.float32)

        self.d_img = d_img

        cm = cl.mem_flags
        self.dSampleFg = cl.Buffer(context, cm.READ_WRITE, size=4 * 10240)
        self.dSampleBg = cl.Buffer(context, cm.READ_WRITE, size=4 * 12000)
        self.dA = cl.Buffer(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hA)
        self.dScoreFg = Buffer2D(context, cm.READ_WRITE, self.dim, np.float32)
        self.dScoreBg = Buffer2D(context, cm.READ_WRITE, self.dim, np.float32)

        #self.points = Set()

        self.capPoints = 200 * 200 * 300 #brush radius 200, stroke length 300
        self.points = np.empty((self.capPoints), np.uint32)

        #		self.colorize = Colorize.Colorize(clContext, clContext.devices)

        #        self.hTriFlat = self.hTri.reshape(-1)

        #        self.probBg(1200)

        self.h_img = np.empty(self.dim, np.uint32)
        self.h_img = self.h_img.ravel()
        cl.enqueue_copy(self.queue, self.h_img, self.d_img, origin=(0, 0), region=self.dim).wait()

        self.samples_bg_idx = np.random.randint(0, self.dim[0] * self.dim[1], 12000)
        self.hSampleBg = self.h_img[self.samples_bg_idx]

        cl.enqueue_copy(self.queue, self.dSampleBg, self.hSampleBg).wait()

        w,m,c = self.gmmBg.fit(self.dSampleBg, 300, retParams=True)

        print w
        print m
        print c

        self.gmmBg.score(self.d_img, self.dScoreBg)

        pass

    def draw(self, p0, p1):
        Brush.draw(self, p0, p1)
        #self.probFg(x1-20, x1+20, y1-20, y1+20)
        #return
        """color = self.colorTri[self.type]

        #self.argsScore[5] = np.int32(self.nComponentsFg)

        #seed = []
        hasSeeds = False
        redoBg = False

        minX = sys.maxint
        maxX = -sys.maxint
        minY = sys.maxint
        maxY = -sys.maxint

        for point in self.points[0:nPoints]:
            #if self.hTriFlat[point] != color:
                self.hTriFlat[point] = color
                #seed += point
                hasSeeds = True

                minX = min(minX, point%self.width)

                maxX = max(maxX, point%self.width)
                minY = min(minY, point/self.width)
                maxY = max(maxY, point/self.width)

                #if (point[1]*self.width + point[0]) in self.randIdx:

                #	redoBg = True
        #if redoBg:
        #	self.probBg(0)

        #if len(seed) == 0:
        if not hasSeeds:
            return

        minX = max(0, minX-DILATE)
        maxX = min(self.width-1, maxX + DILATE)
        minY = max(0, minY-DILATE)
        maxY = min(self.height-1, maxY + DILATE)
        """

        args = [
            np.int32(self.n_points),
            self.d_points,
            cl.Sampler(self.context, False, cl.addressing_mode.NONE,
                cl.filter_mode.NEAREST),
            self.d_img,
            self.dSampleFg
        ]

        gWorksize = roundUp((self.n_points, ), (256, ))

        self.kern_get_samples(self.queue, gWorksize, (256,), *args).wait()

        cl.enqueue_copy(self.queue, self.hSampleFg, self.dSampleFg)
#        print self.hSampleFg.view(np.uint8).reshape(10240, 4)[0:self.n_points, :]

#        print self.n_points
        self.gmmFg.fit(self.dSampleFg, self.n_points)
#        print w
#        print m
#        print c

        self.gmmFg.score(self.d_img, self.dScoreFg)

        #        self.argsSampleBg = [
        #            self.d_labels,
        #            np.int32(self.label),
        #            cl.Sampler(self.context, False, cl.addressing_mode.NONE,
        #                cl.filter_mode.NEAREST),
        #            self.d_img,
        #            self.dSampleFg
        #        ]
        #
        #        gWorksize = roundUp(self.dim, (16, 16))
        #
        #        self.kernSampleBg(self.queue, gWorksize, (16, 16),
        #            *(self.argsSampleBg)).wait()
        #        cl.enqueue_copy(self.queue, self.hSampleBg, self.dSampleBg).wait()

        pass

    def probFg(self, d_samples, n_points):
    #		if True:
    #			tri = self.hTri[minY:maxY, minX:maxX]
    #			b = (tri == self.colorTri[self.type])
    #
    #			samplesFg = self.hSrc[minY:maxY, minX:maxX]
    #			samplesFg = samplesFg[b]
    #		else:
    #			DILATE = 5
    #			samplesFg = self.hSrc[minY:maxY, minX:maxX].ravel()

        #gpu = False
        #self.prob(self.gmmFG, samplesFg, self.dScoreFg, gpu)

        #self.gmmFg_cpu.fit(samplesFg)
        #print 'cpu', self.gmmFg_cpu.weights_
        #a = calcA_cpu(self.gmmFg_cpu.weights_.astype(np.float32), self.gmmFg_cpu.means_.astype(np.float32), self.gmmFg_cpu.covars_.astype(np.float32))
        #cl.enqueue_copy(self.queue, self.gmmFg.dA, a).wait()

        #weights, means, covars = self.gmmFg.fit(samplesFg, retParams=True)
        #a = calcA_cpu(weights, means[:, 0:3], covars[:, 0:3])
        #cl.enqueue_copy(self.queue, self.gmmFg.dA, a).wait()


        w,m,c = self.gmmFg.fit(d_samples, n_points, retParams=True)
        print w
        print m
        print c
        #print 'gpu', weights

        self.gmmFg.score(self.d_img, self.dScoreFg)

    #score returns float64, not float32 -> convert with astype
    #self.hScore = -self.gmmFG.score(self.rgb.reshape(-1, 3)).astype(np.float32)
    """
        def drawCircle(self, xc, yc, points=None):
            r = self.radius

            for y in xrange(-r, r):
                for x in xrange(-r, r):
                    if points != None:
                        points.add((xc+x, yc+y))
        """

    def probBg(self, nSamples):
        #self.kernSampleBg(self.queue, self.gWorksize, self.lWorksize, *(self.argsSampleBg)).wait()
        #cl.enqueue_copy(self.queue, self.hSampleBg, self.dSampleBg).wait()

        self.bgIdx = np.where(self.hTri.ravel() != self.colorTri[self.type])[0]
        self.randIdx = self.bgIdx[np.random.randint(0, len(self.bgIdx), 2000)]
        self.bgIdx = np.setdiff1d(self.bgIdx, self.randIdx)

        self.hSampleBg[0:len(self.randIdx)] = self.hSrc.view(np.uint32).ravel()[
                                              self.randIdx]
        cl.enqueue_copy(self.queue, self.dSampleBg, self.hSampleBg).wait()

        #print self.gmmBg.fit(self.hSrc.view(np.uint32).ravel()[self.randIdx], retParams=True)
        self.gmmBg.fit(self.hSrc.view(np.uint32).ravel()[self.randIdx])
        #self.gmmBg.fit(self.dSampleBg, nSamples=len(self.randIdx))
        self.gmmBg.score(self.dSrc, self.dScoreBg)
