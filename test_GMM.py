import Image
import time
from GMM import GMM
from sklearn import mixture
from clutil import padArray2D
from clutil import roundUp
import numpy as np
import pyopencl as cl

cm = cl.mem_flags

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

a = calcA_cpu(weights, means, covars)
cl.enqueue_copy(queue, gmm.dA, a).wait()

gmm.has_preset_wmc = True
w,m,c = gmm.fit(dSamples, nSamples, retParams=True)
print 'converged: {0}'.format(gmm.has_converged)

print gmm_cpu.weights_
print w
print
print gmm_cpu.means_
print m
print
print gmm_cpu.covars_
print c

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
    gmm.has_converged = False

    t1 = time.time()
    #gmm.fit(samples)
    w,m,c = gmm.fit(dSamples, nSamples, retParams=True)
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

hOut = np.empty((hSrc.shape), np.float32)
dOut = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hOut)


elapsed = 0
t1 = t2 = 0
for i in xrange(iter):
    t1 = time.time()
    gmm.has_converged = False
    #gmm.fit(samples)
#		w,m,c = gmm.fit(queue, samples, nSamples, weights, means, covars, retParams=True)
    gmm.fit(dSamples, nSamples)
    #gmm.fit(dOut, nSamples)
    #gmm.score(dSrc, dOut)
    elapsed += time.time()-t1
print 'gpu fit: ', elapsed/iter

dScore = cl.Buffer(context, cm.READ_WRITE, 4*nSamples)

True