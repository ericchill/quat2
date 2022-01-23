

#include "kernel.h"
#include "memory.h"
#include "iter.h"
#include "parameters.h"
#include "LineCalculator.h"
#include <stdio.h>

#include <crtdbg.h>

__device__ inline int minint(int a, int b) {
    return a < b ? a : b;
}

__constant__ int cudaMaxIter;
__constant__ int cudaMaxOrbit;
__constant__ double cudaBailout;

void setMaxIter(int maxIter) {
    cudaError_t err = cudaMemcpyToSymbol(cudaMaxIter, &maxIter, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw CUDAException("setMaxIter", err);
    }
}

void setMaxOrbit(int maxOrbit) {
    cudaError_t err = cudaMemcpyToSymbol(cudaMaxOrbit, &maxOrbit, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw CUDAException("setMaxOrbit", err);
    }
}

void setBailout(double bailout) {
    cudaError_t err = cudaMemcpyToSymbol(cudaBailout, &bailout, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw CUDAException("setBailout", err);
    }
}


/*
* The weirdness of the loops in iterateX is to avoid what seems to be
* a counterproductive CUDA optimization.
*/

typedef int (*iterate_fn)(const Quat& z0, const Quat& c, const Quat* p, Quat* orbit);

typedef int (*iterate_sans_orbit_fn)(const Quat& z0, const Quat& c, const Quat* p);

template<typename Op>
__device__ int iterate_orbit_cuda(const Quat& z0, const Quat& c, const Quat* p, Quat* orbit) {
    Op op;
    int maxIter = cudaMaxIter;
    double bailout = cudaBailout;
    Quat z = z0;
    int iter = 0;
    orbit[0] = z;
    int firstLoopMax = minint(cudaMaxOrbit-1, cudaMaxIter);
    while (z.magnitudeSquared() < bailout) {
        if (iter >= firstLoopMax) {
            break;
        }
        z = op(z, c, p);
        orbit[++iter] = z;
    }
    while (z.magnitudeSquared() < bailout) {
        if (iter >= maxIter) {
            break;
        }
        z = op(z, c, p);
        ++iter;
    }
    orbit[cudaMaxOrbit - 1] = z;
    return iter;
}

template<typename Op>
__device__ int iterate_sans_orbit_cuda(const Quat& z0, const Quat& c, const Quat* p) {
    Op op;
    Quat z = z0;
    int iter;
    double bailout = cudaBailout;
    int maxIter = cudaMaxIter;
    while (z.magnitudeSquared() < bailout) {
        if (iter >= maxIter) {
            return maxIter;
        }
        z = op(z, c, p);
        iter++;
    }
    return iter;
}

__device__ iterate_fn iterate_cuda[] = {
    iterate_orbit_cuda<Iter0Op>,
    iterate_orbit_cuda<Iter1Op>,
    iterate_orbit_cuda<Iter2Op>,
    iterate_orbit_cuda<Iter3Op>,
    iterate_orbit_cuda<Iter4Op>
};

__device__ iterate_sans_orbit_fn iterate_no_orbit_cuda[] = {
    iterate_sans_orbit_cuda<Iter0Op>,
    iterate_sans_orbit_cuda<Iter1Op>,
    iterate_sans_orbit_cuda<Iter2Op>,
    iterate_sans_orbit_cuda<Iter3Op>,
    iterate_sans_orbit_cuda<Iter4Op>
};


__device__ bool cutaway(const Vec3& x, size_t nCuts, const Vec3* cutNormals, const Vec3* cutPoints) {
    for (unsigned int i = 0; i < nCuts; i++) {
        Vec3 y = x - cutPoints[i];
        if (cutNormals[i].dot(y) > 0) {
            return true;
        }
    }
    return false;
}


__global__ void fillXStarts(
    int N, const obj_distance_kernel_args* args, int y, Quat* xStarts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    Quat xs = args->sBase._O + y * args->sBase._y + args->sBase._x * (i + 1);
    xs[3] = args->lTerm;
    for (int yaa = 0; yaa < args->antialiasing; yaa++) {
        for (int xaa = 0; xaa < args->antialiasing; xaa++) {
            int aaLbufIdx = aaLBufIndex(i, xaa, yaa, args->xRes, args->antialiasing);
            xStarts[aaLbufIdx] = xs + yaa * args->aaBase._y + xaa * args->aaBase._x;
        }
    }
}


__global__ void obj_distances_kernel(
    size_t N, int formula, const obj_distance_kernel_args* args, const Quat* xStarts,
    Quat* orbits, float* distances, float* lastIters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    double refinement = 100.0;
    Quat* orbit = &orbits[i * (cudaMaxOrbit + LineCalculator::orbitSpecial())];
    Quat xStart = xStarts[i];
    int iter = -1;
    int z;
    double z2;
    int zRes = args->zRes;
    iterate_sans_orbit_fn preIterFn = iterate_no_orbit_cuda[formula];
    for (z = 0; iter < cudaMaxIter; z++) {
        if (z >= zRes) {
            break;
        }
        Quat z0 = xStart + static_cast<double>(z) * args->zBase;
        if (!cutaway(Vec3(z0), args->nCuts, args->cutNormals, args->cutPoints)) {
            iter = preIterFn(z0, args->c, args->p);
        } else {
            iter = 0;
        }
    }
    double zDouble = static_cast<double>(z);
    iterate_fn iterFn = iterate_cuda[formula];
    if (z < zRes) {
        zDouble -= 1.0;
        for (z2 = 1.0; z2 <= refinement && iter == cudaMaxIter; z2 += 1.0) {
            Quat z0 = xStart + (zDouble - z2 / refinement) * args->zBase;
            iter = iterFn(z0, args->c, args->p, orbit);
        }
        z2 -= 2;
    } else {
        z2 = 0;
    }
    distances[i] = static_cast<float>(floor((zDouble - z2 / refinement) * 1000.0 + 0.5) / 1000.0);
    lastIters[i] = iter;
}

__global__ void distance_to_lbuf(
    const obj_distance_kernel_args* args, float* distances, float* lastIters,
    float* lBuf, float* bBuf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args->xRes) {
        return;
    }
    int aaSquared = args->antialiasing * args->antialiasing;
    int lBufIdx = (i + args->xRes) * args->antialiasing;
    float dist = distances[lBufIdx];
    lBuf[lBufIdx] = dist;
    if (dist != args->zRes) {
        float itersum = 0;
        for (int yaa = 0; yaa < args->antialiasing; yaa++) {
            for (int xaa = 0; xaa < args->antialiasing; xaa++) {
                int aaLbufIdx = aaLBufIndex(i, xaa, yaa, args->xRes, args->antialiasing);
                lBuf[aaLbufIdx] = distances[aaLbufIdx];
                itersum += lastIters[aaLbufIdx];
            }
        }
        lastIters[lBufIdx] = itersum / aaSquared;
        if (shouldCalculateImage(args->zFlag)) {
            bBuf[i] = 1;
        }
    } else if (shouldCalculateImage(args->zFlag)) {
        float zFloat = static_cast<float>(args->zRes);
        if (shouldCalculateImage(args->zFlag)) {
            for (int yaa = 0; yaa < args->antialiasing; yaa++) {
                for (int xaa = 0; xaa < args->antialiasing; xaa++) {
                    lBuf[aaLBufIndex(i, xaa, yaa, args->xRes, args->antialiasing)] = zFloat;
                }
            }
            bBuf[i] = 0;
        }
    }
}


GPURowCalculator::GPURowCalculator(const LineCalculator& cs, size_t lBufSize, ZFlag zFlag) : _arraySize(lBufSize) {
    cudaError_t cudaStatus = cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaStreamCreate failed", cudaStatus);
    }
    cudaStatus = cudaStreamCreateWithFlags(&_stream2, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaStreamCreate failed", cudaStatus);
    }
    setMaxIter(cs.fractal()._maxiter);
    setBailout(cs.fractal()._bailout);
    setMaxOrbit(cs.fractal()._maxOrbit);
    _kernelArgs.zBase = cs.sbase()._z;
    _kernelArgs.c = cs.fractal()._c;
    copyArray(_kernelArgs.p, cs.fractal()._p, cs.fractal().numPowers);
    _kernelArgs.lTerm = cs.fractal()._lTerm;
    _kernelArgs.sBase = cs.sbase();
    _kernelArgs.aaBase = cs.aabase();
    _kernelArgs.xRes = cs.view()._xres;
    _kernelArgs.zRes = cs.view()._zres;
    _kernelArgs.antialiasing = cs.view()._antialiasing;
    _kernelArgs.zFlag = zFlag;
    _kernelArgs.nCuts = cs.cuts().count();
    copyArray(_kernelArgs.cutNormals, cs.cuts().normals(), cs.cuts().maxCuts);
    copyArray(_kernelArgs.cutPoints, cs.cuts().points(), cs.cuts().maxCuts);

    _xStarts = new CUDAStorage<Quat>(lBufSize);
    _orbits = new CUDAStorage<Quat>(lBufSize * (cs.fractal()._maxOrbit + LineCalculator::orbitSpecial()));
    _distances = new CUDAStorage<float>(lBufSize);
    _lastIters = new CUDAStorage<float>(lBufSize);
    _lBuf = new CUDAStorage<float>(lBufSize);
    _bBuf = new CUDAStorage<float>(cs.view()._xres);

    _kernelArgsGPU.copyToGPU(&_kernelArgs);
}

GPURowCalculator::~GPURowCalculator() {
    delete _xStarts;
    delete _orbits;
    delete _distances;
    delete _lastIters;
    delete _lBuf;
    delete _bBuf;
    cudaError_t cudaStatus;
    cudaStatus = cudaStreamDestroy(_stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaStreamDestroy(_stream) failed." << std::endl;
        assert(false);
    }
    cudaStatus = cudaStreamDestroy(_stream2);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaStreamDestroy(_stream2) failed." << std::endl;
        assert(false);
    }
}

void GPURowCalculator::obj_distances(
    LineCalculator& cs, size_t N, int y,
    Quat* orbits, float* distances, float* lastIters,
    float* lBuf,
    float* bBuf) {
    cudaError_t cudaStatus;
    int gridSize, blockSize;

    int sharedSize = 16384;
    getGridAndBlockCount(cs.view()._xres, fillXStarts, gridSize, blockSize);
    fillXStarts <<< gridSize, blockSize, sharedSize, _stream >>> (
        cs.view()._xres, _kernelArgsGPU.devicePtr(), y, _xStarts->devicePtr());
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("fillXStarts launch failed", cudaStatus);
    }
    getGridAndBlockCount(N, obj_distances_kernel, gridSize, blockSize);
    obj_distances_kernel <<< gridSize, blockSize, sharedSize, _stream >>> (
        N, cs.fractal()._formula, _kernelArgsGPU.devicePtr(), _xStarts->devicePtr(),
        _orbits->devicePtr(), _distances->devicePtr(), _lastIters->devicePtr());
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("obj_distances launch failed", cudaStatus);
    }
    getGridAndBlockCount(cs.view()._xres, distance_to_lbuf, gridSize, blockSize);
    distance_to_lbuf <<< gridSize, blockSize, sharedSize, _stream >>> (
        _kernelArgsGPU.devicePtr(),
        _distances->devicePtr(), _lastIters->devicePtr(), _lBuf->devicePtr(), _bBuf->devicePtr());
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("distance_to_lbuf launch failed", cudaStatus);
    }
    _orbits->copyToCPU(orbits, _stream);
    _distances->copyToCPU(distances, _stream);
    _lastIters->copyToCPU(lastIters, _stream);
    _lBuf->copyToCPU(lBuf, _stream);
    _bBuf->copyToCPU(bBuf, _stream);
#if 1
    cudaStatus = cudaStreamSynchronize(_stream);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaStreamSynchronize(_stream) failed.", cudaStatus);
    }
    //cudaStatus = cudaStreamSynchronize(_stream2);
    //if (cudaStatus != cudaSuccess) {
    //    throw CUDAException("cudaStreamSynchronize(_stream2) failed.", cudaStatus);
    //}
#endif
}