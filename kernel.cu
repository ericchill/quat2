

#include "kernel.h"
#include "cuda_runtime_api.h"
#include <helper_cuda.h>
#include <helper_functions.h>
#include "memory.h"
#include "iter.h"
#include "parameters.h"

#include <stdio.h>

#include <crtdbg.h>


bool haveGPU = false;

constexpr int gpuBlockSize = 256;


__host__ bool initGPU(int argc, char** argv) {
    int devID;
    cudaDeviceProp props;
    cudaError_t err;
    haveGPU = false;
    devID = findCudaDevice(argc, (const char**)argv);
    err = cudaGetDevice(&devID);
    if (cudaSuccess != err) {
        std::cerr << "Couldn't get CUDA device." << std::endl;
        return false;
    }
    err = cudaGetDeviceProperties(&props, devID);
    if (cudaSuccess != err) {
        std::cerr << "Couldn't get CUDA device properties." << std::endl;
        return false;
    }
    haveGPU = true;
    return true;
}


__host__ void getDeviceMemory(void** ptrOut, size_t nBytes) {
    cudaError_t cudaStatus = cudaMalloc(ptrOut, nBytes);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaMalloc failed", cudaStatus);
    }
}

__host__ void freeDeviceMemory(void* ptrIn) {
    cudaError_t cudaStatus = cudaFree(ptrIn);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaFree failed", cudaStatus);
    }
}

__host__ void copyToGPU(void* devPtr, const void *hostPtr, size_t nBytes, cudaStream_t stream) {
    cudaError_t cudaStatus;
    if (0 == stream) {
        cudaStatus = cudaMemcpy(devPtr, hostPtr, nBytes, cudaMemcpyHostToDevice);
    } else {
        cudaStatus = cudaMemcpyAsync(devPtr, hostPtr, nBytes, cudaMemcpyHostToDevice, stream);
    }
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaMemcpy failed", cudaStatus);
    }
}

__host__ void copyToCPU(void* hostPtr, const void* devPtr, size_t nBytes, cudaStream_t stream) {
    cudaError_t cudaStatus;
    if (0 == stream) {
        cudaStatus = cudaMemcpy(hostPtr, devPtr, nBytes, cudaMemcpyDeviceToHost);
    } else {
        cudaStatus = cudaMemcpyAsync(hostPtr, devPtr, nBytes, cudaMemcpyDeviceToHost, stream);
    }
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaMemcpy failed", cudaStatus);
    }
}

void checkAfterKernel() {
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("Kernel launch failed", cudaStatus);
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaSynchronize", cudaStatus);
    }
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


__device__ int iterate_0_cuda(const Quat& z0, const Quat& c, const Quat* p, Quat* orbit) {
    Quat z = z0;
    int iter = 0;
    orbit[0] = z;
    while (iter < cudaMaxOrbit && iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() - c;
        orbit[++iter] = z;
    }
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() - c;
        ++iter;
    }
    orbit[cudaMaxOrbit - 1] = z;
    return iter;
}

__device__ int iterate_0_no_orbit_cuda(const Quat& z0, const Quat& c, const Quat* p) {
    Quat z = z0;
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() - c;
        iter++;
    }
    return iter;
}


__device__ int iterate_1_cuda(const Quat& z0, const Quat& c, const Quat* p, Quat* orbit) {
    Quat z = z0;
    int iter = 0;
    orbit[0] = z;
    while (iter < cudaMaxOrbit && iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = c * z * (1.0 - z);
        orbit[++iter] = z;
    }
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = c * z * (1.0 - z);
        ++iter;
    }
    orbit[cudaMaxOrbit - 1] = z;
    return iter;
}

__device__ int iterate_1_no_orbit_cuda(const Quat& z0, const Quat& c, const Quat* p) {
    Quat z = z0;
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = c * z * (1.0 - z);
        iter++;
    }
    return iter;
}


__device__ int iterate_2_cuda(const Quat& z0, const Quat& c, const Quat* p, Quat* orbit) {
    Quat z = z0;
    int iter = 0;
    orbit[0] = z;
    while (iter < cudaMaxOrbit && iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z * log(z) - c;
        orbit[++iter] = z;
    }
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z * log(z) - c;
        ++iter;
    }
    orbit[cudaMaxOrbit - 1] = z;
    return iter;
}

__device__ int iterate_2_no_orbit_cuda(const Quat& z0, const Quat& c, const Quat* p) {
    Quat z = z0;
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z * log(z) - c;
        iter++;
    }
    return iter;
}


__device__ int iterate_3_cuda(const Quat& z0, const Quat& c, const Quat* p, Quat* orbit) {
    Quat z = z0;
    int iter = 0;
    orbit[0] = z;
    while (iter < cudaMaxOrbit && iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() * z - c;
        orbit[++iter] = z;
    }
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() * z - c;
        ++iter;
    }
    orbit[cudaMaxOrbit - 1] = z;
    return iter;
}

__device__ int iterate_3_no_orbit_cuda(const Quat& z0, const Quat& c, const Quat* p) {
    Quat z = z0;
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() * z - c;
        iter++;
    }
    return iter;
}


__device__ int iterate_4_cuda(const Quat& z0, const Quat& c, const Quat* p, Quat* orbit) {
    Quat z = z0;
    int iter = 0;
    orbit[0] = z;
    while (iter < cudaMaxOrbit && iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() * z - c;
        orbit[++iter] = z;
    }
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() * z - c;
        ++iter;
    }
    orbit[cudaMaxOrbit - 1] = z;
    return iter;
}

__device__ int iterate_4_no_orbit_cuda(const Quat& z0, const Quat& c, const Quat* p) {
    Quat z = z0;
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() * z - c;
        iter++;
    }
    return iter;
}

typedef int (*iterate_fn)(const Quat& z0, const Quat& c, const Quat* p, Quat* orbit);

__device__ iterate_fn iterate_cuda[] = {
    iterate_0_cuda,
    iterate_1_cuda,
    iterate_2_cuda,
    iterate_3_cuda,
    iterate_4_cuda
};

typedef int (*iterate_no_orbit_fn)(const Quat& z0, const Quat& c, const Quat* p);

__device__ iterate_no_orbit_fn iterate_no_orbit_cuda[] = {
    iterate_0_no_orbit_cuda,
    iterate_1_no_orbit_cuda,
    iterate_2_no_orbit_cuda,
    iterate_3_no_orbit_cuda,
    iterate_4_no_orbit_cuda
};


__device__ bool cutaway(const Vec3& x, size_t nCuts, const Vec3* cutNormals, const Vec3* cutPoints) {
    for (unsigned i = 0; i < nCuts; i++) {
        Vec3 y = x - cutPoints[i];
        if (cutNormals[i].dot(y) > 0) {
            return true;
        }
    }
    return false;
}

__global__ void obj_distances_kernel_2(
    size_t N, int formula, int xres, int zres, int antialiasing, const obj_distance_kernel_args* args, const Quat* xStarts,
    Quat* orbits, double* distances, double* lastIters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    double refinement = 20.0;
    Quat* orbit = &orbits[i * (cudaMaxOrbit + 2)];

    Quat xStart = xStarts[i];
    int iter = -1;
    int z;
    double z2;
    for (z = 0; z < zres && iter != cudaMaxIter; z++) {
        Quat z0 = xStart + static_cast<double>(z) * args->zBase;
        if (!cutaway(Vec3(z0), args->nCuts, args->cutNormals, args->cutPoints)) {
            iter = iterate_no_orbit_cuda[formula](z0, args->c, args->p);
        } else {
            iter = 0;
        }
    }
    double zDouble = static_cast<double>(z);
    if (z < zres) {
        zDouble -= 1.0;
        for (z2 = 1.0; z2 <= refinement && iter == cudaMaxIter; z2 += 1.0) {
            Quat z0 = xStart + (zDouble - z2 / refinement) * args->zBase;
            iter = iterate_cuda[formula](z0, args->c, args->p, orbit);
        }
        z2 -= 2;
    } else {
        z2 = 0;
    }
    distances[i] = floor((zDouble - z2 / refinement) * 1000.0 + 0.5) / 1000.0;
    lastIters[i] = iter;
}

GPURowCalculator::GPURowCalculator(const calc_struct& cs, size_t lBufSize) : _arraySize(lBufSize) {
    cudaError_t cudaStatus = cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaStreamCreate failed", cudaStatus);
    }
    cudaStatus = cudaStreamCreateWithFlags(&_stream2, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaStreamCreate 2 failed", cudaStatus);
    }
    setMaxIter(cs._f._maxiter);
    setBailout(cs._f._bailout);
    setMaxOrbit(cs._f._maxOrbit);
    _kernelArgs.zBase = cs._sbase._z;
    _kernelArgs.c = cs._f._c;
    copyArray(_kernelArgs.p, cs._f._p, 4);
    _kernelArgs.nCuts = cs._cuts.count();
    copyArray(_kernelArgs.cutNormals, cs._cuts.normals(), cs._cuts.maxCuts);
    copyArray(_kernelArgs.cutPoints, cs._cuts.points(), cs._cuts.maxCuts);
    _kernelArgsGPU.copyToGPU(&_kernelArgs, _stream);

    _xStarts = new CUDAStorage<Quat>(lBufSize);
    _orbits = new CUDAStorage<Quat>(lBufSize * (cs._f._maxOrbit + 2));
    _distances = new CUDAStorage<double>(lBufSize);
    _lastIters = new CUDAStorage<double>(lBufSize);
}

GPURowCalculator::~GPURowCalculator() {
    delete _xStarts;
    delete _orbits;
    delete _distances;
    delete _lastIters;
    cudaStreamDestroy(_stream);
    cudaStreamDestroy(_stream2);
}

void GPURowCalculator::obj_distances(
    calc_struct& cs, size_t N, const Quat* xStarts,
    Quat* orbits, double* distances, double* lastIters) {

    _xStarts->copyToGPU(xStarts, _stream);
    unsigned int gridSize = (static_cast<unsigned int>(N) + gpuBlockSize - 1) / gpuBlockSize;
    obj_distances_kernel_2 << <gridSize, gpuBlockSize, 0, _stream >> > (
        N, cs._f._formula, cs._v._xres, cs._v._zres, cs._v._antialiasing, _kernelArgsGPU.devicePtr(), _xStarts->devicePtr(),
        _orbits->devicePtr(), _distances->devicePtr(), _lastIters->devicePtr());
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("obj_distances launch failed", cudaStatus);
    }
    _orbits->copyToCPU(orbits, _stream);
    _distances->copyToCPU(distances, _stream2);
    _lastIters->copyToCPU(lastIters, _stream2);
    cudaStatus = cudaStreamSynchronize(_stream);
    cudaStatus = cudaStreamSynchronize(_stream2);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("obj_distance stream synchronize failed.", cudaStatus);
    }
}
