

#include "kernel.h"
#include "cuda_runtime_api.h"
#include <helper_cuda.h>
#include <helper_functions.h>
#include "memory.h"
#include "iter.h"
#include "parameters.h"

#include <stdio.h>

#include <crtdbg.h>


constexpr int gpuBlockSize = 256;


__host__ bool initGPU(int argc, char** argv) {
    int devID;
    cudaDeviceProp props;
    cudaError_t err;

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

__host__ void copyToGPU(void* devPtr, const void *hostPtr, size_t nBytes) {
    cudaError_t cudaStatus = cudaMemcpy(devPtr, hostPtr, nBytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaMemcpy failed", cudaStatus);
    }
}

__host__ void copyToCPU(void* hostPtr, const void* devPtr, size_t nBytes) {
    cudaError_t cudaStatus = cudaMemcpy(hostPtr, devPtr, nBytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaMemcpy failed", cudaStatus);
    }
}

void checkAfterKernel() {
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("addKernel launch failed", cudaStatus);
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaSynchronize", cudaStatus);
    }
}

__constant__ int cudaMaxIter;
__constant__ double cudaBailout;

void setMaxIter(int maxIter) {
    cudaError_t err = cudaMemcpyToSymbol(cudaMaxIter, &maxIter, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw CUDAException("setMaxIter", err);
    }
}

void setBailout(double bailout) {
    cudaError_t err = cudaMemcpyToSymbol(cudaBailout, &bailout, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw CUDAException("setBailout", err);
    }
}


__device__ int iterate_0_cuda(const Quat& z0, const Quat& c, Quat* orbit) {
    Quat z = z0;
    int iter = 0;
    orbit[0] = z;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() - c;
        orbit[++iter] = z;
    } 
    return iter;
}

__device__ int iterate_0_no_orbit_cuda(const Quat& z0, const Quat& c) {
    Quat z = z0;
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() - c;
        iter++;
    }
    return iter;
}

__global__ void iterate_0_search_kernel(int N, const Quat* xStartIn, const Quat* cIn, int* resultOut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || resultOut[i] == -1) {
        return;
    }
    Quat c = *cIn;
    Quat z = xStartIn[i];
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() - c;
        iter++;
    }
    resultOut[i] = iter;
}


__device__ int iterate_1_cuda(const Quat& z0, const Quat& c, Quat* orbit) {
    Quat z = z0;
    int iter = 0;
    orbit[0] = z;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = c * z * (1.0 - z);
        orbit[++iter] = z;
    }
    return iter;
}

__device__ int iterate_1_no_orbit_cuda(const Quat& z0, const Quat& c) {
    Quat z = z0;
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = c * z * (1.0 - z);
        iter++;
    }
    return iter;
}

__global__ void iterate_1_search_kernel(int N, const Quat* xStartIn, const Quat* cIn, int* resultOut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || resultOut[i] == -1) {
        return;
    }
    Quat c = *cIn;
    Quat z = xStartIn[i];
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = c * z * (1.0 - z);
        iter++;
    }
    resultOut[i] = iter;
}


__device__ int iterate_2_cuda(const Quat& z0, const Quat& c, Quat* orbit) {
    Quat z = z0;
    int iter = 0;
    orbit[0] = z;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z * log(z) - c;
        orbit[++iter] = z;
    }
    return iter;
}

__device__ int iterate_2_no_orbit_cuda(const Quat& z0, const Quat& c) {
    Quat z = z0;
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z * log(z) - c;
        iter++;
    }
    return iter;
}

__global__ void iterate_2_search_kernel(int N, const Quat* xStartIn, const Quat* cIn, int* resultOut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || resultOut[i] == -1) {
        return;
    }
    Quat c = *cIn;
    Quat z = xStartIn[i];
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z * log(z) - c;
        iter++;
    }
    resultOut[i] = iter;
}


__device__ int iterate_3_cuda(const Quat& z0, const Quat& c, Quat* orbit) {
    Quat z = z0;
    int iter = 0;
    orbit[0] = z;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() * z - c;
        orbit[++iter] = z;
    }
    return iter;
}

__device__ int iterate_3_no_orbit_cuda(const Quat& z0, const Quat& c) {
    Quat z = z0;
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() * z - c;
        iter++;
    }
    return iter;
}

__global__ void iterate_3_search_kernel(int N, const Quat* xStartIn, const Quat* cIn, int* resultOut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || resultOut[i] == -1) {
        return;
    }
    Quat c = *cIn;
    Quat z = xStartIn[i];
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() * z - c;
        iter++;
    }
    resultOut[i] = iter;
}


__device__ int iterate_4_cuda(const Quat& z0, const Quat& c, Quat* orbit) {
    Quat z = z0;
    int iter = 0;
    orbit[0] = z;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() * z - c;
        orbit[++iter] = z;
    }
    return iter;
}

__device__ int iterate_4_no_orbit_cuda(const Quat& z0, const Quat& c) {
    Quat z = z0;
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() * z - c;
        iter++;
    }
    return iter;
}

__global__ void iterate_4_search_kernel(int N, const Quat* xStartIn, const Quat* cIn, int* resultOut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || resultOut[i] == -1) {
        return;
    }
    Quat c = *cIn;
    Quat z = xStartIn[i];
    int iter = 0;
    while (iter < cudaMaxIter && z.magnitudeSquared() < cudaBailout) {
        z = z.squared() * z - c;
        iter++;
    }
    resultOut[i] = iter;
}


void run_many_search_driver(iter_struct& is, const std::vector<Quat>& positions, std::vector<int>& results) {
    setMaxIter(is.maxiter);
    setBailout(is.bailout);
    const int nElems = static_cast<int>(positions.size());
    CUDAStorage<int> resultBuf(nElems);
    CUDAStorage<Quat> c(1);
    c.copyToGPU(&is.c);
    CUDAStorage<Quat> xStart(positions);
    int gridSize = (nElems + gpuBlockSize - 1) / gpuBlockSize;
    iterate_0_search_kernel <<<gridSize, gpuBlockSize>>> (nElems, xStart.devicePtr(), c.devicePtr(), resultBuf.devicePtr()); 
    checkAfterKernel();
    resultBuf.copyToCPU(results);
}


typedef int (*iterate_fn)(const Quat& z0, const Quat& c, Quat* orbit);

__device__ iterate_fn iterate_cuda[] = {
    iterate_0_cuda,
    iterate_1_cuda,
    iterate_2_cuda,
    iterate_3_cuda,
    iterate_4_cuda
};

typedef int (*iterate_no_orbit_fn)(const Quat& z0, const Quat& c);

__device__ iterate_no_orbit_fn iterate_no_orbit_cuda[] = {
    iterate_0_no_orbit_cuda,
    iterate_1_no_orbit_cuda,
    iterate_2_no_orbit_cuda,
    iterate_3_no_orbit_cuda,
    iterate_4_no_orbit_cuda
};


__global__ void initial_object_distance_kernel(int formula, int N, const Quat* c, const Quat* xStart, const Quat* zBase, const int(*zLimits)[2], int* zResults) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || zLimits[i][0] == -1) {
        return;
    }
    for (int z = zLimits[i][0]; z < zLimits[i][1]; z++) {
        Quat start = xStart[i] + static_cast<double>(z) * *zBase;
        int iter = iterate_no_orbit_cuda[formula](start, *c);
        if (iter == cudaMaxIter) {
            zResults[i] = z;
            return;
        }
    }
    zResults[i] = -1;
}

void row_of_initial_obj_distance_driver(calc_struct& cs, int numXs, const Quat* positions, const Quat& zBase, const int(*zvals)[2], int* zResults) {
    setMaxIter(cs._f._maxiter);
    setBailout(cs._f._bailout);
    CUDAStorage<Quat> c(1, &cs._f._c);
    CUDAStorage<Quat> xStart(numXs, positions);
    CUDAStorage<int[2]> zLimits(numXs, zvals);
    CUDAStorage<Quat> zBaseGPU(1, &zBase);
    CUDAStorage<int> zFound(numXs);
    int gridSize = (numXs + gpuBlockSize - 1) / gpuBlockSize;
    initial_object_distance_kernel << <gridSize, gpuBlockSize >> > (cs._f._formula, numXs, c.devicePtr(), xStart.devicePtr(), zBaseGPU.devicePtr(), zLimits.devicePtr(), zFound.devicePtr());
    checkAfterKernel();
    zFound.copyToCPU(zResults);
}

__global__ void obj_distance_w_orbit_kernel(
    size_t N, int formula, int xres, int antialiasing, const Quat* c, const Quat* xStarts, const Quat* zBase, const int(*zLimits)[2],
    Quat* orbits, double* distances, double* lastIters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    double refinement = 20.0;
    Quat* orbit = &orbits[i * (cudaMaxIter + 2)];

    //for (int i = 0; i < 4; i++) {
    //    is.p[i] = _f._p[i];
    //}
    Quat xStart = xStarts[i];
    int iter = -1;
    int z;
    double z2;
    for (z = zLimits[i][0]; z < zLimits[i][1] && iter != cudaMaxIter; z++) {
        Quat z0 = xStart + static_cast<double>(z) * *zBase;
        iter = iterate_no_orbit_cuda[formula](z0, *c);
    }
    double zDouble = static_cast<double>(z);
    if (z < zLimits[i][1]) {
        zDouble -= 1.0;
        for (z2 = 1.0; z2 <= refinement && iter == cudaMaxIter; z2 += 1.0) {
            Quat z0 = xStart + (zDouble - z2 / refinement) * *zBase;
            iter = iterate_cuda[formula](z0, *c, orbit);
        }
        z2 -= 2;
    } else {
        z2 = 0;
    }
    lastIters[i] = iter;
    distances[i] = floor((zDouble - z2 / refinement) * 1000.0 + 0.5) / 1000.0;
}

void row_of_obj_with_orbits_driver(
    calc_struct& cs, size_t N, const Quat* xStarts, const Quat& zBase, const int(*zLimits)[2],
    Quat* orbits, double* distances, double* lastIters) {
    setMaxIter(cs._f._maxiter);
    setBailout(cs._f._bailout);
    CUDAStorage<Quat> c(1, &cs._f._c);
    CUDAStorage<Quat> xStartsGPU(N, xStarts);
    CUDAStorage<int[2]> zLimitsGPU(N, zLimits);
    CUDAStorage<Quat> zBaseGPU(1, &zBase);
    CUDAStorage<Quat> orbitsGPU(N * (cs._f._maxiter + 1));
    CUDAStorage<double> distancesGPU(N);
    CUDAStorage<double> lastItersGPU(N);
    unsigned int gridSize = (static_cast<unsigned int>(N) + gpuBlockSize - 1) / gpuBlockSize;
    obj_distance_w_orbit_kernel << <gridSize, gpuBlockSize >> > (
        N, cs._f._formula, cs._v._xres, cs._v._antialiasing, c.devicePtr(), xStartsGPU.devicePtr(), zBaseGPU.devicePtr(), zLimitsGPU.devicePtr(),
        orbitsGPU.devicePtr(), distancesGPU.devicePtr(), lastItersGPU.devicePtr());
    checkAfterKernel();
    orbitsGPU.copyToCPU(orbits);
    distancesGPU.copyToCPU(distances);
    lastItersGPU.copyToCPU(lastIters);
}

__global__ void addQuatArraysKernel(int N, Quat* dst, const Quat* a, const Quat* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = a[i] + b[i];
}

void addQuatArrays(Quat* dst, const Quat* a, const Quat* b, int n) {
    CUDAStorage<Quat> dstGPU(n);
    CUDAStorage<Quat> aGPU(n, a);
    CUDAStorage<Quat> bGPU(n, b);
    int gridSize = (n + gpuBlockSize - 1) / gpuBlockSize;
    addQuatArraysKernel << <gridSize, gpuBlockSize >> > (n, dstGPU.devicePtr(), aGPU.devicePtr(), bGPU.devicePtr());
    checkAfterKernel();
    dstGPU.copyToCPU(dst);
}