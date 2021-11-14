

#include "kernel.h"
#include "cuda_runtime_api.h"
#include "memory.h"
#include "iter.h"
#include "parameters.h"

#include <stdio.h>

constexpr int gpuBlockSize = 256;


bool initGPU() {
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
    return cudaStatus == cudaSuccess;
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

__device__ int iterate_z2(const Quat& z0, const Quat& c, int maxIter, double bailout, Quat* orbit) {
    Quat z = z0;
    int iter = 0;
    orbit[0] = z;
    while (iter < maxIter && z.magnitudeSquared() < bailout) {
        z = z.squared() - c;
        orbit[++iter] = z;
    } 
    return iter;
}

__device__ int iterate_z2_no_orbit(const Quat& z0, const Quat& c, int maxIter, double bailout) {
    Quat z = z0;
    int iter = 0;
    while (iter < maxIter && z.magnitudeSquared() < bailout) {
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

void run_many_Z2_driver(iter_struct& is, const std::vector<Quat>& positions, std::vector<int>& results) {
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

__global__ void object_distance_kernel(int N, const Quat* c, const Quat* xStart, const Quat* zBase, const int(*zLimits)[2], int* zResults) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || zLimits[i][0] == -1) {
        return;
    }
    for (int z = zLimits[i][0]; z < zLimits[i][1]; z++) {
        Quat start = xStart[i] + z * zBase[i];
        int iter = iterate_z2_no_orbit(start, *c, cudaMaxIter, cudaBailout);
        if (iter == cudaMaxIter) {
            zResults[i] = z;
            return;
        }
    }
    zResults[i] = -1;
}

void row_of_obj_distance_driver(calc_struct& cs, const Quat* positions, const Quat* zBase, const int(*zvals)[2], int* zResults) {
    setMaxIter(cs.f._maxiter);
    setBailout(cs.f._bailout);
    CUDAStorage<Quat> c(1, &cs.f._c);
    CUDAStorage<Quat> xStart(cs.v._xres, positions);
    CUDAStorage<int[2]> zLimits(cs.v._xres, zvals);
    CUDAStorage<Quat> zBaseGPU(cs.v._xres, zBase);
    CUDAStorage<int> zFound(cs.v._xres);
    int gridSize = (cs.v._xres + gpuBlockSize - 1) / gpuBlockSize;
    object_distance_kernel << <gridSize, gpuBlockSize >> > (cs.v._xres, c.devicePtr(), xStart.devicePtr(), zBaseGPU.devicePtr(), zLimits.devicePtr(), zFound.devicePtr());
    checkAfterKernel();
    zFound.copyToCPU(zResults);
}
