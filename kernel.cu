

#include "kernel.h"

#include <stdio.h>


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

__host__ void copyToGPU(void* devPtr, void *hostPtr, size_t nBytes) {
    cudaError_t cudaStatus = cudaMemcpy(devPtr, hostPtr, nBytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaMemcpy failed", cudaStatus);
    }
}

__host__ void copyToCPU(void* hostPtr, void* devPtr, size_t nBytes) {
    cudaError_t cudaStatus = cudaMemcpy(hostPtr, devPtr, nBytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaMemcpy failed", cudaStatus);
    }
}




__global__ void iterate_0_kernel(const Quat* cIn, int* maxIterIn, double* bailoutIn, Quat* __restrict__ orbit, int* resultOut) {
    Quat z = 0;
    orbit[0] = z;
    double zMag2 = z.magnitudeSquared();
    int iter = 0;
    int maxIter = *maxIterIn;
    double bailout = *bailoutIn;
    Quat c = *cIn;
    while (zMag2 < bailout && iter < maxIter) {
        z = z.squared() - c;
        zMag2 = z.magnitudeSquared();
        orbit[++iter] = z;
    }
    *resultOut = iter;
}

int iterate_0_driver(iter_struct* is) {
    CUDAStorage<int> resultBuf(1);
    CUDAStorage<Quat> orbit(is->maxiter);
    CUDAStorage<Quat> c(1);
    CUDAStorage<int> maxiter(1);
    CUDAStorage<double> bailout(1);
    c.copyToGPU(&is->c, 1);
    maxiter.copyToGPU(&is->maxiter, 1);
    bailout.copyToGPU(&is->bailout, 1);
    iterate_0_kernel <<<1, 1>>> (c.ptr(), maxiter.ptr(), bailout.ptr(), orbit.ptr(), resultBuf.ptr());
    int result;
    resultBuf.copyToCPU(&result, 1);
    return result;
}

__global__ void addKernel(int* a, int* b, int* c) {}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;
    
    // Allocate GPU buffers for three vectors (two input, one output)
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}


bool initGPU() {
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
    return cudaStatus == cudaSuccess;
}