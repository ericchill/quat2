#include "cuda_util.h"

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

int processorCount = 0;
int coresPerProcessor = 0;

bool getSPcores(cudaDeviceProp devProp) {
    processorCount = devProp.multiProcessorCount;
    switch (devProp.major) {
    case 2: // Fermi
        if (devProp.minor == 1) coresPerProcessor = 48;
        else coresPerProcessor = 32;
        break;
    case 3: // Kepler
        coresPerProcessor = 192;
        break;
    case 5: // Maxwell
        coresPerProcessor = 128;
        break;
    case 6: // Pascal
        if ((devProp.minor == 1) || (devProp.minor == 2)) coresPerProcessor = 128;
        else if (devProp.minor == 0) coresPerProcessor = 64;
        else {
            printf("Unknown device type\n");
            return false;
        }
        break;
    case 7: // Volta and Turing
        if ((devProp.minor == 0) || (devProp.minor == 5)) coresPerProcessor = 64;
        else {
            printf("Unknown device type\n");
            return false;
        }
        break;
    case 8: // Ampere
        if (devProp.minor == 0) coresPerProcessor = 64;
        else if (devProp.minor == 6) coresPerProcessor = 128;
        else {
            printf("Unknown device type\n");
            return false;
        }
        break;
    default:
        printf("Unknown device type\n");
        return false;
        break;
    }
    return true;
}

bool initGPU(int argc, char** argv) {
    cudaDeviceProp props;
    int devID = findCudaDevice(argc, (const char**)argv);
    cudaError_t err = cudaGetDevice(&devID);
    if (cudaSuccess != err) {
        std::cerr << "Couldn't get CUDA device." << std::endl;
        return false;
    }
    err = cudaGetDeviceProperties(&props, devID);
    if (cudaSuccess != err) {
        std::cerr << "Couldn't get CUDA device properties." << std::endl;
        return false;
    }
    return getSPcores(props);
}


void getDeviceMemory(void** ptrOut, size_t nBytes) {
    cudaError_t cudaStatus = cudaMalloc(ptrOut, nBytes);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaMalloc failed", cudaStatus);
    }
}

void freeDeviceMemory(void* ptrIn) {
    cudaError_t cudaStatus = cudaFree(ptrIn);
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaFree failed", cudaStatus);
    }
}

void copyToGPU(void* devPtr, const void* hostPtr, size_t nBytes, cudaStream_t stream) {
    cudaError_t cudaStatus;
    if (0 == stream) {
        cudaStatus = cudaMemcpy(devPtr, hostPtr, nBytes, cudaMemcpyHostToDevice);
    } else {
        cudaStatus = cudaMemcpyAsync(devPtr, hostPtr, nBytes, cudaMemcpyHostToDevice, stream);
    }
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaMemcpy to GPU failed", cudaStatus);
    }
}

void copyToCPU(void* hostPtr, const void* devPtr, size_t nBytes, cudaStream_t stream) {
    cudaError_t cudaStatus;
    if (0 == stream) {
        cudaStatus = cudaMemcpy(hostPtr, devPtr, nBytes, cudaMemcpyDeviceToHost);
    } else {
        cudaStatus = cudaMemcpyAsync(hostPtr, devPtr, nBytes, cudaMemcpyDeviceToHost, stream);
    }
    if (cudaStatus != cudaSuccess) {
        throw CUDAException("cudaMemcpy to CPU failed", cudaStatus);
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