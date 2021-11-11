#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "qmath.h"
#include "iter.h"


class CUDAException : public std::exception {
    cudaError_t _cudaError;
    char _fullMsg[1024];
public:
    CUDAException(cudaError_t err) : _cudaError(err) {
        sprintf_s(_fullMsg, sizeof(_fullMsg), "CudaException %d", static_cast<int>(_cudaError));
    }
    CUDAException(const char* const& msg, cudaError_t err) : _cudaError(err) {
        sprintf_s(_fullMsg, sizeof(_fullMsg), "CudaException %d: %s", static_cast<int>(_cudaError), msg);
    }
    virtual ~CUDAException() {}
    virtual const char* what() const {
        return _fullMsg;
    }
};

bool initGPU();


__host__ void getDeviceMemory(void** ptrOut, size_t nBytes);
__host__ void freeDeviceMemory(void* ptrIn);
__host__ void copyToGPU(void* devPtr, void* hostPtr, size_t nBytes);
__host__ void copyToCPU(void* hostPtr, void* devPtr, size_t nBytes);


template<typename T>
class CUDAStorage {
    void* _mem;
public:
    CUDAStorage() : _mem(0) {}
    CUDAStorage(size_t nElems) {
        getDeviceMemory(&_mem, nElems * sizeof(T));
    };
    virtual ~CUDAStorage() {
        if (NULL != _mem) {
            try {
                freeDeviceMemory(_mem);
            } catch (CUDAException& ex) {
                fprintf(stderr, "In CUDAStorage destructor! %s\n", ex.what());
                assert(false);
            }
        }
    }
    T* ptr() {
        return static_cast<T*>(_mem);
    }
    void copyToGPU(T* p, size_t nElems) {
        ::copyToGPU(_mem, p, nElems * sizeof(T));
    }
    void copyToCPU(const T* p, size_t nElems) {
        ::copyToCPU((void*)p, _mem, nElems * sizeof(T));
    }
};


int iterate_0_driver(iter_struct* is);
