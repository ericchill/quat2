#pragma once

#include "common.h"
#include "CReplacements.h"

#include <cuda_runtime.h>
#include <tuple>
#include <vector>

constexpr int gpuBlockSize = 512;

class CUDAException : public QuatException {
    cudaError_t _cudaError;
    char _fullMsg[1024];
public:
    explicit CUDAException(cudaError_t err) : _cudaError(err) {
        sprintf_s(_fullMsg, sizeof(_fullMsg), "CudaException %s", cudaGetErrorString(_cudaError));
    }
    explicit CUDAException(const char* const& msg, cudaError_t err) : _cudaError(err) {
        sprintf_s(_fullMsg, sizeof(_fullMsg), "CudaException %s: %s", cudaGetErrorString(_cudaError), msg);
    }
    virtual ~CUDAException() {}
    virtual const char* what() const {
        return _fullMsg;
    }
};


bool initGPU(int argc, char** argv);

void getDeviceMemory(void** ptrOut, size_t nBytes);
void freeDeviceMemory(void* ptrIn);
void copyToGPU(void* devPtr, const void* hostPtr, size_t nBytes, cudaStream_t stream = 0);
void copyToCPU(void* hostPtr, const void* devPtr, size_t nBytes, cudaStream_t stream = 0);


template<typename T>
class CUDAStorage {
    void* _mem;
    size_t _nElems;
public:
    static T* allocHost(size_t nElems = 1) {
        void* result;
        cudaError_t cudaStatus = cudaMallocHost(&result, nElems * sizeof(T));
        if (cudaSuccess != cudaStatus) {
            throw CUDAException("Can't allocate host memory", cudaStatus);
        }
        return static_cast<T*>(result);
    }
    static void freeHost(T* p) {
        cudaError_t cudaStatus = cudaFreeHost(p);
        if (cudaSuccess != cudaStatus) {
            throw CUDAException("Can't free host memory", cudaStatus);
        }
    }

    explicit CUDAStorage() : _nElems(1) {
        allocate();
    }
    explicit CUDAStorage(size_t nElems) : _nElems(nElems) {
        allocate();
    };
    explicit CUDAStorage(size_t nElems, const T* elems) : _nElems(nElems) {
        allocate();
        copyToGPU(elems);
    }
    explicit CUDAStorage(const std::vector<T>& v) : _nElems(v.size()) {
        allocate();
        copyToGPU(v.data());
    }
    virtual ~CUDAStorage() {
        if (nullptr != _mem) {
            try {
                freeDeviceMemory(_mem);
            }
            catch (CUDAException& ex) {
                fprintf(stderr, "In CUDAStorage destructor! %s\n", ex.what());
                assert(false);
            }
        }
    }
    T* devicePtr() {
        return static_cast<T*>(_mem);
    }
    void copyToGPU(const T* p, cudaStream_t stream = 0) {
        ::copyToGPU(_mem, p, _nElems * sizeof(T), stream);
    }
    void copyToGPU(const std::vector<T>& v, cudaStream_t stream = 0) {
        ::copyToGPU(_mem, v.data(), v.size * sizeof(T), stream);
    }
    void copyToCPU(T* p, cudaStream_t stream = 0) {
        ::copyToCPU(p, _mem, _nElems * sizeof(T), stream);
    }
    void copyToCPU(std::vector<T>& v, cudaStream_t stream = 0) {
        ::copyToCPU((void*)v.data(), _mem, v.size() * sizeof(T), stream);
    }
private:
    void allocate() {
        getDeviceMemory(&_mem, _nElems * sizeof(T));
    }
};

template<typename T>
void getGridAndBlockCount(int n, T fn, int& gridSize, int& blockSize) {
    std::ignore = fn;
    gridSize = (n + gpuBlockSize - 1) / gpuBlockSize;
    blockSize = gpuBlockSize;
    //cudaError_t cudaStatus = cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, fn, 0, n);
    //if (cudaStatus != cudaSuccess) {
    //    throw CUDAException("cudaOccupancyMaxPotentialBlockSize failed", cudaStatus);
    //}
}