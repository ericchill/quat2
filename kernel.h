#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <array>
#include <vector>
#include "qmath.h"
#include "parameters.h"

struct iter_struct;
struct calc_struct;

class CUDAException : public std::exception {
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

extern bool haveGPU;

__host__ void getDeviceMemory(void** ptrOut, size_t nBytes);
__host__ void freeDeviceMemory(void* ptrIn);
__host__ void copyToGPU(void* devPtr, const void* hostPtr, size_t nBytes, cudaStream_t stream=0);
__host__ void copyToCPU(void* hostPtr, const void* devPtr, size_t nBytes, cudaStream_t stream=0);


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
        if (NULL != _mem) {
            try {
                freeDeviceMemory(_mem);
            } catch (CUDAException& ex) {
                fprintf(stderr, "In CUDAStorage destructor! %s\n", ex.what());
                assert(false);
            }
        }
    }
    T* devicePtr() {
        return static_cast<T*>(_mem);
    }
    void copyToGPU(const T* p, cudaStream_t stream=0) {
        ::copyToGPU(_mem, p, _nElems * sizeof(T));
    }
    void copyToGPU(const std::vector<T>& v, cudaStream_t stream=0) {
        ::copyToGPU(_mem, v.data(), v.size * sizeof(T));
    }
    void copyToCPU(T* p, cudaStream_t stream=0) {
        ::copyToCPU(p, _mem, _nElems * sizeof(T));
    }
    void copyToCPU(std::vector<T>& v, cudaStream_t stream=0) {
        ::copyToCPU((void*)v.data(), _mem, v.size() * sizeof(T));
    }
private:
    void allocate() {
        getDeviceMemory(&_mem, _nElems * sizeof(T));
    }
};


struct obj_distance_kernel_args {
    Quat c;
    Quat p[4];
    Quat zBase;
    size_t nCuts;
    Vec3 cutNormals[CutSpec::maxCuts];
    Vec3 cutPoints[CutSpec::maxCuts];
};


class GPURowCalculator {
public:
    GPURowCalculator(const calc_struct& cs, size_t lBufSize);
    virtual ~GPURowCalculator();

    void obj_distances(
        calc_struct& cs, size_t N, const Quat* xStarts,
        Quat* orbits, double* distances, double* lastIters);

private:
    size_t _arraySize;
    obj_distance_kernel_args _kernelArgs;
    CUDAStorage<obj_distance_kernel_args> _kernelArgsGPU;
    CUDAStorage<Quat>* _xStarts;
    CUDAStorage<Quat>* _orbits;
    CUDAStorage<double>* _distances;
    CUDAStorage<double>* _lastIters;
    cudaStream_t _stream;
    cudaStream_t _stream2;
};