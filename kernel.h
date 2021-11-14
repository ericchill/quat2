#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <array>
#include <vector>
#include "qmath.h"
//#include "iter.h"

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

bool initGPU();


__host__ void getDeviceMemory(void** ptrOut, size_t nBytes);
__host__ void freeDeviceMemory(void* ptrIn);
__host__ void copyToGPU(void* devPtr, const void* hostPtr, size_t nBytes);
__host__ void copyToCPU(void* hostPtr, const void* devPtr, size_t nBytes);


template<typename T>
class CUDAStorage {
    void* _mem;
    size_t _nElems;
public:
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
    void copyToGPU(const T* p) {
        ::copyToGPU(_mem, p, _nElems * sizeof(T));
    }
    void copyToGPU(const std::vector<T>& v) {
        ::copyToGPU(_mem, v.data(), v.size * sizeof(T));
    }
    void copyToCPU(T* p) {
        ::copyToCPU(p, _mem, _nElems * sizeof(T));
    }
    void copyToCPU(std::vector<T>& v) {
        ::copyToCPU((void*)v.data(), _mem, v.size() * sizeof(T));
    }
private:
    void allocate() {
        getDeviceMemory(&_mem, _nElems * sizeof(T));
    }
};


void setMaxIter(int maxIter);
void setBailout(int bailout);


void run_many_Z2_driver(iter_struct& is, const std::vector<Quat>& positions, std::vector<int>& results);



void row_of_obj_distance_driver(calc_struct& cs, const Quat* positions, const Quat* zbase, const int (*zvals)[2], int* zResults);