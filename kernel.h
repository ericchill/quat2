#pragma once

#include "cuda_util.h"
#include "memory.h"
#include "qmath.h"
#include "parameters.h"

#include <stdio.h>
#include <array>
#include <vector>

struct iter_struct;
class LineCalculator;


struct obj_distance_kernel_args {
    Quat xs;
    Quat c;
    Quat p[4];
    double lTerm;
    ViewBasis sBase;
    ViewBasis aaBase;
    Quat zBase;
    int xRes;
    int zRes;
    int antialiasing;
    ZFlag zFlag;
    size_t nCuts;
    Vec3 cutNormals[CutSpec::maxCuts];
    Vec3 cutPoints[CutSpec::maxCuts];
};


class GPURowCalculator {
public:
    GPURowCalculator(const LineCalculator& cs, size_t lBufSize, ZFlag zFlag);
    virtual ~GPURowCalculator();

    void obj_distances(
        LineCalculator& cs, size_t N, int y,
        Quat* orbits, float* distances, float* lastIters,
        float* lBuf,
        float* bBuf);

private:
    size_t _arraySize;
    obj_distance_kernel_args _kernelArgs;
    CUDAStorage<obj_distance_kernel_args> _kernelArgsGPU;
    CUDAStorage<Quat>* _xStarts; // This stays on the GPU.
    CUDAStorage<Quat>* _orbits;
    CUDAStorage<float>* _distances; // Will be GPU only.
    CUDAStorage<float>* _lastIters;
    CUDAStorage<float>* _lBuf;
    CUDAStorage<float>* _bBuf;
    cudaStream_t _stream;
    cudaStream_t _stream2;
};