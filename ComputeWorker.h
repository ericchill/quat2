#pragma once

#include "kernel.h"
#include "LineCalculator.h"
#include "parameters.h"
#include "qmath.h"
#include "quat.h"

class PNGFile;

class ComputeWorker {
public:
    ComputeWorker() {}
    virtual ~ComputeWorker() {}
    virtual ComputeWorker& operator=(ComputeWorker&&) {
        return *this;
    }
};

class FractalLineWorker : public ComputeWorker {
    Quater* _quatDriver;
    FractalPreferences _prefs;
    FractalView _view;
    ZFlag _zFlag;

    ViewBasis _srbase;
    ViewBasis _slbase;

    int _xres_st;
    int _xres_aa;
    int _xres_st_aa;

    float* _lBufR;
    float* _lBufL;
    float* _cBuf;
    float* _bBuf;
    uint8_t* _line;
    uint8_t* _line3;
    LineCalculator* _rightCalc;
    LineCalculator* _leftCalc;
    GPURowCalculator* _rightGPUCalc;
    GPURowCalculator* _leftGPUCalc;

    bool _running;
    bool _failed;
    std::string _failed_msg;

public:
    FractalLineWorker();

    FractalLineWorker(
        Quater& quatDriver,
        FractalPreferences& prefs,
        Expression* colorExpr,
        ZFlag zFlag,
        ViewBasis& rbase,
        ViewBasis& srbase,
        ViewBasis& lbase,
        ViewBasis& slbase);

    ~FractalLineWorker();

    FractalLineWorker& operator=(FractalLineWorker&& other) noexcept;

    void readZBuffer(int iy);
    void calcLine(int iy);
    void putLine(int iy);
    void writeToPNG(PNGFile* png_internal, int iy);

    uint8_t* line() { return _line; }

    bool running() { return _running; }
    bool failed() { return _failed; }
    std::string failedMsg() { return _failed_msg; }

private:
    size_t allocBufs();
    void freeStuff();
    void forgetStuff();
};