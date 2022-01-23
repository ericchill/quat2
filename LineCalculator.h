#pragma once

#include "parameters.h"
#include "ExprEval.h"
#include "memory.h"

struct iter_struct;
class GPURowCalculator;
class Expression;
class EvalContext;

class LineCalculator {
public:
    LineCalculator(
        const FractalSpec& fractal,
        const FractalView& view,
        const CutSpec& _cuts,
        Expression* colorExpr,
        ViewBasis& base,
        ViewBasis& sbase,
        ZFlag zflag);
    virtual ~LineCalculator();

    /* calculates a whole line of depths (precision: 1/20th of the base step in z direction), */
    /* Brightnesses and Colors */
    /* y: the number of the "scan"line to calculate */
    /* LBuf: buffer for xres doubles. The depth information will be stored here */
    /* BBuf: buffer for xres floats. The brightnesses will be stored here */
    /* CBuf: buffer for xres floats. The colors will be stored here */
    /* fields in LineCalculator: */
    /* sbase: the specially normalized _base of the screen's coordinate-system */
    /* f, v: structures with the fractal and view information */
    /* zflag: 0..calc image from scratch;
              1..calc ZBuffer from scratch;
              2..calc image from ZBuffer */
              /* if zflag==2, LBuf must be initialized with the depths from the ZBuffer */
    int calcline(GPURowCalculator& rowCalc, int y,
        float* lBuf,
        float* bBuf,
        float* cBuf,
        ZFlag zflag);

    float colorizepoint();

    /* calculate the brightness value of a pixel (0.0 ... 1.0) */
    /* x: x coordinate of pixel on screen */
    /* y: the number of the "scan"line to calculate */
    /* LBuf: depth information for the pixel */
    /* returns brightness value */
    float brightpoint(int x, int y, float* lBuf);

    const FractalSpec& fractal() const { return _f; }
    const FractalView& view() const { return _v; }
    const CutSpec& cuts() const { return _cuts;  }
    const ViewBasis& sbase() const { return _sbase; }
    const ViewBasis& aabase() const { return _aabase; }

    Quat _xp;
    Quat _initialXStart;

    static CUDA_CALLABLE int orbitSpecial() { return 5; }

private:
    Quat _xq, _xcalc;
    FractalSpec _f;
    FractalView _v;
    ViewBasis _sbase, _base, _aabase;
    double _absx, _absy, _absz;
    CutSpec _cuts;

    size_t _lBufSize;
    Quat* _manyOrbits;
    float* _distances;
    float* _lastIters;
    float* _lBuf;
    float* _bBuf;

    Expression* _colorExpr;
    EvalContext _colorContext;

    static EvalSymbol _xSymbol, _ySymbol, _zSymbol, _wSymbol;
    static EvalSymbol _xbSymbol, _ybSymbol, _zbSymbol, _wbSymbol;
    static EvalSymbol _maxIterSymbol, _lastIterSymbol;
    static EvalSymbol _lastOrbitSymbol, _maxOrbitSymbol;
    static EvalSymbol _isInteriorSymbol;

    int (*_iterate_no_orbit) (iter_struct*);
    int (*_iterate) (iter_struct*);
};


inline CUDA_CALLABLE int aaLBufIndex(int x, int xaa, int yaa, int xres, int aa) {
    return ((yaa + 1) * xres + x) * aa + xaa;
}

