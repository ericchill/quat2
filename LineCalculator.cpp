#include "LineCalculator.h"
#include "kernel.h"
#include "memory.h"
#include "iter.h"
#include "quat.h"
#include "ExprEval.h"

#define _USE_MATH_DEFINES // To get M_PI & c.
#include <math.h>

static thread_local Quat* thread_orbit = nullptr;
static thread_local int thread_maxOrbit = 0;

static int orbIndex(double i) {
    return static_cast<int>(clamp<double>(i, 0, thread_maxOrbit));
}

static double orbitComponent(double oi, int compIdx) {
    return thread_orbit[orbIndex(oi) + 1][compIdx];
}

static double orbite(double i) {
    return orbitComponent(i, 0);
}

static double orbitj(double i) {
    return orbitComponent(i, 1);
}

static double orbitk(double i) {
    return orbitComponent(i, 2);
}

static double orbitl(double i) {
    return orbitComponent(i, 3);
}

static double orbitmag(double i) {
    return thread_orbit[orbIndex(i) + 1].magnitude();
}

static double orbitarg(double i) {
    Quat q = thread_orbit[orbIndex(i) + 1];
    return atan2(q[0], q.imag().magnitude());
}

static double orbitcarg(double i) {
    Quat q = thread_orbit[orbIndex(i) + 1];
    return atan2(q[0], q[1]);
}

static double orbitdot(double i, double j) {
    Quat a = thread_orbit[orbIndex(i) + 1];
    Quat b = thread_orbit[orbIndex(j) + 1];
    return a.vectorDot(b);
}

EvalSymbol LineCalculator::_xSymbol("x");
EvalSymbol LineCalculator::_ySymbol("y");
EvalSymbol LineCalculator::_zSymbol("z");
EvalSymbol LineCalculator::_wSymbol("w");
EvalSymbol LineCalculator::_xbSymbol("xb");
EvalSymbol LineCalculator::_ybSymbol("yb");
EvalSymbol LineCalculator::_zbSymbol("zb");
EvalSymbol LineCalculator::_wbSymbol("wb");
EvalSymbol LineCalculator::_maxIterSymbol("maxiter");
EvalSymbol LineCalculator::_lastIterSymbol("lastiter");
EvalSymbol LineCalculator::_lastOrbitSymbol("lastorbit");
EvalSymbol LineCalculator::_maxOrbitSymbol("maxorbit");
EvalSymbol LineCalculator::_isInteriorSymbol("is_interior");


LineCalculator::LineCalculator(
    const FractalSpec& fractal,
    const FractalView& view,
    const CutSpec& cuts,
    Expression *colorExpr,
    ViewBasis& base,
    ViewBasis& sbase,
    ZFlag zflag)
    : _f(fractal), _v(view), _cuts(cuts), _colorExpr(colorExpr) {

    switch (_f._formula) {
    case 0:
        _iterate_no_orbit = basic_iterate_sans_orbit<Iter0Op>;
        _iterate = basic_iterate<Iter0Op>;
        break;
    case 1:
        _iterate_no_orbit = basic_iterate_sans_orbit<Iter1Op>;
        _iterate = basic_iterate<Iter1Op>;
        break;
    case 2:
        _iterate_no_orbit = basic_iterate_sans_orbit<Iter2Op>;
        _iterate = basic_iterate<Iter2Op>;
        break;
    case 3:
        _iterate_no_orbit = basic_iterate_sans_orbit<Iter3Op>;
        _iterate = basic_iterate<Iter3Op>;
        break;
    case 4:
        _iterate_no_orbit = basic_iterate_sans_orbit<Iter3Op>;
        _iterate = basic_iterate<Iter3Op>;
        break;
    default:
        throw std::invalid_argument("Invalid formula number");
    }

    _base = base;
    _sbase = sbase;
    _absx = _sbase._x.magnitude() / _v._antialiasing;
    _absy = _sbase._y.magnitude() / _v._antialiasing;
    _absz = _sbase._z.magnitude();
    _aabase = _sbase;
    if (shouldCalculateDepths(zflag) && _v._antialiasing > 1) {
        _aabase._x /= _v._antialiasing;
        _aabase._y /= _v._antialiasing;
    }

    _lBufSize = KLUDGE_PAD(_v._xres * _v._antialiasing * (_v._antialiasing + 1));

    _manyOrbits = CUDAStorage<Quat>::allocHost(_lBufSize * (fractal._maxOrbit + LineCalculator::orbitSpecial()));
    _distances = CUDAStorage<float>::allocHost(_lBufSize);
    _lastIters = CUDAStorage<float>::allocHost(_lBufSize);
    _lBuf = CUDAStorage<float>::allocHost(_lBufSize);
    _bBuf = CUDAStorage<float>::allocHost(_v._xres);

    EvalContext::setupBasicVariables(_colorContext);
    _colorContext.setUnaryFunc("orbite", orbite);
    _colorContext.setUnaryFunc("orbitj", orbitj);
    _colorContext.setUnaryFunc("orbitk", orbitk);
    _colorContext.setUnaryFunc("orbitl", orbitl);
    _colorContext.setUnaryFunc("orbitmag", orbitmag);
    _colorContext.setUnaryFunc("orbitarg", orbitarg);
    _colorContext.setUnaryFunc("orbitcarg", orbitcarg);
    _colorContext.setBinaryFunc("orbitdot", orbitdot);
}


LineCalculator::~LineCalculator() {
    CUDAStorage<Quat>::freeHost(_manyOrbits);
    CUDAStorage<float>::freeHost(_distances);
    CUDAStorage<float>::freeHost(_lastIters);
    CUDAStorage<float>::freeHost(_lBuf);
    CUDAStorage<float>::freeHost(_bBuf);
}


int LineCalculator::calcline(
    GPURowCalculator& rowCalc,
    int y,
    float* lBuf,
    float* bBuf,
    float* cBuf,
    ZFlag zflag) {
    int x;
    int antialiasing = _v._antialiasing;

    rowCalc.obj_distances(
            *this, _lBufSize, y,
            _manyOrbits, _distances, _lastIters, lBuf, bBuf);
#if 0
    int aaSquared = antialiasing * antialiasing;
    for (x = 0; x < _v._xres; x++) {
        int lbufIdx = (x + _v._xres) * antialiasing;
        lBuf[lbufIdx] = _distances[lbufIdx];
        if (lBuf[lbufIdx] != _v._zres) {
            float itersum = _lastIters[lbufIdx];
            for (int yaa = 0; yaa < antialiasing; yaa++) {
                for (int xaa = 0; xaa < antialiasing; xaa++) {
                    if (xaa != 0 || yaa != 0) {
                        int aaLbufIdx = aaLBufIndex(x, xaa, yaa, _v._xres, antialiasing);
                        lBuf[aaLbufIdx] = _distances[aaLbufIdx];
                        itersum += _lastIters[aaLbufIdx];
                    }
                }
            }
            _lastIters[lbufIdx] = itersum / aaSquared;
            if (shouldCalculateImage(zflag)) {
                bBuf[x] = 1;
            }
        } else if (shouldCalculateImage(zflag)) {
            bBuf[x] = 0;
        }
    }
#endif
    for (x = 0; x < _v._xres; x++) {
        if (bBuf[x] != 0) {
            bBuf[x] = brightpoint(x, y, lBuf);
            int lbufIdx = (x + _v._xres) * antialiasing;
            if (shouldCalculateImage(zflag) && bBuf[x] > 0.0001) {
                thread_maxOrbit = _f._maxOrbit;
                thread_orbit = &_manyOrbits[lbufIdx * (_f._maxOrbit + LineCalculator::orbitSpecial())];
                _f._lastiter = _lastIters[lbufIdx];
                cBuf[x] = colorizepoint();
            }

        }
    }
    return 0;
}

/* finds color for the point c->xq */
float LineCalculator::colorizepoint() {

    _colorContext.setVariable(_xSymbol, _xq[0]);
    _colorContext.setVariable(_ySymbol, _xq[1]);
    _colorContext.setVariable(_zSymbol, _xq[2]);
    _colorContext.setVariable(_wSymbol, _xq[3]);

    _colorContext.setVariable(_xbSymbol, thread_orbit[_f._maxOrbit - 1][0]);
    _colorContext.setVariable(_ybSymbol, thread_orbit[_f._maxOrbit - 1][1]);
    _colorContext.setVariable(_zbSymbol, thread_orbit[_f._maxOrbit - 1][2]);
    _colorContext.setVariable(_wbSymbol, thread_orbit[_f._maxOrbit - 1][3]);

    _colorContext.setVariable(_maxIterSymbol, _f._maxiter);
    _colorContext.setVariable(_lastIterSymbol, _f._lastiter);
    _colorContext.setVariable(_lastOrbitSymbol, std::min<double>(_f._lastiter, _f._maxOrbit));
    _colorContext.setVariable(_maxOrbitSymbol, _f._maxOrbit - 1);

    _colorContext.setVariable(_isInteriorSymbol, _f._lastiter == _f._maxiter);

    double color = _colorExpr->eval(_colorContext);

    /* Make sure result is between 0 and 1 */
    color = fmod(color, 1.0);
    if (color < 0) {
        color = 1 + color;
    }
    return static_cast<float>(color);
}


float LineCalculator::brightpoint(int x, int y, float* lBuf) {
    double invAntialiasing = 1.0 / _v._antialiasing;
    double bBuf = 0.0;
    double sqranti = static_cast<double>(_v._antialiasing * _v._antialiasing);
    for (int ya = 0; ya < _v._antialiasing; ya++) {
        for (int xa = 0; xa < _v._antialiasing; xa++) {
            _xcalc = _sbase._O
                + (x + xa * invAntialiasing) * _sbase._x
                + (y + ya * invAntialiasing) * _sbase._y;
            double depth = lBuf[aaLBufIndex(x, xa, ya, _v._xres, _v._antialiasing)];
            if (depth != _v._zres) {
                Quat xp = _xcalc + depth * _sbase._z;
                xp[3] = _f._lTerm;
                /* Preserve point on object for colorizepoint */
                if (xa == 0 && ya == 0) {
                    _xq = xp;
                }

                double dz1 = lBuf[aaLBufIndex(x, xa, ya - 1, _v._xres, _v._antialiasing)] - depth;
                double dz2;
                if (x + xa > 0) {
                    dz2 = lBuf[aaLBufIndex(x, xa - 1, ya, _v._xres, _v._antialiasing)] - depth;
                } else {
                    dz2 = 0.0;
                }
                Vec3 n = -_v._antialiasing * _absx * _absy / _absz * _sbase._z
                    - dz2 * _absy * _absz / _absx * _sbase._x
                    - dz1 * _absz * _absx / _absy * _sbase._y;
                /* For a correct cross product, each factor must be multiplied
               with c->v.antialiasing, but as n gets normalized afterwards,
               this calculation is not necessary for our purpose. */
                /* ensure that n points to viewer */
                //if (n.dot(z) > 0) n = -n;
                /* ideally there should stand >0 */
                double bright = _v.brightness(xp, n.normalized(), _base._z);
                bBuf += bright / sqranti;
            }
        }
    }
    return static_cast<float>(bBuf);
}
