#include "LineCalculator.h"
#include "calculate.h"
#include "kernel.h"
#include "memory.h"
#include "iter.h"
#include "quat.h"

#define _USE_MATH_DEFINES // To get M_PI & c.
#include <math.h>

LineCalculator::LineCalculator(
    const FractalSpec& fractal,
    const FractalView& view,
    const CutSpec& cuts,
    ViewBasis& base,
    ViewBasis& sbase,
    ZFlag zflag)
    : _f(fractal), _v(view), _cuts(cuts) {

    switch (_f._formula) {
    case 0:
        _iterate_no_orbit = iterate_0_no_orbit;
        _iterate = iterate_0;
        _iternorm = iternorm_0;
        break;
    case 1:
        _iterate_no_orbit = iterate_1_no_orbit;
        _iterate = iterate_1;
        _iternorm = iternorm_1;
        break;
    case 2:
        _iterate_no_orbit = _iterate = iterate_2;
        _iternorm = 0;
        break;
    case 3:
        _iterate_no_orbit = iterate_3_no_orbit;
        _iterate = iterate_3;
        _iternorm = 0;
        break;
    case 4:
        _iterate_no_orbit = iterate_4_no_orbit;
        _iterate = iterate_4;
        _iternorm = 0;;
        break;
    case 5:
        _iterate_no_orbit = _iterate = iterate_bulb;
        _iternorm = 0;
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
    }

    _lBufSize = KLUDGE_PAD(_v._xres * _v._antialiasing * (_v._antialiasing + 1L));

    _xStarts = CUDAStorage<Quat>::allocHost(_lBufSize);
    _manyOrbits = CUDAStorage<Quat>::allocHost(_lBufSize * (fractal._maxOrbit + 2));
    _distances = CUDAStorage<float>::allocHost(_lBufSize);
    _lastIters = CUDAStorage<float>::allocHost(_lBufSize);
    _lBuf = CUDAStorage<float>::allocHost(_lBufSize);
}


LineCalculator::~LineCalculator() {
    CUDAStorage<Quat>::freeHost(_xStarts);
    CUDAStorage<Quat>::freeHost(_manyOrbits);
    CUDAStorage<float>::freeHost(_distances);
    CUDAStorage<float>::freeHost(_lastIters);
}


int LineCalculator::calcline(
    GPURowCalculator& rowCalc,
    int y,
    float* lBuf, float* bBuf, float* cBuf,
    ZFlag zflag) {
    int x, xaa, yaa;
    int antialiasing = _v._antialiasing;
    int aaSquared = antialiasing * antialiasing;

    _xs = _xp;
    _xp[3] = _f._lTerm;
    _xs[3] = _f._lTerm;
    for (x = 0; x < _v._xres; x++) {
        _xs += _sbase._x;
        int lbufIdx = (x + _v._xres) * antialiasing;
        for (yaa = 0; yaa < antialiasing; yaa++) {
            for (xaa = 0; xaa < antialiasing; xaa++) {
                int aaLbufIdx = aaLBufIndex(x, xaa, yaa, _v._xres, antialiasing);
                _xStarts[aaLbufIdx] = _xs + yaa * _aabase._y + xaa * _aabase._x;
            }
        }
    }
    rowCalc.obj_distances(
            *this, _lBufSize, _xStarts,
            _manyOrbits, _distances, _lastIters);
    for (x = 0; x < _v._xres; x++) {
        int lbufIdx = (x + _v._xres) * antialiasing;
        lBuf[lbufIdx] = _distances[lbufIdx];
        if (lBuf[lbufIdx] != _v._zres) {
            float itersum = _lastIters[lbufIdx];
            for (yaa = 0; yaa < antialiasing; yaa++) {
                for (xaa = 0; xaa < antialiasing; xaa++) {
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
    for (x = 0; x < _v._xres; x++) {
        if (bBuf[x] != 0) {
            bBuf[x] = brightpoint(x, y, lBuf);
            int lbufIdx = (x + _v._xres) * antialiasing;
            if (shouldCalculateImage(zflag) && bBuf[x] > 0.0001) {
                GlobalOrbit = &_manyOrbits[lbufIdx * (_f._maxOrbit + 2)];
                _f._lastiter = _lastIters[lbufIdx];
                MAXITER = _f._maxiter;
                LASTITER = _f._lastiter;
                LASTORBIT = std::min<double>(LASTITER, _f._maxOrbit);
                cBuf[x] = colorizepoint();
            }

        }
    }
    return 0;
}


extern progtype prog;

/* finds color for the point c->xq */
float LineCalculator::colorizepoint() {
    static size_t xh = progtype::nullHandle, yh = progtype::nullHandle, zh = progtype::nullHandle, wh = progtype::nullHandle;
    static size_t xbh = progtype::nullHandle, ybh = progtype::nullHandle, zbh = progtype::nullHandle, wbh = progtype::nullHandle;
    static size_t mih = progtype::nullHandle, lih = progtype::nullHandle;
    static size_t pih = progtype::nullHandle;
    char notdef;

    prog.setVariable("pi", &pih, M_PI);

    prog.setVariable("x", &xh, _xq[0]);
    prog.setVariable("y", &yh, _xq[1]);
    prog.setVariable("z", &zh, _xq[2]);
    prog.setVariable("w", &wh, _xq[3]);

    prog.setVariable("xb", &xbh, GlobalOrbit[_f._maxOrbit - 1][0]);
    prog.setVariable("yb", &ybh, GlobalOrbit[_f._maxOrbit - 1][1]);
    prog.setVariable("zb", &zbh, GlobalOrbit[_f._maxOrbit - 1][2]);
    prog.setVariable("wb", &wbh, GlobalOrbit[_f._maxOrbit - 1][3]);

    prog.setVariable("maxiter", &mih, MAXITER);
    prog.setVariable("lastiter", &lih, LASTITER);
    prog.setVariable("lastorbit", &lih, LASTORBIT);

    double CBuf = prog.calculate(&notdef);

    /* Make sure result is between 0 and 1 */
    CBuf = fmod(CBuf, 1.0);
    if (CBuf < 0) {
        CBuf = 1 + CBuf;
    }
    return static_cast<float>(CBuf);
}


float LineCalculator::brightpoint(int x, int y, float* lBuf) {
    Quat xp;
    float absolute = 0.0;
    float depth, bright;

    xp[3] = _f._lTerm;
    float bBuf = 0.0;
    float sqranti = static_cast<float>(_v._antialiasing * _v._antialiasing);
    for (int ya = 0; ya < _v._antialiasing; ya++) {
        for (int xa = 0; xa < _v._antialiasing; xa++) {
            _xcalc = _sbase._O
                + (x + xa / _v._antialiasing) * _sbase._x
                + (y + ya / _v._antialiasing) * _sbase._y;
            depth = lBuf[(x + (ya + 1) * _v._xres) * _v._antialiasing + xa];
            if (depth != _v._zres) {
                xp = _xcalc + depth * _sbase._z;
                /* Preserve point on object for colorizepoint */
                if (xa == 0 && ya == 0) {
                    _xq = xp;
                }
                float dz2;

                float dz1 = lBuf[(x + ya * _v._xres) * _v._antialiasing + xa] - depth;
                if (x + xa > 0) {
                    dz2 = lBuf[(x + (ya + 1) * _v._xres) * _v._antialiasing + (xa - 1)] - depth;
                } else {
                    dz2 = 0.0;
                }
                Vec3 n = -_v._antialiasing * _absx * _absy / _absz * _sbase._z
                    - dz2 * _absy * _absz / _absx * _sbase._x
                    - dz1 * _absz * _absx / _absy * _sbase._y;
                /* For a correct cross product, each factor must be multiplied
               with c->v.antialiasing, but as n gets normalized afterwards,
               this calculation is not necessary for our purpose. */
                absolute = static_cast<float>(n.magnitude());
                /* ensure that n points to viewer */
                /* if (scalar3prod(n, z) > 0) absolute = -absolute; */
                /* ideally there should stand >0 */

                assert(absolute != 0.0);
                n /= absolute;
                bright = _v.brightness(xp, n, _base._z);
                bright /= sqranti;
                bBuf += bright;
            }
        }
    }
    return static_cast<float>(bBuf);
}
