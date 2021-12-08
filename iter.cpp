/* Quat - A 3D fractal generation program */
/* Copyright (C) 1997-2000 Dirk Meyer */
/* (email: dirk.meyer@studserv.uni-stuttgart.de) */
/* mail:  Dirk Meyer */
/*        Marbacher Weg 29 */
/*        D-71334 Waiblingen */
/*        Germany */
/* */
/* This program is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU General Public License */
/* as published by the Free Software Foundation; either version 2 */
/* of the License, or (at your option) any later version. */
/* */
/* This program is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/* GNU General Public License for more details. */
/* */
/* You should have received a copy of the GNU General Public License */
/* along with this program; if not, write to the Free Software */
/* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA. */


#include <ctype.h>

#define _USE_MATH_DEFINES // To get M_PI & c.
#include <math.h>
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
#include "CReplacements.h"
#include "common.h"
#include "calculat.h"
#include "quat.h"
#include "iter.h" 
#include "memory.h"

#include "kernel.h"

#include <crtdbg.h>

static bool cutnorm(const Quat& x1, const Quat& x2, Quat& n, const CutSpec& cutbuf);

int iterate_0(iter_struct* is) {
    Quat* orbit = is->orbit;
    double bailout = is->bailout;
    int maxiter = is->maxiter;
    int maxOrbit = is->maxOrbit;
    Quat z = is->xstart;
    orbit[0] = z;
    int iter = 0;
    while (z.magnitudeSquared() < bailout && iter < maxOrbit && iter < maxiter) {
        z = z.squared() - is->c;
        orbit[++iter] = z;
    }
    while (z.magnitudeSquared() < bailout && iter < maxiter) {
        z = z.squared() - is->c;
        ++iter;
    }
    orbit[maxOrbit - 1] = z;
    return iter;
}

int iterate_0_no_orbit(iter_struct* is) {
    double bailout = is->bailout;
    int maxiter = is->maxiter;
    Quat z = is->xstart;
    int iter = 0;
    while (z.magnitudeSquared() < bailout && iter < maxiter) {
        z = z.squared() - is->c;
        ++iter;
    }
    return iter;
}

int iternorm_0(iter_struct* is, Vec3& norm) {
    Quat x, ox;
    double diff[4][3], odiff[4][3];
    /* differentiations of 1st index to 2nd index */
    /* could be called Jacobi-matrix */

    int iter = 0;

    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 4; i++) {
            odiff[i][j] = 0;
        }
    }

    odiff[0][0] = odiff[1][1] = odiff[2][2] = 1;
    ox = is->xstart;
    Quat c = is->c;
    while (ox.magnitudeSquared() < is->bailout && iter < is->maxiter) {
        /* usual quaternion z^2-c iteration */
        Quat x = ox.squared() - c;

        /* iteration of gradient formula */
        for (int i = 0; i < 3; i++) {
            diff[0][i] = 2 * (ox[0] * odiff[0][i] - ox[1] * odiff[1][i] - ox[2] * odiff[2][i] - ox[3] * odiff[3][i]);
            diff[1][i] = 2 * (ox[0] * odiff[1][i] + ox[1] * odiff[0][i]);
            diff[2][i] = 2 * (ox[0] * odiff[2][i] + ox[2] * odiff[0][i]);
            diff[3][i] = 2 * (ox[0] * odiff[3][i] + ox[3] * odiff[0][i]);
        }

        ox = x;
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 4; i++) {
                odiff[i][j] = diff[i][j];
            }
        }
        iter++;
    }

    double absx = x.magnitude();

    for (int i = 0; i < 3; i++) {
        norm[i] = (x[0] * diff[0][i] + x[1] * diff[1][i] + x[2] * diff[2][i]
            + x[3] * diff[3][i]) / absx;

    }
    return iter;
}


int iterate_1(iter_struct* is) {
    // c * z * (1 - z)
    Quat* orbit = is->orbit;
    double bailout = is->bailout;
    int maxiter = is->maxiter;
    int maxOrbit = is->maxOrbit;
    Quat z = is->xstart;
    orbit[0] = z;
    int iter = 0;
    while (z.magnitudeSquared() < bailout && iter < maxOrbit && iter < maxiter) {
        z = is->c * z * (1.0 - z);
        orbit[++iter] = z;
    }
    while (z.magnitudeSquared() < bailout && iter < maxiter) {
        z = is->c * z * (1.0 - z);
        ++iter;
    }
    orbit[maxOrbit - 1] = z;
    return iter;
}

int iterate_1_no_orbit(iter_struct* is) {
    // c * z * (1 - z)
    double bailout = is->bailout;
    int maxiter = is->maxiter;
    Quat z = is->xstart;

    int iter = 0;
    while (z.magnitudeSquared() < bailout && iter < maxiter) {
        z = is->c * z * (1.0 - z);
        iter++;
    }
    return iter;
}

int iternorm_1(iter_struct* is, Vec3& norm) {

    Quat x2, x3, x4, x5;
    Quat diff[3], odiff[3];

    for (int i = 0; i < 3; i++) {
        odiff[i] = 0;
    }

    /* 1st index: differentiation for x,y,z */
    /* 2nd index: components (e,j,k,l) */
    odiff[0][0] = 1;
    odiff[1][1] = 1;
    odiff[2][2] = 1;

    Quat x1 = is->xstart;
    double xss = x1.magnitudeSquared();

    int iter = 0;
    Quat c = is->c;
    while (x1.magnitudeSquared() < is->bailout && iter < is->maxiter) {
        /* usual iteration */
        x3 = x1;
        x2 = 1 - x1;
        x4 = c * x3;
        x5 = x4 * x2;

        /* iteration of gradient formula */
        for (int i = 0; i < 3; i++) {

            diff[i][0]
                = odiff[i][0] * (2 * (-c[0] * x1[0] + c[1] * x1[1]
                    + c[2] * x1[2] + c[3] * x1[3])
                    + c[0])
                + odiff[i][1] * (2 * c[0] * x1[1] + 2 * c[1] * x1[0] - c[1])
                + odiff[i][2] * (2 * c[0] * x1[2] + 2 * c[2] * x1[0] - c[2])
                + odiff[i][3] * (2 * c[0] * x1[3] + 2 * c[3] * x1[0] - c[3]);
            diff[i][1]
                = odiff[i][0] * (-2 * c[1] * x1[0] - 2 * c[0] * x1[1]
                    + 2 * c[3] * x1[2] - 2 * c[2] * x1[3]
                    + c[1])
                + odiff[i][1] * (2 * c[1] * x1[1] - 2 * c[0] * x1[0] + c[0])
                + odiff[i][2] * (2 * c[1] * x1[2] + 2 * c[3] * x1[0] - c[3])
                + odiff[i][3] * (2 * c[1] * x1[3] - 2 * c[2] * x1[0] + c[2]);
            diff[i][2]
                = odiff[i][0] * (-2 * c[2] * x1[0] - 2 * c[3] * x1[1]
                    - 2 * c[0] * x1[2] + 2 * c[1] * x1[3] + c[2])
                + odiff[i][1] * (2 * c[2] * x1[1] - 2 * c[3] * x1[0] + c[3])
                + odiff[i][2] * (2 * c[2] * x1[2] - 2 * c[0] * x1[0] + c[0])
                + odiff[i][3] * (2 * c[2] * x1[3] + 2 * c[1] * x1[0] - c[1]);
            diff[i][3]
                = odiff[i][0] * (-2 * c[3] * x1[0] + 2 * c[2] * x1[1]
                    - 2 * c[1] * x1[2] - 2 * c[0] * x1[3] + c[3])
                + odiff[i][1] * (2 * c[3] * x1[1] + 2 * c[2] * x1[0] - c[2])
                + odiff[i][2] * (2 * c[3] * x1[2] - 2 * c[1] * x1[0] + c[1])
                + odiff[i][3] * (2 * c[3] * x1[3] - 2 * c[0] * x1[0] + c[0]);
        }
        x1 = x5;

        for (int i = 0; i < 3; i++) {
            odiff[i] = diff[i];
        }

        /* check whether range of double will be enough / underflow? */
        xss = 0;
        for (int i = 0; i < 3; i++) {
            xss += diff[i].magnitudeSquared();
        }

        if (xss < 1E-16) {
            break;  /* normal Vec3 will get zero; do approx. */
        }

        iter++;
    }

    xss = x1.magnitudeSquared();

    norm[0] = 1;
    norm[1] = 0;
    norm[2] = 0;

    if (xss == 0) {
        return 0;
    }

    for (int i = 0; i < 3; i++) {
        Quat ni = 0;
        for (int j = 0; j < 4; j++) {
            Quat xdiff = x1[j] * diff[i][j];
            ni += xdiff;
        }
        norm[i] = ni / xss;
    }

    /* | x[maxiter] | is what we want to have differentiated partially */
    /*   if (norm[0]==0 && norm[1]==0 && norm[2]==0) norm[0] = 1; */

    return iter;
}


int iterate_2(iter_struct* is) {
    // x[n+1] = x[n]*ln(x[n]) - c
    Quat* orbit = is->orbit;
    assert(orbit != NULL);

    double bailout = is->bailout;
    int maxiter = is->maxiter;
    int maxOrbit = is->maxOrbit;

    Quat z = is->xstart;
    orbit[0] = z;

    int iter = 0;

    while (z.magnitudeSquared() < bailout && iter < maxOrbit && iter < maxiter) {
        z = z * log(z) - is->c;
        orbit[++iter] = z;
    }
    while (z.magnitudeSquared() < bailout && iter < maxiter) {
        z = z * log(z) - is->c;
        ++iter;
    }
    orbit[maxOrbit - 1] = z;
    return iter;
}

int iterate_2_no_orbit(iter_struct* is) {
    // x[n+1] = x[n]*ln(x[n]) - c
    double bailout = is->bailout;
    int maxiter = is->maxiter;

    Quat z = is->xstart;

    int iter = 0;
    while (z.magnitudeSquared() < bailout && iter < maxiter) {
        z = z * log(z) - is->c;
        iter++;
    }

    return iter;
}


int iterate_3(iter_struct* is) {
    // z = z^3 - c
    Quat* orbit = is->orbit;
    assert(orbit != NULL);

    double bailout = is->bailout;
    int maxiter = is->maxiter;
    int maxOrbit = is->maxOrbit;

    Quat z = is->xstart;
    orbit[0] = z;

    int iter = 0;
    while (z.magnitudeSquared() < bailout && iter < maxOrbit && iter < maxiter) {
        z = z.squared() * z - is->c;
        orbit[iter++] = z;
    }
    while (z.magnitudeSquared() < bailout && iter < maxiter) {
        z = z.squared() * z - is->c;
        ++iter;
    }
    orbit[maxOrbit - 1] = z;
    return iter;
}

int iterate_3_no_orbit(iter_struct* is) {
    // z = z^3 - c
    double bailout = is->bailout;
    int maxiter = is->maxiter;

    Quat z = is->xstart;

    int iter = 0;
    while (z.magnitudeSquared() < bailout && iter < maxiter) {
        z = z.squared() * z - is->c;
        iter++;
    }
    return iter;
}


int iterate_4(iter_struct* is) {
    double bailout = is->bailout;
    int maxiter = is->maxiter;
    int maxOrbit = is->maxOrbit;

    assert(is->orbit != NULL);
    Quat* orbit = is->orbit;
    Quat z = is->xstart;
    orbit[0] = z;

    int iter = 0;
    int olditer = -1;

    while (!isnan(z[0]) && z.magnitudeSquared() < bailout && iter < maxOrbit && iter < maxiter) {
        Quat px = pow(z, is->p[0]);
        z = px - is->c;
        orbit[++iter] = z;
    }
    while (!isnan(z[0]) && z.magnitudeSquared() < bailout && iter < maxiter) {
        Quat px = pow(z, is->p[0]);
        z = px - is->c;
        ++iter;
    }
    orbit[maxOrbit - 1] = z;
    return iter;
}

int iterate_4_no_orbit(iter_struct* is) {
    double bailout = is->bailout;
    int maxiter = is->maxiter;

    Quat z = is->xstart;

    int iter = 0;
    while (!isnan(z[0]) && z.magnitudeSquared() < bailout && iter < maxiter) {
        Quat px = pow(z, is->p[0]);
        z = px - is->c;
        ++iter;
    }
    return iter;
}


void bulbSquare0(Quat& result, const Quat& value) {
    double x = value[0];
    double y = value[1];
    double z = value[2];
    double x2 = x * x;
    double y2 = y * y;
    double x2y2 = sqrt(x2 + y2);
    result[0] = 2 * (x - y) * (x + y) * z / x2y2;
    result[1] = 4 * z * y * z / x2y2;
    result[2] = -x2 - y2 + z * z;
    result[3] = 0;
}

void bulbSquare1(Quat& result, const Quat& value) {
    double x = value[0];
    double y = value[1];
    double z = value[2];
    double x2 = x * x;
    double y2 = y * y;
    double z2 = z * z;
    double x2y2 = x2 + y2;
    double x2z2 = x2 + z2;
    double x2y2z2 = z2 + y2 + z2;
    double x2y2x2z2 = (x2 + y2) * x2z2;
    result[0] = -x2 + y2 + (x - y) * (x + y) * z2 / x2y2;
    result[1] = 2 * x * y * (z2 / x2y2 - 1);
    result[2] = 2 * sqrt(x2y2) * z;
    result[3] = 0;
}

void buldSquare2(double* result, const double* value) {
}

int iterate_bulb(iter_struct* is) {
    /*
        { cx + (2 (x - y) (x + y) _z) / (x^2 + y^2),
          cy + (4 x y _z) / (x^2 + y^2),
          cz - x^2 - y^2 + _z^2 }
     */

    Quat px = { 0.0, 0.0, 0.0, 0.0 };

    double bailout = is->bailout;
    int maxiter = is->maxiter;

    assert(is->orbit != NULL);
    Quat* orbit = is->orbit;
    orbit[0] = is->xstart;
    double xss = is->xstart.magnitudeSquared();

    int iter = 0;
    int olditer = -1;

    while (xss < bailout && iter < maxiter) {
        ++iter;
        ++olditer;
        bulbSquare1(px, orbit[olditer]);
        orbit[iter] = px - is->c;
        xss = orbit[iter].magnitudeSquared();
    }
    return iter;
}


int FractalView::DoCalcbase(
    ViewBasis* base,
    ViewBasis* sbase,
    bool use_proj_up,
    Vec3 proj_up) {

    double absolute = 0.0, lambda = 0.0;
    Vec3 ss, tmp;

    if (_xres == 0) {
        return -1;
    }
    double leny = _LXR * _yres / _xres;

    /* Norm of z */
    absolute = _s.magnitude();
    if (absolute == 0) {
        return -1;
    }
    base->_z = -_s / absolute;

    /* Norm of up */
    absolute = _up.magnitude();
    if (absolute == 0) {
        return -1;
    }
    _up /= absolute;

    /* check whether up is linearly dependent on z */
    /* cross product != 0 */
    if (_up.cross(base->_z).magnitude() == 0) {
        return -1;
    }

    if (use_proj_up) {
        base->_y = proj_up;
    } else {
        /* Check whether up orthogonal to _z */
        if (base->_z.dot(_up) == 0.0) {
            /* Yes -> norm(up) = -y-Vec3 */
            base->_y = -_up;
        } else {
            /* calculate projection of up onto pi */
            tmp = _s - _up;
            lambda = base->_z.dot(tmp);
            ss = lambda * base->_z + _up;

            /* tmp-Vec3: tmp = s-ss */
            tmp = _s - ss;
            absolute = tmp.magnitude();
            assert(absolute != 0.0);

            /* with this: y-vector */
            base->_y = tmp / absolute;
        }
    }

    /* calculate x-vector (through cross product) */
    base->_x = base->_y.cross(base->_z);

    /* calculate origin */
    leny /= 2;
    _LXR /= 2;
    base->_O = _s - leny * base->_y - _LXR * base->_x;

    /* ready with base, now: calculate a specially pseudo-normed base */
    /* where the length of the base vectors represent 1 pixel */

    if (sbase == NULL) {
        return 0;
    }
    if (_yres == 0) {
        return -1;
    }
    _LXR *= 2;
    leny *= 2;
    sbase->_O = base->_O;
    sbase->_x = _LXR * base->_x / _xres;
    sbase->_y = leny * base->_y / _yres;

    if (_zres == 0) {
        return -1;
    }

    /* how deep into scene */
    absolute = fabs(base->_z.dot(_s));
    absolute *= 2;
    sbase->_z = absolute * base->_z / _zres;

    /* Only thing that has to be done: shift the plane */
    return 0;
}

double whichEyeSign(WhichEye eye) {
    switch (eye) {
    default:  // VS2019 compiler can't tell the rest of the cases are exhaustive.
    case WhichEye::Monocular:
        return 0;
    case WhichEye::Right:
        return 1;
    case WhichEye::Left:
        return -1;
    }
}

int FractalView::calcbase(ViewBasis* base, ViewBasis* sbase, WhichEye viewType) {
    ViewBasis commonbase;
    Vec3 y;

    int e = DoCalcbase(base, sbase, false, y);
    if (e != 0) {
        return e;
    }
    commonbase = *base;
    if (viewType != WhichEye::Monocular) {
        FractalView v2 = *this;
        v2._s += whichEyeSign(viewType) * _interocular / 2 * base->_x;
        y = base->_y;
        e = v2.DoCalcbase(base, sbase, true, y);
        if (e != 0) {
            return e;
        }
    }

    /* Shift plane in direction of common plane */
    base->_O += _Mov[0] * commonbase._x + _Mov[1] * commonbase._y;
    if (sbase != NULL) {
        sbase->_O = base->_O;
    }
    return 0;
}


/* Determines the distance from object to point c->xcalc on viewplane
   in _z direction.
   In other words: looks for the object in _z direction */
double calc_struct::obj_distance(int zStart) {

    int z, z2;
    iter_struct is;

    float refinement = 20.0; // 15 + 5 * v._zres / 480.0;

    assert(GlobalOrbit != NULL);
    is.c = _f._c;

    is.bailout = _f._bailout;
    is.maxiter = _f._maxiter;
    is.maxOrbit = _f._maxOrbit;
    is.exactiter = 0;

    for (int i = 0; i < 4; i++) {
        is.p[i] = _f._p[i];
    }

    is.orbit = GlobalOrbit;
    is.xstart[3] = _f._lTerm;
    int iter = 0;

    for (z = zStart; z < _v._zres && iter != _f._maxiter; z++) {
        is.xstart = _xcalc + z * _sbase._z;
        if (!_cuts.cutaway(is.xstart)) {
            iter = _iterate_no_orbit(&is);
        } else {
            iter = 0;
        }
    }
    if (z < _v._zres) {
        z--;
        for (z2 = 1; z2 <= refinement && iter == _f._maxiter; z2++) {
            is.xstart = _xcalc + (z - z2 / refinement) * _sbase._z;
            if (!_cuts.cutaway(is.xstart)) {
                iter = _iterate(&is);
            } else {
                iter = 0;
            }
        }
        z2 -= 2;
    } else {
        z2 = 0;
    }
    if (iter != 0) {
        _f._lastiter = iter;
    }
    return floorf((z - z2 / refinement) * 1000.0f + 0.5f) / 1000.0f;
}


double FractalView::brightness(const Vec3& p, const Vec3& n, const Vec3& z) const {
    /* values >1 are for phong */
    Vec3 l, r;
    double absolute, result, a, b, c;

    /* vector point -> light source */
    l = _light - p;
    absolute = l.magnitude();
    if (absolute == 0) {
        return 255.0;   /* light source and point are the same! */
    }
    l /= absolute;

    /* Lambert */
    a = n.dot(l);

    /* Distance */
    b = 20 / absolute;
    if (b > 1) {
        b = 1;
    }

    /* Phong */
    if (a > 0) {
        r = 2 * a * n - l;
        /* r...reflected ray */
        c = r.dot(z);
        if (c < 0) {
            c = _phongmax * exp(_phongsharp * log(-c));
        } else {
            c = 0;
        }
    } else {
        c = 0;
    }

    if (a < 0) {
        result = _ambient;
    } else {
        result = a * b + _ambient;
        result = fmin(result, 1.0); /* Lambert and ambient together can't get bigger than 1 */
        result += c;                   /* additional phong to get white */
        result = fmax(result, 0);
    }
    return result;
}


/* calculate a brightness value (0.0 ... 1.0)
   all fields (except xq) in calc_struct MUST be initialized
   should only be called if it´s sure that the object was hit!
   (lBuf != c->v.zres)
   zflag:
      0..calc image from scratch;
      1..calc ZBuffer from scratch;
      2..calc image from ZBuffer
*/
float calc_struct::brightpoint(int x, int y, double* lBuf) {
    long xa, ya;
    Quat xp;
    double absolute = 0.0;
    double depth, bright;

    xp[3] = _f._lTerm;
    double bBuf = 0.0;
    double sqranti = static_cast<double>(_v._antialiasing * _v._antialiasing);
    for (ya = 0; ya < _v._antialiasing; ya++) {
        for (xa = 0; xa < _v._antialiasing; xa++) {
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
                double dz2;

                double dz1 = lBuf[(x + ya * _v._xres) * _v._antialiasing + xa] - depth;
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
                absolute = n.magnitude();
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

void row_of_obj_distance_CPU(calc_struct& cs, size_t N, const Quat& zBase) {
    for (size_t i = 0; i < N; i++) {
        double refinement = 20.0;
        Quat xStart = cs._xStarts[i];
        int iter = -1;

        iter_struct is;
        for (int i = 0; i < 4; i++) {
            is.p[i] = cs._f._p[i];
        }
        is.c = cs._f._c;
        is.bailout = cs._f._bailout;
        is.maxiter = cs._f._maxiter;
        is.maxOrbit = cs._f._maxOrbit;
        is.exactiter = 0;
        is.orbit = &cs._manyOrbits[i * (cs._f._maxOrbit + 2)];
        int z, z2;
        for (z = 0; z < cs._v._zres && iter != is.maxiter; z++) {
            Quat z0 = xStart + static_cast<double>(z) * zBase;
            is.xstart = z0;
            if (!cs._cuts.cutaway(z0)) {
                iter = cs._iterate_no_orbit(&is);
            } else {
                iter = 0;
            }
        }
        if (z < cs._v._zres) {
            z--;
            for (z2 = 1; z2 <= refinement && iter == is.maxiter; z2++) {
                Quat z0 = xStart + static_cast<double>(z - z2 / refinement) * zBase;
                is.xstart = z0;
                iter = cs._iterate(&is);
            }
            z2 -= 2;
        } else {
            z2 = 0;
        }
        cs._lastIters[i] = iter;
        cs._distances[i] = floor(static_cast<double>(z - z2 / refinement) * 1000.0 + 0.5) / 1000.0;
    }
}

inline int aaLBufIndex(int x, int xaa, int yaa, int xres, int aa) {
    return ((yaa + 1) * xres + x) * aa + xaa;
}


/*
   Calculate _a whole line of depths, brightnesses and colors
   All fields (for xs, xq, xcalc only ..[3]=lvalue) in calc_struct
   MUST be filled in!
   Especially xp must be set to the beginning of the line to calculate
   c->xp used for 4d-beginning of line,
   c->xs used for 4d-point in view plane that is worked on,
   c->xq used for 4d-point on object (filled in by brightpoint,
   used by colorizepoint)
   c->xcalc used for 4d point to calculate (fctn. obj_distance)
   (all to avoid double calculations, speed)
   zflag:
      0..calc image from scratch;
      1..calc ZBuffer from scratch;
      2..calc image from ZBuffer
*/
int calc_struct::calcline(
    GPURowCalculator& rowCalc,
    int x1, int x2, int y,
    double* lBuf, float* bBuf, float* cBuf,
    ZFlag zflag) {
    int x, xaa, yaa;
    iter_struct is;
    int antialiasing = _v._antialiasing;
    int aaSquared = antialiasing * antialiasing;

    is.c = _f._c;
    is.bailout = _f._bailout;
    is.maxiter = _f._maxiter;
    is.maxOrbit = _f._maxOrbit;
    is.exactiter = 0;

    for (int i = 0; i < 4; i++) {
        is.p[i] = _f._p[i];
    }
    is.xstart[3] = _f._lTerm;
    _xs = _xp + (x1 - 1) * _sbase._x;
    for (x = x1; x <= x2; x++) {
        _xs += _sbase._x;
        int lbufIdx = (x + _v._xres) * antialiasing;
        for (yaa = 0; yaa < antialiasing; yaa++) {
            for (xaa = 0; xaa < antialiasing; xaa++) {
                    int aaLbufIdx = aaLBufIndex(x, xaa, yaa, _v._xres, antialiasing);
                    _xStarts[aaLbufIdx] = _xs + yaa * _aabase._y + xaa * _aabase._x;
            }
        }
    }
    if (haveGPU) {
        rowCalc.obj_distances(
            *this, _lBufSize, _xStarts,
            _manyOrbits, _distances, _lastIters);
    } else {
        row_of_obj_distance_CPU(*this, _lBufSize, _sbase._z);
    }
    for (x = x1; x <= x2; x++) {
        int lbufIdx = (x + _v._xres) * antialiasing;
        lBuf[lbufIdx] = _distances[lbufIdx];
        if (lBuf[lbufIdx] != _v._zres) {
            double itersum = _lastIters[lbufIdx];
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
                bBuf[x] = brightpoint(x, y, lBuf);
                if (bBuf[x] > 0.0001) {
                    is.xstart = _xq;
                    GlobalOrbit = &_manyOrbits[lbufIdx * (_f._maxOrbit + 2)];
                    _f._lastiter = _lastIters[lbufIdx];
                    MAXITER = _f._maxiter;
                    LASTITER = _f._lastiter;
                    LASTORBIT = std::min<double>(LASTITER, _f._maxOrbit);
                    cBuf[x] = colorizepoint();
                }
            }
        } else if (shouldCalculateImage(zflag)) {
            bBuf[x] = 0;
        }
    }
    return 0;
}

extern progtype prog;

/* finds color for the point c->xq */
float calc_struct::colorizepoint() {
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


calc_struct::calc_struct(
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

    _lBufSize = _v._xres * _v._antialiasing * (_v._antialiasing + 1L) + 10;

    _xStarts = CUDAStorage<Quat>::allocHost(_lBufSize);
    _manyOrbits = CUDAStorage<Quat>::allocHost(_lBufSize * (fractal._maxOrbit + 2));
    _distances = CUDAStorage<double>::allocHost(_lBufSize);
    _lastIters = CUDAStorage<double>::allocHost(_lBufSize);
}

calc_struct::~calc_struct() {
    CUDAStorage<Quat>::freeHost(_xStarts);
    CUDAStorage<Quat>::freeHost(_manyOrbits);
    CUDAStorage<double>::freeHost(_distances);
    CUDAStorage<double>::freeHost(_lastIters);
}