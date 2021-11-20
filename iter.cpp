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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <ctype.h>

#define _USE_MATH_DEFINES // To get M_PI & c.
#include <math.h>
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
#include <assert.h>

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
    Quat z = is->xstart;
    orbit[0] = z;
    double zMag2 = z.magnitudeSquared();
    int iter = 0;
    while (zMag2 < bailout && iter < maxiter) {
        z = z.squared() - is->c;
        zMag2 = z.magnitudeSquared();
        orbit[++iter] = z;
    }
    return iter;
}

int iterate_0_no_orbit(iter_struct* is) {
    double bailout = is->bailout;
    int maxiter = is->maxiter;
    Quat z = is->xstart;
    double zMag2 = z.magnitudeSquared();
    int iter = 0;
    while (zMag2 < bailout && iter < maxiter) {
        z = z.squared() - is->c;
        zMag2 = z.magnitudeSquared();
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


int iterate_1(struct iter_struct* is) {

    double re2, pure2;
    int iter, olditer;
    double ce, ci, cj, ck;
    Quat* orbit;

    Quat isxstart = is->xstart;
    double bailout = is->bailout;
    int maxiter = is->maxiter;

    assert(is->orbit != NULL);
    orbit = is->orbit;

    {
        Quat o0 = orbit[0];

        o0[0] = isxstart[0];
        re2 = isxstart[0] * isxstart[0];
        o0[1] = isxstart[1];
        pure2 = isxstart[1] * isxstart[1];
        o0[2] = isxstart[2];
        pure2 += isxstart[2] * isxstart[2];
        o0[3] = isxstart[3];
        pure2 += isxstart[3] * isxstart[3];
    }

    ce = is->c[0]; ci = is->c[1]; cj = is->c[2]; ck = is->c[3];

    iter = 0;
    olditer = 0;

    while ((re2 + pure2) < bailout && iter < maxiter) {

        double tmp1, tmp2;
        double oo1, oo2, oo3;

        ++iter;

        Quat& orbolditer = orbit[olditer];
        tmp1 = orbolditer[0] * (1.0 - orbolditer[0]) + pure2;
        tmp2 = 1.0 - 2.0 * orbolditer[0];
        Quat& orbiter = orbit[iter];
        oo1 = orbolditer[1];
        oo2 = orbolditer[2];
        oo3 = orbolditer[3];

        orbiter[0] = ce * tmp1 - tmp2 * (ci * oo1 + cj * oo2 + ck * oo3);
        orbiter[1] = ci * tmp1 - tmp2 * (-ce * oo1 + ck * oo2 - cj * oo3);
        orbiter[2] = cj * tmp1 - tmp2 * (-ck * oo1 - ce * oo2 + ci * oo3);
        orbiter[3] = ck * tmp1 - tmp2 * (cj * oo1 - ci * oo2 - ce * oo3);

        re2 = orbiter[0] * orbiter[0];
        pure2 = orbiter[1] * orbiter[1]
            + orbiter[2] * orbiter[2]
            + orbiter[3] * orbiter[3];

        ++olditer;
    }

    return iter;
}


int iterate_1_no_orbit(struct iter_struct* is) {
    double ce, ci, cj, ck;

    Quat isxstart = is->xstart;
    double bailout = is->bailout;
    int maxiter = is->maxiter;


    Quat z = is->xstart;

    ce = is->c[0]; ci = is->c[1]; cj = is->c[2]; ck = is->c[3];

    int iter = 0;
    double zre2 = z[0] * z[0];
    double zpure2 = (z - z.imag()).magnitudeSquared();
    while (zre2 + zpure2 < bailout && iter < maxiter) {

        double tmp1, tmp2;
        double oo1, oo2, oo3;

        tmp1 = z[0] * (1.0 - z[0]) + zpure2;
        tmp2 = 1.0 - 2.0 * z[0];
        oo1 = z[1];
        oo2 = z[2];
        oo3 = z[3];

        z = Quat(
            ce * tmp1 - tmp2 * (ci * oo1 + cj * oo2 + ck * oo3),
            ci * tmp1 - tmp2 * (-ce * oo1 + ck * oo2 - cj * oo3),
            cj * tmp1 - tmp2 * (-ck * oo1 - ce * oo2 + ci * oo3),
            ck * tmp1 - tmp2 * (cj * oo1 - ci * oo2 - ce * oo3));

        zre2 = z[0] * z[0];
        zpure2 = (z - z.imag()).magnitudeSquared();

        ++iter;
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


int iterate_2(struct iter_struct* is) {

    double re2, pure2, pure;
    double lntv;
    double atan2vt, atan2vta;
    int iter, olditer;
    Quat* orbit;

    double bailout = is->bailout;
    int maxiter = is->maxiter;

    assert(is->orbit != NULL);
    orbit = is->orbit;

    orbit[0] = is->xstart;
    re2 = orbit[0][0] * orbit[0][0];
    pure2 = orbit[0][1] * orbit[0][1];
    pure2 += orbit[0][2] * orbit[0][2];
    pure2 += orbit[0][3] * orbit[0][3];

    iter = 0;
    olditer = 0;

    while ((re2 + pure2) < bailout && iter < maxiter) {

        ++iter;
        Quat& orbi = orbit[iter];
        Quat& orboi = orbit[olditer];

        lntv = 0.5 * log(re2 + pure2);
        pure = sqrt(pure2);
        if (pure == 0.0) {
            orbi[0] = lntv * orboi[0] - is->c[0];
            orbi[1] = atan2(0.0, orboi[0]) - is->c[1];
            orbi[2] = -is->c[2];
            orbi[3] = -is->c[3];
        } else {
            atan2vt = atan2(pure, orboi[0]);
            atan2vta = atan2vt * orboi[0] / pure + lntv;
            orbi[0] = lntv * orboi[0] - atan2vt * pure - is->c[0];
            orbi[1] = atan2vta * orboi[1] - is->c[1];
            orbi[2] = atan2vta * orboi[2] - is->c[2];
            orbi[3] = atan2vta * orboi[3] - is->c[3];
        }
        ++olditer;

        re2 = orbi[0] * orbi[0];
        pure2 = orbi[1] * orbi[1];
        pure2 += orbi[2] * orbi[2];
        pure2 += orbi[3] * orbi[3];

        if (re2 + pure2 > 1E100) {
            return -1;
        }
    }

    return iter;
}

int iterate_3(struct iter_struct* is) {
    Quat x2;
    double re2 = 0.0, pure2 = 0.0;
    int iter = 0, olditer = 0;

    double bailout = is->bailout;
    int maxiter = is->maxiter;

    assert(is->orbit != NULL);
    Quat* orbit = is->orbit;

    orbit[0] = is->xstart;
    re2 = x2[0] = orbit[0][0] * orbit[0][0];
    x2[1] = orbit[0][1] * orbit[0][1];
    x2[2] = orbit[0][2] * orbit[0][2];
    x2[3] = orbit[0][3] * orbit[0][3];
    pure2 = x2[1] + x2[2] + x2[3];

    iter = 0;
    olditer = 0;

    while ((re2 + pure2) < bailout && iter < maxiter) {
        double re23mp2 = 3 * re2 - pure2;

        ++iter;

        Quat& oi = orbit[iter];
        Quat& ooi = orbit[olditer];

        oi[0] = ooi[0] * (re2 - 3 * pure2) - is->c[0];
        oi[1] = ooi[1] * re23mp2 - is->c[1];
        oi[2] = ooi[2] * re23mp2 - is->c[2];
        oi[3] = ooi[3] * re23mp2 - is->c[3];

        ++olditer;

        x2[0] = oi[0] * oi[0];
        x2[1] = oi[1] * oi[1];
        x2[2] = oi[2] * oi[2];
        x2[3] = oi[3] * oi[3];

        re2 = x2[0];
        pure2 = x2[1] + x2[2] + x2[3];

        if (re2 + pure2 > 1E100) {
            return -1;
        }
    }

    return iter;
}

int iterate_3_no_orbit(struct iter_struct* is) {
    double bailout = is->bailout;
    int maxiter = is->maxiter;
    Quat c = is->c;
    Quat z = is->xstart;

    double re2 = z[0] * z[0];
    double pure2 = (z - z.imag()).magnitudeSquared();

    int iter = 0;

    while ((re2 + pure2) < bailout && iter < maxiter) {
        double re23mp2 = 3 * re2 - pure2;

        z = Quat(
            z[0] * (re2 - 3 * pure2),
            z[1] * re23mp2,
            z[2] * re23mp2,
            z[3] * re23mp2) - c;

        ++iter;

        re2 = z[0] * z[0];
        pure2 = (z - z.imag()).magnitudeSquared();

        if (re2 + pure2 > 1E100) {
            return -1;
        }
    }

    return iter;
}


int iterate_4(struct iter_struct* is) {
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
        Quat px = pow(orbit[olditer], is->p[0]);
        orbit[iter] = px - is->c;
        xss = orbit[iter].magnitudeSquared();
        if (xss > 1E100) {
            return -1;
        }
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

int iterate_bulb(struct iter_struct* is) {
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

double whichEyeToSign(WhichEye eye) {
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
        v2._s += whichEyeToSign(viewType) * _interocular / 2 * base->_x;
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
void calc_struct::obj_distance_search_range(int& zFrom, int& zTo) {
    bool haveStart = false;
    zFrom = zTo = -1;
    for (int z = 0; z < _v._zres; z++) {
        Quat xstart = _xcalc + z * _sbase._z;
        if (!_cuts.cutaway(xstart)) {
            if (!haveStart) {
                zFrom = z;
                haveStart = true;
            }
        } else if (haveStart) {
            zTo = z;
            return;
        }
    }
    if (haveStart) {
        zTo = _v._zres;
    }
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
    is.exactiter = 0;

    for (int i = 0; i < 4; i++) {
        is.p[i] = _f._p[i];
    }

    is.orbit = GlobalOrbit;
    is.xstart[3] = _f._lTerm;
    int iter = 0;

    try {
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
            for (z2 = 1; z2 < refinement+1 && iter == _f._maxiter; z2++) {
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
    } catch (CUDAException& ex) {
        fprintf(stderr, "In obj_distance: %s\n", ex.what());
        assert(false);
        return 0;
    }
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
float calc_struct::brightpoint(long x, int y, double* lBuf) {
    long xa, ya;
    Quat xp;
    Vec3 n/*,  xp2*/;
    double absolute = 0.0;
    double depth, bright, bBuf, sqranti;

    xp[3] = _f._lTerm;
    bBuf = 0.0;
    sqranti = static_cast<double>(_v._antialiasing * _v._antialiasing);
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
                /*if (shouldCalculateDepths(zflag))    usual calculation of normal vector
              {
              xp2 = xp - 0.05 * c->_sbase._z;
              if (v._cuts.cutaway(xp2))
              c->_iternorm(xp, c->f.c, n, c->f.bailout, c->f.maxiter);
              else v._cuts.cutnorm(xp, xp2, n);
              }
              else */ /* use the ZBuffer */
                if (true) {
                    double dz1, dz2;

                    dz1 = lBuf[(x + ya * _v._xres) * _v._antialiasing + xa] - depth;
                    if (x + xa > 0) {
                        dz2 = lBuf[(x + (ya + 1) * _v._xres) * _v._antialiasing + (xa - 1)] - depth;
                    } else {
                        dz2 = 0.0;
                    }
                    n = -_v._antialiasing * _absx * _absy / _absz * _sbase._z
                        - dz2 * _absy * _absz / _absx * _sbase._x
                        - dz1 * _absz * _absx / _absy * _sbase._y;
                    /* For a correct cross product, each factor must be multiplied
                   with c->v.antialiasing, but as n gets normalized afterwards,
                   this calculation is not necessary for our purpose. */
                }
                absolute = n.magnitude();
                /* ensure that n points to viewer */
                /* if (scalar3prod(n, z) > 0) absolute = -absolute; */
                /* ideally there should stand >0 */

                assert(absolute != 0.0);
                n /= absolute;
                bright = _v.brightness(xp, n, _base._z);

                assert(sqranti != 0.0);

                bright /= sqranti;
                bBuf += bright;
            }
        }
    }
    return static_cast<float>(bBuf);
}

void object_distance_kernel(calc_struct& cs, const Quat& xStart, const Quat& zBase, const int *zLimits, int& zResults) {
    if (zLimits[0] == -1) {
        zResults = -1;
        return;
    }
    iter_struct is;
    is.c = cs._f._c;
    is.bailout = cs._f._bailout;
    is.maxiter = cs._f._maxiter;
    is.exactiter = 0;
    for (int z = zLimits[0]; z < zLimits[1]; z++) {
        Quat xs = xStart + static_cast<double>(z) * zBase;
        is.xstart = xs;
        int iter = cs._iterate_no_orbit(&is);
        if (iter == is.maxiter) {
            zResults = z;
            return;
        }
    }
    zResults = -1;
}

void getObjectDistances(calc_struct& cs, int N, const Quat* xStarts, const Quat& zBase, const int(*zLimits)[2], int* zResults) {
    for (int i = 0; i < N; i++) {
        object_distance_kernel(cs, xStarts[i], zBase, zLimits[i], zResults[i]);
    }
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
int calc_struct::calcline2(
    long x1, long x2, int y,
    double* lBuf, float* bBuf, float* cBuf,
    ZFlag zflag) {
    long x, xaa, yaa;
    struct iter_struct is;
    int antialiasing = _v._antialiasing;

    assert(GlobalOrbit != NULL);

    is.c = _f._c;
    is.bailout = _f._bailout;
    is.maxiter = _f._maxiter;
    is.exactiter = 0;

    for (int i = 0; i < 4; i++) {
        is.p[i] = _f._p[i];
    }
    is.orbit = &GlobalOrbit[1];
    is.xstart[3] = _f._lTerm;
    LexicallyScopedRangeCheckedStorage<int[2]> zRanges(x2 - x1 + 1);
    LexicallyScopedRangeCheckedStorage<Quat> xStarts(x2 - x1 + 1);
    _xs = _xp + (x1 - 1) * _sbase._x;
    for (x = x1; x <= x2; x++) {
        _xs += _sbase._x;
        xStarts[x - x1] = _xs;
        obj_distance_search_range(zRanges[x - x1][0], zRanges[x - x1][1]);
    }
    LexicallyScopedRangeCheckedStorage<int> zStarts(x2 - x1 + 1);
    row_of_obj_distance_driver(*this, x2 - x1 + 1, xStarts.ptr(), _sbase._z, zRanges.ptr(), zStarts.ptr());

    _xs = _xp + (x1 - 1) * _sbase._x;
    for (x = x1; x <= x2; x++) {
        _xs += _sbase._x;
        _xcalc = _xs;
        int zStart = zStarts[x - x1];
        if (-1 != zStart) {
            int lbufIdx = (x + _v._xres) * antialiasing;
            lBuf[lbufIdx] = obj_distance(zStart);
            double itersum = 0;
            for (yaa = 0; yaa < antialiasing; yaa++) {
                for (xaa = 0; xaa < antialiasing; xaa++) {
                    if (xaa != 0 || yaa != 0) {
                        _xcalc = _xs + yaa * _aabase._y + xaa * _aabase._y;
                        lBuf[((yaa + 1) * _v._xres + x) * antialiasing + xaa] = obj_distance(zStart);
                        itersum += _f._lastiter;
                    }
                }
            }
            _f._lastiter = itersum / (antialiasing * antialiasing);
        }
    }
    for (x = x1; x <= x2; x++) {
        if (-1 == zStarts[x - x1]) {
            for (yaa = 0; yaa < antialiasing; yaa++) {
                for (xaa = 0; xaa < antialiasing; xaa++) {
                    lBuf[((yaa + 1) * _v._xres + x) * antialiasing + xaa] = static_cast<float>(_v._zres);
                }
            }
        }
    }
    for (x = x1; x <= x2; x++) {
        if (lBuf[(x + _v._xres) * antialiasing] != _v._zres) {
            if (shouldCalculateImage(zflag)) {
                /* an image has to be calculated */
                bBuf[x] = brightpoint(x, y, lBuf);
                if (bBuf[x] > 0.0001) {
                    is.xstart = _xq;
                    _f._lastiter = _iterate(&is);
                    MAXITER = _f._maxiter;
                    LASTITER = _f._lastiter;
                    cBuf[x] = colorizepoint();
                }
            }
        } else if (shouldCalculateImage(zflag)) {
            bBuf[x] = 0;
        }
    }
    return 0;
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
    long x1, long x2, int y,
    double* lBuf, float* bBuf, float* cBuf,
    ZFlag zflag) {
    long x, xaa, yaa;
    struct iter_struct is;
    int antialiasing = _v._antialiasing;

    assert(GlobalOrbit != NULL);

    is.c = _f._c;
    is.bailout = _f._bailout;
    is.maxiter = _f._maxiter;
    is.exactiter = 0;

    for (int i = 0; i < 4; i++) {
        is.p[i] = _f._p[i];
    }
    is.orbit = &GlobalOrbit[1];
    is.xstart[3] = _f._lTerm;
    _xs = _xp + (x1 - 1) * _sbase._x;
    for (x = x1; x <= x2; x++) {
        _xs += _sbase._x;
        _xcalc = _xs;
        if (shouldCalculateDepths(zflag)) {
            int lbufIdx = (x + _v._xres) * antialiasing;  // +xres gets to right eye
            lBuf[lbufIdx] = obj_distance();
            if (lBuf[lbufIdx] != _v._zres) {
                double itersum = 0;
                for (yaa = 0; yaa < antialiasing; yaa++) {
                    for (xaa = 0; xaa < antialiasing; xaa++) {
                        if (xaa != 0 || yaa != 0) {
                            _xcalc = _xs + yaa * _aabase._y + xaa * _aabase._y;
                            lBuf[((yaa + 1) * _v._xres + x) * antialiasing + xaa] = obj_distance();
                            itersum += _f._lastiter;
                        }
                    }
                }
                _f._lastiter = itersum / (antialiasing * antialiasing);
            } else {
                for (yaa = 0; yaa < antialiasing; yaa++) {
                    for (xaa = 0; xaa < antialiasing; xaa++) {
                        lBuf[((yaa + 1) * _v._xres + x) * antialiasing + xaa] = static_cast<float>(_v._zres);
                    }
                }
            }
        }
        if (lBuf[(x + _v._xres) * antialiasing] != _v._zres) {
            if (shouldCalculateImage(zflag)) {
                /* an image has to be calculated */
                bBuf[x] = brightpoint(x, y, lBuf);
                if (bBuf[x] > 0.0001) {
                    is.xstart = _xq;
                    _f._lastiter = _iterate(&is);
                    MAXITER = _f._maxiter;
                    LASTITER = _f._lastiter;
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
    /* Handles for variables */
    static unsigned char xh = 255, yh = 255, zh = 255, wh = 255;
    static unsigned char xbh = 255, ybh = 255, zbh = 255, wbh = 255;
    static unsigned char mih = 255, lih = 255;
    static unsigned char pih = 255;
    char notdef;
    /*, diffrh=255, eorbith = 255, jorbith = 255, korbith = 255; */
    /* unsigned char lorbith = 255, closestith = 255; */
    //progtype prog;

    prog.setVariable("pi", &pih, M_PI);

    prog.setVariable("x", &xh, _xq[0]);
    prog.setVariable("y", &yh, _xq[1]);
    prog.setVariable("z", &zh, _xq[2]);
    prog.setVariable("w", &wh, _xq[3]);

    prog.setVariable("xb", &xbh, GlobalOrbit[(int)LASTITER][0]);
    prog.setVariable("yb", &ybh, GlobalOrbit[(int)LASTITER][1]);
    prog.setVariable("zb", &zbh, GlobalOrbit[(int)LASTITER][2]);
    prog.setVariable("wb", &wbh, GlobalOrbit[(int)LASTITER][3]);

    prog.setVariable("maxiter", &mih, MAXITER);
    prog.setVariable("lastiter", &lih, LASTITER);

    /*  prog.setVariable("diffr", &diffrh, diffr);
      prog.setVariable("eorbit", &eorbith, lastorbit[0]);
      prog.setVariable("jorbit", &jorbith, lastorbit[1]);
      prog.setVariable("korbit", &korbith, lastorbit[2]);
      prog.setVariable("lorbit", &lorbith, lastorbit[3]);
      prog.setVariable("closestit", &closestith, closest_iteration);
   */
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
    : _f(fractal), _v(view), _cuts(cuts)
{

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
        _iterate_no_orbit = _iterate = iterate_4;
        _iternorm = 0;;
        break;
    case 5:
        _iterate_no_orbit = _iterate = iterate_bulb;
        _iternorm = 0;
        break;
    default:
        assert(1 == 0);
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
}