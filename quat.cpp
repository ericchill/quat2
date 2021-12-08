/* Quat - A 3D fractal generation program */
/* Copyright (C) 1997-2000 Dirk Meyer */
/* (email: dirk.meyer@studserv.uni-stuttgart.de) */
/* mail:  Dirk Meyer */
/*        Marbacher Weg 29 */
/*        D-71334 Waiblingen */
/*        Germany */
/* */
/* Further work 2004 by Eric C. Hill (eric@stochastic.com) */
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
/* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */


#include <algorithm>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES // To get M_PI & c.
#include <math.h>
#include <time.h>
#include <ctype.h>
#include "common.h"
#include "quat.h"
#include "iter.h"
#include "files.h"
#include "colors.h"
#include "calculat.h"   
#include "ver.h"
#include "memory.h"
#include "kernel.h"

Quat* GlobalOrbit = nullptr;

char orbite(double* a, const double* b);
char orbitj(double* a, const double* b);
char orbitk(double* a, const double* b);
char orbitl(double* a, const double* b);

int PrepareKeywords(
    ColorScheme* colorscheme,
    FractalSpec* f,
    FractalView* v);

int LBuf_ZBuf(
    FractalView& v,
    double* LBuf,
    unsigned char* line,
    long ii,
    long imax,
    long aa_line,
    int offs);

progtype prog;

time_t calc_time;

Quat* globalOrbit = NULL;


static char orbitComponent(double* a, const double* b, int i) {
    int orbIndex = static_cast<int>(clamp<double>(*b, 0, MAXITER));
    *a = GlobalOrbit[orbIndex + 1][i];
    return 0;
}

char orbite(double* a, const double* b) {
    return orbitComponent(a, b, 0);
}

char orbitj(double* a, const double* b) {
    return orbitComponent(a, b, 1);
}

char orbitk(double* a, const double* b) {
    return orbitComponent(a, b, 2);
}

char orbitl(double* a, const double* b) {
    return orbitComponent(a, b, 3);
}

int TranslateColorFormula(std::ostream& errorMsg, const char* colscheme) {
    size_t dummy;
    double ddummy;

    prog.reset();
    prog.declareFunction("orbite", orbite);
    prog.declareFunction("orbitj", orbitj);
    prog.declareFunction("orbitk", orbitk);
    prog.declareFunction("orbitl", orbitl);

    ddummy = 0;

    dummy = progtype::nullHandle; prog.setVariable("maxiter", &dummy, ddummy);
    dummy = progtype::nullHandle; prog.setVariable("lastiter", &dummy, ddummy);

    dummy = progtype::nullHandle; prog.setVariable("x", &dummy, ddummy);
    dummy = progtype::nullHandle; prog.setVariable("y", &dummy, ddummy);
    dummy = progtype::nullHandle; prog.setVariable("z", &dummy, ddummy);
    dummy = progtype::nullHandle; prog.setVariable("w", &dummy, ddummy);

    dummy = progtype::nullHandle; prog.setVariable("xb", &dummy, ddummy);
    dummy = progtype::nullHandle; prog.setVariable("yb", &dummy, ddummy);
    dummy = progtype::nullHandle; prog.setVariable("zb", &dummy, ddummy);
    dummy = progtype::nullHandle; prog.setVariable("wb", &dummy, ddummy);

    dummy = progtype::nullHandle; prog.setVariable("pi", &dummy, M_PI);

    return prog.compile(errorMsg, colscheme) != 0 ? -1 : 0;
}

int formatExternToIntern(FractalSpec& spec, FractalView& view) {
    /* This is to change the input format to internal format */
    /* Input: light relative to viewers position (for convenience of user) */
    /* Internal: light absolute in space (for convenience of programmer) */
    ViewBasis base;

    if (view.calcbase(&base, NULL, WhichEye::Monocular) != 0) {
        return -1;
    }
    Vec3 ltmp = view._light[0] * base._x + view._light[1] * base._y + view._light[2] * base._z;
    view._light = view._s + view._Mov[0] * base._x + view._Mov[1] * base._y;
    /* bailout */
    /* Input: real bailout value */
    /* Internal: square of bailout value (saves some time) */
    spec._bailout *= spec._bailout;
    return 0;
}

int formatInternToExtern(FractalSpec& frac, FractalView& view) {
    /* Reverse process of "formatExternToIntern". see above */
    ViewBasis base;

    if (view.calcbase(&base, NULL, WhichEye::Monocular) != 0) {
        return -1;
    }
    frac._bailout = sqrt(frac._bailout);
    Vec3 ltmp = view._light - view._s;
    view._light = Vec3(
        ltmp.dot(base._x) - view._Mov[0],
        ltmp.dot(base._y) - view._Mov[1],
        ltmp.dot(base._z));
    return 0;
}

int InitGraphics(
    std::ostream& errorMsg,
    FractalPreferences& fractal,
    bool ready,
    int* xadd, int* yadd,
    bool useZBuf) {
    /* ready ... 0: not ready */
    /* useZBuf ... 0: no zbuffer (reason: AA) */

    *yadd = 0; 
    *xadd = 0;
    int i = fractal.view()._xres;
    if (fractal.view().isStereo()) {
        i *= 2;
    }
    int j = fractal.view()._yres;
    if (useZBuf) {
        i *= fractal.view()._antialiasing;
        j *= fractal.view()._antialiasing;
    }

    if (Initialize != nullptr && Initialize(errorMsg, i, j) != 0) {
        return -1;
    }
    fractal.realPalette().computeWeightSum();

    return 0;
}

void allocBufs(
    FractalView& v,
    ZFlag zflag, 
    LexicallyScopedPtr<float>& cBuf,
    LexicallyScopedPtr<float>& bBuf,
    LexicallyScopedPtr<double>& lBufR,
    LexicallyScopedPtr<double>& lBufL,
    LexicallyScopedPtr<unsigned char>& line,
    LexicallyScopedPtr<unsigned char>& line2,
    LexicallyScopedPtr<unsigned char>& line3,
    size_t& lBufSize)
    /* zflag ... whether img (0), zbuf (1) or zbuf->img (2) is calculated */
    /* pal ... whether palette or not */
{
    unsigned int st = v.isStereo() ? 2 : 1;
    lBufSize = v._xres * v._antialiasing * (v._antialiasing + 1L) + 10;
    switch (zflag) {
    case ZFlag::NewImage:   /* create an image without ZBuffer */
        lBufR = new double[lBufSize]();
        line = new unsigned char[3 * v._xres * st + 1 + 10]();
        cBuf = new float[v._xres * st + 10]();
        bBuf = new float[v._xres * st + 10]();
        if (st == 2) {
            lBufL = new double[lBufSize + 10]();
        }
        break;
    case ZFlag::NewZBuffer:   /* create a ZBuffer only */
        /* lBufR and lBufL each hold aa lines (mono) */
        lBufR = new double[lBufSize + 10]();
        /* line only holds a single stereo line for transferring of
           lBuf->global ZBuffer */
        line = new unsigned char[3 * v._xres * v._antialiasing * st + 1 + 10]();
        if (st == 2) {
            lBufL = new double[lBufSize + 10]();
        }
        break;
    case ZFlag::ImageFromZBuffer:   /* create an image using a ZBuffer */
        lBufR = new double[lBufSize + 10]();
        line = new unsigned char[3 * v._xres * st + 1 + 10]();
        cBuf = new float[v._xres * st + 10]();
        bBuf = new float[v._xres * st + 10]();
        line3 = new unsigned char[3 * v._xres * v._antialiasing * st + 1 + 10]();
        if (st == 2) {
            lBufL = new double[lBufSize]();
        }
        break;
    }
}

int LBuf_ZBuf(
    FractalView& v,
    double* lBuf,
    unsigned char* line,
    long ii,
    long imax,
    long aa_line, 
    int offs) {
    int tmp;
    long l;

    for (int k = 0; k < v._antialiasing; k++) {
        for (int i = ii; i <= imax; i++) {
            tmp = i * v._antialiasing + k + offs;
            l = static_cast<long>(floor(lBuf[(i + aa_line * v._xres) * v._antialiasing + k] * 100 + 0.5));;
            line[tmp * 3 + 1] = (unsigned char)(l >> 16 & 0xFF);
            line[tmp * 3 + 2] = (unsigned char)(l >> 8 & 0xFF);
            line[tmp * 3 + 3] = (unsigned char)(l & 0xFF);
        }
    }
    return 0;
}

int CalculateFractal(
    std::ostream& errorMsg,
    char* pngfile,
    FILE** png,
    PNGFile* png_internal,
    ZFlag zflag,
    int* xstart,
    int* ystart,
    ViewBasis* rbase,
    ViewBasis* srbase,
    ViewBasis* lbase,
    ViewBasis* slbase,
    FractalPreferences& fractal,
    LinePutter& lineDst)
    /* pngfile: string of basename, without path (only for title bar) */
    /* png: _opened_ png filename */
    /* png filename returns _closed_ */
    /*      and initialized (png_info, png_internal) */
    /* if png==NULL, no filename is written */
    /* also closes graphics, resets keyboard mode */
    /* wants external format for light-source and bailout */
    /* zflag: decides what to calculate,
          0..image from scratch, size view.xres*view.yres
          1..ZBuffer from scratch, size view.xres*view.yres*AA^2
          2..image from ZBuffer, size img: xres*yres, buffer *AA^2
          for every case take into account that images can be stereo! */
{
    LexicallyScopedPtr<double> lBufR;
    LexicallyScopedPtr<double> lBufL;
    LexicallyScopedPtr<float> cBuf;
    LexicallyScopedPtr<float> bBuf;
    LexicallyScopedPtr<unsigned char> line;
    LexicallyScopedPtr<unsigned char> line2;
    LexicallyScopedPtr<unsigned char> line3;

    FractalSpec frac = fractal.fractal();
    FractalView view = fractal.view();
    if (formatExternToIntern(frac, view) != 0) {
        errorMsg << "Error in view struct!";
        return -1;
    }

    calc_struct cr(frac, view, fractal.cuts(), *rbase, *srbase, zflag);
    calc_struct cl(frac, view, fractal.cuts(), *lbase, *slbase, zflag);
    char MSG[300];


    long xres_st = view._xres;
    long xres_aa = view._xres * view._antialiasing;
    long xres_st_aa = xres_aa;
    if (view.isStereo()) {
        xres_st *= 2;
        xres_st_aa *= 2;
    }

    int pixelsPerCheck = haveGPU ? xres_aa : 16;

    size_t lBufSize;
    allocBufs(view, zflag, cBuf, bBuf, lBufR, lBufL, line, line2, line3, lBufSize);

    LexicallyScopedPtr<GPURowCalculator> rightRowCalculator = new GPURowCalculator(cr, lBufSize);
    LexicallyScopedPtr<GPURowCalculator> leftRowCalculator = new GPURowCalculator(cl, lBufSize);

    time_t my_time = time(NULL);

    /* recalculate last line when resuming an image without ZBuffer */
    bool firstLine = false;
    if (ZFlag::NewImage == zflag) {
        (*ystart)--;
        *xstart = 0;
        firstLine = true;
        Change_Name("recalc.");
    }
 
    for (int iy = *ystart; iy < view._yres; iy++) {
        /* Initialize variables for calcline (which change from line to line) */
        cr._xp = srbase->_O + iy * srbase->_y;
        cr._xp[3] = frac._lTerm;
        cr._xs[3] = frac._lTerm;
        memcpy(&lBufR[0], &lBufR[view._antialiasing * xres_aa], xres_aa * sizeof(lBufR[0]));
        if (view.isStereo()) {
            cl._xp = slbase->_O + iy * slbase->_y;
            cl._xp[3] = frac._lTerm;
            cl._xs[3] = frac._lTerm;
            memcpy(&lBufL[0], &lBufL[view._antialiasing * xres_aa], xres_aa * sizeof(lBufL[0]));
        }

        if (ZFlag::ImageFromZBuffer == zflag) {
            for (int ii = 0; ii < view._antialiasing + 1; ii++) {
                if (iy + ii > 0) {  /* this doesn´t work for the 1st line */
                    QU_getline(line3, iy * view._antialiasing + ii - 1, xres_st_aa, ZFlag::NewZBuffer);
                    for (int i = 0; i < xres_aa; i++) {
                        lBufR[i + ii * xres_aa] = static_cast<double>(threeBytesToLong(&line[i * 3])) / 100.0;
                        if (view.isStereo()) {
                            lBufL[i + ii * xres_aa] = static_cast<double>(threeBytesToLong(&line3[(i + xres_aa) * 3])) / 100.0;
                        }
                    }
                } else {
                    fillArray<double>(lBufR, static_cast<double>(view._zres), xres_aa);
                    if (view.isStereo()) {
                        fillArray<double>(lBufL, static_cast<double>(view._zres), xres_aa);
                    }
                }
            }
        }
        for (int ii = *xstart; ii < view._xres; ii += pixelsPerCheck) {
            long imax = ii + pixelsPerCheck - 1;
            imax = std::min<long>(imax, view._xres - 1);
            cr.calcline(*rightRowCalculator, ii, imax, iy, lBufR, bBuf, cBuf, zflag);  // right eye or monocular

            /* image to calculate */
            if (shouldCalculateImage(zflag) && !firstLine) {
                fractal.realPalette().pixelValue(ii, imax, 255, 255, 255,  &line[1], cBuf, bBuf);
            }
            if (view.isStereo()) {
                cl.calcline(*leftRowCalculator, ii, imax, iy, lBufL, &bBuf[view._xres], &cBuf[view._xres], zflag);   /* left eye image */
               /* image to calculate */
                if (shouldCalculateImage(zflag) && !firstLine) {
                    fractal.realPalette().pixelValue(
                        ii, imax, 255, 255, 255,
                        &line[3 * view._xres + 1],
                        &cBuf[view._xres],
                        &bBuf[view._xres]);
                }
            }
            /* Display and Transfer */
            if (ZFlag::NewZBuffer == zflag) {
                for (int kk = 1; kk < view._antialiasing + 1; kk++) {
                    LBuf_ZBuf(view, lBufR, line, ii, imax, kk, 0);
                    lineDst.putLine(ii * view._antialiasing,
                        (imax + 1) * view._antialiasing - 1,
                        xres_st_aa,
                        iy * view._antialiasing + kk - 1,
                        line + 1,
                        true);
                }
                if (view.isStereo()) {
                    for (int kk = 1; kk < view._antialiasing + 1; kk++) {
                        LBuf_ZBuf(view, lBufL, line, ii, imax, kk, xres_aa);
                        lineDst.putLine(ii * view._antialiasing + xres_aa,
                            (imax + 1) * view._antialiasing - 1 + xres_aa,
                            xres_st_aa,
                            iy * view._antialiasing + kk - 1,
                            line + 1,
                            true);
                    }
                }
            } else if (!firstLine) {   /* the image */
                lineDst.putLine(ii, imax, xres_st, iy, line + 1, false);
                if (view.isStereo()) {
                    lineDst.putLine(ii + view._xres, imax + view._xres, xres_st, iy, line + 1, false);
                }
            }
            if ((imax == view._xres - 1) && !firstLine) {
                eol(iy + 1);
            }
            int rrv = check_event();
            if (rrv != 0) {
                int i = view._xres * 3;
                if (view.isStereo()) {
                    i *= 2;
                }
                memset(line, 0, i + 1);
                calc_time += time(NULL) - my_time;
                if (png != NULL) {
                    PNGEnd(*png_internal, line, 0, iy);
                    fclose(*png);
                }
                *ystart = iy;
                *xstart = ii + 1;
                if (*xstart == view._xres) {
                    *xstart = 0;
                    (*ystart)++;
                }
                Done();
                return rrv;
            }
        }
        *xstart = 0;
        if (NULL != png) {
            switch (zflag) {
            case ZFlag::NewImage:
                if (!firstLine) {
                    line[0] = '\0';       /* Set filter method */
                    png_internal->doFiltering(line);
                    png_internal->writePNGLine(line);
                }
                break;
            case ZFlag::NewZBuffer:
                if (!firstLine) {
                    for (int kk = 1; kk < view._antialiasing + 1; kk++) {
                        LBuf_ZBuf(view, lBufR, line, 0, view._xres - 1, kk, 0);
                        if (view.isStereo()) {
                            LBuf_ZBuf(view, lBufL, line, 0, view._xres - 1, kk, xres_aa);
                        }
                    }
                    line[0] = '\0';
                    png_internal->doFiltering(line);
                    png_internal->writePNGLine(line);
                }
                break;
            case ZFlag::ImageFromZBuffer:
                line[0] = '\0';      /* Set filter method */
                png_internal->doFiltering(line);
                png_internal->writePNGLine(line);
            }
        }
        firstLine = false;
    }

    calc_time += time(NULL) - my_time;
    if (png != NULL) {
        PNGEnd(*png_internal, line, 0, view._yres);
        fclose(*png);
    }

    if (pngfile != NULL) {
        sprintf_s(MSG, sizeof(MSG), "%s", pngfile);
        Change_Name(MSG);
    }
    if (Done != NULL) {
        Done();
    }
    *xstart = 0;
    *ystart = view._yres;
    return 0;
}

int CreateImage(
    std::ostream& errorMsg,
    int* xstart, int* ystart, 
    FractalPreferences& fractal,
    ZFlag zflag,
    LinePutter& lineDst)
    /* Creates/Continues image from given parameters.
       Wants external format of frac & view */
{
    ViewBasis rbase, srbase, lbase, slbase, cbase;
    int xadd, yadd;
    std::stringstream nestedErrorMsg;

    if (TranslateColorFormula(nestedErrorMsg, fractal.colorScheme().get()) != 0) {
        errorMsg << "Error in color scheme formula:" << std::endl;
        errorMsg << nestedErrorMsg.str();
        return -1;
    }
    if (fractal.view().calcbase(&cbase, &srbase, WhichEye::Monocular) != 0) {
        errorMsg << "Error in FractalView!" << std::endl;
        return -1;
    }

    rbase = cbase;

    if (fractal.view().isStereo()) {
        if (fractal.view().calcbase(&rbase, &srbase, WhichEye::Right) != 0) {
            errorMsg << "Error in FractalView (right eye)!" << std::endl;
            return -1;
        }
        if (fractal.view().calcbase(&lbase, &slbase, WhichEye::Left) != 0) {
            errorMsg << "Error in FractalView (left eye)!" << std::endl;
            return -1;
        }
    }

    if (InitGraphics(errorMsg, fractal, false, &xadd, &yadd, zflag == ZFlag::NewZBuffer) != 0) {
        return -1;
    }

    if (0 == ystart && 0 == xstart) {
        calc_time = 0;
    }
    return CalculateFractal(
        errorMsg, 
        NULL, NULL, NULL,
        zflag,
        xstart, ystart,
        &rbase, &srbase, &lbase, &slbase, 
        fractal,
        lineDst);
}

int CreateZBuf(std::ostream& errorMsg, int* xstart, int* ystart, FractalPreferences& fractal, LinePutter& lineDst)
    /* Creates/Continues ZBuffer from given parameters. Wants external format of frac & view */
{
    ViewBasis rbase, srbase, lbase, slbase, cbase;
    RealPalette realpal;
    int xadd, yadd;
    int i;

    /* Set a grayscale palette */
    realpal._nColors = 1;
    realpal._cols[0].weight = 1;
    for (i = 0; i < 3; i++) {
        realpal._cols[0].col1[i] = 1;
        realpal._cols[0].col2[i] = 1;
    }

    if (fractal.view().calcbase(&cbase, &srbase, WhichEye::Monocular) != 0) {
        errorMsg << "Error in FractalView!" << std::endl;
        return -1;
    }

    rbase = cbase;

    if (fractal.view().isStereo()) {
        if (fractal.view().calcbase(&rbase, &srbase, WhichEye::Right) != 0) {
            errorMsg << "Error in FractalView (right eye)!" << std::endl;
            return -1;
        }
        if (fractal.view().calcbase(&lbase, &slbase, WhichEye::Left) != 0) {
            errorMsg << "Error in FractalView (left eye)!" << std::endl;
            return -1;
        }
    }
    
    if (InitGraphics(errorMsg, fractal, false, &xadd, &yadd, true) != 0) {
        return -1;
    }

    i = CalculateFractal(errorMsg, NULL, NULL, NULL,
        ZFlag::NewZBuffer, xstart, ystart, &rbase, &srbase, &lbase,
        &slbase, fractal, lineDst);

    return i;
}
