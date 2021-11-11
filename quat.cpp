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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <algorithm>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES // To get M_PI & c.
#include <math.h>  /* fmod */
#include <time.h> /* time */
#include <ctype.h> /* tolower */
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

struct progtype prog;

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

int TranslateColorFormula(const char* colscheme, char* ErrorMSG, size_t maxErrorLen) {

    unsigned char dummy;
    double ddummy;

    InitProg(&prog);
    DeclareFunction("orbite", orbite, &prog);
    DeclareFunction("orbitj", orbitj, &prog);
    DeclareFunction("orbitk", orbitk, &prog);
    DeclareFunction("orbitl", orbitl, &prog);

    ddummy = 0;

    dummy = 255; SetVariable("maxiter", &dummy, ddummy, &prog);
    dummy = 255; SetVariable("lastiter", &dummy, ddummy, &prog);

    dummy = 255; SetVariable("x", &dummy, ddummy, &prog);
    dummy = 255; SetVariable("y", &dummy, ddummy, &prog);
    dummy = 255; SetVariable("z", &dummy, ddummy, &prog);
    dummy = 255; SetVariable("w", &dummy, ddummy, &prog);

    dummy = 255; SetVariable("xb", &dummy, ddummy, &prog);
    dummy = 255; SetVariable("yb", &dummy, ddummy, &prog);
    dummy = 255; SetVariable("zb", &dummy, ddummy, &prog);
    dummy = 255; SetVariable("wb", &dummy, ddummy, &prog);

    /*  SetVariable("diffr", &dummy, ddummy, &prog); */
    /*  SetVariable("eorbit", &dummy, ddummy, &prog);
    SetVariable("jorbit", &dummy, ddummy, &prog);
    SetVariable("korbit", &dummy, ddummy, &prog);
    SetVariable("lorbit", &dummy, ddummy, &prog);
    SetVariable("closestit", &dummy, ddummy, &prog); */

    dummy = 255; SetVariable("pi", &dummy, M_PI, &prog);

    return (Translate(ErrorMSG, maxErrorLen, colscheme, &prog) != 0) ? -1 : 0;
}

int FormatExternToIntern(FractalSpec& spec, FractalView& view) {
    /* This is to change the input format to internal format */
    /* Input: light relative to viewers position (for convenience of user) */
    /* Internal: light absolute in space (for convenience of programmer) */
    base_struct base;

    if (view.calcbase(&base, NULL, WhichEye::Monocular) != 0) {
        return -1;
    }
    vec3 ltmp = view._light[0] * base._x + view._light[1] * base._y + view._light[2] * base._z;
    view._light = view._s + view._Mov[0] * base._x + view._Mov[1] * base._y;
    /* bailout */
    /* Input: real bailout value */
    /* Internal: square of bailout value (saves some time) */
    spec._bailout *= spec._bailout;
    return 0;
}

int FormatInternToExtern(FractalSpec& frac, FractalView& view) {
    /* Reverse process of "FormatExternToIntern". see above */
    base_struct base;

    if (view.calcbase(&base, NULL, WhichEye::Monocular) != 0) {
        return -1;
    }
    frac._bailout = sqrt(frac._bailout);
    vec3 ltmp = view._light - view._s;
    view._light = vec3(
        ltmp.dot(base._x) - view._Mov[0],
        ltmp.dot(base._y) - view._Mov[1],
        ltmp.dot(base._z));
    return 0;
}

int InitGraphics(
    char* Error,
    size_t maxErrorLen,
    FractalPreferences& fractal,
    bool ready,
    vidinfo_struct* vidinfo,
    disppal_struct* disppal,
    int* xadd, int* yadd,
    bool useZBuf) {
    /* ready ... 0: not ready */
    /* zbufflag ... 0: no zbuffer (reason: AA) */
    int i, j;

    /* Do graphics initialization & return mode info (in vidinfo) in different versions */
    *yadd = 0; *xadd = 0;
    i = fractal.view()._xres;
    if (fractal.view().isStereo()) {
        i *= 2;
    }
    j = fractal.view()._yres;
    if (useZBuf) {
        i *= fractal.view()._antialiasing;
        j *= fractal.view()._antialiasing;
    }

    if (Initialize != NULL && Initialize(i, j, Error) != 0) {
        return -1;
    }
    if (ReturnVideoInfo != NULL) {
        ReturnVideoInfo(vidinfo);
    }
    if (vidinfo->maxcol != -1) {
        int r = 0;
        disppal->btoc = 1;
        if (!useZBuf) {
            r = CreateDispPal(disppal,
                fractal.realPalette(),
                vidinfo->maxcol,
                fractal.view()._phongmax,
                vidinfo->rdepth,
                vidinfo->gdepth,
                vidinfo->bdepth);
        } else {
            RealPalette rp;
            rp._nColors = 1;
            rp._cols[0].weight = 1;
            for (int i = 0; i < 3; i++) {
                rp._cols[0].col1[i] = 1;
                rp._cols[0].col1[i] = 1;
            }
            r = CreateDispPal(disppal,
                rp,
                vidinfo->maxcol,
                fractal.view()._phongmax,
                vidinfo->rdepth,
                vidinfo->gdepth,
                vidinfo->bdepth);
        }
        if (r == -1) {
            sprintf_s(Error, maxErrorLen,
                "Too many colours for this display (%i).\nUse text only version.\n",
                vidinfo->maxcol);
            return -1;
        }
        if (SetColors != NULL) {
            SetColors(disppal);
        }
    } else {
        CalcWeightsum(fractal.realPalette());
    }

    return 0;
}

void AllocBufs(
    FractalView& v,
    ZFlag zflag, 
    ColorMode colorMode,
    LexicallyScopedPtr<float>& CBuf,
    LexicallyScopedPtr<float>& BBuf,
    LexicallyScopedPtr<double>& LBufR,
    LexicallyScopedPtr<double>& LBufL,
    LexicallyScopedPtr<unsigned char>& line,
    LexicallyScopedPtr<unsigned char>& line2,
    LexicallyScopedPtr<unsigned char>& line3)
    /* zflag ... whether img (0), zbuf (1) or zbuf->img (2) is calculated */
    /* pal ... whether palette or not */
{
    unsigned int st = v.isStereo() ? 2 : 1;
    size_t LBufSize = v._xres * v._antialiasing * (v._antialiasing + 1L) + 10;
    switch (zflag) {
    case ZFlag::NewImage:   /* create an image without ZBuffer */
        LBufR = new double[LBufSize]();
        line = new unsigned char[3 * v._xres * st + 1 + 10]();
        if (colorMode == ColorMode::Indexed) {
            line2 = new unsigned char[v._xres * st + 10]();
        }
        CBuf = new float[v._xres * st + 10]();
        BBuf = new float[v._xres * st + 10]();
        if (st == 2) {
            LBufL = new double[LBufSize + 10]();
        }
        break;
    case ZFlag::NewZBuffer:   /* create a ZBuffer only */
        /* LBufR and LBufL each hold aa lines (mono) */
        LBufR = new double[LBufSize + 10]();
        /* line only holds a single stereo line for transferring of
           LBuf->global ZBuffer */
        line = new unsigned char[3 * v._xres * v._antialiasing * st + 1 + 10]();
        if (st == 2) {
            LBufL = new double[LBufSize + 10]();
        }
        break;
    case ZFlag::ImageFromZBuffer:   /* create an image using a ZBuffer */
        LBufR = new double[LBufSize + 10]();
        line = new unsigned char[3 * v._xres * st + 1 + 10]();
        if (colorMode == ColorMode::Indexed) {
            line2 = new unsigned char[3 * v._xres * st + 1 + 10]();
        }
        CBuf = new float[v._xres * st + 10]();
        BBuf = new float[v._xres * st + 10]();
        line3 = new unsigned char[3 * v._xres * v._antialiasing * st + 1 + 10]();
        if (st == 2) {
            LBufL = new double[LBufSize]();
        }
        break;
    }
}

int LBuf_ZBuf(
    FractalView& v,
    double* LBuf,
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
            l = static_cast<long>(floor(LBuf[(i + aa_line * v._xres) * v._antialiasing + k] * 100 + 0.5));;
            line[tmp * 3 + 1] = (unsigned char)(l >> 16 & 0xFF);
            line[tmp * 3 + 2] = (unsigned char)(l >> 8 & 0xFF);
            line[tmp * 3 + 3] = (unsigned char)(l & 0xFF);
        }
    }
    return 0;
}

int CalculateFractal(
    char* Error,
    size_t maxErrorLen,
    char* pngfile,
    FILE** png,
    struct png_internal_struct* png_internal,
    ZFlag zflag,
    int* xstart,
    int* ystart,
    int pixelsPerCheck,
    base_struct* rbase,
    base_struct* srbase,
    base_struct* lbase,
    base_struct* slbase,
    FractalPreferences& fractal,
    vidinfo_struct* vidinfo,
    disppal_struct* disppal,
    LinePutter& lineDst)
    /* pngfile: string of filename, without path (only for title bar) */
    /* png: _opened_ png file */
    /* png file returns _closed_ */
    /*      and initialized (png_info, png_internal) */
    /* if png==NULL, no file is written */
    /* also closes graphics, resets keyboard mode */
    /* wants external format for light-source and bailout */
    /* zflag: decides what to calculate,
          0..image from scratch, size view.xres*view.yres
          1..ZBuffer from scratch, size view.xres*view.yres*AA^2
          2..image from ZBuffer, size img: xres*yres, buffer *AA^2
          for every case take into account that images can be stereo! */
{
    //long i;
    long xres_st, xres_aa, xres_st_aa;
    LexicallyScopedPtr<double> LBufR;
    LexicallyScopedPtr<double> LBufL;
    LexicallyScopedPtr<float> CBuf;
    LexicallyScopedPtr<float> BBuf;
    LexicallyScopedPtr<unsigned char> line;
    LexicallyScopedPtr<unsigned char> line2;
    LexicallyScopedPtr<unsigned char> line3;
    calc_struct cr, cl;
    time_t my_time;
    char MSG[300];
    long ii, kk;
    FractalSpec frac;
    FractalView view;
    int rrv = 0;

    frac = fractal.fractal(); 
    view = fractal.view();

    xres_st = view._xres;
    xres_aa = view._xres * view._antialiasing;
    xres_st_aa = xres_aa;
    if (view.isStereo()) {
        xres_st *= 2;
        xres_st_aa *= 2;
    }

    if (FormatExternToIntern(frac, view) != 0) {
        sprintf_s(Error, maxErrorLen, "Error in view struct!");
        return -1;
    }

    if (nullptr != GlobalOrbit) {
        delete[] GlobalOrbit;
    }
    GlobalOrbit = new Quat[frac._maxiter + 2]();
    /* "+2", because orbit[0] is special flag for orbite,j,k,l */
    AllocBufs(
        view, zflag, (vidinfo->maxcol != -1) ? ColorMode::Indexed : ColorMode::RGB,
        CBuf, BBuf, LBufR, LBufL,
        line, line2, line3);

    my_time = time(NULL);

    switch (frac._formula) {
    case 0:
        cr.iterate_no_orbit = iterate_0_no_orbit; cr.iterate = iterate_0; cr.iternorm = iternorm_0;
        cl.iterate_no_orbit = iterate_0_no_orbit; cl.iterate = iterate_0; cl.iternorm = iternorm_0;
        break;
    case 1:
        cr.iterate_no_orbit = iterate_1_no_orbit; cr.iterate = iterate_1; cr.iternorm = iternorm_1;
        cl.iterate_no_orbit = iterate_1_no_orbit; cl.iterate = iterate_1; cl.iternorm = iternorm_1;
        break;
    case 2:
        cr.iterate_no_orbit = cr.iterate = iterate_2; cr.iternorm = 0;
        cl.iterate_no_orbit = cl.iterate = iterate_2; cl.iternorm = 0;
        break;
    case 3:
        cr.iterate_no_orbit = iterate_3_no_orbit; cr.iterate = iterate_3; cr.iternorm = 0;
        cl.iterate_no_orbit = iterate_3_no_orbit; cl.iterate = iterate_3; cl.iternorm = 0;
        break;
    case 4:
        cr.iterate_no_orbit = cr.iterate = iterate_4; cr.iternorm = 0;
        cl.iterate_no_orbit = cl.iterate = iterate_4; cl.iternorm = 0;
        break;
    case 5:
        cr.iterate_no_orbit = cr.iterate = iterate_bulb; cr.iternorm = 0;
        cl.iterate_no_orbit = cl.iterate = iterate_bulb; cl.iternorm = 0;
        break;
    default:
        assert(1 == 0);
    }

    /* Initialize variables for calcline (those which don´t change from
       line to line) */
    cr.v = view; 
    cr.f = frac;
    cr.sbase = *srbase; 
    cr.base = *rbase;
    cr.cuts = fractal.cuts();
    cr.absx = cr.sbase._x.magnitude() / cr.v._antialiasing;
    cr.absy = cr.sbase._y.magnitude() / cr.v._antialiasing;
    cr.absz = cr.sbase._z.magnitude();
    cr.aabase = cr.sbase;

    if (shouldCalculateDepths(zflag) && cr.v._antialiasing > 1) {
        cr.aabase._x /= cr.v._antialiasing;
        cr.aabase._y /= cr.v._antialiasing;
    }

    if (view.isStereo()) {
        cl.v = view; cl.f = frac;
        cl.sbase = *slbase;
        cl.base = *lbase;
        cl.cuts = fractal.cuts();
        cl.absx = cl.sbase._x.magnitude() / cl.v._antialiasing;
        cl.absy = cl.sbase._y.magnitude() / cl.v._antialiasing;
        cl.absz = cl.sbase._z.magnitude();
        cl.aabase = cl.sbase;
        if (shouldCalculateDepths(zflag) && cl.v._antialiasing > 1) {
            cl.aabase._x /= cl.v._antialiasing;
            cl.aabase._y /= cl.v._antialiasing;
        }
    }

    /* recalculate last line when resuming an image without ZBuffer */
    bool firstline = false;
    if (zflag == ZFlag::NewImage) {
        (*ystart)--;
        *xstart = 0;
        firstline = true;
        Change_Name("recalc.");
    }
    int j;
    for (j = *ystart; j < view._yres; j++) {
        /* Initialize variables for calcline (which change from line to line) */
        cr.xp = srbase->_O + j * srbase->_y;
        cr.xp[3] = frac._lvalue;
        cr.xs[3] = frac._lvalue;
        if (view.isStereo()) {
            cl.xp = slbase->_O + j * slbase->_y;
            cl.xp[3] = frac._lvalue;
            cl.xs[3] = frac._lvalue;
        }

        /* Initialize LBufs from ZBuffer */
        memcpy(&LBufR[0], &LBufR[view._antialiasing * xres_aa], xres_aa * sizeof(LBufR[0]));
        if (view.isStereo()) {
            memcpy(&LBufL[0], &LBufL[view._antialiasing * xres_aa], xres_aa * sizeof(LBufL[0]));
        }

        if (zflag == ZFlag::ImageFromZBuffer) {
            for (ii = 0; ii < view._antialiasing + 1; ii++) {
                if (j + ii > 0) {  /* this doesn´t work for the 1st line */
                    QU_getline(line3, j * view._antialiasing + ii - 1, xres_st_aa, ZFlag::NewZBuffer);
                    for (int i = 0; i < xres_aa; i++) {
                        LBufR[i + ii * xres_aa] = static_cast<double>(threeBytesToLong(&line[i * 3])) / 100.0;
                        if (view.isStereo()) {
                            LBufL[i + ii * xres_aa] = static_cast<double>(threeBytesToLong(&line3[(i + xres_aa) * 3])) / 100.0;
                        }
                    }
                } else {
                    fillArray<double>(LBufR, xres_aa, static_cast<double>(view._zres));
                    if (view.isStereo()) {
                        fillArray<double>(LBufL, xres_aa, static_cast<double>(view._zres));
                    }
                }
            }
        }
        for (ii = *xstart; ii < view._xres; ii += pixelsPerCheck) {
            long imax = ii + pixelsPerCheck - 1;
            if (imax > view._xres - 1) {
                imax = view._xres - 1;
            }

            cr.calcline(ii, imax, j, LBufR, BBuf, CBuf, zflag);  // right eye or monocular

            /* image to calculate */
            if (shouldCalculateImage(zflag) && !firstline) {
                if (vidinfo->maxcol != -1) {
                    PixelvaluePalMode(ii, imax, disppal->_nColors - 1,
                        disppal->brightnum - 1, line2, CBuf, BBuf);
                }
                PixelvalueTrueMode(ii, imax, 255, 255, 255,
                    fractal.realPalette(), &(line[1]), CBuf, BBuf);
            }
            if (view.isStereo()) {
                cl.calcline(ii, imax, j, LBufL, &BBuf[view._xres], &CBuf[view._xres], zflag);   /* left eye image */
               /* image to calculate */
                if (shouldCalculateImage(zflag) && !firstline) {
                    if (vidinfo->maxcol != -1) {
                        PixelvaluePalMode(ii, imax, disppal->_nColors - 1,
                            disppal->brightnum - 1,
                            &line2[view._xres],
                            &CBuf[view._xres], &BBuf[view._xres]);
                    }
                    PixelvalueTrueMode(ii, imax, 255, 255, 255,
                        fractal.realPalette(), &(line[3 * view._xres + 1]),
                        &CBuf[view._xres], &BBuf[view._xres]);
                }
            }
            /* Display and Transfer */
            if (zflag == ZFlag::NewZBuffer) {  /* the ZBuffer */
                for (kk = 1; kk < view._antialiasing + 1; kk++) {
                    LBuf_ZBuf(view, LBufR, line, ii, imax, kk, 0);
                    lineDst.putLine(ii * view._antialiasing,
                        (imax + 1) * view._antialiasing - 1,
                        xres_st_aa,
                        j * view._antialiasing + kk - 1,
                        line + 1,
                        true);
                    update_bitmap(ii * view._antialiasing,
                        (imax + 1) * view._antialiasing - 1,
                        xres_st_aa,
                        j * view._antialiasing + kk - 1,
                        line + 1,
                        view._zres);
                }
                if (view.isStereo()) {
                    for (kk = 1; kk < view._antialiasing + 1; kk++) {
                        LBuf_ZBuf(view, LBufL, line, ii, imax, kk, xres_aa);
                        lineDst.putLine(ii * view._antialiasing + xres_aa,
                            (imax + 1) * view._antialiasing - 1 + xres_aa,
                            xres_st_aa,
                            j * view._antialiasing + kk - 1,
                            line + 1,
                            true);
                        update_bitmap(ii * view._antialiasing + xres_aa,
                            (imax + 1) * view._antialiasing - 1 + xres_aa,
                            xres_st_aa,
                            j * view._antialiasing + kk - 1,
                            line + 1,
                            view._zres);
                    }
                }
            } else if (!firstline) {   /* the image */
                lineDst.putLine(ii, imax, xres_st, j, line + 1, false);
                if (view.isStereo()) {
                    lineDst.putLine(ii + view._xres, imax + view._xres, xres_st,
                        j, line + 1, false);
                }
                update_bitmap(ii, imax, xres_st, j, line + 1, 0);
                if (view.isStereo()) {
                    update_bitmap(ii + view._xres, imax + view._xres, xres_st,
                        j, &(line[1]),
                        0);
                }
            }
            if ((imax == view._xres - 1) && !firstline) {
                eol(j + 1);
            }
            rrv = check_event();
            if (rrv != 0) {
                int i = view._xres * 3;
                if (view.isStereo()) {
                    i *= 2;
                }
                memset(line, 0, i + 1);
                calc_time += time(NULL) - my_time;
                if (png != NULL) {
                    PNGEnd(png_internal, line, 0, j);
                    fclose(*png);
                    EndPNG(png_internal);
                }
                *ystart = j;
                *xstart = ii + 1;
                if (*xstart == view._xres) {
                    *xstart = 0; (*ystart)++;
                }
                Done();
                /* return code from check event.
                   (win destroyed / calc stopped) */
                return rrv;
            }
        }
        *xstart = 0;
        /* Write to PNG file */
        if (NULL != png) {
            switch (zflag) {
            case ZFlag::NewImage:
                if (!firstline) {
                    line[0] = '\0';       /* Set filter method */
                    DoFiltering(png_internal, line);
                    WritePNGLine(png_internal, line);
                }
                break;
            case ZFlag::NewZBuffer:
                if (!firstline) {
                    for (kk = 1; kk < view._antialiasing + 1; kk++) {
                        LBuf_ZBuf(view, LBufR, line, 0, view._xres - 1, kk, 0);
                        if (view.isStereo()) {
                            LBuf_ZBuf(view, LBufL, line, 0, view._xres - 1, kk, xres_aa);
                        }
                    }
                    line[0] = '\0';
                    DoFiltering(png_internal, line);
                    WritePNGLine(png_internal, line);
                }
                break;
            case ZFlag::ImageFromZBuffer:
                line[0] = '\0';      /* Set filter method */
                DoFiltering(png_internal, line);
                WritePNGLine(png_internal, line);
            }
        }
        firstline = false;
    }

    calc_time += time(NULL) - my_time;
    if (png != NULL) {
        PNGEnd(png_internal, line, 0, j);
        fclose(*png);
        EndPNG(png_internal);
    }

    if (pngfile != NULL) {
        sprintf_s(MSG, sizeof(MSG), "%s", pngfile);
        Change_Name(MSG);
    }
    if (Done != NULL) {
        Done();
    }
    *xstart = 0;
    *ystart = j;
    return 0;
}

int CreateImage(
    char* Error,
    size_t maxErrorLen,
    int* xstart, int* ystart, 
    FractalPreferences& fractal,
    int pixelsPerCheck,
    ZFlag zflag,
    LinePutter& lineDst)
    /* Creates/Continues image from given parameters.
       Wants external format of frac & view */
{
    base_struct rbase, srbase, lbase, slbase, cbase;
    vidinfo_struct vidinfo;
    disppal_struct disppal;
    int xadd, yadd;
    char ErrorMSG[256];

    Error[0] = '\0';
    if (TranslateColorFormula(fractal.colorScheme().get(), ErrorMSG, sizeof(ErrorMSG)) != 0) {
        sprintf_s(Error, maxErrorLen, "Error in color scheme formula:\n%s\n", ErrorMSG);
        return -1;
    }
    if (fractal.view().calcbase(&cbase, &srbase, WhichEye::Monocular) != 0) {
        sprintf_s(Error, maxErrorLen, "Error in FractalView!\n");
        return -1;
    }

    rbase = cbase;

    if (fractal.view().isStereo()) {
        if (fractal.view().calcbase(&rbase, &srbase, WhichEye::Right) != 0) {
            sprintf_s(Error, maxErrorLen, "Error in FractalView (right eye)!\n");
            return -1;
        }
        if (fractal.view().calcbase(&lbase, &slbase, WhichEye::Left) != 0) {
            sprintf_s(Error, maxErrorLen, "Error in FractalView (left eye)!\n");
            return -1;
        }
    }

    /* MSWIN_Initialize will recognize if it is already initialized.
       Needing vidinfo, disppal...
    */
    if (InitGraphics(Error, maxErrorLen, fractal, false, &vidinfo, &disppal,
        &xadd, &yadd, zflag == ZFlag::NewZBuffer) != 0) {
        return -1;
    }

    if (ystart == 0 && xstart == 0) {
        calc_time = 0;
    }
    return CalculateFractal(
        Error, maxErrorLen, 
        NULL, NULL, NULL,
        zflag,
        xstart, ystart,
        pixelsPerCheck,
        &rbase, &srbase, &lbase, &slbase, 
        fractal,
        &vidinfo, &disppal,
        lineDst);
}

int CreateZBuf(char* Error, size_t maxErrorLen, int* xstart, int* ystart, FractalPreferences& fractal, int pixelsPerCheck, LinePutter& lineDst)
    /* Creates/Continues ZBuffer from given parameters. Wants external format of frac & view */
{
    base_struct rbase, srbase, lbase, slbase, cbase;
    RealPalette realpal;
    vidinfo_struct vidinfo;
    disppal_struct disppal;
    int xadd, yadd;
    int i;

    /* Set a grayscale palette */
    realpal._nColors = 1;
    realpal._cols[0].weight = 1;
    for (i = 0; i < 3; i++) {
        realpal._cols[0].col1[i] = 1;
        realpal._cols[0].col2[i] = 1;
    }

    Error[0] = '\0';
    if (fractal.view().calcbase(&cbase, &srbase, WhichEye::Monocular) != 0) {
        sprintf_s(Error, maxErrorLen, "Error in FractalView!\n");
        return -1;
    }

    rbase = cbase;

    if (fractal.view().isStereo()) {
        if (fractal.view().calcbase(&rbase, &srbase, WhichEye::Right) != 0) {
            sprintf_s(Error, maxErrorLen, "Error in FractalView (right eye)!\n");
            return -1;
        }
        if (fractal.view().calcbase(&lbase, &slbase, WhichEye::Left) != 0) {
            sprintf_s(Error, maxErrorLen, "Error in FractalView (left eye)!\n");
            return -1;
        }
    }
    /* MSWIN_Initialize will recognize if it is already initialized.
       Needing vidinfo, disppal...
    */
    if (InitGraphics(Error, maxErrorLen, fractal, false, &vidinfo, &disppal,
        &xadd, &yadd, true) != 0) {
        return -1;
    }

    i = CalculateFractal(Error, maxErrorLen, NULL, NULL, NULL,
        ZFlag::NewZBuffer, xstart, ystart, pixelsPerCheck, &rbase, &srbase, &lbase,
        &slbase, fractal, &vidinfo, &disppal, lineDst);

    return i;
}


