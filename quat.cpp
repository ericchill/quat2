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
#include <ctype.h>
#include "common.h"
#include "quat.h"
#include "iter.h"
#include "files.h"
#include "colors.h" 
#include "memory.h"
#include "kernel.h"
#include "LineCalculator.h"
#include "ComputeWorker.h"
#include "ExprParse.h"
#include "ExprEval.h"
#include <chrono>
#include <functional>
#include <thread>


constexpr size_t nThreads = 4;

Expression* TranslateColorFormula(std::ostream& errorMsg, const char* colscheme) {
    try {
        return ExprCompiler::translate(colscheme);
    } catch (ParseException& ex) {
        errorMsg << ex.what();
        return nullptr;
    }
}

void InitGraphics(
    Quater& quatDriver,
    std::ostream& errorMsg,
    FractalPreferences& fractal,
    bool useZBuf) {
    /* ready ... 0: not ready */
    /* useZBuf ... 0: no zbuffer (reason: AA) */

    int i = fractal.view()._xres;
    if (fractal.view().isStereo()) {
        i *= 2;
    }
    int j = fractal.view()._yres;
    if (useZBuf) {
        i *= fractal.view()._antialiasing;
        j *= fractal.view()._antialiasing;
    }
    quatDriver.initialize(errorMsg, i, j);
    fractal.realPalette().computeWeightSum();
}


int LBuf_ZBuf(
    FractalView& v,
    float* lBuf,
    uint8_t* line,
    int aa_line, 
    int offs) {

    for (int k = 0; k < v._antialiasing; k++) {
        for (int i = 0; i < v._xres; i++) {
            int tmp = i * v._antialiasing + k + offs;
            int l = static_cast<int>(floor(lBuf[(i + aa_line * v._xres) * v._antialiasing + k] * 100 + 0.5));;
            line[tmp * 3 + 1] = static_cast<uint8_t>(l >> 16 & 0xFF);
            line[tmp * 3 + 2] = static_cast<uint8_t>(l >> 8 & 0xFF);
            line[tmp * 3 + 3] = static_cast<uint8_t>(l & 0xFF);
        }
    }
    return 0;
}


int CalculateFractal(
    Quater& quatDriver,
    std::ostream& errorMsg,
    char* pngfile,
    FILE** png,
    PNGFile* png_internal,
    ZFlag zFlag,
    int* ystart,
    ViewBasis& rbase,
    ViewBasis& srbase,
    ViewBasis& lbase,
    ViewBasis& slbase,
    FractalPreferences& fractal)
    /* pngfile: string of basename, without path (only for title bar) */
    /* png: _opened_ png filename */
    /* png filename returns _closed_ */
    /*      and initialized (png_info, png_internal) */
    /* if png==NULL, no filename is written */
    /* also closes graphics, resets keyboard mode */
    /* wants external format for light-source and bailout */
    /* zFlag: decides what to calculate,
          0..image from scratch, size view.xres*view.yres
          1..ZBuffer from scratch, size view.xres*view.yres*AA^2
          2..image from ZBuffer, size img: xres*yres, buffer *AA^2
          for every case take into account that images can be stereo! */
{
    std::stringstream nestedErrorMsg;
    LexicallyScopedPtr<Expression> colorExpr = TranslateColorFormula(nestedErrorMsg, fractal.colorScheme().get());
    if (nullptr == colorExpr) {
        errorMsg << "Error in color scheme formula:" << std::endl;
        errorMsg << nestedErrorMsg.str();
        return -1;
    }
    FractalLineWorker workers[nThreads];
    for (int i = 0; i < nThreads; i++) {
        workers[i] = FractalLineWorker(quatDriver, fractal, colorExpr, zFlag, rbase, srbase, lbase, slbase);
    }

    /* recalculate last line when resuming an image without ZBuffer */
    bool firstLine = false;
    if (ZFlag::NewImage == zFlag) {
        if (*ystart > 0) {
            (*ystart)--;
        }
        firstLine = true;
        quatDriver.changeName("recalc.");
    }
 
    for (int iy = *ystart; iy < fractal.view()._yres; iy += nThreads) {

        if (ZFlag::ImageFromZBuffer == zFlag) {
            for (int i = 0; i < nThreads; i++) {
                workers[i].readZBuffer(iy + i);
            }
        }
        std::thread threads[nThreads];
        for (int i = 0; i < nThreads; i++) {
            threads[i] = std::thread([&workers, i, iy] { workers[i].calcLine(iy + i); });
        }
        int nRunning = nThreads;
        while (nRunning > 0) {
            for (int i = 0; i < nThreads; i++) {
                if (threads[i].joinable() && !workers[i].running()) {
                    threads[i].join();
                    --nRunning;
                }
            }
            if (quatDriver.checkEvent()) {
                for (int i = 0; i < nThreads; i++) {
                    if (threads[i].joinable()) {
                        threads[i].join();
                    }
                }
                if (png != nullptr) {
                    PNGEnd(*png_internal, workers[nThreads - 1].line(), 0, iy);
                    fclose(*png);
                }
                *ystart = iy;
                quatDriver.done();
                return -128;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        for (int i = 0; i < nThreads; i++) {
            if (workers[i].failed()) {
                throw QuatException(workers[i].failedMsg());
            }
            workers[i].putLine(iy + i);
            if (nullptr != png) {
                workers[i].writeToPNG(png_internal, iy + i);
            }
            quatDriver.eol(iy + i);
        }
        firstLine = false;
    }

    if (png != nullptr) {
        PNGEnd(*png_internal, workers[nThreads-1].line(), 0, fractal.view()._yres);
        fclose(*png);
    }

    if (pngfile != nullptr) {
        char footerString[BUFSIZ];
        sprintf_s(footerString, sizeof(footerString), "%s", pngfile);
        quatDriver.changeName(footerString);
    }
    quatDriver.done();
    *ystart = fractal.view()._yres;
    return 0;
}

int CreateImage(
    Quater& quatDriver,
    std::ostream& errorMsg,
    int* ystart, 
    FractalPreferences& fractal,
    ZFlag zflag)
    /* Creates/Continues image from given parameters.
       Wants external format of frac & view */
{
    ViewBasis rbase, srbase, lbase, slbase, cbase;

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

    InitGraphics(quatDriver, errorMsg, fractal, zflag == ZFlag::NewZBuffer);

    return CalculateFractal(
        quatDriver,
        errorMsg,
        nullptr, nullptr, nullptr,
        zflag,
        ystart,
        rbase, srbase, lbase, slbase,
        fractal);
}

int CreateZBuf(
    Quater& quatDriver,
    std::ostream& errorMsg,
    int* ystart,
    FractalPreferences& fractal)
    /* Creates/Continues ZBuffer from given parameters. Wants external format of frac & view */
{
    ViewBasis rbase, srbase, lbase, slbase, cbase;
    RealPalette realpal;
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
    
    InitGraphics(quatDriver, errorMsg, fractal, true);

    i = CalculateFractal(quatDriver, errorMsg, nullptr, nullptr, nullptr,
        ZFlag::NewZBuffer, ystart,
        rbase, srbase, lbase, slbase,
        fractal);

    return i;
}
