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

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "quat.h"
#include "files.h"
#include "quatfiles.h"
#include "iter.h"
#include "calculat.h"
#include "ver.h"
#include "memory.h"


int CopyIDAT(bool copy,
    PNGFile* i,
    PNGFile* i2,
    int ystart,
    ZFlag zflag,
    int zres,
    LinePutter& lineDst)
    /* "copy" decides whether a new IDAT chunk is written */
{

    LexicallyScopedPtr<unsigned char> lineBuf = new unsigned char[i->width() * 3 + 1];

    while (!(i->checkChunkType(image_data_label) || i->checkChunkType(image_end_label))) {
        i->GetNextChunk();
    }
    if (i->checkChunkType(image_end_label)) {
        return -1; /* No IDAT chunk found */
    }
    for (int j = 0; j < ystart; j++) {
        if (i->ReadPNGLine(lineBuf)) {
            return -2;
        }
        if (copy) {
            i2->WritePNGLine(lineBuf);
        }

        lineDst.putLine(0L, i->width() - 1, i->width(), j, &lineBuf[1], zflag != ZFlag::NewImage);
        check_event();
        /*eol(j+1);*/
    }
    return 0;
}


int CalculatePNG(
    const char* pngf1,
    const char* pngf2,
    char* Error,
    size_t maxErrorLen,
    const char* ini,
    ZFlag zflag,
    LinePutter& lineDst)
    /* shouldCalculateZBuffer(zflag): */
    /* pngf1 ... file to read from */
    /* pngf2 ... file to create ("" if no name available) */
    /* else: */
    /* pngf1 ... ZBuffer */
    /* pngf2 ... file to create */
    /* ini      ... ini file with parameters to replace ZBuffer parameters */
{
    base_struct rbase, srbase, lbase, slbase, cbase;
    FractalPreferences fractal;
    int xadd, yadd, xstart, ystart;
    FILE* png2;
    png_info_struct png_info1;
    PNGFile png_internal2;
    int i;
    char ErrorMSG[256], pngfile1[256], pngfile2[256], ready;
    int ys;

    strcpy_s(pngfile1, sizeof(pngfile1), pngf1);
    strcpy_s(pngfile2, sizeof(pngfile2), pngf2);
    ConvertPath(pngfile1);
    ConvertPath(pngfile2);

    /* Do PNG-Init for loading */

    LexicallyScopedFile png1 = fopen(pngfile1, "rb");
    if (nullptr == static_cast<FILE*>(png1)) {
        sprintf_s(Error, maxErrorLen, "Cannot open file '%s'.'\n", pngfile1);
        return -1;
    }
    try {
        PNGFile png_internal1(png1, &png_info1);
        i = ReadParameters(Error, maxErrorLen, &xstart, &ystart, png_internal1, fractal);
        if (i < 0 && i > -128) {
            return -1;
        }
        if (zflag == ZFlag::ImageFromZBuffer && ystart != fractal.view()._yres) {
            sprintf_s(Error, maxErrorLen, "Calculation of ZBuffer '%s' is not ready.\n", pngfile1);
            return -1;
        } else if (ystart == fractal.view()._yres) {
            sprintf_s(Error, maxErrorLen, "Calculation of image '%s' was complete.\n", pngfile1);
            ready = 1;
        } else {
            ready = 0;
        }

        if (!ready && strlen(pngfile2) == 0) {
            sprintf_s(Error, maxErrorLen, "Couldn't find a free filename based on '%s'\n", pngfile1);
            return -1;
        }

        if (ZFlag::ImageFromZBuffer == zflag && strlen(ini) != 0) {
            i = ParseINI(ini, Error, maxErrorLen, fractal);
            if (0 != i) {
                return -1;
            }
        }

        if (TranslateColorFormula(fractal.colorScheme().get(), ErrorMSG, sizeof(ErrorMSG)) != 0) {
            sprintf_s(Error, maxErrorLen,
                "Strange error:\nPNG-File '%s':\nError in color scheme formula:\n%s\n", pngfile1, ErrorMSG);
            return -1;
        }

        if (fractal.view().calcbase(&cbase, &srbase, WhichEye::Monocular) != 0) {
            sprintf_s(Error, maxErrorLen, "file '%s':\nError in FractalView!\n", pngfile1);
            return -1;
        }
        rbase = cbase;
        if (fractal.view().isStereo()) {
            if (fractal.view().calcbase(&rbase, &srbase, WhichEye::Right) != 0) {
                sprintf_s(Error, maxErrorLen, "file '%s':\nError in FractalView (right eye)!\n", pngfile1);
                return -1;
            }
            if (fractal.view().calcbase(&lbase, &slbase, WhichEye::Left) != 0) {
                sprintf_s(Error, maxErrorLen, "file '%s':\nError in FractalView (left eye)!\n", pngfile1);
                return -1;
            }
        }

        if (InitGraphics(Error, maxErrorLen, fractal, ready,
            &xadd, &yadd, zflag != ZFlag::NewImage) != 0) {
            return -1;
        }

        ys = ystart;
        if (shouldCalculateImage(zflag)) {
            ys *= fractal.view()._antialiasing;
        }

        if (ZFlag::ImageFromZBuffer == zflag) {
            if (writeQuatPNGHead(pngfile2, &png2, png_internal2, 0, 0, 0L, fractal, zflag)) {
                sprintf_s(Error, maxErrorLen, "Could not create file '%s'\n", pngfile2);
                return -1;
            }
            sprintf_s(Error, maxErrorLen, "\nCreated new file '%s'\n", pngfile2);
            ystart = 0;
            xstart = 0;
        }

        if (!ready) {
            if (writeQuatPNGHead(pngfile2, &png2, png_internal2, 0, 0, 0L, fractal, zflag)) {
                sprintf_s(Error, maxErrorLen, "Could not create file '%s'\n", pngfile2);
                return -1;
            }
            CopyIDAT(true, &png_internal1, &png_internal2, ys, zflag, fractal.view()._zres, lineDst);
            sprintf_s(Error, maxErrorLen, "\nCreated new file '%s'\n", pngfile2);
        } else {
            if (ZFlag::NewImage == zflag) {
                CopyIDAT(false, &png_internal1, &png_internal2, ys, zflag, fractal.view()._zres, lineDst);
            } else {
                CopyIDAT(false, &png_internal1, &png_internal2, ys, ZFlag::NewZBuffer, fractal.view()._zres, lineDst);
            }
            if (shouldCalculateDepths(zflag)) {
                Done();
            }
        }

        i = 0;
        if (!ready || ZFlag::ImageFromZBuffer == zflag) {
            i = CalculateFractal(Error, maxErrorLen, pngfile2, &png2,
                &png_internal2,
                zflag, &xstart, &ystart, /* noev */16,
                &rbase, &srbase, &lbase, &slbase, fractal,
                lineDst);
        }
    } catch (PNGException&) {
        sprintf_s(Error, maxErrorLen, "File '%s' is not a valid png-file.\n", pngfile1);
        return -1;
    }

    return i;
}

int ParseINI(
    const char* file,
    char* Error,
    size_t maxErrorLen,
    FractalPreferences& fractal) {

    return ParseFile(file, fractal, Error, maxErrorLen);
}

/* Opens pngfile, reads parameters from it (no checking if valid)
   it initializes graphics and shows image data line per line
*/
int ReadParametersAndImage(
    char* Error,
    size_t maxErrorLen,
    const char* pngf,
    bool* ready,
    int* xstart,
    int* ystart,
    FractalPreferences& fractal,
    ZFlag zflag,
    LinePutter& lineDst) {

    png_info_struct png_info;
    int xadd, yadd, i, xres, yr;
    base_struct base, sbase;
    char pngfile[256];

    strcpy_s(pngfile, sizeof(pngfile), pngf);
    ConvertPath(pngfile);

    /* Do PNG-Init for loading */
    LexicallyScopedFile png = fopen(pngfile, "rb");
    if (nullptr == png) {
        return -1;
    }
    try {
        PNGFile png_internal(png, &png_info);
        i = ReadParameters(Error, maxErrorLen, xstart, ystart, png_internal, fractal);
        /* if (i == -128) sprintf_s(Error, maxErrorLen, "Warning: File version higher than %s.\n",PROGVERSION); */
        if (i < 0 && i > -128) {
            return -1;
        }
        if (fractal.view().calcbase(&base, &sbase, WhichEye::Monocular) != 0) {
            sprintf_s(Error, maxErrorLen, "file '%s':\nError in FractalView!\n", pngfile);
            return -1;
        }
        *ready = *ystart == fractal.view()._yres;
        xres = fractal.view()._xres;
        if (fractal.view().isStereo()) {
            xres *= 2;
        }
        if (ZFlag::NewZBuffer == zflag) {
            xres *= fractal.view()._antialiasing;
        }
        LexicallyScopedPtr<unsigned char> line = new unsigned char[xres * 3 + 1];
        if (InitGraphics(Error, maxErrorLen, fractal, *ready,
            &xadd, &yadd, zflag != ZFlag::NewImage) != 0) {
            return -1;
        }
        yr = *ystart;
        if (ZFlag::NewZBuffer == zflag) {
            yr *= fractal.view()._antialiasing;
        }
        if (CopyIDAT(false, &png_internal, NULL, yr,
            zflag, fractal.view()._zres, lineDst) != 0) {
            sprintf_s(Error, maxErrorLen, "Error reading file '%s'.", pngfile);
            return -1;
        }
    } catch (PNGException&) {
        sprintf_s(Error, maxErrorLen, "File '%s' is not a valid PNG file.\n", pngfile);
        return -1;
    }
    return 0;
}


int SavePNG(
    char* Error,
    size_t maxErrorLen,
    const char* pngf,
    int xstart,
    int ystart,
    disppal_struct* disppal,
    const FractalPreferences& fractal,
    ZFlag zflag) {

    FILE* png;
    FractalSpec frac;
    FractalView view;
    unsigned char dummy;
    int xres, yres, i;
    char pngfile[256];

    strcpy_s(pngfile, sizeof(pngfile), pngf);
    ConvertPath(pngfile);

    if (ystart < 0) {
        ystart = 0;	/* Else bug when saving empty image. */
    }

    frac = fractal.fractal();
    view = fractal.view();
    xres = view._xres;
    if (view.isStereo()) {
        xres *= 2;
    }
    if (ZFlag::NewZBuffer == zflag) {
        xres *= view._antialiasing;
    }
    yres = view._yres;
    if (ZFlag::NewZBuffer == zflag) {
        yres *= view._antialiasing;
    }
    LexicallyScopedPtr<unsigned char> line = new unsigned char[xres * 3 + 2];
    PNGFile png_internal;
    if (writeQuatPNGHead(
        pngfile, &png, png_internal,
        xstart, ystart, static_cast<int>(calc_time),
        fractal,
        zflag)) {
        sprintf_s(Error, maxErrorLen, "Error writing file '%s' in PNGInitialization\n", pngfile);
        return -1;
    }

    for (i = 0; i < yres; i++) {
        QU_getline(&line[1], i, xres, zflag);
        line[0] = 0; /* Set filter method */
        png_internal.DoFiltering(line);
        if (png_internal.WritePNGLine(line)) {
            sprintf_s(Error, maxErrorLen, "Error writing file '%s' in WritePNGLine\n", pngfile);
            return -1;
        }
    }

    i = png_internal.EndIDAT();
    png_internal.setChunkType(image_end_label);

    if (i != 0 || !png_internal.WriteChunk(&dummy, 0)) {
        sprintf_s(Error, maxErrorLen, "Error writing file '%s' after EndPNG\n", pngfile);
        return -1;
    }

    return 0;
}

int BuildName(char* name, char* namewop, size_t maxNameLen, const char* ext, const char* file,
    char* Error, size_t maxErrorLen) {

    strncpy_s(name, maxNameLen, file, 256);
    char* c = strrchr(name, '.');
    if (c != NULL) {
        strcpy_s(c, maxNameLen - (c - name), ext);
    } else {
        strcat_s(name, maxNameLen, ext);
    }
    if (GetNextName(name, namewop, maxNameLen)) {
        sprintf_s(Error, maxErrorLen, "Couldn't find a free filename based on '%s'.\n", name);
        return -1;
    }
    return 0;
}


int CleanNumberString(char* s, size_t maxLen) {
    size_t i;

    if (strrchr(s, '.') != NULL) {
        i = strlen(s) - 1;
        while (s[i] == '0') {
            i--;
        }
        s[i + 1] = '\0';
        if (s[i] == '.') {
            s[i] = '\0';
        }
    }
    if (strlen(s) == 0) {
        strcpy_s(s, maxLen, "0");
    }
    return 0;
}

int ReadParametersPNG(
    const char* fil,
    char* Error,
    size_t maxErrorLen,
    FractalPreferences& fractal) {

    png_info_struct png_info;
    int xstart, ystart;
    int i;
    base_struct base, sbase;
    char file[256];

    strcpy_s(file, sizeof(file), fil);
    ConvertPath(file);

    LexicallyScopedFile png = fopen(file, "rb");
    if (nullptr == png) {
        sprintf_s(Error, maxErrorLen, "Cannot open png-file '%s'.'", file);
        return -1;
    }
    try {
        PNGFile png_internal(png, &png_info);
        i = ReadParameters(Error, maxErrorLen, &xstart, &ystart, png_internal, fractal);
        if (i < 0 && i > -128) {
            return -1;
        }
        if (fractal.view().calcbase(&base, &sbase, WhichEye::Monocular) != 0) {
            sprintf_s(Error, maxErrorLen, "Strange error in '%s'\nError in FractalView!", file);
            return -1;
        }
    } catch (PNGException&) {
        sprintf_s(Error, maxErrorLen, "File '%s' is not a valid png-file.", file);
        return -1;
    }
    return 0;
}

constexpr size_t numBufSize = 30;

static void formatDouble(char* buf, double num) {
    sprintf_s(buf, numBufSize, "%.15f", num);
    CleanNumberString(buf, numBufSize);
}

int WriteINI(
    char* Error,
    size_t maxErrorLen,
    const char* fil,
    const FractalPreferences& fractal) {

    char file[256];

    strcpy_s(file, sizeof(file), fil);
    ConvertPath(file);

    std::ofstream txt;
    txt.open(file, std::ios::out);
    if (!txt.is_open()) {
        sprintf_s(Error, maxErrorLen, "Could not create file '%s'.", file);
        return -1;
    }
    txt << "# This file was generated by '" << PROGNAME << " " << PROGNAME << "'" << std::endl << std::endl;
    json::value jsonForm = fractal.toJSON();
    std::string indent("    ");
    pretty_print(txt, jsonForm, &indent);
    txt << std::endl;
    txt.close();
    return 0;
}

int GetParameters(const char* afile, char* Error, size_t maxErrorLen) {
    char file[256], ini[256], * filewop;
    FractalPreferences fractal;

    strcpy_s(file, sizeof(file), afile);
    ConvertPath(file);
    filewop = strrchr(file, GetSlash());
    if (nullptr == filewop) {
        filewop = file;
    } else {
        filewop++;
    }

    if (ReadParametersPNG(file, Error, maxErrorLen, fractal) != 0) {
        return -1;
    }

    if (BuildName(ini, NULL, sizeof(ini), ".ini", filewop, Error, maxErrorLen)) {
        return -1;
    }
    if (!WriteINI(Error, maxErrorLen, ini, fractal)) {
        sprintf_s(Error, maxErrorLen, "Generated file '%s'.", ini);
    } else {
        return -1;
    }
    return 0;
}

int ImgFromZBuf(const char* file, const char* file2, char* Error, size_t maxErrorLen, LinePutter& lineDst)
/* file is a ZBuffer, which gets turned into an image */
/* called by command-line versions (non-Windows), which are given just a file */
{
    char zpnfile[256], pngfile[256];
    char* s, * s2;

    strcpy_s(zpnfile, sizeof(zpnfile), file);
    ConvertPath(zpnfile);
    strcpy_s(pngfile, sizeof(pngfile), zpnfile);
    s2 = FilenameWithoutPath(pngfile);

    /* Delete suffixes and append .png */
    if ((s = strstr(s2, ".")) == nullptr) {
        strcat_s(pngfile, sizeof(pngfile), ".png");
    } else {
        strcpy_s(s, sizeof(s), ".png");
    }
    if (GetNextName(pngfile, NULL, sizeof(pngfile)) != 0) {
        sprintf_s(Error, maxErrorLen, "Couldn't find a free filename based on '%s'\n", pngfile);
        return -1;
    }

    return CalculatePNG(zpnfile, pngfile, Error, maxErrorLen, file2, ZFlag::ImageFromZBuffer, lineDst);
}
