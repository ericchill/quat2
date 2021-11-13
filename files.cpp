/* Quat - A 3D fractal generation program */
/* Copyright (C) 1997,98 Dirk Meyer */
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
/* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <fstream>
#include <boost/json/stream_parser.hpp>

#include "common.h"
#include "quat.h"  /* includes common.h */
#include "files.h"   
#include "memory.h"
#include "colors.h"

static int ReadNext(z_stream* d, unsigned char* s);
static int ReadNextDouble(z_stream* s, double* d, char* Error, size_t maxErrorLen);
/*static int ReadNextLong(z_stream *s, long *i, char *Error);*/
static int ReadNextInt(z_stream* s, int* i, char* Error, size_t maxErrorLen);


static char slash;


void SetSlash(char s) {
    slash = s;
}

char GetSlash() {
    return slash;
}

void ConvertPath(char* Path) {
    for (size_t i = 0; i < strlen(Path); i++) {
        if (Path[i] == '/' || Path[i] == '\\') {
            Path[i] = slash;
        }
    }
}

void TruncateFileExt(char* filename) {
    char* s = strrchr(filename, slash);
    if (s == NULL) {
        filename[0] = '\0';
    } else {
        *(++s) = '\0';
    }
}

/* Make a pointer to the filename (without path) */
char* FilenameWithoutPath(char* filename) {
    char* s;

    if ((s = strrchr(filename, slash)) == NULL) {
        s = filename;
    } else {
        s++;
    }
    return s;
}

int ParseFile(
    const char* name,
    FractalPreferences& fractal,
    char* Error_Msg,
    size_t maxErrorLen) {
    std::ifstream input;
    input.open(name, std::ios::in);
    if (input.is_open()) {
        json::stream_parser parser;
        for (std::string line; std::getline(input >> std::ws, line); ) {
            if ('#' != line[0]) {
                parser.write_some(line);
            }
        }
        json::value result = parser.release();
        fractal = json::value_to<FractalPreferences>(result);
    } else {
        sprintf_s(Error_Msg, maxErrorLen, "Can't open file '%s' for reading.", name);
        return -1;
    }
    return 0;
}


int writeQuatPNGHead(
    const char* name,
    FILE** png,
    PNGFile& png_internal,
    int xstart, int ystart,
    long calctime,
    const FractalPreferences& fractal,
    disppal_struct* pal,
    ZFlag zflag)
    /* rewrites file "name" */
{
    char pngname[256], * s;
    z_stream c_stream;
    time_t dtime;
    unsigned char longBuf[4];
    FractalSpec frac;
    FractalView view;

    frac = fractal.fractal();
    view = fractal.view();
    FormatExternToIntern(frac, view);

    strcpy_s(pngname, sizeof(pngname), name);
    ConvertPath(pngname);

    *png = fopen(pngname, "w+b");
    if (*png == NULL) {
        return -1;
    }

    int xPixelCount = view._xres;
    int yPixelCount = view._yres;
    if (view.isStereo()) {
        xPixelCount *= 2;
    }
    if (zflag == ZFlag::NewZBuffer) {
        xPixelCount *= view._antialiasing;
        yPixelCount *= view._antialiasing;
    }
    png_internal.setDimensions(xPixelCount, yPixelCount);

    if (png_internal.InitWritePNG(*png)) {
        return -1;
    }

    /* Write a gAMA chunk */
    png_internal.setChunkType(gamma_chunk_label);
    ulong2bytes(static_cast<long>(GAMMA * 100000), longBuf);
    if (!png_internal.WriteChunk(longBuf, sizeof(longBuf))) {
        return -1;
    }

    constexpr size_t bufAllocSize = 10240;
    LexicallyScopedPtr<char> Buf = new char[bufAllocSize];
    LexicallyScopedPtr<unsigned char> uBuf = new unsigned char[bufAllocSize];
    png_internal.setChunkType(text_chunk_label);
    strcpy_s(Buf, bufAllocSize, "Title");
    strcpy_s(&Buf[6], bufAllocSize - 6, "Quaternion fractal image");
    if (!png_internal.WriteChunk((unsigned char*)(char*)Buf, 6 + strlen(&Buf[6]) + 1)) {
        return -1;
    }

    png_internal.setChunkType(text_chunk_label);
    strcpy_s(Buf, bufAllocSize, "Software");
    sprintf_s(&Buf[9], bufAllocSize - 9, "%s %s", PROGNAME, PROGVERSION);
    if (!png_internal.WriteChunk((unsigned char*)(char*)Buf, strlen(Buf) + strlen(&Buf[9]) + 1)) {
        return -1;
    }

    png_internal.setChunkType(text_chunk_label);
    strcpy_s(Buf, bufAllocSize, "Creation Time");
    time(&dtime);
    s = ctime(&dtime);
    s[strlen(s) - 1] = '\0';
    sprintf_s(&Buf[14], bufAllocSize - 14, "%s", s);
    if (!png_internal.WriteChunk((unsigned char*)(char*)Buf, 14 + strlen(&Buf[14]) + 1)) {
        return -1;
    }

    /* Write quAt chunk */
    png_internal.setChunkType(quat_chunk_label);
    sprintf_s(Buf, bufAllocSize, "%s %s", PROGNAME, PROGVERSION);
    memcpy(uBuf, Buf, bufAllocSize);
    size_t i = strlen(Buf);
    uBuf[i + 1] = (unsigned char)(xstart >> 8 & 0xffL); /* MSB of actx */
    uBuf[i + 2] = (unsigned char)(xstart & 0xffL);    /* LSB of actx */
    uBuf[i + 3] = (unsigned char)(ystart >> 8 & 0xffL); /* MSB of acty */
    uBuf[i + 4] = (unsigned char)(ystart & 0xffL);    /* LSB of acty */
    uBuf[i + 5] = (unsigned char)(calctime >> 24 & 0xffL);
    uBuf[i + 6] = (unsigned char)(calctime >> 16 & 0xffL);
    uBuf[i + 7] = (unsigned char)(calctime >> 8 & 0xffL);
    uBuf[i + 8] = (unsigned char)(calctime & 0xffL);
    i = i + 8;

    /* convert FractalPreferences to ascii */
    std::string fracJSON = json::serialize(fractal.toJSON());
    c_stream.zalloc = nullptr;
    c_stream.zfree = nullptr;
    deflateInit(&c_stream, Z_DEFAULT_COMPRESSION);
    c_stream.next_out = &uBuf[i + 1];
    c_stream.avail_out = static_cast<uInt>(bufAllocSize - i - 1);
    c_stream.next_in = (unsigned char*)fracJSON.c_str();
    c_stream.avail_in = static_cast<uInt>(fracJSON.size());
    deflate(&c_stream, Z_FINISH);
    deflateEnd(&c_stream);
    if (!png_internal.WriteChunk(uBuf, c_stream.total_out + i)) {
        return -1;
    }

    return 0;
}

int ReadAll(z_stream* d, unsigned char* s, size_t maxBytes) {
    d->avail_out = static_cast<uInt>(maxBytes);
    d->next_out = s;
    return inflate(d, Z_FINISH);
}

int ReadNext(z_stream* d, unsigned char* s) {
    int err = Z_OK;
    int i = 0;

    do {
        d->avail_out = 1;
        d->next_out = &s[i];
        err = inflate(d, Z_PARTIAL_FLUSH);
    } while (err == Z_OK && s[i++] != 'N');
    return err;
}

int ReadNextDouble(z_stream* s, double* d, char* Error, size_t maxErrorLen) {
    int err;
    char string[100];

    err = ReadNext(s, (unsigned char*)string);
    if (err == Z_DATA_ERROR || err == Z_STREAM_ERROR || err == Z_MEM_ERROR) {
        if (err == Z_MEM_ERROR) {
            sprintf_s(Error, maxErrorLen, "Not enough memory for zlib inflation.\n");
        } else {
            sprintf_s(Error, maxErrorLen, "Corrupted PNG file.\n");
        }
        return -1;
    }
    *d = atof(string);

    return 0;
}

int ReadNextInt(z_stream* s, int* i, char* Error, size_t maxErrorLen) {
    int err;
    char string[100];

    err = ReadNext(s, (unsigned char*)string);
    if (err == Z_DATA_ERROR || err == Z_STREAM_ERROR || err == Z_MEM_ERROR) {
        sprintf_s(Error, maxErrorLen, "Corrupted PNG file or no memory.\n");
        return -1;
    }
    *i = atoi(string);

    return 0;
}

/*
int ReadNextLong(z_stream *s, long *i, char *Error)
{
    int err;
    char string[100];

    err = ReadNext(s, (unsigned char*)string);
    if (err == Z_DATA_ERROR || err == Z_STREAM_ERROR || err == Z_MEM_ERROR) {
    sprintf(Error, "Corrupted PNG file or no memory.\n");
    return -1;
    }
    *i = atol(string);

   return 0;
}
*/

/* RC :  -3 No quat chunk; -4 No memory; -5 No Quat-PNG; -128 file ver higher
   than progver */
int ReadParameters(
    char* Error,
    size_t maxErrorLen,
    int* xstart,
    int* ystart,
    PNGFile& internal,
    FractalPreferences& fractal) {
\
    char s[40];
    z_stream d;
    int err;
    float ver, thisver;
    FractalPreferences maybeFractal;
    unsigned char jsonBuf[10240];

    /* search until quAt chunk found */
    while (!(internal.checkChunkType(quat_chunk_label) || internal.checkChunkType(image_end_label))) {
        internal.GetNextChunk();
    }
    /* No quAt chunk? */
    if (internal.checkChunkType(image_end_label)) {
        sprintf_s(Error, maxErrorLen, "PNG file has no QUAT chunk.\n");
        return -3;
    }
    sprintf_s(s, sizeof(s), "%s %s", PROGNAME, PROGVERSION);
    LexicallyScopedPtr<unsigned char> Buf = new unsigned char[internal.length()];
    internal.ReadChunkData(Buf);
    if (strncmp((char*)(unsigned char*)Buf, s, 4) != 0) {
        sprintf_s(Error, maxErrorLen, "No QUAT signature in QUAT chunk.\n");
        return -5;
    } else {
        sprintf_s(Error, maxErrorLen, "%c%c%c%c", Buf[5], Buf[6], Buf[7], Buf[8]);
    }
    /* Determine version of QUAT which wrote file */
    thisver = static_cast<float>(atof(PROGVERSION));
    ver = static_cast<float>(atof((char*)&Buf[5]));
    if (ver == 0) {
        sprintf_s(Error, maxErrorLen, "No QUAT signature in QUAT chunk.\n");
        return -5;
    }
    *xstart = ((Buf[strlen(s) + 1] << 8) & 0xff00) | Buf[strlen(s) + 2];
    *ystart = ((Buf[strlen(s) + 3] << 8) & 0xff00) | Buf[strlen(s) + 4];
    d.zalloc = nullptr;
    d.zfree = nullptr;
    err = inflateInit(&d);
    if (ver >= 0.92) {
        calc_time = ((Buf[strlen(s) + 5] << 24) & 0xff000000L)
            | ((Buf[strlen(s) + 6] << 16) & 0xff0000L)
            | ((Buf[strlen(s) + 7] << 8) & 0xff00L)
            | Buf[strlen(s) + 8];
        d.next_in = &(Buf[strlen(s) + 9]);
        d.avail_in = static_cast<uInt>(internal.length() - strlen(s) - 9);
    } else {
        d.next_in = &(Buf[strlen(s) + 5]);
        d.avail_in = static_cast<uInt>(internal.length() - strlen(s) - 5);
    }

    err = ReadAll(&d, jsonBuf, sizeof(jsonBuf));
    std::string jsonStr((char*)jsonBuf);
    json::stream_parser parser;
    parser.write(jsonStr);
    json::value parsed = parser.release();
    fractal = json::value_to<FractalPreferences>(parsed);
    err = inflateEnd(&d);

    if (FormatInternToExtern(fractal.fractal(), fractal.view()) != 0) {
        sprintf_s(Error, maxErrorLen, "Strange: Error in view struct!");
        return -1;
    }
    return thisver >= ver ? 0 : -128;
}

int UpdateQUATChunk(PNGFile& internal, int actx, int acty) {
    char s[41];
    long QuatPos;

    /* Back to beginning */
    internal.flush();
    internal.PosOverIHDR();

    /* search until quAt chunk found */
    while (!(internal.checkChunkType(quat_chunk_label) || internal.checkChunkType(image_end_label))) {
        internal.GetNextChunk();
    }
    /* No quAt chunk? */
    if (internal.checkChunkType(image_end_label)) {
        return -3;
    }
    sprintf_s(s, sizeof(s), "%s %s", PROGNAME, PROGVERSION);
    QuatPos = internal.position() - 8;
    LexicallyScopedPtr<unsigned char> Buf = new unsigned char[internal.length()];
    internal.ReadChunkData(Buf);
    if (strncmp((char*)(unsigned char*)Buf, s, strlen(s)) != 0) {
        return -5;
    }
    Buf[strlen(s) + 1] = (unsigned char)(actx >> 8 & 0xffL);   /* MSB of actx */
    Buf[strlen(s) + 2] = (unsigned char)(actx & 0xffL);      /* LSB of actx */
    Buf[strlen(s) + 3] = (unsigned char)(acty >> 8 & 0xffL);   /* MSB of acty */
    Buf[strlen(s) + 4] = (unsigned char)(acty & 0xffL);      /* LSB of acty */
    Buf[strlen(s) + 5] = (unsigned char)(calc_time >> 24 & 0xffL);
    Buf[strlen(s) + 6] = (unsigned char)(calc_time >> 16 & 0xffL);
    Buf[strlen(s) + 7] = (unsigned char)(calc_time >> 8 & 0xffL);
    Buf[strlen(s) + 8] = (unsigned char)(calc_time & 0xffL);
    internal.flush();
    internal.position(QuatPos);
    if (!internal.WriteChunk(Buf, -1)) { // length is left over from last ReadChunkData
        return -1;
    }
    internal.flush();

    return 0;
}

int PNGEnd(PNGFile& png_internal,
    unsigned char* buf,
    int actx, int acty) {
    unsigned char dummy;

    for (unsigned int i = acty; i < png_internal.height(); i++) {
        png_internal.WritePNGLine(buf);
    }

    png_internal.EndIDAT();
    png_internal.setChunkType(image_end_label);
    if (!png_internal.WriteChunk(&dummy, 0)) {
        return -1;
    }
    UpdateQUATChunk(png_internal, actx, acty);

    return 0;
}

int GetNextName(char* nowname, char* namewop, size_t maxNameSize) {
    char onlyname[256], * onlyextp, * pathendp, onlyext[256], name[260];
    int n = 0;
    size_t pathlen;
    FILE* f;

    ConvertPath(nowname);
    /* Delete everything after first point */
    pathendp = strrchr(nowname, slash);
    if (pathendp != NULL) {
        pathlen = pathendp - nowname + 1;
    } else {
        pathlen = 0;
        pathendp = nowname - 1;
    }
    strncpy_s(onlyname, sizeof(onlyname), pathendp + 1, 256);
    if ((f = fopen(nowname, "r")) == NULL) {
        if (namewop != NULL) {
            strncpy_s(namewop, maxNameSize, onlyname, 80);
        }
        return 0;
    }
    fclose(f);
    onlyextp = strrchr(onlyname, '.');
    strcpy_s(onlyext, sizeof(onlyext), ".png");
    if (onlyextp != NULL) {
        strcpy_s(onlyext, sizeof(onlyext), onlyextp);
        onlyextp[0] = '\0';
    } else {
        onlyextp = onlyext;
    }
    while (strlen(onlyname) < 8) {
        strcat_s(onlyname, sizeof(onlyname), "_");
    }
    onlyname[strlen(onlyname) - 2] = '\0';

    strncpy_s(name, sizeof(name), nowname, pathlen); 
    name[pathlen] = '\0';
    sprintf_s(name, sizeof(name), "%s%s%02i%s", name, onlyname, n, onlyext);
    for (n = 1; n <= 99 && (f = fopen(name, "r")) != NULL; n++) {
        fclose(f);
        strncpy_s(name, sizeof(name), nowname, pathlen);
        name[pathlen] = '\0';
        sprintf_s(name, sizeof(name), "%s%s%02i%s", name, onlyname, n, onlyext);
    }

    /* No free name found */
    if (n > 99) {
        return -1;
    }
    if (strlen(name) > 256) {
        return -1;
    }
    strncpy_s(nowname, maxNameSize, name, 256);
    if (namewop != NULL) {
        sprintf_s(namewop, maxNameSize, "%s%02i%s", onlyname, n - 1, onlyext);
    }
    return 0;
}

