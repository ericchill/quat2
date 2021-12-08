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

#include <iostream>
#include <fstream>
#include <boost/json/stream_parser.hpp>

#include "common.h"
#include "quat.h"
#include "files.h"   
#include "memory.h"
#include "colors.h"

static int readNextCompressed(z_stream* d, unsigned char* s);
static int ReadNextDouble(std::ostream& errorMsg, z_stream* s, double* d);
static int ReadNextInt(std::ostream& errorMsg, z_stream* s, int* i);


static char slash;


void SetSlash(char s) {
    slash = s;
}

char GetSlash() {
    return slash;
}

void ConvertPath(char* Path) {
    for (size_t i = 0; i < strlen(Path); i++) {
        if ('/' == Path[i] || '\\' == Path[i]) {
            Path[i] = slash;
        }
    }
}

void TruncateFileExt(char* filename) {
    char* s = strrchr(filename, slash);
    if (nullptr == s) {
        filename[0] = '\0';
    } else {
        *(++s) = '\0';
    }
}

/* Make a pointer to the basename (without path) */
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
    std::ostream& errorMsg,
    const char* name,
    FractalPreferences& fractal) {
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
        errorMsg << "Can't open file '" << name << "' for reading.";
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
    ZFlag zflag)
    /* rewrites filename "name" */
{
    char pngname[256];
    time_t dtime;
    unsigned char longBuf[4];
    FractalSpec frac;
    FractalView view;

    frac = fractal.fractal();
    view = fractal.view();
    formatExternToIntern(frac, view);

    strcpy_s(pngname, sizeof(pngname), name);
    ConvertPath(pngname);

    errno_t err = fopen_s(png, pngname, "w+b");
    if (nullptr == *png) {
        return -1;
    }

    int xPixelCount = view._xres;
    int yPixelCount = view._yres;
    if (view.isStereo()) {
        xPixelCount *= 2;
    }
    if (ZFlag::NewZBuffer == zflag) {
        xPixelCount *= view._antialiasing;
        yPixelCount *= view._antialiasing;
    }
    png_internal.setDimensions(xPixelCount, yPixelCount);

    if (png_internal.initWritePNG(*png)) {
        return -1;
    }

    /* Write a gAMA chunk */
    png_internal.setChunkType(gamma_chunk_label);
    ulong2bytes(static_cast<long>(GAMMA * 100000), longBuf);
    if (!png_internal.writeChunk(longBuf, sizeof(longBuf))) {
        return -1;
    }

    constexpr size_t bufAllocSize = 10240;
    LexicallyScopedPtr<char> Buf = new char[bufAllocSize];
    LexicallyScopedPtr<unsigned char> uBuf = new unsigned char[bufAllocSize];
    png_internal.setChunkType(text_chunk_label);
    strcpy_s(Buf, bufAllocSize, "Title");
    strcpy_s(&Buf[6], bufAllocSize - 6, "Quaternion fractal image");
    if (!png_internal.writeChunk((unsigned char*)(char*)Buf, 6 + strlen(&Buf[6]) + 1)) {
        return -1;
    }

    png_internal.setChunkType(text_chunk_label);
    strcpy_s(Buf, bufAllocSize, "Software");
    sprintf_s(&Buf[9], bufAllocSize - 9, "%s %s", PROGNAME, PROGVERSION);
    if (!png_internal.writeChunk((unsigned char*)(char*)Buf, strlen(Buf) + strlen(&Buf[9]) + 1)) {
        return -1;
    }

    png_internal.setChunkType(text_chunk_label);
    strcpy_s(Buf, bufAllocSize, "Creation Time");
    time(&dtime);
    char timeStr[256];
    ctime_s(timeStr, sizeof(timeStr), &dtime);
    timeStr[strlen(timeStr) - 1] = '\0';
    sprintf_s(&Buf[14], bufAllocSize - 14, "%s", timeStr);
    if (!png_internal.writeChunk((unsigned char*)(char*)Buf, 14 + strlen(&Buf[14]) + 1)) {
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
#if 0
    z_stream c_stream;
    c_stream.zalloc = nullptr;
    c_stream.zfree = nullptr;
    deflateInit(&c_stream, Z_DEFAULT_COMPRESSION);
    c_stream.next_out = &uBuf[i + 1];
    c_stream.avail_out = static_cast<uInt>(bufAllocSize - i - 1);
    c_stream.next_in = (unsigned char*)fracJSON.c_str();
    c_stream.avail_in = static_cast<uInt>(fracJSON.size());
    deflate(&c_stream, Z_FINISH);
    deflateEnd(&c_stream);
    if (!png_internal.writeChunk(uBuf, c_stream.total_out + i)) {
        return -1;
    }
#else
    // Don't compress so we can get it back with strings(1) if need be.
    memcpy(&uBuf[i+1], fracJSON.c_str(), fracJSON.size());
    if (!png_internal.writeChunk(uBuf, i + 1 + fracJSON.size())) {
        return -1;
    }
#endif
    return 0;
}

int readAllCompressed(z_stream* d, unsigned char* s, size_t maxBytes) {
    d->avail_out = static_cast<uInt>(maxBytes);
    d->next_out = s;
    return inflate(d, Z_FINISH);
}

int readNextCompressed(z_stream* d, unsigned char* s) {
    int err = Z_OK;
    int i = 0;

    do {
        d->avail_out = 1;
        d->next_out = &s[i];
        err = inflate(d, Z_PARTIAL_FLUSH);
    } while (err == Z_OK && s[i++] != 'N');
    return err;
}

int ReadNextDouble(std::ostream& errorMsg, z_stream* s, double* d) {
    int err;
    char string[100];

    err = readNextCompressed(s, (unsigned char*)string);
    if (Z_DATA_ERROR == err || Z_STREAM_ERROR == err || Z_MEM_ERROR == err) {
        if (Z_MEM_ERROR == err) {
            errorMsg << "Not enough memory for zlib inflation." << std::endl;
        } else {
            errorMsg << "Corrupted PNG file." << std::endl;
        }
        return -1;
    }
    *d = atof(string);

    return 0;
}

int ReadNextInt(std::ostream& errorMsg, z_stream* s, int* i) {
    int err;
    char string[100];

    err = readNextCompressed(s, (unsigned char*)string);
    if (Z_DATA_ERROR == err || Z_STREAM_ERROR == err || Z_MEM_ERROR == err) {
        errorMsg << "Corrupted PNG file or no memory." << std::endl;
        return -1;
    }
    *i = atoi(string);

    return 0;
}

/* RC :  -3 No quat chunk; -4 No memory; -5 No Quat-PNG; -128 filename ver higher
   than progver */
int ReadParameters(
    std::ostream& errorMsg,
    int* xstart,
    int* ystart,
    PNGFile& internal,
    FractalPreferences& fractal) {
    char s[40];
    z_stream d;
    int err;
    float ver, thisver;
    FractalPreferences maybeFractal;

    /* search until quAt chunk found */
    while (!(internal.checkChunkType(quat_chunk_label) || internal.checkChunkType(image_end_label))) {
        internal.getNextChunk();
    }
    /* No quAt chunk? */
    if (internal.checkChunkType(image_end_label)) {
        errorMsg << "PNG file has no QUAT chunk." << std::endl;
        return -3;
    }
    sprintf_s(s, sizeof(s), "%s %s", PROGNAME, PROGVERSION);
    LexicallyScopedPtr<unsigned char> Buf = new unsigned char[internal.length()];
    internal.readChunkData(Buf);
    if (strncmp((char*)(unsigned char*)Buf, s, 4) != 0) {
        errorMsg << "No QUAT signature in QUAT chunk." << std::endl;
        return -5;
    } else {
        errorMsg << Buf[5] << Buf[6] << Buf[7] << Buf[8];
    }
    /* Determine version of QUAT which wrote filename */
    thisver = static_cast<float>(atof(PROGVERSION));
    ver = static_cast<float>(atof((char*)&Buf[5]));
    if (0 == ver) {
        errorMsg << "No QUAT signature in QUAT chunk." << std::endl;
        return -5;
    }
    if (ver < 2.0f) {
        errorMsg << "PNG is from an unsupported older version of Quat." << std::endl;
        return -5;
    }
    *xstart = ((Buf[strlen(s) + 1] << 8) & 0xff00) | Buf[strlen(s) + 2];
    *ystart = ((Buf[strlen(s) + 3] << 8) & 0xff00) | Buf[strlen(s) + 4];
    d.zalloc = nullptr;
    d.zfree = nullptr;
    err = inflateInit(&d);

    calc_time = ((Buf[strlen(s) + 5] << 24) & 0xff000000L)
        | ((Buf[strlen(s) + 6] << 16) & 0xff0000L)
        | ((Buf[strlen(s) + 7] << 8) & 0xff00L)
        | Buf[strlen(s) + 8];
    d.next_in = &(Buf[strlen(s) + 9]);
    d.avail_in = static_cast<uInt>(internal.length() - strlen(s) - 9);

    LexicallyScopedPtr<unsigned char> jsonBuf = new unsigned char[10240];
    readAllCompressed(&d, jsonBuf, sizeof(jsonBuf));
    std::string jsonStr((char*)(unsigned char*)jsonBuf);
    json::stream_parser parser;
    parser.write(jsonStr);
    json::value parsed = parser.release();
    fractal = json::value_to<FractalPreferences>(parsed);
    inflateEnd(&d);

    if (formatInternToExtern(fractal.fractal(), fractal.view()) != 0) {
        errorMsg << "Strange: Error in view struct!";
        return -1;
    }
    return thisver >= ver ? 0 : -128;
}

int updateQUATChunk(PNGFile& internal, int actx, int acty) {
    char s[41];
    int quatPos;

    /* Back to beginning */
    internal.flush();
    internal.posOverIHDR();

    /* search until quAt chunk found */
    while (!(internal.checkChunkType(quat_chunk_label) || internal.checkChunkType(image_end_label))) {
        internal.getNextChunk();
    }
    /* No quAt chunk? */
    if (internal.checkChunkType(image_end_label)) {
        return -3;
    }
    sprintf_s(s, sizeof(s), "%s %s", PROGNAME, PROGVERSION);
    quatPos = internal.position() - 8;
    LexicallyScopedPtr<unsigned char> Buf = new unsigned char[internal.length()];
    internal.readChunkData(Buf);
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
    internal.position(quatPos);
    if (!internal.writeChunk(Buf)) { // length is left over from last readChunkData
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
        png_internal.writePNGLine(buf);
    }

    png_internal.endIDAT();
    png_internal.setChunkType(image_end_label);
    if (!png_internal.writeChunk(&dummy, 0)) {
        return -1;
    }
    updateQUATChunk(png_internal, actx, acty);

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
    if (pathendp != nullptr) {
        pathlen = pathendp - nowname + 1;
    } else {
        pathlen = 0;
        pathendp = nowname - 1;
    }
    strncpy_s(onlyname, sizeof(onlyname), pathendp + 1, 256);
    {
        LexicallyScopedFile f(nowname, "r");
        if (!f.isOpen()) {
            if (namewop != nullptr) {
                strncpy_s(namewop, maxNameSize, onlyname, strlen(onlyname));
            }
            return 0;
            fclose(f);
        }
    }
    onlyextp = strrchr(onlyname, '.');
    strcpy_s(onlyext, sizeof(onlyext), ".png");
    if (onlyextp != nullptr) {
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
    for (n = 1; n <= 99 && (fopen_s(&f, name, "r"), f) != nullptr; n++) {
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

