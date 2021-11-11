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
        for (std::string line; std::getline(input, line); ) {
            parser.write(line);
        }
        json::value result = parser.release();
        fractal = json::value_to<FractalPreferences>(result);
    } else {
        sprintf_s(Error_Msg, maxErrorLen, "Can't open file '%s' for reading.", name);
        return -1;
    }
    return 0;
}


int PNGInitialization(
    const char* name,
    ColorMode mode, 
    FILE** png,
    png_info_struct* png_info,
    png_internal_struct* png_internal,
    int xstart, int ystart,
    long calctime,
    const FractalPreferences& fractal,
    disppal_struct* pal,
    ZFlag zflag)
    /* rewrites file "name" */
{
    size_t i;
    unsigned char pal2[256][3];
    char pngname[256], * s;
    unsigned long l;
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

    png_info->width = view._xres;
    if (view.isStereo()) {
        png_info->width *= 2;
    }
    png_info->height = view._yres;
    if (zflag == ZFlag::NewZBuffer) {
        png_info->width *= view._antialiasing;
        png_info->height *= view._antialiasing;
    }
    if (mode == ColorMode::Indexed) {  /* indexed color */
        png_info->bit_depth = 8;
        png_info->color_type = 3;
    } else {  /* true color */
        png_info->bit_depth = 8;
        png_info->color_type = 2;
    }
    png_info->compression = 0;
    png_info->interlace = 0;
    png_info->filter = 0;

    if (InitWritePNG(*png, png_info, png_internal)) {
        return -1;
    }

    /* Write a gAMA chunk */
    setChunkType(png_internal, gamma_chunk_label);
    png_internal->length = sizeof(uint32_t);
    l = (long)(0.45455 * 100000);
    ulong2bytes(l, longBuf);
    if (WriteChunk(png_internal, longBuf)) {
        return -1;
    }

    /* Write a PLTE chunk */
    if (mode == ColorMode::Indexed) {
        for (i = 0; i < pal->maxcol; i++) {
            pal2[i][0] = pal->cols[i].r << 2;
            pal2[i][1] = pal->cols[i].g << 2;
            pal2[i][2] = pal->cols[i].b << 2;
        }
        setChunkType(png_internal, palette_chunk_label);
        png_internal->length = pal->maxcol * 3;
        if (WriteChunk(png_internal, pal2[0])) {
            return -1;
        }
    }
    constexpr size_t bufAllocSize = 10240;
    LexicallyScopedPtr<char> Buf = new char[bufAllocSize];
    LexicallyScopedPtr<unsigned char> uBuf = new unsigned char[bufAllocSize];
    setChunkType(png_internal, text_chunk_label);
    strcpy_s(Buf, bufAllocSize, "Title");
    strcpy_s(&Buf[6], bufAllocSize - 6, "Quaternion fractal image");
    png_internal->length = 30;
    if (WriteChunk(png_internal, (unsigned char*)(char*)Buf)) {
        return -1;
    }

    setChunkType(png_internal, text_chunk_label);
    strcpy_s(Buf, bufAllocSize, "Software");
    sprintf_s(&Buf[9], bufAllocSize - 9, "%s %s", PROGNAME, PROGVERSION);
    png_internal->length = strlen(Buf) + strlen(&Buf[9]) + 1;
    if (WriteChunk(png_internal, (unsigned char*)(char*)Buf)) {
        return -1;
    }

    setChunkType(png_internal, text_chunk_label);
    strcpy_s(Buf, bufAllocSize, "Creation Time");
    time(&dtime);
    s = ctime(&dtime);
    s[strlen(s) - 1] = '\0';
    sprintf_s(&Buf[14], bufAllocSize - 14, "%s", s);
    png_internal->length = 14 + strlen(&Buf[14]);
    if (WriteChunk(png_internal, (unsigned char*)(char*)Buf)) {
        return -1;
    }

    /* Write quAt chunk */
    setChunkType(png_internal, quat_chunk_label);
    sprintf_s(Buf, bufAllocSize, "%s %s", PROGNAME, PROGVERSION);
    memcpy(uBuf, Buf, bufAllocSize);
    i = strlen(Buf);
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
    c_stream.zalloc = (alloc_func)NULL;
    c_stream.zfree = (free_func)NULL;
    deflateInit(&c_stream, Z_DEFAULT_COMPRESSION);
    c_stream.next_out = &uBuf[i + 1];
    c_stream.avail_out = static_cast<uInt>(bufAllocSize - i - 1);
    c_stream.next_in = (unsigned char*)fracJSON.c_str();
    c_stream.avail_in = static_cast<uInt>(fracJSON.size());
    deflate(&c_stream, Z_FINISH);
    png_internal->length = c_stream.total_out + i;
    deflateEnd(&c_stream);
    i = WriteChunk(png_internal, uBuf);

    return i ? -1 : 0;
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
    png_internal_struct* internal,
    FractalPreferences& fractal) {
\
    unsigned char* Buf;
    char s[40];
    z_stream d;
    int err;
    float ver, thisver;
    FractalPreferences maybeFractal;
    unsigned char jsonBuf[10240];

    /* search until quAt chunk found */
    while (!(checkChunkType(internal, quat_chunk_label) || checkChunkType(internal, image_end_label))) {
        GetNextChunk(internal);
    }
    /* No quAt chunk? */
    if (checkChunkType(internal, image_end_label)) {
        sprintf_s(Error, maxErrorLen, "PNG file has no QUAT chunk.\n");
        return -3;
    }
    sprintf_s(s, sizeof(s), "%s %s", PROGNAME, PROGVERSION);
    Buf = new unsigned char[internal->length];
    if (Buf == NULL) {
        sprintf_s(Error, maxErrorLen, "No memory.\n");
        return -4;
    }
    ReadChunkData(internal, Buf);
    if (strncmp((char*)Buf, s, 4) != 0) {
        delete [] Buf;
        sprintf_s(Error, maxErrorLen, "No QUAT signature in QUAT chunk.\n");
        return -5;
    } else {
        sprintf_s(Error, maxErrorLen, "%c%c%c%c", Buf[5], Buf[6], Buf[7], Buf[8]);
    }
    /* Determine version of QUAT which wrote file */
    ver = 0.0;
    thisver = 0.0;
    thisver = static_cast<float>(atof(PROGVERSION));
    ver = static_cast<float>(atof((char*)&Buf[5]));
    if (ver == 0) {
        delete Buf;
        sprintf_s(Error, maxErrorLen, "No QUAT signature in QUAT chunk.\n");
        return -5;
    }
    *xstart = ((Buf[strlen(s) + 1] << 8) & 0xff00) | Buf[strlen(s) + 2];
    *ystart = ((Buf[strlen(s) + 3] << 8) & 0xff00) | Buf[strlen(s) + 4];
    d.zalloc = (alloc_func)NULL;
    d.zfree = (free_func)NULL;
    err = inflateInit(&d);
    if (ver >= 0.92) {
        calc_time = ((Buf[strlen(s) + 5] << 24) & 0xff000000L)
            | ((Buf[strlen(s) + 6] << 16) & 0xff0000L)
            | ((Buf[strlen(s) + 7] << 8) & 0xff00L)
            | Buf[strlen(s) + 8];
        d.next_in = &(Buf[strlen(s) + 9]);
        d.avail_in = static_cast<uInt>(internal->length - strlen(s) - 9);
    } else {
        d.next_in = &(Buf[strlen(s) + 5]);
        d.avail_in = static_cast<uInt>(internal->length - strlen(s) - 5);
    }

    err = ReadAll(&d, jsonBuf, sizeof(jsonBuf));
    std::string jsonStr((char*)jsonBuf);
    json::stream_parser parser;
    parser.write(jsonStr);
    json::value parsed = parser.release();
    fractal = json::value_to<FractalPreferences>(parsed);
    err = inflateEnd(&d);

    delete Buf;

    if (FormatInternToExtern(fractal.fractal(), fractal.view()) != 0) {
        sprintf_s(Error, maxErrorLen, "Strange: Error in view struct!");
        return -1;
    }
    return thisver >= ver ? 0 : -128;
}

int UpdateQUATChunk(png_internal_struct* internal, int actx, int acty) {
    char s[41];
    unsigned char* Buf;
    long QuatPos;

    /* Back to beginning */
    fflush(internal->png);
    PosOverIHDR(internal);

    /* search until quAt chunk found */
    while (!(checkChunkType(internal, quat_chunk_label) || checkChunkType(internal, image_end_label))) {
        GetNextChunk(internal);
    }
    /* No quAt chunk? */
    if (checkChunkType(internal, image_end_label)) {
        return -3;
    }
    sprintf_s(s, sizeof(s), "%s %s", PROGNAME, PROGVERSION);
    QuatPos = ftell(internal->png) - 8;
    Buf = new unsigned char[internal->length];
    /* No memory? Bad! */
    if (Buf == NULL) return -4;
    ReadChunkData(internal, Buf);
    if (strncmp((char*)Buf, s, strlen(s)) != 0) {
        delete Buf;
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
    fflush(internal->png);
    fseek(internal->png, QuatPos, SEEK_SET);
    if (WriteChunk(internal, Buf)) {
        return -1;
    }
    fflush(internal->png);
    delete [] Buf;

    return 0;
}

int PNGEnd(png_internal_struct* png_internal,
    unsigned char* buf,
    int actx, int acty) {
    unsigned char dummy;

    for (unsigned int i = acty; i < png_internal->height; i++) {
        WritePNGLine(png_internal, buf);
    }

    EndIDAT(png_internal);
    png_internal->length = 0;
    setChunkType(png_internal, image_end_label);
    if (WriteChunk(png_internal, &dummy)) {
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

