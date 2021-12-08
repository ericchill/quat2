#pragma once

#include <ostream>

#define T_INTEGER 0
#define T_DOUBLE 1
#define T_STRING 2
#define STDDOUBLE(z,i) { z[i].type = T_DOUBLE; z[i].min = -1000000; z[i].max = 1000000; }
#define STDINT(z,i) { z[i].type = T_INTEGER; z[i].min = 0; z[i].max = 65535; }
#define VVPTR (void *)&

#include "png.h"

#include "parameters.h"

#define NO_BUF 0
#define COL_BUF 1
#define CUT_BUF 2

typedef char String_t[256];


/* name: name of INI-filename
   altpath: alternative path to search for files
   kwords: array of keyword_struct, which describes keywords and
           their parameters
   Col|CutBuf: Buffers which data from files is written to according to the
               field "format_string" in keyword_struct
   Col|CutBufLen: maximal number of bytes in buffer
   *Col|*CutBufCount: number of bytes written to buffer.
                      Has to be set to zero on call
   rc: recursion counter. Has to be zero when called.
   Error_MSG: string, Error-message is returned here

   returns:
   0 , no errors
   -1 , if parsing error
*/
int ParseFile(
    std::ostream& errorMsg,
    const char* name,
    FractalPreferences& prefs);

/* Wants external format, writes internal to PNG */
int writeQuatPNGHead(
    const char* name,
    FILE** png,
    PNGFile& png_internal,
    int xstart,
    int ystart,
    long calctime,
    const FractalPreferences& fractal,
    ZFlag zflag);

/* returns external format, reads internal from PNG */
int ReadParameters(
    std::ostream& errorMsg,
    int* xstart,
    int* ystart,
    PNGFile& internal,
    FractalPreferences& fractal);

int updateQUATChunk(
    PNGFile& internal,
    int actx,
    int acty);

int PNGEnd(
    PNGFile& png_internal,
    unsigned char* buf,
    int actx,
    int acty);

/* returns next name in numbering, namewop ... name without path */
int GetNextName(char* nowname, char* namewop, size_t maxNameSize);

void SetSlash(char s);
char GetSlash();
void ConvertPath(char* Path);
void TruncateFileExt(char* filename);
char* FilenameWithoutPath(char* filename);
