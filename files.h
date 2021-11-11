#pragma once

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

struct disppal_struct;


/* name: name of INI-file
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
   -1 , if parsing error (Error_Msg says which)
*/
int ParseFile(
    const char* name,
    FractalPreferences& prefs,
    char* Error_Msg,
    size_t maxErrorLen);

enum class ColorMode {
    Indexed,
    RGB
};

/* Wants external format, writes internal to PNG */
int PNGInitialization(
    const char* name,
    ColorMode mode,
    FILE** png,
    png_info_struct* png_info,
    png_internal_struct* png_internal,
    int xstart,
    int ystart,
    long calctime,
    const FractalPreferences& fractal,
    disppal_struct* disppal,
    ZFlag zflag);

/* returns external format, reads internal from PNG */
int ReadParameters(
    char* Error,
    size_t maxErrorLen,
    int* xstart,
    int* ystart,
    png_internal_struct* internal,
    FractalPreferences& fractal);

int UpdateQUATChunk(
    png_internal_struct* internal,
    int actx,
    int acty);

int PNGEnd(
    png_internal_struct* png_internal,
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