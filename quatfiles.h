#pragma once

class FractalSpec;
class FractalView;
struct disppal_struct;
class RealPalette;

class FractalPreferences;
class LinePutter;

int CalculatePNG(const char* pngfile1,
    const char* pngfile2,
    char* Error,
    size_t maxErrorLen,
    const char* ini,
    ZFlag zflag,
    LinePutter& lineDst);

int ParseINI(const char* file,
    char* Error,
    size_t maxErrorLen,
    FractalPreferences& fractal);

int GetParameters(const char* file, char* Error);

int ReadParametersPNG(const char* file,
    char* Error,
    size_t maxErrorLen,
    FractalPreferences& fractal);

int ReadParametersAndImage(
    char* Error,
    size_t maxErrorLen,
    const char* pngfile,
    unsigned char* ready,
    int* xstart,
    int* ystart,
    FractalPreferences& fractal,
    ZFlag zflag,
    LinePutter& lineDst);

int SavePNG(char* Error,
    size_t maxErrorLen,
    const char* pngfile,
    int xstart,
    int ystart,
    disppal_struct* disppal,
    const FractalPreferences& fractal,
    ZFlag zflag);

/* Saves the whole QUAT-PNG (retrieved by XXX_getline) */
int BuildName(
    char* name, 
    char* namewop,
    size_t maxNameLen,
    const char* ext, 
    const char* file,
    char* Error,
    size_t maxErrorLen);

int WriteINI(
    char* Error,
    size_t maxErrorLen,
    const char* file,
    const FractalPreferences& fractal);

int ImgFromZBuf(
    const char* file,
    const char* file2, 
    char* Error,
    size_t maxErrorLen,
    LinePutter& lineDst);

extern int CleanNumberString(char* s);
