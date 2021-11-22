#pragma once

class FractalSpec;
class FractalView;
struct disppal_struct;
class RealPalette;

class FractalPreferences;
class LinePutter;

int CalculatePNG(
    std::ostream& errorMsg,
    const char* pngfile1,
    const char* pngfile2,
    const char* ini,
    ZFlag zflag,
    LinePutter& lineDst);

int ParseINI(
    std::ostream& errorMsg,
    const char* file,
    FractalPreferences& fractal);

int GetParameters(std::ostream& errorMsg, const char* file);

int ReadParametersPNG(
    std::ostream& errorMsg,
    const char* file,
    FractalPreferences& fractal);

int ReadParametersAndImage(
    std::ostream& errorMsg,
    const char* pngfile,
    bool* ready,
    int* xstart,
    int* ystart,
    FractalPreferences& fractal,
    ZFlag zflag,
    LinePutter& lineDst);

int SavePNG(
    std::ostream& errorMsg,
    const char* pngfile,
    int xstart,
    int ystart,
    disppal_struct* disppal,
    const FractalPreferences& fractal,
    ZFlag zflag);

/* Saves the whole QUAT-PNG (retrieved by XXX_getline) */
int BuildName(
    std::ostream& errorMsg,
    char* name, 
    char* namewop,
    size_t maxNameLen,
    const char* ext, 
    const char* file);

int WriteINI(
    std::ostream& errorMsg,
    const char* file,
    const FractalPreferences& fractal);

int ImgFromZBuf(
    std::ostream& errorMsg,
    const char* file,
    const char* file2,
    LinePutter& lineDst);

extern int CleanNumberString(char* s);
