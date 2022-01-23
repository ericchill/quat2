#pragma once

#include "parameters.h"

class Quater;

int CalculatePNG(
    Quater& quatDriver,
    std::ostream& errorMsg,
    const char* pngfile1,
    const char* pngfile2,
    const char* ini,
    ZFlag zflag);

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
    Quater& quatDriver,
    std::ostream& errorMsg,
    const char* pngfile,
    bool* ready,
    int* ystart,
    FractalPreferences& fractal,
    ZFlag zflag);

int SavePNG(
    Quater& quatDriver,
    std::ostream& errorMsg,
    const char* pngfile,
    int xstart,
    int ystart,
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

/* define codes for "mode" in WriteINI */
#define PS_OBJ 1
#define PS_VIEW 2
#define PS_COL 4
#define PS_OTHER 8
#define PS_USEFILE 128
#define PS_ALL (-1)

int WriteINI(
    std::ostream& errorMsg,
    const char* file,
    const FractalPreferences& fractal,
    int saveChoices);

int ImgFromZBuf(
    Quater& quatDriver,
    std::ostream& errorMsg,
    const char* file,
    const char* file2);

extern int CleanNumberString(char* s);
