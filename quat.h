#ifndef QUAT_KERNEL_QUAT_H
#define QUAT_KERNEL_QUAT_H 1

#include "common.h"
#include "qmath.h" 
#include "parameters.h"

/*#include "png.h"*/


/* DUMMY function prototypes */
int DUMMY_Initialize(int x, int y, char* Error);
int DUMMY_Done();
int DUMMY_getline(unsigned char* line, int y, long xres, ZFlag whichbuf);
int DUMMY_check_event(void);
int DUMMY_Change_Name(const char* s);
void DUMMY_eol(int line);

int ParseAndCalculate(const char* file, char* Error, char zflag);
int TranslateColorFormula(const char* colscheme, char* ErrorMSG, size_t maxErrorLen);
int formatExternToIntern(FractalSpec& frac, FractalView& view);
int formatInternToExtern(FractalSpec& frac, FractalView& view);

class LinePutter {
public:
    virtual int putLine(long x1, long x2, long xres, int Y, unsigned char* Buf, bool useZBuf) = 0;
    virtual void eol(int) = 0;
};

/* Creates image from given parameters. Wants external format of frac & view */
int CreateImage(
    char* Error,
    size_t maxErrorLen,
    int* xstart, int* ystart,
    FractalPreferences& prefs,
    int pixelsPerCheck,
    ZFlag zflag,
    LinePutter& lineDst);

/* Creates ZBuffer from given parameters. Wants external format of
   frac & view
*/
int CreateZBuf(
    char* Error,
    size_t maxErrorLen,
    int* xstart,
    int* ystart,
    FractalPreferences& fractal,
    int pixelsPerCheck,
    LinePutter& lineDst);

int InitGraphics(
    char* Error,
    size_t maxErrorLen,
    FractalPreferences& fractal,
    bool ready,
    int* xadd, int* yadd, 
    bool useZBuf);

/* pngfile: string of filename, without path (only for title bar) */
/* png: _opened_ png file */
/* png file returns _closed_ */
/*      and initialized (png_info, png_internal) */
/* if png==NULL, no file is written */
/* also closes graphics, resets keyboard mode */
/* wants external format for light-source and bailout */
/* zflag: decides what to calculate,
      0..image from scratch, size view.xres*view.yres
      1..ZBuffer from scratch, size view.xres*view.yres*AA^2
      2..image from ZBuffer, size img: xres*yres, buffer *AA^2
      for every case take into account that images can be stereo! */
class PNGFile;

int CalculateFractal(
    char* Error,
    size_t maxErrorLen,
    char* pngfile,
    FILE** png,
    PNGFile* png_internal,
    ZFlag zflag,
    int* xstart, int* ystart,
    /* int xadd, int yadd, */
    int noev,
    ViewBasis* rbase,
    ViewBasis* srbase,
    ViewBasis* lbase,
    ViewBasis* slbase,
    FractalPreferences& fractal,
    LinePutter& lineDst);


extern time_t calc_time;

extern Quat* GlobalOrbit;
#define MAXITER (GlobalOrbit[0][0])
#define LASTITER (GlobalOrbit[0][1])


#endif /* QUAT_KERNEL_QUAT_H */
