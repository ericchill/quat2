#ifndef QUAT_KERNEL_QUAT_H
#define QUAT_KERNEL_QUAT_H 1

#include "common.h"
#include "qmath.h" 
#include "parameters.h"

/*#include "png.h"*/


/* DUMMY function prototypes */
int DUMMY_Initialize(std::ostream& errorMsg, int x, int y);
int DUMMY_Done();
int DUMMY_getline(unsigned char* line, int y, long xres, ZFlag whichbuf);
int DUMMY_check_event(void);
int DUMMY_Change_Name(const char* s);
void DUMMY_eol(int line);

int TranslateColorFormula(std::ostream& errorMsg, const char* colscheme);
int formatExternToIntern(FractalSpec& frac, FractalView& view);
int formatInternToExtern(FractalSpec& frac, FractalView& view);

class LinePutter {
public:
    virtual int putLine(long x1, long x2, long xres, int Y, unsigned char* Buf, bool useZBuf) = 0;
    virtual void eol(int) = 0;
};

/* Creates image from given parameters. Wants external format of frac & view */
int CreateImage(
    std::ostream& errorMsg,
    int* xstart, int* ystart,
    FractalPreferences& prefs,
    ZFlag zflag,
    LinePutter& lineDst);

/* Creates ZBuffer from given parameters. Wants external format of
   frac & view
*/
int CreateZBuf(
    std::ostream& errorMsg,
    int* xstart,
    int* ystart,
    FractalPreferences& fractal,
    LinePutter& lineDst);

int InitGraphics(
    std::ostream& errorMsg,
    FractalPreferences& fractal,
    bool ready,
    int* xadd, int* yadd, 
    bool useZBuf);

/* pngfile: string of basename, without path (only for title bar) */
/* png: _opened_ png filename */
/* png filename returns _closed_ */
/*      and initialized (png_info, png_internal) */
/* if png==NULL, no filename is written */
/* also closes graphics, resets keyboard mode */
/* wants external format for light-source and bailout */
/* zflag: decides what to calculate,
      0..image from scratch, size view.xres*view.yres
      1..ZBuffer from scratch, size view.xres*view.yres*AA^2
      2..image from ZBuffer, size img: xres*yres, buffer *AA^2
      for every case take into account that images can be stereo! */
class PNGFile;

int CalculateFractal(
    std::ostream& errorMsg,
    char* pngfile,
    FILE** png,
    PNGFile* png_internal,
    ZFlag zflag,
    int* xstart, int* ystart,
    /* int xadd, int yadd, */
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
#define LASTORBIT (GlobalOrbit[0][2])


#endif /* QUAT_KERNEL_QUAT_H */
