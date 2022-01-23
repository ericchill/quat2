#ifndef QUAT_KERNEL_QUAT_H
#define QUAT_KERNEL_QUAT_H 1

#include "common.h"
#include "qmath.h" 
#include "parameters.h"

class Expression;


class Quater {
public:
    virtual void initialize(std::ostream& errorMsg, int x, int y) {
        std::ignore = errorMsg;
        std::ignore = x;
        std::ignore = y;
    }
    virtual void done() {}
    virtual void getline(uint8_t* line, int y, long xres, ZFlag whichBuf) {
        std::ignore = line;
        std::ignore = y;
        std::ignore = xres;
        std::ignore = whichBuf;
    }
    virtual void putLine(long x1, long x2, long xres, int Y, uint8_t* Buf, bool useZBuf) = 0;
    virtual bool checkEvent() { return false; }
    virtual void changeName(const char* s) {
        std::ignore = s;
    }
    virtual void eol(int line) {
        std::ignore = line;
    }
};

/* Creates image from given parameters. Wants external format of frac & view */
int CreateImage(
    Quater& quatDriver,
    std::ostream& errorMsg,
    int* ystart,
    FractalPreferences& prefs,
    ZFlag zflag);

/* Creates ZBuffer from given parameters. Wants external format of
   frac & view
*/
int CreateZBuf(
    Quater& quatDriver,
    std::ostream& errorMsg,
    int* ystart,
    FractalPreferences& fractal);

void InitGraphics(
    Quater& quatDriver,
    std::ostream& errorMsg,
    FractalPreferences& fractal,
    bool useZBuf);

/* pngfile: string of basename, without path (only for title bar) */
/* png: _opened_ png filename */
/* png filename returns _closed_ */
/*      and initialized (png_info, png_internal) */
/* if png==nullptr, no filename is written */
/* also closes graphics, resets keyboard mode */
/* wants external format for light-source and bailout */
/* zflag: decides what to calculate,
      0..image from scratch, size view.xres*view.yres
      1..ZBuffer from scratch, size view.xres*view.yres*AA^2
      2..image from ZBuffer, size img: xres*yres, buffer *AA^2
      for every case take into account that images can be stereo! */
class PNGFile;

int CalculateFractal(
    Quater& quatDriver,
    std::ostream& errorMsg,
    char* pngfile,
    FILE** png,
    PNGFile* png_internal,
    ZFlag zflag,
    int* ystart,
    ViewBasis& rbase,
    ViewBasis& srbase,
    ViewBasis& lbase,
    ViewBasis& slbase,
    FractalPreferences& fractal);


template<typename T>
inline T KLUDGE_PAD(T x) { return x + 10; }


#endif /* QUAT_KERNEL_QUAT_H */
