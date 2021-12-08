#pragma once


#include <sstream>

#ifndef NO_NAMESPACE
//using namespace std;
#endif

#include "common.h"
#include "parameters.h"

#include "CReplacements.h"

#include "quat.h"

class Fl_Window;
class Fl_Widget;
class Fl_Box;
class ScrollWid;
class Fl_Menu_Bar;
class PixWid;
class Fl_Help_Dialog;

class MainWindow : public LinePutter {
public:

    static MainWindow* mainWindowPtr;

    MainWindow(int argc, char** argv, int x = 308, int y = 296, const char* label = PROGNAME);
    ~MainWindow();
    bool shown() const;

    static int FLTK_Initialize(std::ostream& errorMsg, int x, int y);
    static int FLTK_Done();
    static int FLTK_getline(unsigned char* line, int y, long xres, ZFlag whichbuf);
    static int FLTK_check_event();
    static int FLTK_Change_Name(const char* s);
    static void FLTK_Debug(const char* s);

    ZFlag _zflag;
    void eol(int line);
    void FLTK_eol_4(int line);

    int putLine(long x1, long x2, long xres, int y, unsigned char* Buf, bool useZBuf);

    void Image_Open();
    void Image_Close();
    void Image_Save();
    void Image_SaveAs();
    void Image_AdjustWindow();
    void Help_About();
    void Image_Exit();
    void Calculation_StartImage();
    void Calculation_StartZBuf();
    void Parameters_Edit();
    void Parameters_Reset();
    bool Parameters_ReadINI(
        FractalPreferences&,
        const char* fn = NULL);
    bool Parameters_ReadPNG(
        FractalPreferences&,
        bool zbuf);
    void Parameters_SaveAs(const FractalPreferences&);
    void ZBuffer_Open();
    void ZBuffer_Close();
    void ZBuffer_SaveAs();
    void Help_Manual();
    FractalPreferences _fractal;
    bool auto_resize;
private:
    void MakeTitle();
    void DoImgOpen(const char* givenfile, ZFlag zflag);
    int DoInitMem(std::ostream& errorMsg, int xres, int yres, ZFlag zflag);
    void DoStartCalc(ZFlag zflag);
    static void Image_Open_cb(Fl_Widget*, void*);
    static void Image_Close_cb(Fl_Widget*, void*);
    static void Image_Save_cb(Fl_Widget*, void*);
    static void Image_SaveAs_cb(Fl_Widget*, void*);
    static void Image_AdjustWindow_cb(Fl_Widget*, void*);
    static void Help_About_cb(Fl_Widget*, void*);
    static void Image_Exit_cb(Fl_Widget*, void*);
    static void Calculation_StartImage_cb(Fl_Widget*, void*);
    static void Calculation_StartZBuf_cb(Fl_Widget*, void*);
    static void Calculation_Stop_cb(Fl_Widget*, void*);
    static void Parameters_Edit_cb(Fl_Widget*, void*);
    static void Parameters_Reset_cb(Fl_Widget*, void*);
    static void Parameters_ReadINI_cb(Fl_Widget*, void*);
    static void Parameters_ReadPNG_cb(Fl_Widget*, void*);
    static void Parameters_SaveAs_cb(Fl_Widget*, void*);
    static void ZBuffer_Open_cb(Fl_Widget*, void*);
    static void ZBuffer_Close_cb(Fl_Widget*, void*);
    static void ZBuffer_SaveAs_cb(Fl_Widget*, void*);
    static void Help_Manual_cb(Fl_Widget*, void*);
    int minsizeX, minsizeY;
    int imgxstart, imgystart, zbufxstart, zbufystart;
    bool stop;
    void menuItemEnabled(int, bool);
    void update();
    bool ImgInMem, ZBufInMem, ImgChanged, ZBufChanged,
        ImgReady, ZBufReady, InCalc;
    Fl_Window* MainWin;
    std::ostringstream status_text;
    Fl_Box* status;
    ScrollWid* scroll;
    Fl_Widget* pix;
    Fl_Menu_Bar* menubar;
    Fl_Help_Dialog* help;
    unsigned char* ZBuf;
    pathname act_file;
    std::string ini_path, png_path;
    char* _status_text_char;
    //	const unsigned long _type;
};

