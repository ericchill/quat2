#pragma once


#include <sstream>

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

class MainWindow : public Quater {
public:

    static MainWindow* mainWindowPtr;

    MainWindow(int argc, char** argv, int x = 308, int y = 296, const char* label = PROGNAME);
    ~MainWindow();

    bool shown() const;

    void initialize(std::ostream& errorMsg, int x, int y);
    void getline(unsigned char* line, int y, long xres, ZFlag whichbuf);
    bool checkEvent();
    void changeName(const char* s);
    static void FLTK_Debug(const char* s);

    ZFlag _zflag;
    void eol(int line);
    void FLTK_eol_4(int line);

    void putLine(long x1, long x2, long xres, int y, unsigned char* Buf, bool useZBuf);

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
        const char* fn = nullptr);
    bool Parameters_ReadPNG(
        FractalPreferences&,
        bool zbuf);
    void Parameters_SaveAs(const FractalPreferences&);
    void ZBuffer_Open();
    void ZBuffer_Close();
    void ZBuffer_SaveAs();
    void Help_Manual();
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

    void menuItemEnabled(int, bool);
    void update();

    FractalPreferences _fractal;
    int _minSizeX, _minSizeY;
    int _imgYStart, _zbufYStart;
    bool _stop;
    bool _imgInMem, _zBufInMem, _imgChanged, _zBufChanged,
        _imgReady, _zBufReady, _inCalc;
    Fl_Window* _mainWin;
    std::ostringstream _statusText;
    char* _statusText_cstr;
    Fl_Box* _status;
    ScrollWid* _scroll;
    Fl_Widget* _pix;
    Fl_Menu_Bar* _menubar;
    Fl_Help_Dialog* _help;
    uint8_t* _zBuf;
    pathname _actFile;
    std::string _iniPath, _pngPath;
    bool _autoResize;
};

