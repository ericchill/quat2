/* Quat - A 3D fractal generation program */
/* Copyright (C) 1997-2002 Dirk Meyer */
/* (email: dirk.meyer@studserv.uni-stuttgart.de) */
/* mail:  Dirk Meyer */
/*        Marbacher Weg 29 */
/*        D-71334 Waiblingen */
/*        Germany */
/* */
/* This program is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU General Public License */
/* as published by the Free Software Foundation; either version 2 */
/* of the License, or (at your option) any later version. */
/* */
/* This program is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/* GNU General Public License for more details. */
/* */
/* You should have received a copy of the GNU General Public License */
/* along with this program; if not, write to the Free Software */
/* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA. */


#include "CReplacements.h"
#include "memory.h"

#include <cstdlib>	// atof
#include <new>

#include <sstream>
#include <iostream>

#include "MainWindow.h"
#include "colors.h"
#include "quat.h"
#include "files.h"
#include "quatfiles.h"
#include "ScrollWid.h"
#include "PixWid.h"
#include "about.h"
#include "ParameterEditor.h"
#include "WriteIni.h"
#include "title.xpm"
#include "ImageWid.h"

#pragma warning(push, 0)
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/fl_draw.H>
#include <FL/fl_message.H>
#include <FL/fl_file_chooser.H>
#include <FL/Fl_Help_Dialog.H>
#include <FL/x.H>
#pragma warning(pop)

#ifdef WIN32
#include "resources.h"
#elif __unix__
#include "icon.xbm"
#endif


MainWindow* MainWindow::mainWindowPtr = nullptr;

static char* decygwinify(char* filename) {
    constexpr char cygpfx[] = "/cygdrive/";
    size_t cygpfxlen = strlen(cygpfx);

    if (0 == strncmp(filename, cygpfx, cygpfxlen)) {
        /* Turn /cygdrive/a/ into a:/ */
        filename = filename + cygpfxlen - 1;
        filename[0] = filename[1];
        filename[1] = ':';  // instead of '/'
    }
    return filename;
}

static char* main_file_chooser(
    const char* message,
    const char* pat,
    const char* fname,
    int relative = 0) {

    char* filename = fl_file_chooser(message, pat, fname, relative);
    if (filename != nullptr) {
        char* fixedfilename = decygwinify(filename);
        return fixedfilename;
    } else {
        return filename;
    }
}

void MainWindow::Image_Open_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->Image_Open();
}

void MainWindow::Image_Close_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->Image_Close();
}

void MainWindow::Image_Save_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->Image_Save();
}

void MainWindow::Image_SaveAs_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->Image_SaveAs();
}

void MainWindow::Help_About_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->Help_About();
}

void MainWindow::Image_AdjustWindow_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->Image_AdjustWindow();
}

void MainWindow::Image_Exit_cb(Fl_Widget*, void* v) {
    if (FL_SHORTCUT == Fl::event() && FL_Escape == Fl::event_key()) {
        return; // ignore Escape
    }
    reinterpret_cast<MainWindow*>(v)->Image_Exit();
}

void MainWindow::Calculation_StartImage_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->Calculation_StartImage();
}

void MainWindow::Calculation_StartZBuf_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->Calculation_StartZBuf();
}

void MainWindow::Calculation_Stop_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->_stop = true;
}

void MainWindow::Parameters_Edit_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->Parameters_Edit();
}

void MainWindow::Parameters_Reset_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->Parameters_Reset();
}

void MainWindow::Parameters_ReadINI_cb(Fl_Widget*, void* v) {
    MainWindow* p = reinterpret_cast<MainWindow*>(v);
    p->Parameters_ReadINI(p->_fractal);
}

void MainWindow::Parameters_ReadPNG_cb(Fl_Widget*, void* v) {
    MainWindow* p = reinterpret_cast<MainWindow*>(v);
    p->Parameters_ReadPNG(p->_fractal, p->_zBufInMem);
}

void MainWindow::Parameters_SaveAs_cb(Fl_Widget*, void* v) {
    MainWindow* p = reinterpret_cast<MainWindow*>(v);
    p->Parameters_SaveAs(p->_fractal);
}

void MainWindow::ZBuffer_Open_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->ZBuffer_Open();
}

void MainWindow::ZBuffer_Close_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->ZBuffer_Close();
}

void MainWindow::ZBuffer_SaveAs_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->ZBuffer_SaveAs();
}

void MainWindow::Help_Manual_cb(Fl_Widget*, void* v) {
    reinterpret_cast<MainWindow*>(v)->Help_Manual();
}

int switch_callback(int argc, char* argv[], int& i) {
    // Ugly method to get parameters without global variables...
    static int paramidx[] = { -1 };
    static char* param[] = { nullptr };

    if (argc < 0) {
        int idx = -argc - 1;
        assert(0 == idx);
        if (-1 == paramidx[idx]) {
            return -1;
        }
        argv[0] = param[idx];
        return 0;
    }

    // regular callback
    std::string s(argv[i]);

    // Seems not to be a filename, but an option?
    if ('-' == s[0]) {
        return 0;
    }

    // File already given?
    if (paramidx[0] != -1) {
        return 0;
    }

    paramidx[0] = i;
    param[0] = argv[i++];

    return 1;
}

MainWindow::MainWindow(int argc, char** argv, int w, int h, const char* label)
    : _autoResize(false),
    _minSizeX(680), _minSizeY(530), 
    _imgYStart(0),
    _zbufYStart(0), _stop(false),
    _imgInMem(false), _zBufInMem(false), _imgChanged(false), _zBufChanged(false),
    _imgReady(false), _zBufReady(false), _inCalc(false),
    _pix(0), _help(new Fl_Help_Dialog), _zBuf(0),
    _actFile("Noname.png"), _iniPath("./"), _pngPath("./"),
    _statusText_cstr(0) {

    assert(nullptr == mainWindowPtr);
    mainWindowPtr = this;

#ifdef WIN32
    SetSlash('\\');
#endif
    _statusText << "Dummy."; _statusText.seekp(0);
    Fl_Menu_Item menuItems[] = {
    {"&Image", FL_ALT + 'f', 0, 0, FL_SUBMENU, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"&Open...", 0, Image_Open_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"&Close", 0, Image_Close_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"&Save", 0, Image_Save_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"Save &As...", 0, Image_SaveAs_cb, this, FL_MENU_DIVIDER, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"Adjust &Window...", 0, Image_AdjustWindow_cb, this, FL_MENU_DIVIDER, FL_NORMAL_LABEL,
     FL_HELVETICA, 12},
    {"E&xit...", 0, Image_Exit_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {"&Calculation", FL_ALT + 'c', 0, 0, FL_SUBMENU, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"Start / Resume an &image", 0, Calculation_StartImage_cb, this, 0, FL_NORMAL_LABEL,
     FL_HELVETICA, 12},
    {"Start / Resume a &ZBuffer", 0, Calculation_StartZBuf_cb, this, 0, FL_NORMAL_LABEL,
     FL_HELVETICA, 12},
    {"Stop", 0, Calculation_Stop_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {"&Parameters", FL_ALT + 'p', 0, 0, FL_SUBMENU, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"&Edit...", 0, Parameters_Edit_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"&Reset", 0, Parameters_Reset_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"Read from &INI...", 0, Parameters_ReadINI_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"Read from &PNG...", 0, Parameters_ReadPNG_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"Save &As...", 0, Parameters_SaveAs_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {"&ZBuffer", FL_ALT + 'f', 0, 0, FL_SUBMENU, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"&Open...", 0, ZBuffer_Open_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"&Close", 0, ZBuffer_Close_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"Save &As...", 0, ZBuffer_SaveAs_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {"&Help", FL_ALT + 'h', 0, 0, FL_SUBMENU, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"&Manual", 0, Help_Manual_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {"A&bout...", 0, Help_About_cb, this, 0, FL_NORMAL_LABEL, FL_HELVETICA, 12},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0} };

    _mainWin = new Fl_Double_Window(w, h, label);
    _mainWin->callback(MainWindow::Image_Exit_cb, this);
    _mainWin->size_range(_minSizeX, _minSizeY);
    _mainWin->box(FL_NO_BOX);
    _status = new Fl_Box(0, _mainWin->h() - 20, _mainWin->w(), 20, "Status");
    _status->box(FL_UP_BOX);
    _status->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
    _status->labelsize(12);
    _scroll = new ScrollWid(0, 25, _mainWin->w(), _mainWin->h() - 25 - _status->h());
    MakeTitle();
    _scroll->end();
    _menubar = new Fl_Menu_Bar(0, 0, _mainWin->w(), 25);
    _menubar->copy(menuItems);
    _mainWin->end();
    _mainWin->resizable(_scroll);

    int sw_i = 0;
    if (Fl::args(argc, argv, sw_i, switch_callback) < argc) {
        std::cout << "Quat options:" << std::endl <<
            "  filename.[png|zpn|ini|col] Open file." << std::endl <<
            "FLTK options:" << std::endl << Fl::help << std::endl;
    }

    _actFile.auto_name();
    static std::string title;
    title = PROGNAME; title += " - "; title += _actFile.filename();
    _mainWin->label(title.c_str());
    update();
    Image_AdjustWindow();
#ifdef WIN32
    _mainWin->icon((char*)LoadIcon(fl_display, MAKEINTRESOURCE(IDI_ICON1)));
#elif __unix__
    Fl::w();	// Just to open the display.
    Pixmap p = XCreateBitmapFromData(fl_display, DefaultRootWindow(fl_display),
        (char*)icon_bits, icon_width, icon_height);

    MainWin->icon((char*)p);
#endif

    pathname helpfile;
#ifdef DOCDIR
    helpfile = DOCDIR "/quat-us.html";
#else
    helpfile = "/quat-us.html";
#endif
    pathname prgpath(argv[0]);
    prgpath = prgpath.path();

    if (!helpfile.exists()) {
        constexpr char const* search[] = { "../doc/quat-us.html",
                     "doc/quat-us.html", "quat-us.html",
                     "../share/doc/quat/quat-us.html",
                     "../doc/quat/quat-us.html",
                     "../share/quat/quat-us.html" };
        int no = sizeof(search) / sizeof(const char*);
        int idx = 0;
        do {
            helpfile = prgpath + search[idx];
            ++idx;
        } while (!helpfile.exists() && idx < no);
    }
#ifdef DOCDIR
    // Simply to let Fl_Help_Dialog output an errorMsg message
    // with the *most important* path.
    if (!helpfile.exists()) helpfile = DOCDIR "/quat-us.html";
#endif

    _help->load(helpfile.c_str());
    _mainWin->show(argc, argv);

    char* param[1];
    if (0 == switch_callback(-1, param, sw_i)) {
        pathname tmppath(param[0]);
        pathname tmp_upper(tmppath);
        tmp_upper.uppercase();
        if (tmp_upper.ext() == ".PNG") {
            DoImgOpen(tmppath.c_str(), ZFlag::NewImage);
        } else if (tmp_upper.ext() == ".ZPN") {
            DoImgOpen(tmppath.c_str(), ZFlag::NewZBuffer);
        } else if (tmp_upper.ext() == ".INI" || tmppath.ext() == ".OBJ" || tmppath.ext() == ".COL") {
            Parameters_ReadINI(_fractal, tmppath.c_str());
        } else {
            DoImgOpen(tmppath.c_str(), ZFlag::NewImage);
        }
        update();
    }
}

MainWindow::~MainWindow() {
    delete _mainWin;
    delete _help;
    mainWindowPtr = nullptr;
    operator delete[](_zBuf, std::nothrow);
    delete[] _statusText_cstr;
}

void MainWindow::MakeTitle() {
    int wp = 0, hp = 0;
    int suc = fl_measure_pixmap(titlePixmap, wp, hp);
    if (suc) {
        _pix = new PixWid(0, 25, wp, hp);
        reinterpret_cast<PixWid*>(_pix)->setPixmap(const_cast<char* const*>(titlePixmap));
    } else {
        _pix = new Fl_Button(0, 0, 300, 200, "There was a problem\nanalyzing the XPM image.");
    }
    _scroll->widget(_pix);
}

bool MainWindow::shown() const {
    return _mainWin->shown();
}

void MainWindow::DoImgOpen(const char* givenfile, ZFlag zflag) {
    std::stringstream errorMsg;
    const char* filename;

    _statusText.seekp(0);
    if (nullptr == givenfile) {
        if (zflag == ZFlag::NewImage) {
            filename = main_file_chooser("Open Image", "Images (*.{png,PNG,Png})", _pngPath.c_str());
        } else {
            filename = main_file_chooser("Open ZBuffer", "ZBuffers (*.{zpn,ZPN,Zpn})", _pngPath.c_str());
        }
        if (!filename) {
            return;
        }
        _pngPath = pathname(filename).path();
    } else {
        filename = givenfile;
    }
    if (0 == filename[0]) {
        return;
    }

    FractalPreferences maybeFractal;
    if (ReadParametersPNG(errorMsg, filename, maybeFractal)) {
        fl_alert("%s", errorMsg.str().c_str());
        if (zflag == ZFlag::NewImage) {
            _imgChanged = false; 
        } else {
            _zBufChanged = false;
        }
        return;
    }
    _fractal = maybeFractal;

    int xres = _fractal.view().renderedXRes();
    int yres = _fractal.view().renderedYRes();
    if (DoInitMem(errorMsg, xres, yres, zflag)) {
        fl_alert("%s", errorMsg.str().c_str());
        return;
    }
    int ys;
    bool ready;
    if (ReadParametersAndImage(*this, errorMsg, filename, &ready, &ys, _fractal, zflag)) {
        fl_alert("%s", errorMsg.str().c_str());
        if (ZFlag::NewImage == zflag) {
            _imgChanged = false;
        } else {
            _zBufChanged = false;
        }
        return;
    }
    if (ZFlag::NewImage == zflag) {
        static std::string title;
        _actFile = filename;
        _imgReady = ready;
        _imgChanged = false;
        _imgInMem = true;
        _imgYStart = ys;
        title = PROGNAME;
        title += " - ";
        title += _actFile.filename();
        _mainWin->label(title.c_str());
    } else {
        _zBufReady = ready;
        _zBufChanged = false;
        _zBufInMem = true;
        _zbufYStart = ys;
    }
    _mainWin->redraw();
    pathname tmp(filename);
    if ((zflag == ZFlag::NewImage && _imgReady) || (zflag != ZFlag::NewImage && _zBufReady)) {
        _statusText.seekp(0);
        _statusText << tmp.filename() << ": Finished. Created by Quat "
            << errorMsg.str() << ".";
    } else {
        _statusText.seekp(0);
        _statusText << tmp.filename() << ": In line " << ys
            << ", created by Quat " << errorMsg.str() << ".";
    }
    if (atof(errorMsg.str().c_str()) > atof(PROGVERSION)) {
        fl_message("Warning: File created by a higher version of Quat!\n"
            "Unpredictable if it will work...");
    }
}

int MainWindow::DoInitMem(std::ostream& errorMsg, int xres, int yres, ZFlag zflag) {
    // Maximum size of Fl_Widget is 32767...
    if (xres <= 0 || yres <= 0 || xres >= 32768 || yres >= 32768) {
        errorMsg << "Resolution out of range. The GUI version (currently "
            "running)\nhas a limit of 32,767.";
        return -1;
    }
    ImageWid* pixtmp = new ImageWid(0, 25, xres, yres);

    if (!pixtmp->valid()) {
        if (_zBufInMem) {
            errorMsg << "Couldn't create image (probably not enough memory.)"
                " Try to close the ZBuffer and calculate directly.";
        } else {
            errorMsg << "Couldn't create image (probably not enough memory.)";
        }
        delete pixtmp;
        return -1;
    }

    pixtmp->gray(150);
    _scroll->widget(pixtmp);
    _scroll->redraw();
    _pix = pixtmp;

    if (zflag != ZFlag::NewImage) {
        _zBuf = new (std::nothrow) unsigned char[xres * yres * 3];
        if (nullptr == _zBuf) {
            errorMsg << "Couldn't allocate memory for ZBuffer.";
            MakeTitle();
            return -1;
        }
    }
    return 0;
}

void MainWindow::DoStartCalc(ZFlag zflag) {
    _zflag = zflag;
    _inCalc = true;
    update();
}

void MainWindow::Image_Open() {
    _mainWin->cursor(FL_CURSOR_WAIT);
    _menubar->deactivate();
    DoImgOpen(nullptr, ZFlag::NewImage);
    update();
    _menubar->activate();
    _mainWin->cursor(FL_CURSOR_DEFAULT);
}

void MainWindow::Image_Close() {
    assert(_imgInMem);
    _statusText.seekp(0);
    if (_imgChanged)
        if (fl_choice("Image: There are unsaved modifications. Do you really want to close?",
            "No", "Yes", nullptr) == 0) {
            update();
            return;
        }
    if (_zBufInMem) {
        _mainWin->cursor(FL_CURSOR_WAIT);
        int picx = _fractal.view().renderedXRes();
        int picy = _fractal.view().renderedYRes();
        ImageWid* pixtmp = new ImageWid(0, 25, picx, picy);
        if (!pixtmp->valid()) {
            fl_alert("Couldn't create ZBuffer pixmap (probably not enough memory.)");
            delete pixtmp;
            update();
            return;
        }
        _pix = reinterpret_cast<Fl_Widget*>(pixtmp);
        for (int j = 0; j < picy; j++) {
            for (int i = 0; i < picx; i++) {
                float l = static_cast<float>(threeBytesToLong(&_zBuf[3 * (i + j * picx)]));
                unsigned char R = 255 - static_cast<unsigned char>
                    (2.55 * l / static_cast<float>(_fractal.view()._zres));
                pixtmp->set_pixel(i, j, R, R, R);
            }
        }
        _scroll->widget(_pix);
        if (_autoResize) {
            Image_AdjustWindow();
        }
        _mainWin->cursor(FL_CURSOR_DEFAULT);
    } else {
        MakeTitle();
    }
    _imgInMem = false; _imgReady = false; _imgChanged = false;
    _imgYStart = 0;

    _actFile.auto_name();
    static std::string title;
    title = PROGNAME;
    title += " - "; 
    title += _actFile.filename();
    _mainWin->label(title.c_str());

    if (!_zBufInMem && _autoResize) {
        Image_AdjustWindow();
    }
    _mainWin->redraw();

    update();
}

void MainWindow::Image_Save() {
    std::stringstream error;
    _mainWin->cursor(FL_CURSOR_WAIT);

    _statusText.seekp(0);
    if (SavePNG(*this, error, _actFile.c_str(), 0, _imgYStart, _fractal, ZFlag::NewImage) != 0) {
        fl_alert("%s", error.str().c_str());
        _mainWin->cursor(FL_CURSOR_DEFAULT);
    } else {
        _mainWin->cursor(FL_CURSOR_DEFAULT);
        _imgChanged = false;
        _statusText.seekp(0);
        _statusText << "Image \""
            << _actFile.c_str()
            << "\" written successfully.";
    }
    update();
}

void MainWindow::Image_SaveAs() {
    std::stringstream error;
    _statusText.seekp(0);
    const char* fn = main_file_chooser("Save Image As",
        "Images (*.{png,PNG,Png})",
        _actFile.c_str());
    if (nullptr == fn) {
        update();
        return;
    }
    _pngPath = pathname(fn).path();

    fprintf(stderr, "fn = \"%s\"\n", fn);
    fprintf(stderr, "png_path = \"%s\"\n", _pngPath.c_str());

    pathname file(fn);
    file.ext(".png");
    if (file.exists()) {
        if (fl_choice("Selected file already exists. Overwrite?", "No", "Yes", nullptr) == 0) {
            update();
            return;
        }
    }

    _mainWin->cursor(FL_CURSOR_WAIT);
    if (SavePNG(*this, error, file.c_str(), 0, _imgYStart, _fractal, ZFlag::NewImage) != 0) {
        fl_alert("%s", error.str().c_str());
        _mainWin->cursor(FL_CURSOR_DEFAULT);
        update();
        return;
    }
    _mainWin->cursor(FL_CURSOR_DEFAULT);
    _imgChanged = false;

    _actFile = file;
    static std::string title;
    title = PROGNAME; title += " - "; title += _actFile.filename();
    _mainWin->label(title.c_str());
    _statusText.seekp(0);
    _statusText << "Image '" << file.c_str() << "' written successfully.";
    update();
}

void MainWindow::Image_AdjustWindow() {
    int w = _pix->w(), 
        h = _pix->h() + _menubar->h() + _status->h();
    w = std::max<int>(w, _minSizeX);
    h = std::max<int>(h, _minSizeY);
    w = std::min<int>(w, Fl::w());
    h = std::min<int>(h, Fl::h());
    _mainWin->size(w, h);
    _statusText.seekp(0);
    update();
}

void MainWindow::Help_About() {
    static std::string a;
    AboutBox* about = new AboutBox();
    a = PROGNAME;
    a += " "; a += PROGVERSION; a += PROGSUBVERSION; a += PROGSTATE;
    about->state->label(a.c_str());
    about->note->label(COMMENT);
    about->run();
    delete about;

    _statusText.seekp(0);
    update();
}

void MainWindow::Image_Exit() {
    if (_inCalc) {
        if (fl_choice("Render in progress. Do you really want to exit?",
            "No", "Yes", nullptr) == 0) {
            return;
        }
    }
    if (_imgChanged || _zBufChanged) {
        if (fl_choice("There are unsaved modifications. Do you really want to exit?",
            "No", "Yes", nullptr) == 0) {
            return;
        }
    }
    _help->hide();
    _mainWin->hide();
}

void MainWindow::Help_Manual() {
    _help->show();
}

void MainWindow::Calculation_StartImage() {
    std::stringstream errorMsg;
    ZFlag zflag;
    int xres, i;

    if (_zBufInMem) {
        zflag = ZFlag::ImageFromZBuffer;
    } else {
        zflag = ZFlag::NewImage;
    }
    DoStartCalc(zflag);
    xres = _fractal.view()._xres;
    if (_fractal.view().isStereo()) {
        xres *= 2;
    }
    try {
        if (!_imgInMem && DoInitMem(errorMsg, xres, _fractal.view()._yres, ZFlag::NewImage)) {
            fl_alert("%s", errorMsg.str().c_str());
            _inCalc = false;
            _statusText.seekp(0);
            update();
            return;
        }
        i = CreateImage(*this, errorMsg, &_imgYStart, _fractal, zflag);
        if (0 == i || -2 == i || -128 == i) {
            _imgInMem = true;
            _imgReady = false;
            _imgChanged = true;
        }
        if (_imgYStart == _fractal.view()._yres) {
            _imgReady = true;
        }
    } catch (std::exception& ex) {
        _imgReady = false;
        errorMsg << "Caught exception: " << ex.what();
        i = -128;
    }
    _inCalc = false;
    if (i != -2) {
        if (errorMsg.str().size() != 0) {
            fl_alert("%s", errorMsg.str().c_str());
        }
        _statusText.seekp(0);
        update();
    }
    _statusText.seekp(0);
    update();
}

void MainWindow::Calculation_StartZBuf() {
    std::stringstream errorMsg;
    assert(!_imgInMem);
    int xres = _fractal.view().renderedXRes();
    int yres = _fractal.view().renderedYRes();
    if (!_zBufInMem) {
        if (DoInitMem(errorMsg, xres, yres, ZFlag::NewZBuffer)) {
            fl_alert("%s", errorMsg.str().c_str());
            _statusText.seekp(0);
            update();
            return;
        }
    }
    DoStartCalc(ZFlag::NewZBuffer);
    _zBufChanged = true;
    int i = CreateZBuf(*this, errorMsg, &_zbufYStart, _fractal);
    if (0 == i || -2 == i || -128 == i) {
        _zBufInMem = true;
    }
    if (_zbufYStart == _fractal.view()._yres) {
        _zBufReady = true;
    }
    if (i != -2) {
        if (errorMsg.str().size() != 0) {
            fl_alert("%s", errorMsg.str().c_str());
        }
        _inCalc = false;
    }
    _statusText.seekp(0);
    update();
    return;
}

void MainWindow::Parameters_Edit() {
    ParameterEditor* editor = new ParameterEditor();
    editor->set(_fractal);

    if (_imgInMem) {
        editor->SetState(1);
    } else if (_zBufInMem) {
        editor->SetState(2);
    }

    _menubar->deactivate();
    if (editor->run()) {
        editor->get(_fractal);
    }
    delete editor;
    _menubar->activate();

    _statusText.seekp(0);
    update();
}

void MainWindow::Parameters_Reset() {
    if (fl_choice("This will reset all parameters to their default values. Continue?",
        "No", "Yes", nullptr)) {
        _fractal.reset();
    }
    _statusText.seekp(0);
    update();
}

bool MainWindow::Parameters_ReadINI(
    FractalPreferences& fractal,
    const char* fn) {
    std::stringstream errorMsg;

    _statusText.seekp(0);
    char* filename;
    if (fn != nullptr) {
        filename = new char[strlen(fn) + 1];
        strcpy_s(filename, strlen(fn) + 1, fn);
    } else {
        filename =
            main_file_chooser("Read parameters from INI file", "INI files (*.{ini,col,obj,INI,COL,OBJ,Ini,Col,Obj})", _iniPath.c_str());
    }
    if (nullptr == filename) {
        update();
        return false;
    }
    _iniPath = pathname(filename).path();
    int reset = fl_choice("Do you want to start from default parameters? (Recommended.)",
        "No", "Yes", nullptr);

    FractalPreferences maybeFractal;
    if (!reset) {
        maybeFractal = fractal;
    }
    int i = ParseINI(errorMsg, filename, maybeFractal);
    if (i != 0) {
        fl_alert("%s", errorMsg.str().c_str());
        update();
        return false;
    }
    if (_zBufInMem) {
        fractal.view() = maybeFractal.view();
        fractal.realPalette() = maybeFractal.realPalette();
        fractal.colorScheme() = maybeFractal.colorScheme();
        _statusText.seekp(0);
        _statusText << "Read only the modifyable parameters for a Z Buffer.";
        update();
        return true;
    }
    fractal = maybeFractal;
    _statusText.seekp(0);
    if (reset) {
        _statusText << "Parameters read successfully.";
    } else {
        _statusText << "Parameters added to current ones.";
    }
    update();
    return true;
}

bool MainWindow::Parameters_ReadPNG(
    FractalPreferences& fractal, 
    bool useZBuf) {
    std::stringstream errorMsg;

    _statusText.seekp(0);
    const char* fn =
        main_file_chooser("Import parameters from PNG file", "Images (*.{png,PNG,Png})", _pngPath.c_str());
    if (nullptr == fn) {
        update();
        return false;
    }

    _pngPath = pathname(fn).path();
    FractalPreferences maybeFractal = fractal;
    int i = ReadParametersPNG(errorMsg, fn, maybeFractal);
    if (i != 0) {
        fl_alert("%s", errorMsg.str().c_str());
        update();
        return false;
    }
    if (useZBuf) {
        fractal.view() = maybeFractal.view();
        fractal.realPalette() = maybeFractal.realPalette();
        fractal.colorScheme() = maybeFractal.colorScheme();
        _statusText.seekp(0);
        _statusText << "Imported only the modifyable parameters for a Z Buffer.";
        update();
        return true;
    }
    _fractal = maybeFractal;
    _statusText.seekp(0);
    _statusText << "Parameters imported successfully.";
    return true;
}

void MainWindow::Parameters_SaveAs(const FractalPreferences& fractal) {
    std::stringstream errorMsg;

    int mode;
    _statusText.seekp(0);
    if (!WriteINI(mode)) {
        update();
        return;
    }
    std::string canonical_ext(".ini");
    if (PS_OBJ == mode) {
        canonical_ext = ".obj";
    } else if (PS_COL == mode) {
        canonical_ext = ".col";
    }

    std::string ininame(_iniPath);
    ininame += _actFile.basename();
    ininame += canonical_ext;

    std::string filter("INI files (*");
    filter += canonical_ext;
    filter += ")";
    char* file = main_file_chooser("Write parameters to INI file",
        filter.c_str(),
        ininame.c_str());
    if (nullptr == file) {
        update();
        return;
    }
    _iniPath = pathname(file).path();

    pathname fn(file);
    fn.ext(canonical_ext.c_str());
    if (fn.exists()) {
        char strbuf[BUFSIZ];
        sprintf_s(strbuf, sizeof(strbuf), "Selected file '%s' already exists. Overwrite?", fn.c_str());
        if (0 == fl_choice(strbuf, "No", "Yes", nullptr)) {
            update();
            return;
        }
    }
    _statusText.seekp(0);
    if (WriteINI(errorMsg, fn.c_str(), fractal, mode)) {
        fl_alert("Couldn't write file '%s',\nError = \"%s\".", fn.c_str(), errorMsg.str().c_str());
    } else {
        _statusText << "File '" << fn.c_str() << "' was written successfully.";
    }
    update();
}

void MainWindow::ZBuffer_Open() {
    _mainWin->cursor(FL_CURSOR_WAIT);
    _menubar->deactivate();
    DoImgOpen(nullptr, ZFlag::NewZBuffer);
    update();
    _menubar->activate();
    _mainWin->cursor(FL_CURSOR_DEFAULT);
}

void MainWindow::ZBuffer_Close() {
    assert(_zBufInMem);
    assert(!_imgInMem);
    _statusText.seekp(0);
    if (_zBufChanged && fl_choice("ZBuffer: There are unsaved modifications. Do you want to close?", "No", "Yes", nullptr) == 0) {
        update();
    }
    operator delete[](_zBuf, std::nothrow);
    _zBuf = nullptr;
    _zBufInMem = false;
    _zBufChanged = false;
    _zBufReady = false;
    _zbufYStart = 0;
    MakeTitle();
    if (_autoResize) {
        Image_AdjustWindow();
    }
    update();
    _mainWin->redraw();
}

void MainWindow::ZBuffer_SaveAs() {
    std::stringstream error;

    _statusText.seekp(0);

    pathname zpnname(_pngPath);
    zpnname += _actFile.basename();
    zpnname += ".zpn";

    const char* fn = main_file_chooser("Save ZBuffer As", "ZBuffers (*.{zpn,ZPN,Zpn})", zpnname.c_str());
    if (nullptr == fn) {
        update();
        return;
    }
    _pngPath = pathname(fn).path();

    pathname filename(fn);
    filename.ext(".zpn");
    if (filename.exists() && fl_choice("Selected file already exists. Overwrite?", "No", "Yes", nullptr)) {
        update();
        return;
    }
    _mainWin->cursor(FL_CURSOR_WAIT);
    if (SavePNG(*this, error, filename.c_str(), 0, _zbufYStart, _fractal, ZFlag::NewZBuffer) != 0) {
        fl_alert("%s", error.str().c_str());
        _mainWin->cursor(FL_CURSOR_DEFAULT);
        update();
        return;
    }
    _mainWin->cursor(FL_CURSOR_DEFAULT);
    _zBufChanged = false;
    _statusText.seekp(0);
    _statusText << "ZBuffer '" << filename << "' was written successfully.";
    update();
}

void MainWindow::menuItemEnabled(int i, bool b) {
    if (!b) {
        _menubar->mode(i, _menubar->mode(i) | FL_MENU_INACTIVE);
    } else {
        _menubar->mode(i, _menubar->mode(i) & ~FL_MENU_INACTIVE);
    }
}

void MainWindow::update() {
    assert(_actFile.length() > 0);
    // Image = 0
    menuItemEnabled(1, !_imgInMem && !_zBufInMem && !_inCalc);
    menuItemEnabled(2, _imgInMem && !_inCalc);
    menuItemEnabled(3, _imgInMem && _imgChanged && !_inCalc /*&& act_file[0]!=0*/);
    menuItemEnabled(4, _imgInMem && !_inCalc);
    menuItemEnabled(6, !_inCalc);
    // Calculation = 8
    menuItemEnabled(9, !_imgReady && (_zBufReady || !_zBufInMem) && !_inCalc);
    menuItemEnabled(10, !_zBufReady && !_imgInMem && !_inCalc);
    menuItemEnabled(11, _inCalc);
    // Parameters = 13
    menuItemEnabled(14, !_inCalc);
    menuItemEnabled(15, !_zBufInMem && !_imgInMem && !_inCalc);
    menuItemEnabled(16, !_imgInMem && !_inCalc);
    menuItemEnabled(17, !_imgInMem && !_inCalc);
    menuItemEnabled(18, !_inCalc);
    // ZBuffer = 20
    menuItemEnabled(21, !_imgInMem && !_zBufInMem && !_inCalc);
    menuItemEnabled(22, !_imgInMem && _zBufInMem && !_inCalc);
    menuItemEnabled(23, _zBufInMem && !_inCalc);
    // Help = 25

    if (!_inCalc) {
        if (0 == _statusText.str().length()) {
            _statusText << "Image: ";
            if (_imgInMem && !_imgReady) {
                _statusText << "stopped.";
            } else if (_imgReady && _imgChanged) {
                _statusText << "ready.";
            } else if (_imgReady && !_imgChanged) {
                _statusText << "saved.";
            } else {
                _statusText << "none.";
            }
            _statusText << " ZBuffer: ";
            if (_zBufInMem && !_zBufReady) {
                _statusText << "stopped.";
            } else if (_zBufInMem && _zBufChanged) {
                _statusText << "ready.";
            } else if (_zBufInMem && !_zBufChanged) {
                _statusText << "saved.";
            } else {
                _statusText << "none.";
            }
        }
        _statusText << std::ends;
        delete[] _statusText_cstr;
        size_t new_str_len = _statusText.str().length() + 1;
        _statusText_cstr = new char[new_str_len];
        strcpy_s(_statusText_cstr, new_str_len, _statusText.str().c_str());
        _status->label(_statusText_cstr); 
        _status->redraw();
    }
}

void MainWindow::initialize(std::ostream& /*errorMsg*/, int x, int y) {
#ifndef NDEBUG
    ImageWid* p = reinterpret_cast<ImageWid*>(_pix);
    assert(p != nullptr);
    assert(p->w() == x && p->h() == y);
#else
    std::ignore = x;
    std::ignore = y;
#endif
    if (_autoResize) {
        Image_AdjustWindow();
    }
}


void MainWindow::getline(unsigned char* line, int y, long xres, ZFlag whichbuf) {
    switch (whichbuf) {
    case ZFlag::NewImage:
        reinterpret_cast<ImageWid*>(_pix)->get_line(y, line);
        break;
    case ZFlag::NewZBuffer:
        memcpy(line, _zBuf + y * xres * 3L, xres * 3L);
        break;
    case ZFlag::ImageFromZBuffer:
        break;
    }
}

void MainWindow::putLine(long x1, long x2, long xres, int Y, unsigned char* Buf, bool useZBuf) {
    if (useZBuf) {
        memcpy(_zBuf + 3L * (Y * xres + x1), Buf + 3L * x1, (x2 - x1 + 1) * 3L);
        for (long i = x1; i < x2; i++) {
            float l = static_cast<float>(threeBytesToLong(&Buf[3 * i]));
            unsigned char R = 255 - static_cast<unsigned char>(
                2.55 * l / static_cast<float>(_fractal.view()._zres));
            reinterpret_cast<ImageWid*>(_pix)->set_pixel(i, Y, R, R, R);
        }
    } else {
        reinterpret_cast<ImageWid*>(_pix)->set_line_segment(x1, x2, Y, Buf + x1 * 3);
    }
}

bool MainWindow::checkEvent() {
    Fl::check();
    if (_stop) {
        _stop = false;
        return true;
    }
    return false;
}

void MainWindow::changeName(const char* s) {
    _status->label(s);
    _status->redraw();
}

void MainWindow::FLTK_Debug(const char* s) {
    fl_message("Quat - Debug: %s", s);
}

void MainWindow::eol(int line) {
    if (shouldCalculateDepths(_zflag)) {
        std::ostringstream s;
        s << (int)line << " lines calculated." << std::ends;
        changeName(s.str().c_str());
    }
}
