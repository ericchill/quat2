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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "CReplacements.h"
#include "memory.h"
#include "ver.h"

//#include <cassert>
#include <ctime>	// time_t
#include <cstdlib>	// atof
#include <new>
#ifndef WIN32
#	include <sys/time.h>
#endif

#include <sstream>
#include <iostream>

#ifndef NO_NAMESPACE
using namespace std;
#endif

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

#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/fl_draw.H>
#include <FL/fl_message.H>
#include <FL/fl_file_chooser.H>
#include <FL/Fl_Help_Dialog.H>
#include <FL/x.H>
#ifdef WIN32
#include "resources.h"
#elif __unix__
#include "icon.xbm"
#endif

// Those ugly global variables are neccessary for communication with C source
MainWindow* MainWinPtr = nullptr;
char Error[1024];
time_t old_time = 0;

static char* decygwinify(char* filename) {
    static const char cygpfx[] = "/cygdrive/";
    size_t cygpfxlen = strlen(cygpfx);

    if (0 == strncmp(filename, cygpfx, cygpfxlen)) {
        /* Turn /cygdrive/a/ into a:/ */
        filename = filename + cygpfxlen - 1;
        filename[0] = filename[1];
        filename[1] = ':';  // instead of '/'
    }
    return filename;
}

static char* main_file_chooser(const char* message,
    const char* pat,
    const char* fname,
    int relative = 0) {
    char* filename;

    filename = fl_file_chooser(message, pat, fname, relative);
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
    reinterpret_cast<MainWindow*>(v)->stop = true;
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
    p->Parameters_ReadPNG(p->_fractal, p->ZBufInMem);
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
        assert(idx == 0);
        if (-1 == paramidx[idx]) {
            return -1;
        }
        argv[0] = param[idx];
        return 0;
    }

    // regular callback
    string s(argv[i]);

    // Seems not to be _a file, but an option?
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
    : auto_resize(false),
    minsizeX(680), minsizeY(530), 
    imgxstart(0), imgystart(0),
    zbufxstart(0), zbufystart(0), stop(false),
    ImgInMem(false), ZBufInMem(false), ImgChanged(false), ZBufChanged(false),
    ImgReady(false), ZBufReady(false), InCalc(false),
    pix(0), help(new Fl_Help_Dialog), ZBuf(0),
    act_file("Noname.png"), ini_path("./"), png_path("./"),
    _status_text_char(0) {
    assert(MainWinPtr == 0);
    MainWinPtr = this;
#ifdef WIN32
    SetSlash('\\');
#endif
    status_text << "Dummy."; status_text.seekp(0);
    Fl_Menu_Item tmp[] = {
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

    MainWin = new Fl_Double_Window(w, h, label);
    MainWin->callback(MainWindow::Image_Exit_cb, this);
    MainWin->size_range(minsizeX, minsizeY);
    MainWin->box(FL_NO_BOX);
    status = new Fl_Box(0, MainWin->h() - 20, MainWin->w(), 20, "Status");
    status->box(FL_UP_BOX);
    status->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
    status->labelsize(12);
    scroll = new ScrollWid(0, 25, /*320*/MainWin->w(), MainWin->h() - 25 - status->h());
    MakeTitle();
    scroll->end();
    menubar = new Fl_Menu_Bar(0, 0, MainWin->w(), 25);
    menubar->copy(tmp);
    MainWin->end();
    MainWin->resizable(scroll);

    int sw_i = 0;
    if (Fl::args(argc, argv, sw_i, switch_callback) < argc) {
        cout << "Quat options:" << endl <<
            "  filename.[png|zpn|ini|col] Open file." << endl <<
            "FLTK options:" << endl << Fl::help << endl;
    }

    act_file.auto_name();
    static string title;
    title = PROGNAME; title += " - "; title += act_file.file();
    MainWin->label(title.c_str());
    update();
    Image_AdjustWindow();
#ifdef WIN32
    MainWin->icon((char*)LoadIcon(fl_display, MAKEINTRESOURCE(IDI_ICON1)));
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
        const char* search[] = { "../doc/quat-us.html",
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
    // Simply to let Fl_Help_Dialog output an error message
    // with the *most important* path.
    if (!helpfile.exists()) helpfile = DOCDIR "/quat-us.html";
#endif

    help->load(helpfile.c_str());
    MainWin->show(argc, argv);

    char* param[1];
    if (0 == switch_callback(-1, param, sw_i)) {
        pathname tmp(param[0]);
        pathname tmp_upper(tmp);
        tmp_upper.uppercase();
        if (tmp_upper.ext() == ".PNG") {
            DoImgOpen(tmp.c_str(), ZFlag::NewImage);
        } else if (tmp_upper.ext() == ".ZPN") {
            DoImgOpen(tmp.c_str(), ZFlag::NewZBuffer);
        } else if (tmp_upper.ext() == ".INI" || tmp.ext() == ".OBJ" || tmp.ext() == ".COL") {
            Parameters_ReadINI(_fractal, tmp.c_str());
        } else {
            DoImgOpen(tmp.c_str(), ZFlag::NewImage);
        }
        update();
    }
}

MainWindow::~MainWindow() {
    delete MainWin;
    delete help;
    MainWinPtr = 0;
    operator delete[](ZBuf, nothrow);
    delete[] _status_text_char;
}

void MainWindow::MakeTitle() {
    int wp = 0, hp = 0;
    int suc = fl_measure_pixmap(title, wp, hp);
    if (suc) {
        pix = new PixWid(0, 25, wp, hp);
        reinterpret_cast<PixWid*>(pix)->setPixmap(const_cast<char* const*>(title));
    } else {
        pix = new Fl_Button(0, 0, 300, 200, "There was a problem\nanalyzing the XPM image.");
    }
    scroll->widget(pix);
}

bool MainWindow::shown() const {
    return MainWin->shown();
}

void MainWindow::DoImgOpen(const char* givenfile, ZFlag zflag) {
    Error[0] = 0;
    const char* filename;

    status_text.seekp(0);
    if (nullptr == givenfile) {
        if (zflag == ZFlag::NewImage) {
            filename = main_file_chooser("Open Image", "Images (*.{png,PNG,Png})", png_path.c_str());
        } else {
            filename = main_file_chooser("Open ZBuffer", "ZBuffers (*.{zpn,ZPN,Zpn})", png_path.c_str());
        }
        if (!filename) {
            return;
        }
        png_path = pathname(filename).path();
    } else {
        filename = givenfile;
    }
    if (0 == filename[0]) {
        return;
    }

    Initialize = FLTK_Initialize;
    Done = FLTK_Done;
    QU_getline = FLTK_getline;
    check_event = FLTK_check_event;
    Change_Name = FLTK_Change_Name;
    FractalPreferences maybeFractal;
    if (ReadParametersPNG(filename, Error, sizeof(Error), maybeFractal)) {
        fl_alert("%s", Error);
        if (zflag == ZFlag::NewImage) {
            ImgChanged = false; 
        } else {
            ZBufChanged = false;
        }
        return;
    }
    _fractal = maybeFractal;

    int xres = _fractal.view().renderedXRes();
    int yres = _fractal.view().renderedYRes();
    string error;
    if (DoInitMem(xres, yres, error, zflag)) {
        fl_alert("%s", error.c_str());
        return;
    }
    int xs, ys;
    bool ready;
    if (ReadParametersAndImage(Error, sizeof(Error), filename, &ready, &xs, &ys,
        _fractal, zflag, *this)) {
        fl_alert("%s", Error);
        if (ZFlag::NewImage == zflag) {
            ImgChanged = false;
        } else {
            ZBufChanged = false;
        }
        return;
    }
    if (ZFlag::NewImage == zflag) {
        static string title;
        act_file = filename;
        ImgReady = ready;
        ImgChanged = false;
        ImgInMem = true;
        imgxstart = xs;
        imgystart = ys;
        title = PROGNAME;
        title += " - ";
        title += act_file.file();
        MainWin->label(title.c_str());
    } else {
        ZBufReady = (ready != 0);
        ZBufChanged = false;
        ZBufInMem = true;
        zbufxstart = xs;
        zbufystart = ys;
    }
    MainWin->redraw();
    pathname tmp(filename);
    if ((zflag == ZFlag::NewImage && ImgReady) || (zflag != ZFlag::NewImage && ZBufReady)) {
        status_text.seekp(0);
        status_text << tmp.file() << ": Finished. Created by Quat "
            << Error << ".";
    } else {
        status_text.seekp(0);
        status_text << tmp.file() << ": In line " << ys
            << ", created by Quat " << Error << ".";
    }
    if (atof(Error) > atof(PROGVERSION)) {
        fl_message("Warning: File created by a higher version of Quat!\n"
            "Unpredictable if it will work...");
    }
    return;
}

int MainWindow::DoInitMem(int xres, int yres, string& error, ZFlag zflag) {
    // Maximum size of Fl_Widget is 32767...
    if (xres <= 0 || yres <= 0 || xres >= 32768 || yres >= 32768) {
        error = "Resolution out of range. The GUI version (currently "
            "running)\nhas a limit of 32,767.";
        return -1;
    }
    ImageWid* pixtmp = new ImageWid(0, 25, xres, yres);

    if (!pixtmp->valid()) {
        if (ZBufInMem) {
            error = "Couldn't create image (probably not enough memory.)"
                " Try to close the ZBuffer and calculate directly.";
        } else {
            error = "Couldn't create image (probably not enough memory.)";
        }
        delete pixtmp;
        return -1;
    }

    pixtmp->white();
    scroll->widget(pixtmp);
    scroll->redraw();
    pix = pixtmp;

    if (zflag != ZFlag::NewImage) {
        ZBuf = new (nothrow) unsigned char[static_cast<long>(xres) * static_cast<long>(yres) * 3L];
        if (nullptr == ZBuf) {
            error = "Couldn't allocate memory for ZBuffer.";
            MakeTitle();
            return -1;
        }
    }
    return 0;
}

void MainWindow::DoStartCalc(ZFlag zflag) {
    Initialize = FLTK_Initialize;
    Done = FLTK_Done;
    QU_getline = FLTK_getline;
    check_event = FLTK_check_event;
    Change_Name = FLTK_Change_Name;
    _zflag = zflag;
    InCalc = true;
    update();
}

void MainWindow::Image_Open() {
    MainWin->cursor(FL_CURSOR_WAIT);
    menubar->deactivate();
    DoImgOpen(nullptr, ZFlag::NewImage);
    update();
    menubar->activate();
    MainWin->cursor(FL_CURSOR_DEFAULT);
}

int MainWindow::Image_Close() {
    assert(ImgInMem);
    status_text.seekp(0);
    if (ImgChanged)
        if (fl_choice("Image: There are unsaved modifications. Do you really want to close?",
            "No", "Yes", nullptr) == 0) {
            update();
            return 0;
        }
    if (ZBufInMem) {
        MainWin->cursor(FL_CURSOR_WAIT);
        int picx = _fractal.view().renderedXRes();
        int picy = _fractal.view().renderedYRes();
        ImageWid* pixtmp = new ImageWid(0, 25, picx, picy);
        if (!pixtmp->valid()) {
            fl_alert("Couldn't create ZBuffer pixmap (probably not enough memory.)");
            delete pixtmp;
            update();
            return 1;
        }
        pix = reinterpret_cast<Fl_Widget*>(pixtmp);
        for (int j = 0; j < picy; j++) {
            for (int i = 0; i < picx; i++) {
                float l = static_cast<float>(threeBytesToLong(&ZBuf[3 * (i + j * picx)]));
                unsigned char R = 255 - static_cast<unsigned char>
                    (2.55 * l / static_cast<float>(_fractal.view()._zres));
                pixtmp->set_pixel(i, j, R, R, R);
            }
        }
        scroll->widget(pix);
        if (auto_resize) Image_AdjustWindow();
        MainWin->cursor(FL_CURSOR_DEFAULT);
    } else {
        MakeTitle();
    }
    ImgInMem = false; ImgReady = false; ImgChanged = false;
    calc_time = old_time;
    imgxstart = 0; 
    imgystart = 0;

    act_file.auto_name();
    static string title = PROGNAME;
    title += " - "; 
    title += act_file.file();
    MainWin->label(title.c_str());

    if (!ZBufInMem && auto_resize) {
        Image_AdjustWindow();
    }
    MainWin->redraw();

    update();
    return 1;
}

void MainWindow::Image_Save() {
    Error[0] = '\0';
    MainWin->cursor(FL_CURSOR_WAIT);

    status_text.seekp(0);
    if (SavePNG(Error, sizeof(Error), act_file.c_str(), 0, imgystart, nullptr, _fractal, ZFlag::NewImage) != 0) {
        fl_alert("%s", Error);
        MainWin->cursor(FL_CURSOR_DEFAULT);
        update();
        return;
    }
    MainWin->cursor(FL_CURSOR_DEFAULT);
    ImgChanged = false;
    status_text.seekp(0);
    status_text << "Image '"
        << act_file.c_str()
        << "' written successfully.";
    update();
    return;
}

void MainWindow::Image_SaveAs() {
    Error[0] = '\0';

    status_text.seekp(0);
    const char* fn = main_file_chooser("Save Image As",
        "Images (*.{png,PNG,Png})",
        act_file.c_str());
    if (nullptr == fn) {
        update();
        return;
    }
    png_path = pathname(fn).path();

    fprintf(stderr, "fn = \"%s\"\n", fn);
    fprintf(stderr, "png_path = \"%s\"\n", png_path.c_str());

    pathname file(fn);
    file.ext(".png");
    if (file.exists()) {
        if (fl_choice("Selected file already exists. Overwrite?", "No", "Yes", nullptr) == 0) {
            update();
            return;
        }
    }

    MainWin->cursor(FL_CURSOR_WAIT);
    if (SavePNG(Error, sizeof(Error), file.c_str(), 0, imgystart, nullptr, _fractal, ZFlag::NewImage) != 0) {
        fl_alert("%s", Error);
        MainWin->cursor(FL_CURSOR_DEFAULT);
        update();
        return;
    }
    MainWin->cursor(FL_CURSOR_DEFAULT);
    ImgChanged = false;

    act_file = file;
    static string title;
    title = PROGNAME; title += " - "; title += act_file.file();
    MainWin->label(title.c_str());

    status_text.seekp(0);
    status_text << "Image '" << file.c_str() << "' written successfully.";

    update();
    return;
}

void MainWindow::Image_AdjustWindow() {
    int w = pix->w(), 
        h = pix->h() + menubar->h() + status->h();
    w = std::max<int>(w, minsizeX);
    h = std::max<int>(h, minsizeY);
    w = std::min<int>(w, Fl::w());
    h = std::min<int>(h, Fl::h());
    MainWin->size(w, h);

    status_text.seekp(0);
    update();
}

void MainWindow::Help_About() {
    static string a;
    AboutBox* about = new AboutBox();
    a = PROGNAME;
    a += " "; a += PROGVERSION; a += PROGSUBVERSION; a += PROGSTATE;
    about->state->label(a.c_str());
    about->note->label(COMMENT);
    about->run();
    delete about;

    status_text.seekp(0);
    update();
}

void MainWindow::Image_Exit() {
    if (InCalc) {
        if (fl_choice("Render in progress. Do you really want to exit?",
            "No", "Yes", nullptr) == 0) {
            return;
        }
    }
    if (ImgChanged || ZBufChanged) {
        if (fl_choice("There are unsaved modifications. Do you really want to exit?",
            "No", "Yes", nullptr) == 0) {
            return;
        }
    }
    help->hide();
    MainWin->hide();
}

void MainWindow::Help_Manual() {
    help->show();
}

void MainWindow::Calculation_StartImage() {
    ZFlag zflag;
    int xres, i;

    if (ZBufInMem) {
        zflag = ZFlag::ImageFromZBuffer;
        old_time = calc_time;
    } else {
        zflag = ZFlag::NewImage;
        old_time = 0;
    }
    DoStartCalc(zflag);
    xres = _fractal.view()._xres;
    if (_fractal.view().isStereo()) {
        xres *= 2;
    }

    string error;
    if (!ImgInMem && DoInitMem(xres, _fractal.view()._yres, error, ZFlag::NewImage)) {
        fl_alert("%s", error.c_str());
        InCalc = false;
        status_text.seekp(0);
        update();
        return;
    }
    Error[0] = '\0';
#ifdef WIN32
    const int pixels_per_check = 8;
#else
    const int pixels_per_check = 1;
#endif
    i = CreateImage(Error, sizeof(Error), &imgxstart, &imgystart, _fractal, pixels_per_check, zflag, *this);
    if (0 == i || -2 == i || -128 == i) {
        ImgInMem = true;
        ImgReady = false;
        ImgChanged = true;
    }
    if (imgystart == _fractal.view()._yres) {
        ImgReady = true;
    }
    InCalc = false;
    if (i != -2) {
        if (Error[0] != 0) {
            fl_alert("%s", Error);
        }
        status_text.seekp(0);
        update();
    }
    status_text.seekp(0);
    update();
    return;
}

void MainWindow::Calculation_StartZBuf() {
    string error;
    assert(!ImgInMem);
    Error[0] = 0;
    int xres = _fractal.view().renderedXRes();
    int yres = _fractal.view().renderedYRes();
    if (!ZBufInMem) {
        if (DoInitMem(xres, yres, error, ZFlag::NewZBuffer)) {
            fl_alert("%s", error.c_str());
            status_text.seekp(0);
            update();
            return;
        }
    }
    DoStartCalc(ZFlag::NewZBuffer);
    ZBufChanged = true;
    int i = CreateZBuf(Error, sizeof(Error), &zbufxstart, &zbufystart, _fractal, /* pixelsPerCheck */ 8, *this);
    if (0 == i || -2 == i || -128 == i) {
        ZBufInMem = true;
    }
    if (zbufystart == _fractal.view()._yres) {
        ZBufReady = true;
    }
    if (i != -2) {
        if (Error[0] != 0)
            fl_alert("%s", Error);
        InCalc = false;
    }
    status_text.seekp(0);
    update();
    return;
}

void MainWindow::Parameters_Edit() {
    ParameterEditor* editor = new ParameterEditor();
    editor->set(_fractal);

    if (ImgInMem) {
        editor->SetState(1);
    } else if (ZBufInMem) {
        editor->SetState(2);
    }

    menubar->deactivate();
    if (editor->run()) {
        editor->get(_fractal);
    }
    delete editor;
    menubar->activate();

    status_text.seekp(0);
    update();
}

void MainWindow::Parameters_Reset() {
    if (fl_choice("This will reset all parameters to their default values. Continue?",
        "No", "Yes", nullptr)) {
        _fractal.reset();
    }
    status_text.seekp(0);
    update();
}

bool MainWindow::Parameters_ReadINI(
    FractalPreferences& fractal,
    const char* fn) {
    Error[0] = '\0';

    status_text.seekp(0);
    char* filename;
    if (fn != nullptr) {
        filename = new char[strlen(fn) + 1];
        strcpy_s(filename, strlen(fn) + 1, fn);
    } else {
        filename =
            main_file_chooser("Read parameters from INI file", "INI files (*.{ini,col,obj,INI,COL,OBJ,Ini,Col,Obj})", ini_path.c_str());
    }
    if (nullptr == filename) {
        update();
        return false;
    }
    ini_path = pathname(filename).path();
    int reset = fl_choice("Do you want to start from default parameters? (Recommended.)",
        "No", "Yes", nullptr);

    FractalPreferences maybeFractal;
    if (!reset) {
        maybeFractal = fractal;
    }
    int i = ParseINI(filename, Error, sizeof(Error), maybeFractal);
    if (i != 0) {
        fl_alert("%s", Error);
        update();
        return false;
    }
    if (ZBufInMem) {
        fractal.view() = maybeFractal.view();
        fractal.realPalette() = maybeFractal.realPalette();
        fractal.colorScheme() = maybeFractal.colorScheme();
        status_text.seekp(0);
        status_text << "Read only the modifyable parameters for a Z Buffer.";
        update();
        return true;
    }
    fractal = maybeFractal;
    status_text.seekp(0);
    if (reset) {
        status_text << "Parameters read successfully.";
    } else {
        status_text << "Parameters added to current ones.";
    }
    update();
    return true;
}

bool MainWindow::Parameters_ReadPNG(
    FractalPreferences& fractal, 
    bool useZBuf) {
    Error[0] = '\0';

    status_text.seekp(0);
    const char* fn =
        main_file_chooser("Import parameters from PNG file", "Images (*.{png,PNG,Png})", png_path.c_str());
    if (nullptr == fn) {
        update();
        return false;
    }

    png_path = pathname(fn).path();
    FractalPreferences maybeFractal = fractal;
    int i = ReadParametersPNG(fn, Error, sizeof(Error), maybeFractal);
    if (i != 0) {
        fl_alert("%s", Error);
        update();
        return false;
    }
    if (useZBuf) {
        fractal.view() = maybeFractal.view();
        fractal.realPalette() = maybeFractal.realPalette();
        fractal.colorScheme() = maybeFractal.colorScheme();
        status_text.seekp(0);
        status_text << "Imported only the modifyable parameters for a Z Buffer.";
        update();
        return true;
    }
    _fractal = maybeFractal;
    status_text.seekp(0);
    status_text << "Parameters imported successfully.";
    return true;
}

void MainWindow::Parameters_SaveAs(const FractalPreferences& fractal) {
    Error[0] = '\0';

    int mode;
    status_text.seekp(0);
    if (!WriteINI(mode)) {
        update();
        return;
    }
    string canonical_ext(".ini");
    if (PS_OBJ == mode) {
        canonical_ext = ".obj";
    } else if (PS_COL == mode) {
        canonical_ext = ".col";
    }

    string ininame(ini_path);
    ininame += act_file.filename();
    ininame += canonical_ext;

    string filter("INI files (*"); filter += canonical_ext;
    filter += ")";
    char* file = main_file_chooser("Write parameters to INI file",
        filter.c_str(),
        ininame.c_str());
    if (nullptr == file) {
        update();
        return;
    }
    ini_path = pathname(file).path();

    pathname fn(file);
    fn.ext(canonical_ext.c_str());
    if (fn.exists()) {
        char strbuf[BUFSIZ];
        sprintf_s(strbuf, sizeof(strbuf), "Selected file '%s' already exists. Overwrite?", fn.c_str());
        if (0 == fl_choice(strbuf, "No", "Yes", NULL)) {
            update();
            return;
        }
    }
    status_text.seekp(0);
    if (WriteINI(Error, sizeof(Error), fn.c_str(), _fractal)) {
        fl_alert("Couldn't write file '%s',\nError = \"%s\".", fn.c_str(), Error);
    } else {
        status_text << "File '" << fn.c_str() << "' was written successfully.";
    }
    update();
}

void MainWindow::ZBuffer_Open() {
    MainWin->cursor(FL_CURSOR_WAIT);
    menubar->deactivate();
    DoImgOpen(nullptr, ZFlag::NewZBuffer);
    update();
    menubar->activate();
    MainWin->cursor(FL_CURSOR_DEFAULT);
}

int MainWindow::ZBuffer_Close() {
    assert(ZBufInMem);
    assert(!ImgInMem);
    status_text.seekp(0);
    if (ZBufChanged && fl_choice("ZBuffer: There are unsaved modifications. Do you want to close?", "No", "Yes", nullptr) == 0) {
        update();
        return 0;
    }
    operator delete[](ZBuf, nothrow);
    ZBuf = nullptr;
    ZBufInMem = false;
    ZBufChanged = false;
    ZBufReady = false;
    calc_time = 0;
    old_time = 0;
    zbufxstart = 0; 
    zbufystart = 0;
    MakeTitle();
    if (auto_resize) {
        Image_AdjustWindow();
    }
    update();
    MainWin->redraw();
    return 1;
}

void MainWindow::ZBuffer_SaveAs() {
    Error[0] = '\0';

    status_text.seekp(0);

    pathname zpnname(png_path);
    zpnname += act_file.filename();
    zpnname += ".zpn";

    const char* fn = main_file_chooser("Save ZBuffer As", "ZBuffers (*.{zpn,ZPN,Zpn})", zpnname.c_str());
    if (nullptr == fn) {
        update();
        return;
    }
    png_path = pathname(fn).path();

    pathname filename(fn);
    filename.ext(".zpn");
    if (filename.exists() && fl_choice("Selected file already exists. Overwrite?", "No", "Yes", nullptr)) {
        update();
        return;
    }
    MainWin->cursor(FL_CURSOR_WAIT);
    if (SavePNG(Error, sizeof(Error), filename.c_str(), 0, zbufystart, NULL, _fractal, ZFlag::NewZBuffer) != 0) {
        fl_alert("%s", Error);
        MainWin->cursor(FL_CURSOR_DEFAULT);
        update();
        return;
    }
    MainWin->cursor(FL_CURSOR_DEFAULT);
    ZBufChanged = false;
    status_text.seekp(0);
    status_text << "ZBuffer '" << filename << "' was written successfully.";
    update();
    return;
}

void MainWindow::menuItemEnabled(int i, bool b) {
    if (!b) {
        menubar->mode(i, menubar->mode(i) | FL_MENU_INACTIVE);
    } else {
        menubar->mode(i, menubar->mode(i) & ~FL_MENU_INACTIVE);
    }
}

void MainWindow::update() {
    assert(act_file.length() > 0);
    // Image = 0
    menuItemEnabled(1, !ImgInMem && !ZBufInMem && !InCalc);
    menuItemEnabled(2, ImgInMem && !InCalc);
    menuItemEnabled(3, ImgInMem && ImgChanged && !InCalc /*&& act_file[0]!=0*/);
    menuItemEnabled(4, ImgInMem && !InCalc);
    menuItemEnabled(6, !InCalc);
    // Calculation = 8
    menuItemEnabled(9, !ImgReady && (ZBufReady || !ZBufInMem) && !InCalc);
    menuItemEnabled(10, !ZBufReady && !ImgInMem && !InCalc);
    menuItemEnabled(11, InCalc);
    // Parameters = 13
    menuItemEnabled(14, !InCalc);
    menuItemEnabled(15, !ZBufInMem && !ImgInMem && !InCalc);
    menuItemEnabled(16, !ImgInMem && !InCalc);
    menuItemEnabled(17, !ImgInMem && !InCalc);
    menuItemEnabled(18, !InCalc);
    // ZBuffer = 20
    menuItemEnabled(21, !ImgInMem && !ZBufInMem && !InCalc);
    menuItemEnabled(22, !ImgInMem && ZBufInMem && !InCalc);
    menuItemEnabled(23, ZBufInMem && !InCalc);
    // Help = 25

    if (!InCalc) {
        if (0 == status_text.str().length()) {
            status_text << "Image: ";
            if (ImgInMem && !ImgReady) {
                status_text << "stopped.";
            } else if (ImgReady && ImgChanged) {
                status_text << "ready.";
            } else if (ImgReady && !ImgChanged) {
                status_text << "saved.";
            } else {
                status_text << "none.";
            }
            status_text << " ZBuffer: ";
            if (ZBufInMem && !ZBufReady) {
                status_text << "stopped.";
            } else if (ZBufInMem && ZBufChanged) {
                status_text << "ready.";
            } else if (ZBufInMem && !ZBufChanged) {
                status_text << "saved.";
            } else {
                status_text << "none.";
            }
        }
        status_text << ends;
        delete[] _status_text_char;
        size_t new_str_len = status_text.str().length() + 1;
        _status_text_char = new char[new_str_len];
        strcpy_s(_status_text_char, new_str_len, status_text.str().c_str());
        status->label(_status_text_char); status->redraw();
    }
}

int MainWindow::FLTK_Initialize(int x, int y, char*) {
    ImageWid* p = reinterpret_cast<ImageWid*>(MainWinPtr->pix);
    assert(p != NULL);
    assert(p->w() == x && p->h() == y);
    if (MainWinPtr->auto_resize) {
        MainWinPtr->Image_AdjustWindow();
    }
    return 0;
}

int MainWindow::FLTK_Done() {
    return 0;
}

int MainWindow::FLTK_getline(unsigned char* line, int y, long xres, ZFlag whichbuf) {
    switch (whichbuf) {
    case ZFlag::NewImage:
        reinterpret_cast<ImageWid*>(MainWinPtr->pix)->get_line(y, line);
        break;
    case ZFlag::NewZBuffer:
        memcpy(line, MainWinPtr->ZBuf + y * xres * 3L, xres * 3L);
        break;
    case ZFlag::ImageFromZBuffer:
        break;
    }
    return 0;
}

int MainWindow::putLine(long x1, long x2, long xres, int Y, unsigned char* Buf, bool useZBuf) {
    if (useZBuf) {
        memcpy(ZBuf + 3L * (Y * xres + x1), Buf + 3L * x1, (x2 - x1 + 1) * 3L);
        for (long i = x1; i <= x2; i++) {
            float l = static_cast<float>(threeBytesToLong(&Buf[3 * i]));
            unsigned char R = 255 - static_cast<unsigned char>(
                2.55 * l / static_cast<float>(_fractal.view()._zres));
            reinterpret_cast<ImageWid*>(MainWinPtr->pix)->set_pixel(i, Y, R, R, R);
        }
    } else {
        reinterpret_cast<ImageWid*>(pix)->set_area(x1, x2, Y, Buf + x1 * 3);
    }
    return 0;
}

int MainWindow::FLTK_check_event() {
    // For Windows, conventional method is faster.
    //#ifdef WIN32
    //	unsigned long newclock = GetTickCount();
    //	static unsigned long prevclock = 0;
    //	double elapsed = (newclock-prevclock)/1000.0;
    //#else   // unix and mac (?)
#ifndef WIN32
    static struct timeval prevclock = { 0L, 0L };
    struct timeval newclock;
    gettimeofday(&newclock, NULL);
    double elapsed = newclock.tv_sec - prevclock.tv_sec +
        (newclock.tv_usec - prevclock.tv_usec) / 1000000.0;
    //#endif
    if (elapsed < 0.05) return 0;

    //#ifdef WIN32
    //	prevclock = newclock;
    //#else   // unix and mac (?)
    prevclock.tv_sec = newclock.tv_sec;
    prevclock.tv_usec = newclock.tv_usec;
#endif

    Fl::check();
    if (MainWinPtr->stop) {
        MainWinPtr->stop = false;
        return -128;
    }
    return 0;
}

int MainWindow::FLTK_Change_Name(const char* s) {
    MainWinPtr->status->label(s); MainWinPtr->status->redraw();
    return 0;
}

void MainWindow::FLTK_Debug(const char* s) {
    fl_message("Quat - Debug: %s", s);
}

void MainWindow::eol(int line) {
    if (shouldCalculateDepths(_zflag)) {
        ostringstream s;
        s << (int)line << " lines calculated." << ends;
        FLTK_Change_Name(s.str().c_str());
    }
}
