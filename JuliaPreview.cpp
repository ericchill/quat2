/* Quat - A 3D fractal generation program */
/* Copyright (C) 1997-2000 Dirk Meyer */
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

#include "JuliaPreview.h"
#include "CReplacements.h"
#include "quat.h"
#include "MainWindow.h"
#include "ver.h"

#include <FL/Fl.H>
#include <FL/Fl_Button.H>
#include <FL/fl_ask.H>

#include <math.h>
#include <ctime>

extern time_t old_time;

int JuliaPreview::putLine(long x1, long x2, long, int y, unsigned char *Buf, bool)
{
	set_area(x1, x2, y, Buf+x1*3);
	return 0;
}

void JuliaPreview::eol(int)
{}

JuliaPreview::JuliaPreview(int X, int Y, int W, int H, const char *label)
	: ImageWid(X, Y, W, H, label), _updated(false), _stereo(false), _ownView(false),
	_imagestereo(false), _pic_ownView(false)
{
	white();
}

JuliaPreview::~JuliaPreview()
{}

void JuliaPreview::CalcImage3D()
{
	int xs = 0, ys = 0;
	std::stringstream errorMsg;

	Initialize = DUMMY_Initialize;
	Done = DUMMY_Done;
	QU_getline = DUMMY_getline;
	check_event = MainWindow::FLTK_check_event;
	Change_Name = DUMMY_Change_Name;
	_calcFractal.reset();
	for (int j = 0; j < 4; ++j) {
		_calcFractal.fractal()._p[j] = _fractal.fractal()._p[j];
	}
	_calcFractal.view()._LXR = 4.0;
	_calcFractal.realPalette()._cols[0].col1[0] = 1.0;
	_calcFractal.realPalette()._cols[0].col1[1] = 1.0;
	_calcFractal.realPalette()._cols[0].col1[2] = 0.0;

	if (_ownView) {
		_calcFractal = _fractal;
	}
	if (!_stereo) {
		_calcFractal.view()._interocular = 0.0;
	}
	_calcFractal.view()._xres = w(); 
	if (_stereo) {
		_calcFractal.view()._xres /= 2;
	}
	_calcFractal.view()._yres = h();
	_calcFractal.view()._zres = 60;
	_calcFractal.view()._antialiasing = 1;

	button3d->deactivate();

	bool oldstereo = _imagestereo;
	_imagestereo = _calcFractal.view().isStereo();
	old_time = calc_time;
	CreateImage(errorMsg, &xs, &ys, _calcFractal, 80, ZFlag::NewImage, *this);
	calc_time = old_time;
	if (errorMsg.str().size() != 0) {
		_imagestereo = oldstereo;
		fl_alert("%s", errorMsg.str().c_str());
	}

	_pic_ownView = _ownView;
	_picFractal = _calcFractal;

	button3d->activate();
	_updated = true;
}

