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

#include "JuliaPreview.h"
#include "CReplacements.h"
#include "quat.h"
#include "MainWindow.h"

#pragma warning(push, 0)
#include <FL/Fl.H>
#include <FL/Fl_Button.H>
#include <FL/fl_ask.H>
#pragma warning(pop)

#include <math.h>
//#include <ctime>

void JuliaPreview::putLine(long x1, long x2, long, int y, uint8_t *lineData, bool)
{
	set_line_segment(x1, x2, y, lineData+x1*3);
}

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
	int ys = 0;
	std::stringstream errorMsg;

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
	try {
		CreateImage(*this, errorMsg, &ys, _calcFractal, ZFlag::NewImage);
		if (errorMsg.str().size() != 0) {
			_imagestereo = oldstereo;
			fl_alert("%s", errorMsg.str().c_str());
		}
	}
	catch (QuatException& ex) {
		_imagestereo = oldstereo;
		fl_alert("%s", ex.what());
	}

	_pic_ownView = _ownView;
	_picFractal = _calcFractal;

	button3d->activate();
	_updated = true;
}

