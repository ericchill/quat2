/* Quat - A 3D fractal generation program */
/* Copyright (C) 2000 Dirk Meyer */
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

#include <cstdlib> 	// atoi
#include <cstring>	// memset
#include <iostream>

#ifndef NO_NAMESPACE
using namespace std;
#endif

#include <FL/fl_draw.H>
#include <FL/Fl_Int_Input.H>
#include <FL/Fl.H>

#include "common.h"
#include "memory.h"
#include "iter.h"
#include "MandelPreview.h"
#include "CReplacements.h"

#define for4 for(int i=0; i<4; i++)

MandelPreview::MandelPreview(int x, int y, int w, int h, const char *label = 0)
	: Fl_Widget(x, y, w, h, label),
	  _Maxiter(12), _Formula(0), _Bailout(16), _BailoutSQR(256),
	  _cx(0.0), _cy(0.0), _cj(0.0), _ck(0.0),
	  _mandel_x(-1.0), _mandel_y(-1.5),
	  _mandel_lx(4), _mandel_ly(3), _len_x(w-2), _len_y(h-2), _updated(true),
	  _pic_Maxiter(12), _pic_Formula(0), _pic_Bailout(16),
	  _pic_cj(0.0), _pic_ck(0.0)
{
	// Use only that part of the widget area that the maximum
	// rectangle with the correct aspect ration can fill
	if (static_cast<double>(_len_x) / static_cast<double>(_len_y) > _mandel_lx/_mandel_ly) {
		_len_x = static_cast<int>(_len_y * _mandel_lx / _mandel_ly) - 2;
	} else if (static_cast<double>(_len_x) / static_cast<double>(_len_y) < _mandel_lx / _mandel_ly) {
		_len_y = static_cast<int>(_len_x * _mandel_ly / _mandel_lx)-2;
	}

	_pixmap = new unsigned char[_len_x*_len_y];

	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4; ++i) {
			_pic_p[j][i] = _p[j][i] = 0.0;
		}
	}

	C2Coo(_cx, _cy, _CursorX, _CursorY);
}

MandelPreview::~MandelPreview()
{
	delete[] _pixmap;
}

void MandelPreview::draw()
{
	bool cursor = (damage() & 2) != 0;
	if (!isValid()) {
		fl_color(255, 255, 255);
		fl_rectf(x() + 1, y() + 1, _len_x, _len_y);
		fl_color(0, 0, 0);
		fl_rect(x(), y(), _len_x + 2, _len_y + 2);
		fl_draw("invalid", x() + 2, y() + 16);
		fl_draw("Parameters", x() + 2, y() + 30);
	}
	if (!cursor) {
		fl_draw_image_mono(_pixmap, x()+1, y()+1, _len_x, _len_y);
	}
	if (_updated) {
		fl_color(0, 0, 0); 
	} else {
		fl_color(255, 0, 0);
	}
	fl_rect(x(), y(), _len_x+2, _len_y+2);

	fl_push_clip(x()+1, y()+1, _len_x, _len_y);
	// Delete old cursor
	if (cursor)	{
		fl_draw_image_mono(
			_pixmap + (_oldCursorY - 4) * _len_x + _oldCursorX - 4,
			x() + _oldCursorX - 3, y() + _oldCursorY - 3, 7, 7, 1, _len_x);
	}
	// Draw the cursor
	fl_color(255, 0, 0);
	fl_line(x() + _CursorX, y() + _CursorY-1, x() + _CursorX, y() + _CursorY - 3);
	fl_line(x() + _CursorX - 1, y() + _CursorY, x() + _CursorX - 3, y() + _CursorY);
	fl_line(x() + _CursorX, y() + _CursorY+1, x() + _CursorX, y() + _CursorY + 3);
	fl_line(x() + _CursorX+1, y() + _CursorY, x() + _CursorX + 3, y() + _CursorY);
	fl_pop_clip();
}

int MandelPreview::handle(int event)
{
	static bool LButtonPressed = false;
	switch (event) {
	  case FL_ENTER:
	  case FL_LEAVE:
		return 1;
	  case FL_PUSH:
		if (FL_RIGHT_MOUSE == Fl::event_button()) {
			CalcImage();
			redraw();
			CheckUpdate();
			return 1;
		}
		if (FL_LEFT_MOUSE == Fl::event_button()) {
			LButtonPressed = true;
			fl_cursor(FL_CURSOR_NONE);
			int mx = Fl::event_x(), my = Fl::event_y();
			if (x() < mx && mx < x() + _len_x + 1 && y() < my && my < y() + _len_y + 1) {
				_oldCursorX = _CursorX;
				_CursorX = mx - x();
				_oldCursorY = _CursorY;
				_CursorY = my - y();
				double cre, ci;
				Coo2C(_CursorX, _CursorY, cre, ci);
				if (_input_cre != NULL) {
					_input_cre->value(cre);
					_input_cre->do_callback(); 
				}
				if (_input_ci != NULL) {
					_input_ci->value(ci);
					_input_ci->do_callback();
				}
				_cx = cre;
				_cy = ci;
				damage(2);
				return 1;
			} else {
				return 0;
			}
		}
	  case FL_RELEASE:
		if (FL_LEFT_MOUSE == Fl::event_button()) {
			fl_cursor(FL_CURSOR_DEFAULT);
			LButtonPressed = false;
			return 1;
		}
	  case FL_DRAG: {
		  if (!LButtonPressed) {
			  return 0;
		  }
		  int xm = Fl::event_x() - x();
		  int ym = Fl::event_y() - y();
		  if (xm >= 1 && xm < _len_x + 1 && ym >= 1 && ym < _len_y + 1) {
			  fl_cursor(FL_CURSOR_NONE);
		  } else {
			  fl_cursor(FL_CURSOR_DEFAULT);
		  }
		  xm = clamp<int>(xm, 1, _len_x);
		  ym = clamp<int>(ym, 1, _len_y);
		  Coo2C(xm, ym, _cx, _cy);
		  _input_cre->value(_cx);
		  _input_ci->value(_cy);
		  _input_cre->do_callback();
		  _input_ci->do_callback();
		  SetMyCursor(xm, ym);
		  redraw();
		  break;
	  }
	  default: return Fl_Widget::handle(event);
	}
	return Fl_Widget::handle(event);
}

void MandelPreview::MoveLeft()
{
	int x = 0, y = 0;

	_mandel_x -= _mandel_lx / 3.0;
	C2Coo(_cx, _cy, x, y);
	SetMyCursor(x, y);
	CalcImage();
	redraw();
}

void MandelPreview::MoveRight()
{
	int x = 0, y = 0;

	_mandel_x += _mandel_lx / 3.0;
	C2Coo(_cx, _cy, x, y);
	SetMyCursor(x, y);
	CalcImage();
	redraw();
}

void MandelPreview::MoveUp()
{
	int x = 0, y = 0;

	_mandel_y -= _mandel_ly / 3.0;
	C2Coo(_cx, _cy, x, y);
	SetMyCursor(x, y);
	CalcImage();
	redraw();
}

void MandelPreview::MoveDown()
{
	int x = 0, y = 0;

	_mandel_y += _mandel_ly / 3.0;
	C2Coo(_cx, _cy, x, y);
	SetMyCursor(x, y);
	CalcImage();
	redraw();
}

void MandelPreview::ZoomIn()
{
	_mandel_lx /= 2.0;
	_mandel_ly /= 2.0;
	CenterCursor();
	CalcImage();
	redraw();
}

void MandelPreview::ZoomOut()
{

	_mandel_lx *= 2.0;
	_mandel_ly *= 2.0;
	CenterCursor();
	CalcImage();
	redraw();
}

void MandelPreview::CenterCursor()
{
	int x = 0, y = 0;

	_mandel_x = _cx - _mandel_lx / 2.0;
	_mandel_y = _cy - _mandel_ly / 2.0;
	C2Coo(_cx, _cy, x, y);
	SetMyCursor(x, y);
}

void MandelPreview::SetMyCursor(int x, int y)
{
	_oldCursorX = _CursorX;
	_oldCursorY = _CursorY;
	_CursorX = x;
	_CursorY = y;
}

void MandelPreview::C2Coo(double cx, double cy, int& x, int& y)
{
	assert(_mandel_lx != 0.0);
	assert(_mandel_ly != 0.0);
	assert(_len_x != 0.0);
	assert(_len_y != 0.0);
	x = static_cast<int>((cx - _mandel_x) / _mandel_lx * _len_x + 0.5);
	y = static_cast<int>((cy - _mandel_y) / _mandel_ly * _len_y + 0.5);
}

void MandelPreview::Coo2C(int x, int y, double& cx, double& cy)
{
	assert(_len_x != 0.0);
	assert(_len_y != 0.0);
	cx = _mandel_x + static_cast<double>(x) / _len_x * _mandel_lx;
	cy = _mandel_y + static_cast<double>(y) / _len_y * _mandel_ly;
}

int MandelPreview::CalcMPixel(int x, int y)
{
	int (*iter) (struct iter_struct*);
	static struct iter_struct is;
	LexicallyScopedPtr<Quat> orbit = new Quat[_Maxiter+1];

	is.xstart = 0;

	switch (_Formula) {
	  case 0: 
		  iter = iterate_0;
		break;
	  case 1:
		  iter = iterate_1;
		  break;
	  case 2: 
		  iter = iterate_2;
		  is.xstart = 0.367879441;
		  break;
	  case 3: 
		  iter = iterate_3;
		  break;
	  case 4: iter = iterate_4;
		  is.p[0] = _p[0];
		  break;
	  case 5:
		  iter = iterate_bulb;
		  break;
	  default:
		  throw std::invalid_argument("Invalid formula number");
	}
	
	Coo2C(x, y, is.c[0], is.c[1]);
	is.c[2] = _cj;
	is.c[3] = _ck;
	is.bailout = _BailoutSQR;
	is.maxiter = _Maxiter;
	is.exactiter = 0;
	is.orbit = &orbit[0];
	
	return iter(&is) == _Maxiter ? 1 : 0;
}

void MandelPreview::CalcImage()
{
	if (!isValid()) {
		return;
	}
	fl_cursor(FL_CURSOR_WAIT);
	memset(_pixmap, 255, _len_x * _len_y);
	for (int yz = 0; yz < _len_y; yz++) {
		for (int x = 0; x < _len_x; x++) {
			if (CalcMPixel(x + 1, yz + 1)) {
				_pixmap[x + _len_x * yz] = 0;
			}
		}
	}
	_pic_cj = _cj;
	_pic_ck = _ck;
	_pic_Maxiter = _Maxiter;
	_pic_Bailout = _Bailout;
	_pic_Formula = _Formula;
	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4; ++i) {
			_pic_p[j][i] = _p[j][i];
		}
	}
	fl_cursor(FL_CURSOR_DEFAULT);
	_updated = true;
}

void MandelPreview::CheckUpdate()
{
	_updated = !(
		_pic_cj != _cj
		|| _pic_ck != _ck 
		|| _pic_Maxiter != _Maxiter
		|| _pic_Bailout != _Bailout
		|| _pic_Formula != _Formula);
	if (_updated) {
		for (int j = 0; j < 4; ++j) {
			for (int i = 0; i < 4; ++i) {
				if (_p[j][i] != _pic_p[j][i]) {
					_updated = false;
				}
			}
		}
	}
	damage(1);
	if (nullptr != _button_update) {
		if (_updated) {
			_button_update->deactivate();
		} else {
			_button_update->activate();
		}
	}
}

bool MandelPreview::isValid()
{
	return (_Maxiter > 0) && (_Bailout >= 0.0) && (_Formula >= 0) && (_Formula <= 5);
}
